// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU audit: does the vendored FlashAttention-2 launcher (forced via `prefer_backend`) follow the attention NaN table
//! that our own kernels carry (ROADMAP E0k)?
//!
//! Table: a NaN in a query row gives that row NaN and no other; a NaN attended key or value poisons every row that attends it; a `+inf`
//! score gives a NaN row; a fully masked row (here: a causal row with no visible key, Q > K, bottom-right aligned) is all zeros.
#![cfg(feature = "fa2")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    BackendKind, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan, PlanPreference,
    TensorMut, TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

const D: usize = 64;
const Q: usize = 128;
const K: usize = 128;

#[derive(Clone)]
struct In {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn base() -> In {
    In {
        q: (0..Q * D).map(|i| 0.05 * ((i % 5) as f32 - 2.0)).collect(),
        // Positive keys so an `inf` query row scores +inf on every key.
        k: (0..K * D).map(|i| 0.05 * ((i % 7) as f32 + 1.0)).collect(),
        v: (0..K * D).map(|i| 1.0 + (i % 9) as f32).collect(),
    }
}

macro_rules! suite {
    ($m:ident, $T:ty, $kind:expr) => {
        mod $m {
            use super::*;

            fn run(inp: &In, q_len: usize, k_len: usize, causal: bool) -> Vec<f32> {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let cv = |x: &[f32]| x.iter().map(|v| <$T>::from_f32(*v)).collect::<Vec<$T>>();
                let dq = DeviceBuffer::from_slice(&ctx, &cv(&inp.q[..q_len * D])).unwrap();
                let dk = DeviceBuffer::from_slice(&ctx, &cv(&inp.k[..k_len * D])).unwrap();
                let dv = DeviceBuffer::from_slice(&ctx, &cv(&inp.v[..k_len * D])).unwrap();
                let mut dy: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, q_len * D).unwrap();
                let mut dl: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, q_len).unwrap();
                let (sq, sk, sy, sl) = (
                    [1, 1, q_len as i32, D as i32],
                    [1, 1, k_len as i32, D as i32],
                    [1, 1, q_len as i32, D as i32],
                    [1, 1, q_len as i32],
                );
                let desc = FlashSdpaDescriptor::new(
                    1,
                    1,
                    q_len as i32,
                    k_len as i32,
                    D as i32,
                    D as i32,
                    1.0 / (D as f32).sqrt(),
                    causal,
                    $kind,
                );
                let pref = PlanPreference {
                    prefer_backend: Some(BackendKind::FlashAttentionV2),
                    ..Default::default()
                };
                let plan = FlashSdpaPlan::<$T>::select(&stream, &desc, pref).expect("select");
                assert_eq!(plan.backend(), BackendKind::FlashAttentionV2);
                let mut ws: DeviceBuffer<u8> =
                    DeviceBuffer::zeros(&ctx, plan.workspace_size()).unwrap();
                plan.run(
                    &stream,
                    Workspace::Borrowed(ws.as_slice_mut()),
                    FlashSdpaArgs {
                        q: TensorRef::new(dq.as_slice(), sq, contiguous_stride(sq)).unwrap(),
                        k: TensorRef::new(dk.as_slice(), sk, contiguous_stride(sk)).unwrap(),
                        v: TensorRef::new(dv.as_slice(), sk, contiguous_stride(sk)).unwrap(),
                        y: TensorMut::new(dy.as_slice_mut(), sy, contiguous_stride(sy)).unwrap(),
                        lse: TensorMut::new(dl.as_slice_mut(), sl, contiguous_stride(sl)).unwrap(),
                        mask: None,
                        alibi_slopes: None,
                    },
                )
                .expect("fa2 run");
                stream.synchronize().expect("sync");
                let mut y = vec![<$T>::from_f32(0.0); q_len * D];
                dy.copy_to_host(&mut y).unwrap();
                y.iter().map(|v| v.to_f32()).collect()
            }

            fn row(y: &[f32], r: usize) -> &[f32] {
                &y[r * D..(r + 1) * D]
            }
            fn nan(y: &[f32], r: usize) -> bool {
                row(y, r).iter().all(|v| v.is_nan())
            }
            fn fin(y: &[f32], r: usize) -> bool {
                row(y, r).iter().all(|v| v.is_finite())
            }

            #[test]
            fn clean_baseline_is_finite() {
                let y = run(&base(), Q, K, false);
                assert!((0..Q).all(|r| fin(&y, r)));
            }

            #[test]
            fn nan_query_row_poisons_that_row_only() {
                let mut c = base();
                c.q[5 * D] = f32::NAN;
                let y = run(&c, Q, K, false);
                assert!(nan(&y, 5), "row 5: {:?}", &row(&y, 5)[..4]);
                assert!((0..Q).filter(|r| *r != 5).all(|r| fin(&y, r)));
            }

            #[test]
            fn nan_key_poisons_every_row() {
                let mut c = base();
                c.k[2 * D] = f32::NAN;
                let y = run(&c, Q, K, false);
                let bad: Vec<usize> = (0..Q).filter(|r| !nan(&y, *r)).collect();
                assert!(
                    bad.is_empty(),
                    "rows not NaN: {} of {Q}, first {:?}",
                    bad.len(),
                    &bad[..bad.len().min(5)]
                );
            }

            #[test]
            fn nan_value_poisons_every_row() {
                let mut c = base();
                c.v[2 * D] = f32::NAN;
                let y = run(&c, Q, K, false);
                let bad: Vec<usize> = (0..Q).filter(|r| !y[r * D].is_nan()).collect();
                assert!(
                    bad.is_empty(),
                    "rows not NaN in col 0: {} of {Q}",
                    bad.len()
                );
            }

            #[test]
            fn plus_inf_score_gives_a_nan_row() {
                let mut c = base();
                for d in 0..D {
                    c.q[7 * D + d] = f32::INFINITY;
                }
                let y = run(&c, Q, K, false);
                assert!(nan(&y, 7), "row 7: {:?}", &row(&y, 7)[..4]);
                assert!((0..Q).filter(|r| *r != 7).all(|r| fin(&y, r)));
            }

            /// Q > K, causal, bottom-right aligned: rows 0..Q-K see no key at all.
            #[test]
            fn fully_masked_rows_are_zero() {
                let q_len = 128;
                let k_len = 64;
                let y = run(&base(), q_len, k_len, true);
                let bad: Vec<(usize, f32)> = (0..q_len - k_len)
                    .filter(|r| !row(&y, *r).iter().all(|v| *v == 0.0))
                    .map(|r| (r, row(&y, r)[0]))
                    .collect();
                assert!(
                    bad.is_empty(),
                    "{} fully masked rows not zero, first {:?}",
                    bad.len(),
                    &bad[..bad.len().min(3)]
                );
                assert!(
                    (q_len - k_len..q_len).all(|r| fin(&y, r)),
                    "visible rows must be finite"
                );
            }
        }
    };
}

suite!(f16_, f16, ElementKind::F16);
suite!(bf16_, bf16, ElementKind::Bf16);
