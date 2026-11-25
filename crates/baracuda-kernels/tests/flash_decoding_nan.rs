// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `FlashDecodingPlan` propagates NaN instead of swallowing it.
//!
//! Same contract as the other attention kernels (`attention_nan_rows.rs`): a NaN in a head's query, or in a key or value that head attends to
//! (weight above 0), makes that head's output NaN, and only that head's. `k_len = 600` spans three split-K chunks, so a NaN key in a later chunk
//! is covered too. f16 and bf16 are the dtypes the plan supports. The decoding API has no mask and no causal flag, so the fully masked row (all
//! `-inf`, zeros) cannot be expressed here; every other contract case is covered (query NaN, attended key and value NaN, a `+inf` score).
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, FlashDecodingArgs, FlashDecodingDescriptor, FlashDecodingPlan, PlanPreference,
    TensorMut, TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

const B: i32 = 2;
const H: i32 = 2;
const KL: i32 = 600;
const D: i32 = 64;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

struct In {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn base() -> In {
    let n = |len: usize, a: f32, b: f32| -> Vec<f32> {
        (0..len).map(|i| ((i as f32) * a + b).sin() * 0.3).collect()
    };
    In {
        q: n((B * H * D) as usize, 0.013, -0.5),
        k: n((B * H * KL * D) as usize, 0.017, 0.2),
        v: n((B * H * KL * D) as usize, 0.011, -0.1),
    }
}

fn head_nan(y: &[f32], b: i32, h: i32) -> bool {
    let o = ((b * H + h) * D) as usize;
    y[o..o + D as usize].iter().all(|v| v.is_nan())
}
fn head_finite(y: &[f32], b: i32, h: i32) -> bool {
    let o = ((b * H + h) * D) as usize;
    y[o..o + D as usize].iter().all(|v| v.is_finite())
}

macro_rules! tests {
    ($name:ident, $T:ty, $elem:expr, $to:expr, $from:expr) => {
        #[test]
        fn $name() {
            let (ctx, stream) = setup();
            let run = |x: &In| -> Vec<f32> {
                let q: Vec<$T> = x.q.iter().map(|v| $to(*v)).collect();
                let k: Vec<$T> = x.k.iter().map(|v| $to(*v)).collect();
                let v: Vec<$T> = x.v.iter().map(|v| $to(*v)).collect();
                let dq = DeviceBuffer::from_slice(&ctx, &q).unwrap();
                let dk = DeviceBuffer::from_slice(&ctx, &k).unwrap();
                let dv = DeviceBuffer::from_slice(&ctx, &v).unwrap();
                let mut dy: DeviceBuffer<$T> =
                    DeviceBuffer::zeros(&ctx, (B * H * D) as usize).unwrap();
                let desc = FlashDecodingDescriptor::new(B, H, KL, D, $elem);
                let plan =
                    FlashDecodingPlan::<$T>::select(&stream, &desc, PlanPreference::default())
                        .unwrap();
                let mut ws: DeviceBuffer<u8> =
                    DeviceBuffer::zeros(&ctx, plan.workspace_size()).unwrap();
                let (sq, sk, sy) = ([B, H, D], [B, H, KL, D], [B, H, D]);
                plan.run(
                    &stream,
                    Workspace::Borrowed(ws.as_slice_mut()),
                    FlashDecodingArgs::<$T> {
                        q: TensorRef::new(dq.as_slice(), sq, contiguous_stride(sq)).unwrap(),
                        k: TensorRef::new(dk.as_slice(), sk, contiguous_stride(sk)).unwrap(),
                        v: TensorRef::new(dv.as_slice(), sk, contiguous_stride(sk)).unwrap(),
                        y: TensorMut::new(dy.as_slice_mut(), sy, contiguous_stride(sy)).unwrap(),
                        a: None,
                        a_mean: None,
                    },
                )
                .unwrap();
                stream.synchronize().unwrap();
                let mut y = vec![$to(0.0); (B * H * D) as usize];
                dy.copy_to_host(&mut y).unwrap();
                y.iter().map(|v| $from(*v)).collect()
            };

            let ok = base();
            let y = run(&ok);
            assert!(
                (0..B).all(|b| (0..H).all(|h| head_finite(&y, b, h))),
                "clean baseline must be finite"
            );

            // nan_q_head: NaN in the query of (b 0, h 1).
            let mut c = In {
                q: ok.q.clone(),
                k: ok.k.clone(),
                v: ok.v.clone(),
            };
            c.q[(1 * D) as usize] = f32::NAN;
            let y = run(&c);
            assert!(
                head_nan(&y, 0, 1),
                "NaN in a query head must give a NaN output head, got {:?}",
                &y[(1 * D) as usize..(1 * D + 4) as usize]
            );
            assert!(
                head_finite(&y, 0, 0) && head_finite(&y, 1, 0) && head_finite(&y, 1, 1),
                "other heads must stay finite"
            );

            // nan_k_first_chunk: NaN key 5 of (b 0, h 0).
            let mut c = In {
                q: ok.q.clone(),
                k: ok.k.clone(),
                v: ok.v.clone(),
            };
            c.k[(5 * D) as usize] = f32::NAN;
            let y = run(&c);
            assert!(
                head_nan(&y, 0, 0),
                "a NaN key in the first chunk must poison its head"
            );
            assert!(
                head_finite(&y, 0, 1) && head_finite(&y, 1, 0) && head_finite(&y, 1, 1),
                "other heads must stay finite"
            );

            // nan_k_late_chunk: NaN key 580 (third chunk) of (b 1, h 1).
            let mut c = In {
                q: ok.q.clone(),
                k: ok.k.clone(),
                v: ok.v.clone(),
            };
            c.k[(((1 * H + 1) * KL + 580) * D) as usize] = f32::NAN;
            let y = run(&c);
            assert!(
                head_nan(&y, 1, 1),
                "a NaN key in a late chunk must poison its head"
            );
            assert!(
                head_finite(&y, 0, 0) && head_finite(&y, 0, 1) && head_finite(&y, 1, 0),
                "other heads must stay finite"
            );

            // inf_q: +inf in the query of (b 0, h 0): scores of +inf or -inf, so a chunk maximum of +inf (exp(inf - inf) = NaN): NaN head.
            let mut c = In {
                q: ok.q.clone(),
                k: ok.k.clone(),
                v: ok.v.clone(),
            };
            c.q[0] = f32::INFINITY;
            let y = run(&c);
            assert!(
                head_nan(&y, 0, 0),
                "a +inf query component must give a NaN head (torch: softmax with +inf is NaN)"
            );
            assert!(
                head_finite(&y, 0, 1) && head_finite(&y, 1, 0) && head_finite(&y, 1, 1),
                "other heads must stay finite"
            );

            // nan_v_attended: NaN value at key 3 of (b 0, h 0): every key has weight > 0.
            let mut c = In {
                q: ok.q.clone(),
                k: ok.k.clone(),
                v: ok.v.clone(),
            };
            c.v[(3 * D) as usize] = f32::NAN;
            let y = run(&c);
            assert!(
                y[0].is_nan(),
                "a NaN value the head attends to must reach y[0]"
            );
            assert!(
                head_finite(&y, 0, 1) && head_finite(&y, 1, 0),
                "other heads must stay finite"
            );
        }
    };
}

tests!(
    flash_decoding_f16_nan_heads,
    f16,
    ElementKind::F16,
    f16::from_f32,
    |v: f16| v.to_f32()
);
tests!(
    flash_decoding_bf16_nan_heads,
    bf16,
    ElementKind::Bf16,
    bf16::from_f32,
    |v: bf16| v.to_f32()
);
