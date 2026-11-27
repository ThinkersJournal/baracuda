// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU audit: does the vendored FlashAttention-2 launcher (forced via `prefer_backend`) follow the attention NaN table
//! that our own kernels carry (ROADMAP E0k)?
//!
//! Table: a NaN in a query row gives that row NaN and no other; a NaN attended key or value poisons every row that attends it; a `+inf`
//! score gives a NaN row; a fully masked row (a causal row with no visible key, or one outside a sliding window) is all zeros.
//! Extensions measured here: GQA (`H_k < H`), head dims 64/128/256, sliding window, and softcap. With a softcap the score is bounded
//! (`cap * tanh(s / cap)`), so a `+inf` score becomes `cap` and the row is finite, as in torch; that is asserted, not NaN.
//! A NaN key or value at a position a row does NOT attend is UNSPECIFIED and only printed (`eprintln`), never asserted.
#![cfg(feature = "fa2")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    BackendKind, ElementKind, FlashSdpaArgs, FlashSdpaDescriptor, FlashSdpaPlan, PlanPreference,
    TensorMut, TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

#[derive(Clone, Copy)]
struct Cfg {
    h: usize,
    hk: usize,
    d: usize,
    q: usize,
    k: usize,
    causal: bool,
    wl: Option<i32>,
    wr: Option<i32>,
    cap: f32,
}

const BASIC: Cfg = Cfg {
    h: 1,
    hk: 1,
    d: 64,
    q: 128,
    k: 128,
    causal: false,
    wl: None,
    wr: None,
    cap: 0.0,
};

#[derive(Clone)]
struct In {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn base(c: &Cfg) -> In {
    In {
        q: (0..c.h * c.q * c.d)
            .map(|i| 0.05 * ((i % 5) as f32 - 2.0))
            .collect(),
        // Positive keys so an `inf` query row scores +inf on every key.
        k: (0..c.hk * c.k * c.d)
            .map(|i| 0.05 * ((i % 7) as f32 + 1.0))
            .collect(),
        v: (0..c.hk * c.k * c.d)
            .map(|i| 1.0 + (i % 9) as f32)
            .collect(),
    }
}

macro_rules! suite {
    ($m:ident, $T:ty, $kind:expr) => {
        mod $m {
            use super::*;

            fn run(inp: &In, c: &Cfg) -> Vec<f32> {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let cv = |x: &[f32]| x.iter().map(|v| <$T>::from_f32(*v)).collect::<Vec<$T>>();
                let dq = DeviceBuffer::from_slice(&ctx, &cv(&inp.q)).unwrap();
                let dk = DeviceBuffer::from_slice(&ctx, &cv(&inp.k)).unwrap();
                let dv = DeviceBuffer::from_slice(&ctx, &cv(&inp.v)).unwrap();
                let mut dy: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, c.h * c.q * c.d).unwrap();
                let mut dl: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, c.h * c.q).unwrap();
                let (sq, sk, sl) = (
                    [1, c.h as i32, c.q as i32, c.d as i32],
                    [1, c.hk as i32, c.k as i32, c.d as i32],
                    [1, c.h as i32, c.q as i32],
                );
                let desc = FlashSdpaDescriptor::new(
                    1,
                    c.h as i32,
                    c.q as i32,
                    c.k as i32,
                    c.d as i32,
                    c.d as i32,
                    1.0 / (c.d as f32).sqrt(),
                    c.causal,
                    $kind,
                )
                .with_window_size_left(c.wl)
                .with_window_size_right(c.wr)
                .with_softcap(c.cap);
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
                        y: TensorMut::new(dy.as_slice_mut(), sq, contiguous_stride(sq)).unwrap(),
                        lse: TensorMut::new(dl.as_slice_mut(), sl, contiguous_stride(sl)).unwrap(),
                        mask: None,
                        alibi_slopes: None,
                    },
                )
                .expect("fa2 run");
                stream.synchronize().expect("sync");
                let mut y = vec![<$T>::from_f32(0.0); c.h * c.q * c.d];
                dy.copy_to_host(&mut y).unwrap();
                y.iter().map(|v| v.to_f32()).collect()
            }

            /// Global row index `(head, row)`.
            fn row<'a>(y: &'a [f32], c: &Cfg, h: usize, r: usize) -> &'a [f32] {
                let o = (h * c.q + r) * c.d;
                &y[o..o + c.d]
            }
            fn nan(y: &[f32], c: &Cfg, h: usize, r: usize) -> bool {
                row(y, c, h, r).iter().all(|v| v.is_nan())
            }
            fn fin(y: &[f32], c: &Cfg, h: usize, r: usize) -> bool {
                row(y, c, h, r).iter().all(|v| v.is_finite())
            }
            fn zero(y: &[f32], c: &Cfg, h: usize, r: usize) -> bool {
                row(y, c, h, r).iter().all(|v| *v == 0.0)
            }
            fn all_rows(c: &Cfg) -> Vec<(usize, usize)> {
                (0..c.h)
                    .flat_map(|h| (0..c.q).map(move |r| (h, r)))
                    .collect()
            }
            /// Rows that must be listed as failing the predicate.
            fn bad(
                c: &Cfg,
                rows: &[(usize, usize)],
                y: &[f32],
                p: impl Fn(&[f32], &Cfg, usize, usize) -> bool,
            ) -> String {
                let b: Vec<_> = rows.iter().filter(|(h, r)| !p(y, c, *h, *r)).collect();
                format!("{} of {} rows failed, first {:?}", b.len(), rows.len(), &b[..b.len().min(4)])
            }
            fn ok(
                c: &Cfg,
                rows: &[(usize, usize)],
                y: &[f32],
                p: impl Fn(&[f32], &Cfg, usize, usize) -> bool,
            ) -> bool {
                rows.iter().all(|(h, r)| p(y, c, *h, *r))
            }

            // ---- the table on a configuration -------------------------------------------------------------------------------------
            fn table(c: Cfg, name: &str) {
                let all = all_rows(&c);
                let y = run(&base(&c), &c);
                assert!(ok(&c, &all, &y, fin), "{name}: clean baseline: {}", bad(&c, &all, &y, fin));

                // NaN query at (head 0, row 5): that row only (rows are independent in every head).
                let mut i = base(&c);
                i.q[5 * c.d] = f32::NAN;
                let y = run(&i, &c);
                assert!(nan(&y, &c, 0, 5), "{name}: NaN query row 5: {:?}", &row(&y, &c, 0, 5)[..4]);
                let rest: Vec<_> = all.iter().copied().filter(|x| *x != (0, 5)).collect();
                assert!(ok(&c, &rest, &y, fin), "{name}: NaN query leaked: {}", bad(&c, &rest, &y, fin));

                // +inf score at (last head, row 7): NaN row without softcap, `cap`-bounded (finite) with softcap.
                let hl = c.h - 1;
                let mut i = base(&c);
                for d in 0..c.d {
                    i.q[(hl * c.q + 7) * c.d + d] = f32::INFINITY;
                }
                let y = run(&i, &c);
                if c.cap == 0.0 {
                    assert!(nan(&y, &c, hl, 7), "{name}: +inf score row: {:?}", &row(&y, &c, hl, 7)[..4]);
                    let rest: Vec<_> = all.iter().copied().filter(|x| *x != (hl, 7)).collect();
                    assert!(ok(&c, &rest, &y, fin), "{name}: +inf leaked: {}", bad(&c, &rest, &y, fin));
                } else {
                    assert!(ok(&c, &all, &y, fin), "{name}: with softcap +inf is bounded, all rows finite: {}", bad(&c, &all, &y, fin));
                }
            }

            /// NaN in key/value position `pos` of kv head 0: every row of the query heads mapping to kv head 0 that ATTENDS `pos`
            /// must be NaN. Rows that do not attend it are only printed.
            fn kv_poison(c: Cfg, name: &str, pos: usize, attends: impl Fn(usize) -> bool) {
                let group = c.h / c.hk;
                for what in ["key", "value"] {
                    let mut i = base(&c);
                    if what == "key" {
                        i.k[pos * c.d] = f32::NAN;
                    } else {
                        i.v[pos * c.d] = f32::NAN;
                    }
                    let y = run(&i, &c);
                    let must: Vec<_> = (0..group)
                        .flat_map(|h| (0..c.q).filter(|r| attends(*r)).map(move |r| (h, r)))
                        .collect();
                    let isnan = |y: &[f32], c: &Cfg, h: usize, r: usize| {
                        if what == "key" { nan(y, c, h, r) } else { row(y, c, h, r)[0].is_nan() }
                    };
                    assert!(ok(&c, &must, &y, isnan), "{name}: NaN {what} at {pos}: {}", bad(&c, &must, &y, isnan));
                    // Other kv heads' query rows are unrelated: finite.
                    let other: Vec<_> = (group..c.h).flat_map(|h| (0..c.q).map(move |r| (h, r))).collect();
                    assert!(ok(&c, &other, &y, fin), "{name}: NaN {what} leaked to another kv head: {}", bad(&c, &other, &y, fin));
                    // Informational: rows that do not attend `pos`.
                    let skip: Vec<_> = (0..group)
                        .flat_map(|h| (0..c.q).filter(|r| !attends(*r)).map(move |r| (h, r)))
                        .collect();
                    if !skip.is_empty() {
                        let n = skip.iter().filter(|(h, r)| !fin(&y, &c, *h, *r)).count();
                        eprintln!("{name}: NaN {what} at non-attended position {pos}: {n} of {} non-attending rows non-finite (UNSPECIFIED)", skip.len());
                    }
                }
            }

            // ---- basic (D=64, MHA, no window) ---------------------------------------------------------------------------------------
            #[test]
            fn basic_table() {
                table(BASIC, "basic");
                kv_poison(BASIC, "basic", 2, |_| true);
            }

            /// Q > K, causal, bottom-right aligned: rows 0..Q-K see no key at all.
            #[test]
            fn fully_masked_rows_by_causal_are_zero() {
                let c = Cfg { q: 128, k: 64, causal: true, ..BASIC };
                let y = run(&base(&c), &c);
                let masked: Vec<_> = (0..64).map(|r| (0, r)).collect();
                let vis: Vec<_> = (64..128).map(|r| (0, r)).collect();
                assert!(ok(&c, &masked, &y, zero), "causal masked: {}", bad(&c, &masked, &y, zero));
                assert!(ok(&c, &vis, &y, fin), "causal visible: {}", bad(&c, &vis, &y, fin));
            }

            // ---- GQA --------------------------------------------------------------------------------------------------------------
            #[test]
            fn gqa_table() {
                let c = Cfg { h: 4, hk: 2, ..BASIC };
                table(c, "gqa");
                kv_poison(c, "gqa", 2, |_| true);
            }

            // ---- head dims ---------------------------------------------------------------------------------------------------------
            #[test]
            fn head_dim_128_table() {
                let c = Cfg { d: 128, ..BASIC };
                table(c, "d128");
                kv_poison(c, "d128", 2, |_| true);
            }

            #[test]
            fn head_dim_256_table() {
                let c = Cfg { d: 256, ..BASIC };
                table(c, "d256");
                kv_poison(c, "d256", 2, |_| true);
            }

            // ---- sliding window --------------------------------------------------------------------------------------------------
            /// Non-causal, window_left 16 / right 16: row r attends keys r-16..=r+16, so key 2 is attended by rows 0..=18.
            #[test]
            fn window_table() {
                let c = Cfg { wl: Some(16), wr: Some(16), ..BASIC };
                table(c, "window");
                kv_poison(c, "window", 2, |r| r <= 18);
            }

            /// Window with GQA and D=128, the combination a Mistral-style model uses (causal + left window).
            #[test]
            fn window_causal_gqa_d128() {
                let c = Cfg { h: 4, hk: 2, d: 128, causal: true, wl: Some(32), ..BASIC };
                table(c, "window+causal+gqa+d128");
                // causal + left window 32: key 2 is attended by rows 2..=34.
                kv_poison(c, "window+causal+gqa+d128", 2, |r| (2..=34).contains(&r));
            }

            /// Rows whose window holds no key are zero. Q=192, K=64, not causal, window `[r - 128, r - 120]` (FA2 aligns the window
            /// bottom-right, offset K - Q = -128): rows 0..=119 see no key, rows 120..192 see at least one.
            #[test]
            fn fully_masked_rows_by_window_are_zero() {
                let c = Cfg { q: 192, k: 64, wl: Some(0), wr: Some(8), ..BASIC };
                let y = run(&base(&c), &c);
                let masked: Vec<_> = (0..120).map(|r| (0, r)).collect();
                let vis: Vec<_> = (120..192).map(|r| (0, r)).collect();
                assert!(ok(&c, &masked, &y, zero), "window masked: {}", bad(&c, &masked, &y, zero));
                assert!(ok(&c, &vis, &y, fin), "window visible: {}", bad(&c, &vis, &y, fin));
            }

            // ---- softcap ---------------------------------------------------------------------------------------------------------
            #[test]
            fn softcap_table() {
                let c = Cfg { cap: 30.0, ..BASIC };
                table(c, "softcap");
                kv_poison(c, "softcap", 2, |_| true);
            }

            #[test]
            fn softcap_window_gqa_d128() {
                let c = Cfg { h: 4, hk: 2, d: 128, causal: true, wl: Some(32), cap: 30.0, ..BASIC };
                table(c, "softcap+window+gqa+d128");
                kv_poison(c, "softcap+window+gqa+d128", 2, |r| (2..=34).contains(&r));
            }
        }
    };
}

suite!(f16_, f16, ElementKind::F16);
suite!(bf16_, bf16, ElementKind::Bf16);
