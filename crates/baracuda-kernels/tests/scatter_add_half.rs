// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `ScatterAddPlan` for f16 and bf16 (new atomic family: add by compare-and-swap on the 32-bit word that holds the 2-byte cell).
//!
//! `out[..., index[..., j, ...], ...] += updates[..., j, ...]` along `scatter_dim`. With duplicate target indices the ORDER of additions into a
//! cell is undefined, so every partial sum in the exact-value cases is chosen to be exactly representable in the dtype (multiples of 0.25 of
//! magnitude well under 32): any order then gives the same bits, and the test compares against an f64 accumulation rounded to the dtype.
//! Also pinned: heavy duplication (200 updates into 3 rows), a NaN update poisons its cell, `+inf + -inf` is NaN, `+inf` plus a finite value is
//! `+inf`, untouched cells keep their bits (including NaN payloads), i32 and i64 indices, and an out-of-range or negative index is skipped
//! (the shared policy of every indexing plan).
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, ScatterAddArgs, ScatterAddDescriptor, ScatterAddPlan, TensorMut,
    TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

trait H: Copy {
    fn from64(v: f64) -> Self;
    fn to64(self) -> f64;
    fn bits(self) -> u16;
}
impl H for f16 {
    fn from64(v: f64) -> Self {
        f16::from_f64(v)
    }
    fn to64(self) -> f64 {
        f16::to_f64(self)
    }
    fn bits(self) -> u16 {
        self.to_bits()
    }
}
impl H for bf16 {
    fn from64(v: f64) -> Self {
        bf16::from_f64(v)
    }
    fn to64(self) -> f64 {
        bf16::to_f64(self)
    }
    fn bits(self) -> u16 {
        self.to_bits()
    }
}

macro_rules! suite {
    ($m:ident, $T:ty, $kind:expr) => {
        mod $m {
            use super::*;

            /// Runs one scatter_add on `out [r, c]` with `updates`/`index` of shape `[u, c]` along dim 0; returns the output bits.
            fn run<I: baracuda_kernels::IndexElement + Copy>(
                r: usize,
                c: usize,
                u: usize,
                out0: &[$T],
                upd: &[$T],
                idx: &[I],
            ) -> Vec<u16> {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let (ri, ci, ui) = (r as i32, c as i32, u as i32);
                let o_sh = [ri, ci];
                let u_sh = [ui, ci];
                let dupd = DeviceBuffer::from_slice(&ctx, upd).unwrap();
                let didx = DeviceBuffer::from_slice(&ctx, idx).unwrap();
                let mut dout = DeviceBuffer::from_slice(&ctx, out0).unwrap();
                let desc = ScatterAddDescriptor {
                    upd_shape: u_sh,
                    scatter_dim: 0,
                    out_dim_size: ri,
                    element: $kind,
                };
                let plan =
                    ScatterAddPlan::<$T, 2>::select(&stream, &desc, PlanPreference::default())
                        .expect("select");
                plan.run(
                    &stream,
                    Workspace::None,
                    ScatterAddArgs::<$T, 2, I> {
                        updates: TensorRef::new(dupd.as_slice(), u_sh, contiguous_stride(u_sh))
                            .unwrap(),
                        index: TensorRef::new(didx.as_slice(), u_sh, contiguous_stride(u_sh))
                            .unwrap(),
                        out: TensorMut::new(dout.as_slice_mut(), o_sh, contiguous_stride(o_sh))
                            .unwrap(),
                    },
                )
                .expect("run");
                stream.synchronize().expect("sync");
                let mut got = vec![<$T>::from64(0.0); r * c];
                dout.copy_to_host(&mut got).unwrap();
                got.iter().map(|v| v.bits()).collect()
            }

            /// f64 accumulation of the same scatter, rounded once to the dtype.
            fn reference(
                r: usize,
                c: usize,
                u: usize,
                out0: &[$T],
                upd: &[$T],
                idx: &[i64],
            ) -> Vec<u16> {
                let mut acc: Vec<f64> = out0.iter().map(|v| v.to64()).collect();
                for j in 0..u {
                    for k in 0..c {
                        let t = idx[j * c + k];
                        if t < 0 || t as usize >= r {
                            continue;
                        }
                        acc[t as usize * c + k] += upd[j * c + k].to64();
                    }
                }
                acc.iter().map(|v| <$T>::from64(*v).bits()).collect()
            }

            fn fill(n: usize, f: impl Fn(usize) -> f64) -> Vec<$T> {
                (0..n).map(|i| <$T>::from64(f(i))).collect()
            }

            #[test]
            fn heavy_duplicates_with_exactly_representable_sums() {
                let (r, c, u) = (3usize, 4usize, 200usize);
                let out0 = fill(r * c, |i| 1.5 + (i % 3) as f64 * 0.25);
                let upd = fill(u * c, |i| {
                    [1.0, -1.0, 0.5, -0.5, 0.25, -0.25][(i / c + i % c) % 6]
                });
                let idx: Vec<i64> = (0..u * c)
                    .map(|i| ((i / c) * 7 + (i % c)) as i64 % 3)
                    .collect();
                let want = reference(r, c, u, &out0, &upd, &idx);
                let i32s: Vec<i32> = idx.iter().map(|v| *v as i32).collect();
                for rep in 0..20 {
                    assert_eq!(
                        run(r, c, u, &out0, &upd, &i32s),
                        want,
                        "i32 index, run {rep}"
                    );
                    assert_eq!(
                        run(r, c, u, &out0, &upd, &idx),
                        want,
                        "i64 index, run {rep}"
                    );
                }
            }

            #[test]
            fn nan_and_inf_follow_ieee_addition() {
                // 4 target rows x 1 column; updates [6, 1].
                let out0 = fill(4, |_| 1.0);
                // row 0: +inf then -inf -> NaN; row 1: NaN update -> NaN; row 2: +inf + finite -> +inf; row 3: untouched NaN payload.
                let mut o = out0.clone();
                o[3] = <$T>::from64(f64::NAN);
                let upd = fill(6, |i| {
                    [
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::NAN,
                        f64::INFINITY,
                        2.0,
                        0.0,
                    ][i]
                });
                let idx: Vec<i32> = vec![0, 0, 1, 2, 2, 1];
                let got = run(4, 1, 6, &o, &upd, &idx);
                let dec = decode;
                assert!(
                    dec(got[0]).is_nan(),
                    "+inf + -inf must be NaN, got {:#06x}",
                    got[0]
                );
                assert!(
                    dec(got[1]).is_nan(),
                    "a NaN update poisons its cell, got {:#06x}",
                    got[1]
                );
                assert_eq!(dec(got[2]), f64::INFINITY, "+inf + finite is +inf");
                assert_eq!(
                    got[3],
                    o[3].bits(),
                    "an untouched cell keeps its exact bits (NaN payload)"
                );
            }

            fn decode(b: u16) -> f64 {
                <$T as Dec>::dec(b)
            }

            #[test]
            fn out_of_range_and_negative_indices_are_skipped() {
                let out0 = fill(3 * 2, |i| i as f64);
                let upd = fill(5 * 2, |_| 1.0);
                let idx: Vec<i32> = vec![0, 1, -1, 2, 99, 0, 2, 1, 1, -7];
                let want = reference(
                    3,
                    2,
                    5,
                    &out0,
                    &upd,
                    &idx.iter().map(|v| *v as i64).collect::<Vec<_>>(),
                );
                assert_eq!(run(3, 2, 5, &out0, &upd, &idx), want, "OOB i32");
                let idx64: Vec<i64> = idx.iter().map(|v| *v as i64).collect();
                assert_eq!(run(3, 2, 5, &out0, &upd, &idx64), want, "OOB i64");
            }
        }
    };
}

trait Dec {
    fn dec(b: u16) -> f64;
}
impl Dec for f16 {
    fn dec(b: u16) -> f64 {
        f16::from_bits(b).to_f64()
    }
}
impl Dec for bf16 {
    fn dec(b: u16) -> f64 {
        bf16::from_bits(b).to_f64()
    }
}

suite!(f16_, f16, ElementKind::F16);
suite!(bf16_, bf16, ElementKind::Bf16);
