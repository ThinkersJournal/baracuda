// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `GatherBackwardPlan` and `IndexSelectBackwardPlan` for f16 and bf16 (new compare-and-swap add kernels).
//!
//! Both are scatter-adds of the upstream gradient into `dsrc`: `dsrc[index[...]] += dout[...]`. With duplicate target rows the ORDER of
//! additions into a cell is undefined, so every partial sum in these cases is exactly representable in the dtype (multiples of 0.25 of
//! small magnitude), any order gives the same bits, and the expectation is an f64 accumulation rounded once. Covered: heavy duplicates (200
//! updates into 3 rows, 20 runs), i32 and i64 indices, a NaN gradient poisoning its cell only, an untouched cell keeping its bits, and an
//! out-of-range or negative index skipped (the shared policy of every indexing plan).
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, GatherBackwardArgs, GatherBackwardDescriptor, GatherBackwardPlan, IndexElement,
    IndexSelectBackwardArgs, IndexSelectBackwardDescriptor, IndexSelectBackwardPlan,
    PlanPreference, TensorMut, TensorRef, Workspace, contiguous_stride,
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

/// f64 accumulation of `dsrc[idx[j]][k] += dout[j][k]`, rounded once to the dtype. `idx_at(j, k)` is the target row.
fn reference<T: H>(
    r: usize,
    c: usize,
    u: usize,
    start: &[T],
    dout: &[T],
    idx_at: impl Fn(usize, usize) -> i64,
) -> Vec<u16> {
    let mut acc: Vec<f64> = start.iter().map(|v| v.to64()).collect();
    for j in 0..u {
        for k in 0..c {
            let t = idx_at(j, k);
            if t < 0 || t as usize >= r {
                continue;
            }
            acc[t as usize * c + k] += dout[j * c + k].to64();
        }
    }
    acc.iter().map(|v| T::from64(*v).bits()).collect()
}

fn fill<T: H>(n: usize, f: impl Fn(usize) -> f64) -> Vec<T> {
    (0..n).map(|i| T::from64(f(i))).collect()
}

fn ctx_stream() -> (Context, Stream) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

macro_rules! suite {
    ($m:ident, $T:ty, $kind:expr) => {
        mod $m {
            use super::*;

            fn gather_bw<I: IndexElement + Copy>(
                r: usize,
                c: usize,
                u: usize,
                start: &[$T],
                dout: &[$T],
                idx: &[I],
            ) -> Vec<u16> {
                let (ctx, stream) = ctx_stream();
                let (ri, ci, ui) = (r as i32, c as i32, u as i32);
                let (d_sh, s_sh) = ([ui, ci], [ri, ci]);
                let ddout = DeviceBuffer::from_slice(&ctx, dout).unwrap();
                let didx = DeviceBuffer::from_slice(&ctx, idx).unwrap();
                let mut ddsrc = DeviceBuffer::from_slice(&ctx, start).unwrap();
                let desc = GatherBackwardDescriptor {
                    out_shape: d_sh,
                    gather_dim: 0,
                    src_dim_size: ri,
                    element: $kind,
                };
                let plan =
                    GatherBackwardPlan::<$T, 2>::select(&stream, &desc, PlanPreference::default())
                        .expect("select");
                plan.run(
                    &stream,
                    Workspace::None,
                    GatherBackwardArgs::<$T, 2, I> {
                        dout: TensorRef::new(ddout.as_slice(), d_sh, contiguous_stride(d_sh))
                            .unwrap(),
                        index: TensorRef::new(didx.as_slice(), d_sh, contiguous_stride(d_sh))
                            .unwrap(),
                        dsrc: TensorMut::new(ddsrc.as_slice_mut(), s_sh, contiguous_stride(s_sh))
                            .unwrap(),
                    },
                )
                .expect("run");
                stream.synchronize().unwrap();
                let mut got = vec![<$T>::from64(0.0); r * c];
                ddsrc.copy_to_host(&mut got).unwrap();
                got.iter().map(|v| v.bits()).collect()
            }

            fn select_bw<I: IndexElement + Copy>(
                r: usize,
                c: usize,
                j: usize,
                start: &[$T],
                dout: &[$T],
                idx: &[I],
            ) -> Vec<u16> {
                let (ctx, stream) = ctx_stream();
                let (ri, ci, ji) = (r as i32, c as i32, j as i32);
                let (d_sh, s_sh) = ([ji, ci], [ri, ci]);
                let ddout = DeviceBuffer::from_slice(&ctx, dout).unwrap();
                let didx = DeviceBuffer::from_slice(&ctx, idx).unwrap();
                let mut ddsrc = DeviceBuffer::from_slice(&ctx, start).unwrap();
                let desc = IndexSelectBackwardDescriptor {
                    out_shape: d_sh,
                    select_dim: 0,
                    src_dim_size: ri,
                    element: $kind,
                };
                let plan = IndexSelectBackwardPlan::<$T, 2>::select(
                    &stream,
                    &desc,
                    PlanPreference::default(),
                )
                .expect("select");
                plan.run(
                    &stream,
                    Workspace::None,
                    IndexSelectBackwardArgs::<$T, 2, I> {
                        dout: TensorRef::new(ddout.as_slice(), d_sh, contiguous_stride(d_sh))
                            .unwrap(),
                        idx: TensorRef::new(didx.as_slice(), [ji], contiguous_stride([ji]))
                            .unwrap(),
                        dsrc: TensorMut::new(ddsrc.as_slice_mut(), s_sh, contiguous_stride(s_sh))
                            .unwrap(),
                    },
                )
                .expect("run");
                stream.synchronize().unwrap();
                let mut got = vec![<$T>::from64(0.0); r * c];
                ddsrc.copy_to_host(&mut got).unwrap();
                got.iter().map(|v| v.bits()).collect()
            }

            #[test]
            fn gather_backward_heavy_duplicates() {
                let (r, c, u) = (3usize, 4usize, 200usize);
                let start = fill::<$T>(r * c, |i| 0.25 * (i % 3) as f64);
                let dout = fill::<$T>(u * c, |i| {
                    [1.0, -1.0, 0.5, -0.5, 0.25, -0.25][(i / c + i % c) % 6]
                });
                let idx: Vec<i64> = (0..u * c)
                    .map(|i| ((i / c) * 7 + (i % c)) as i64 % 3)
                    .collect();
                let want = reference(r, c, u, &start, &dout, |j, k| idx[j * c + k]);
                let i32s: Vec<i32> = idx.iter().map(|v| *v as i32).collect();
                for rep in 0..20 {
                    assert_eq!(
                        gather_bw(r, c, u, &start, &dout, &i32s),
                        want,
                        "i32, run {rep}"
                    );
                    assert_eq!(
                        gather_bw(r, c, u, &start, &dout, &idx),
                        want,
                        "i64, run {rep}"
                    );
                }
            }

            #[test]
            fn gather_backward_nan_and_out_of_range() {
                let (r, c, u) = (3usize, 2usize, 4usize);
                let mut start = fill::<$T>(r * c, |i| i as f64);
                start[5] = <$T>::from64(f64::NAN); // an untouched cell with a NaN payload
                let mut dout = fill::<$T>(u * c, |_| 1.0);
                dout[0] = <$T>::from64(f64::NAN); // row idx[0][0] = 1, column 0 becomes NaN
                let idx: Vec<i32> = vec![1, 0, 2, -1, 99, 1, 0, 0];
                let got = gather_bw(r, c, u, &start, &dout, &idx);
                let want = reference(r, c, u, &start, &dout, |j, k| idx[j * c + k] as i64);
                // Compare everything except NaN cells by bits; NaN cells just need to be NaN on both sides.
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let dec = |b: u16| <$T as Dec>::dec(b);
                    if dec(*w).is_nan() {
                        assert!(dec(*g).is_nan(), "cell {i}: expected NaN, got {g:#06x}");
                    } else {
                        assert_eq!(g, w, "cell {i}");
                    }
                }
                assert_eq!(
                    got[5],
                    start[5].bits(),
                    "an untouched NaN cell keeps its exact bits"
                );
            }

            #[test]
            fn index_select_backward_heavy_duplicates() {
                let (r, c, j) = (3usize, 4usize, 200usize);
                let start = fill::<$T>(r * c, |i| 0.25 * (i % 3) as f64);
                let dout = fill::<$T>(j * c, |i| {
                    [1.0, -1.0, 0.5, -0.5, 0.25, -0.25][(i / c + i % c) % 6]
                });
                let idx: Vec<i64> = (0..j).map(|x| (x * 7 % 5) as i64 % 3).collect();
                let want = reference(r, c, j, &start, &dout, |jj, _| idx[jj]);
                let i32s: Vec<i32> = idx.iter().map(|v| *v as i32).collect();
                for rep in 0..20 {
                    assert_eq!(
                        select_bw(r, c, j, &start, &dout, &i32s),
                        want,
                        "i32, run {rep}"
                    );
                    assert_eq!(
                        select_bw(r, c, j, &start, &dout, &idx),
                        want,
                        "i64, run {rep}"
                    );
                }
            }

            #[test]
            fn index_select_backward_out_of_range_is_skipped() {
                let (r, c, j) = (3usize, 2usize, 5usize);
                let start = fill::<$T>(r * c, |i| i as f64);
                let dout = fill::<$T>(j * c, |_| 1.0);
                let idx: Vec<i32> = vec![0, -1, 2, 99, 1];
                let want = reference(r, c, j, &start, &dout, |jj, _| idx[jj] as i64);
                assert_eq!(select_bw(r, c, j, &start, &dout, &idx), want);
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
