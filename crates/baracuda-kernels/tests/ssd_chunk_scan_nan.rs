// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `SsdChunkScanPlan` (Mamba-2 SSD forward) propagates NaN along the recurrence exactly as the reference does, and does not
//! leak it across batch or head (NaN sweep row, ROADMAP E0).
//!
//! Recurrence per (b, h): `state[d][n] = decay_t * state[d][n] + dt_t * B[t][n] * x[t][d]`, `y[t][d] = sum_n C[t][n] * state[d][n]`.
//! So a NaN in `x[t, d]` poisons column `d` from `t` on; a NaN in `B[t, n]` or in `dt[t]` poisons every column from `t` on; a NaN in `C[t, n]`
//! poisons row `t` only; a NaN in `A[h]` poisons head `h` from the first step. Nothing reaches another batch or head, nor an earlier step.
#![cfg(feature = "mamba")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, SsdChunkScanArgs, SsdChunkScanDescriptor, SsdChunkScanPlan,
    TensorMut, TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

trait Conv: Copy {
    fn from32(v: f32) -> Self;
    fn to32(self) -> f32;
}
impl Conv for f32 {
    fn from32(v: f32) -> Self {
        v
    }
    fn to32(self) -> f32 {
        self
    }
}
impl Conv for f16 {
    fn from32(v: f32) -> Self {
        f16::from_f32(v)
    }
    fn to32(self) -> f32 {
        self.to_f32()
    }
}
impl Conv for bf16 {
    fn from32(v: f32) -> Self {
        bf16::from_f32(v)
    }
    fn to32(self) -> f32 {
        self.to_f32()
    }
}

const B: usize = 2;
const L: usize = 8;
const H: usize = 2;
const D: usize = 4;
const N: usize = 4;

#[derive(Clone)]
struct In {
    x: Vec<f32>,
    dt: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
}

fn base() -> In {
    In {
        x: (0..B * L * H * D)
            .map(|i| 0.5 + (i % 5) as f32 * 0.1)
            .collect(),
        dt: (0..B * L * H)
            .map(|i| 0.1 + (i % 3) as f32 * 0.05)
            .collect(),
        a: vec![-0.5, -0.7],
        b: (0..B * L * H * N)
            .map(|i| 0.3 + (i % 4) as f32 * 0.1)
            .collect(),
        c: (0..B * L * H * N)
            .map(|i| 0.2 + (i % 3) as f32 * 0.1)
            .collect(),
    }
}

fn yi(b: usize, t: usize, h: usize, d: usize) -> usize {
    ((b * L + t) * H + h) * D + d
}

macro_rules! suite {
    ($m:ident, $T:ty, $kind:expr) => {
        mod $m {
            use super::*;

            fn run(inp: &In) -> Vec<f32> {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let cv = |x: &[f32]| {
                    x.iter()
                        .map(|v| <$T as Conv>::from32(*v))
                        .collect::<Vec<$T>>()
                };
                let dx = DeviceBuffer::from_slice(&ctx, &cv(&inp.x)).unwrap();
                let ddt = DeviceBuffer::from_slice(&ctx, &cv(&inp.dt)).unwrap();
                let da = DeviceBuffer::from_slice(&ctx, &cv(&inp.a)).unwrap();
                let db = DeviceBuffer::from_slice(&ctx, &cv(&inp.b)).unwrap();
                let dc = DeviceBuffer::from_slice(&ctx, &cv(&inp.c)).unwrap();
                let mut dy: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, B * L * H * D).unwrap();
                let desc = SsdChunkScanDescriptor {
                    batch_size: B as i32,
                    seq_len: L as i32,
                    num_heads: H as i32,
                    head_dim: D as i32,
                    state_dim: N as i32,
                    chunk_size: 4,
                    element: $kind,
                };
                let plan =
                    SsdChunkScanPlan::<$T>::select(&stream, &desc, PlanPreference::default())
                        .expect("select");
                let sx = [B as i32, L as i32, H as i32, D as i32];
                let sdt = [B as i32, L as i32, H as i32];
                let sa = [H as i32];
                let sbn = [B as i32, L as i32, H as i32, N as i32];
                plan.run(
                    &stream,
                    Workspace::None,
                    SsdChunkScanArgs {
                        x: TensorRef::new(dx.as_slice(), sx, contiguous_stride(sx)).unwrap(),
                        dt: TensorRef::new(ddt.as_slice(), sdt, contiguous_stride(sdt)).unwrap(),
                        a: TensorRef::new(da.as_slice(), sa, contiguous_stride(sa)).unwrap(),
                        b: TensorRef::new(db.as_slice(), sbn, contiguous_stride(sbn)).unwrap(),
                        c: TensorRef::new(dc.as_slice(), sbn, contiguous_stride(sbn)).unwrap(),
                        y: TensorMut::new(dy.as_slice_mut(), sx, contiguous_stride(sx)).unwrap(),
                    },
                )
                .expect("run");
                stream.synchronize().expect("sync");
                let mut y = vec![<$T as Conv>::from32(0.0); B * L * H * D];
                dy.copy_to_host(&mut y).unwrap();
                y.iter().map(|v| v.to32()).collect()
            }

            /// Every output cell must be NaN exactly where `expect_nan` says so, and finite elsewhere.
            fn check(
                y: &[f32],
                expect_nan: impl Fn(usize, usize, usize, usize) -> bool,
                what: &str,
            ) {
                let mut wrong = Vec::new();
                for b in 0..B {
                    for t in 0..L {
                        for h in 0..H {
                            for d in 0..D {
                                let v = y[yi(b, t, h, d)];
                                if expect_nan(b, t, h, d) != v.is_nan()
                                    || (!v.is_nan() && !v.is_finite())
                                {
                                    wrong.push((b, t, h, d, v));
                                }
                            }
                        }
                    }
                }
                assert!(
                    wrong.is_empty(),
                    "{what}: {} cells wrong, first {:?}",
                    wrong.len(),
                    &wrong[..wrong.len().min(4)]
                );
            }

            #[test]
            fn clean_baseline_is_finite() {
                check(&run(&base()), |_, _, _, _| false, "baseline");
            }

            #[test]
            fn nan_x_poisons_one_column_from_t_on() {
                let mut c = base();
                c.x[((0 * L + 3) * H + 0) * D + 2] = f32::NAN;
                check(
                    &run(&c),
                    |b, t, h, d| b == 0 && h == 0 && d == 2 && t >= 3,
                    "NaN x",
                );
            }

            #[test]
            fn nan_b_poisons_every_column_from_t_on() {
                let mut c = base();
                c.b[((0 * L + 3) * H + 0) * N + 1] = f32::NAN;
                check(&run(&c), |b, t, h, _| b == 0 && h == 0 && t >= 3, "NaN B");
            }

            #[test]
            fn nan_c_poisons_one_step_only() {
                let mut c = base();
                c.c[((0 * L + 3) * H + 0) * N] = f32::NAN;
                check(&run(&c), |b, t, h, _| b == 0 && h == 0 && t == 3, "NaN C");
            }

            #[test]
            fn nan_dt_poisons_the_state_from_t_on() {
                let mut c = base();
                c.dt[(0 * L + 3) * H + 0] = f32::NAN;
                check(&run(&c), |b, t, h, _| b == 0 && h == 0 && t >= 3, "NaN dt");
            }

            #[test]
            fn nan_a_poisons_its_head() {
                let mut c = base();
                c.a[1] = f32::NAN;
                check(&run(&c), |_, _, h, _| h == 1, "NaN A");
            }
        }
    };
}

suite!(f32_, f32, ElementKind::F32);
suite!(f16_, f16, ElementKind::F16);
suite!(bf16_, bf16, ElementKind::Bf16);
