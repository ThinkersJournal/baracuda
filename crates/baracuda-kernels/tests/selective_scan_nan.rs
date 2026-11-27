// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `SelectiveScanPlan` (Mamba-1 forward) propagates NaN along the recurrence and does not leak it across batch, channel or an
//! earlier step (NaN sweep row, ROADMAP E0). Per (b, d) the state is `h[n] = exp(delta[t,d] * A[d,n]) * h[n] + delta[t,d] * B[t,n] * u[t,d]`,
//! `y[t,d] = sum_n C[t,n] * h[n] (+ D[d] * u[t,d]) (* silu(z[t,d]))`. So a NaN in `u[t,d]` or `delta[t,d]` poisons channel `d` from step `t` on,
//! a NaN in `B[t,n]` poisons every channel from `t` on, a NaN in `C[t,n]` poisons step `t` only, a NaN in `A[d,n]` poisons channel `d`
//! entirely, and a NaN in `z[t,d]` poisons `(t, d)` only. f32, f16, bf16.
#![cfg(feature = "mamba")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, SelectiveScanArgs, SelectiveScanDescriptor, SelectiveScanPlan,
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

const BSZ: usize = 2;
const L: usize = 8;
const DIM: usize = 4;
const DSTATE: usize = 4;

#[derive(Clone)]
struct In {
    u: Vec<f32>,
    delta: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    z: Vec<f32>,
}

fn base() -> In {
    In {
        u: (0..BSZ * L * DIM)
            .map(|i| 0.5 + (i % 5) as f32 * 0.1)
            .collect(),
        delta: (0..BSZ * L * DIM)
            .map(|i| 0.1 + (i % 3) as f32 * 0.05)
            .collect(),
        a: (0..DIM * DSTATE)
            .map(|i| -0.5 - (i % 4) as f32 * 0.1)
            .collect(),
        b: (0..BSZ * L * DSTATE)
            .map(|i| 0.3 + (i % 4) as f32 * 0.1)
            .collect(),
        c: (0..BSZ * L * DSTATE)
            .map(|i| 0.2 + (i % 3) as f32 * 0.1)
            .collect(),
        z: (0..BSZ * L * DIM)
            .map(|i| 0.4 + (i % 5) as f32 * 0.1)
            .collect(),
    }
}

fn yi(b: usize, t: usize, d: usize) -> usize {
    (b * L + t) * DIM + d
}

macro_rules! suite {
    ($m:ident, $T:ty, $kind:expr) => {
        mod $m {
            use super::*;

            fn run(inp: &In, softplus: bool, gated: bool) -> Vec<f32> {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let cv = |x: &[f32]| {
                    x.iter()
                        .map(|v| <$T as Conv>::from32(*v))
                        .collect::<Vec<$T>>()
                };
                let du = DeviceBuffer::from_slice(&ctx, &cv(&inp.u)).unwrap();
                let ddelta = DeviceBuffer::from_slice(&ctx, &cv(&inp.delta)).unwrap();
                let da = DeviceBuffer::from_slice(&ctx, &cv(&inp.a)).unwrap();
                let db = DeviceBuffer::from_slice(&ctx, &cv(&inp.b)).unwrap();
                let dc = DeviceBuffer::from_slice(&ctx, &cv(&inp.c)).unwrap();
                let dz = DeviceBuffer::from_slice(&ctx, &cv(&inp.z)).unwrap();
                let mut dy: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, BSZ * L * DIM).unwrap();
                let desc = SelectiveScanDescriptor {
                    batch_size: BSZ as i32,
                    seq_len: L as i32,
                    dim: DIM as i32,
                    dstate: DSTATE as i32,
                    delta_softplus: softplus,
                    element: $kind,
                };
                let plan =
                    SelectiveScanPlan::<$T>::select(&stream, &desc, PlanPreference::default())
                        .expect("select");
                let s_ud = [BSZ as i32, L as i32, DIM as i32];
                let s_a = [DIM as i32, DSTATE as i32];
                let s_bc = [BSZ as i32, L as i32, DSTATE as i32];
                plan.run(
                    &stream,
                    Workspace::None,
                    SelectiveScanArgs {
                        u: TensorRef::new(du.as_slice(), s_ud, contiguous_stride(s_ud)).unwrap(),
                        delta: TensorRef::new(ddelta.as_slice(), s_ud, contiguous_stride(s_ud))
                            .unwrap(),
                        a: TensorRef::new(da.as_slice(), s_a, contiguous_stride(s_a)).unwrap(),
                        b: TensorRef::new(db.as_slice(), s_bc, contiguous_stride(s_bc)).unwrap(),
                        c: TensorRef::new(dc.as_slice(), s_bc, contiguous_stride(s_bc)).unwrap(),
                        d_skip: None,
                        z: gated.then(|| {
                            TensorRef::new(dz.as_slice(), s_ud, contiguous_stride(s_ud)).unwrap()
                        }),
                        delta_bias: None,
                        y: TensorMut::new(dy.as_slice_mut(), s_ud, contiguous_stride(s_ud))
                            .unwrap(),
                        last_state: None,
                    },
                )
                .expect("run");
                stream.synchronize().expect("sync");
                let mut y = vec![<$T as Conv>::from32(0.0); BSZ * L * DIM];
                dy.copy_to_host(&mut y).unwrap();
                y.iter().map(|v| v.to32()).collect()
            }

            /// Each output cell must be NaN exactly where `expect_nan` says, and finite elsewhere.
            fn check(y: &[f32], expect_nan: impl Fn(usize, usize, usize) -> bool, what: &str) {
                let mut wrong = Vec::new();
                for b in 0..BSZ {
                    for t in 0..L {
                        for d in 0..DIM {
                            let v = y[yi(b, t, d)];
                            if expect_nan(b, t, d) != v.is_nan() || (!v.is_nan() && !v.is_finite())
                            {
                                wrong.push((b, t, d, v));
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
                check(&run(&base(), false, false), |_, _, _| false, "baseline");
                check(
                    &run(&base(), true, true),
                    |_, _, _| false,
                    "softplus+gate baseline",
                );
            }

            #[test]
            fn nan_u_poisons_one_channel_from_t_on() {
                let mut c = base();
                c.u[yi(0, 3, 2)] = f32::NAN;
                check(
                    &run(&c, false, false),
                    |b, t, d| b == 0 && d == 2 && t >= 3,
                    "NaN u",
                );
            }

            #[test]
            fn nan_delta_poisons_one_channel_from_t_on() {
                let mut c = base();
                c.delta[yi(0, 3, 2)] = f32::NAN;
                check(
                    &run(&c, false, false),
                    |b, t, d| b == 0 && d == 2 && t >= 3,
                    "NaN delta",
                );
                // With the softplus branch (its `<= 20` guard takes the else arm for NaN): same cells.
                check(
                    &run(&c, true, false),
                    |b, t, d| b == 0 && d == 2 && t >= 3,
                    "NaN delta, softplus",
                );
            }

            #[test]
            fn nan_b_poisons_every_channel_from_t_on() {
                let mut c = base();
                c.b[(0 * L + 3) * DSTATE + 1] = f32::NAN;
                check(&run(&c, false, false), |b, t, _| b == 0 && t >= 3, "NaN B");
            }

            #[test]
            fn nan_c_poisons_one_step_only() {
                let mut c = base();
                c.c[(0 * L + 3) * DSTATE] = f32::NAN;
                check(&run(&c, false, false), |b, t, _| b == 0 && t == 3, "NaN C");
            }

            #[test]
            fn nan_a_poisons_its_channel() {
                let mut c = base();
                c.a[DSTATE] = f32::NAN; // channel 1
                check(&run(&c, false, false), |_, _, d| d == 1, "NaN A");
            }

            #[test]
            fn nan_z_poisons_one_cell() {
                let mut c = base();
                c.z[yi(1, 5, 3)] = f32::NAN;
                check(
                    &run(&c, false, true),
                    |b, t, d| b == 1 && t == 5 && d == 3,
                    "NaN z",
                );
            }
        }
    };
}

suite!(f32_, f32, ElementKind::F32);
suite!(f16_, f16, ElementKind::F16);
suite!(bf16_, bf16, ElementKind::Bf16);
