// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `ScanBackwardPlan + ScanKind::LogCumsumExp` with `-inf` in the input (torch convention, ROADMAP E0).
//!
//! `dx[k] = sum_{i in range(k)} dy[i] * w(k, i)`, `w = exp(x[k] - y[i])`. When a prefix is all `-inf` both `x[k]` and `y[i]` are `-inf`: the
//! weight is zero-mass, 0 (torch), not `exp(-inf - -inf) = NaN`. `+inf / +inf` stays NaN and a NaN in `x` still poisons. The host reference
//! below applies exactly that rule in f64.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, ScanBackwardArgs, ScanBackwardDescriptor, ScanBackwardPlan,
    ScanKind, TensorMut, TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

trait Conv: Copy {
    fn from64(v: f64) -> Self;
    fn to64(self) -> f64;
}
macro_rules! conv {
    ($t:ty, $from:expr, $to:expr) => {
        impl Conv for $t {
            fn from64(v: f64) -> Self {
                $from(v)
            }
            fn to64(self) -> f64 {
                $to(self)
            }
        }
    };
}
conv!(f32, |v| v as f32, |s: f32| s as f64);
conv!(f64, |v| v, |s: f64| s);
conv!(f16, f16::from_f64, f16::to_f64);
conv!(bf16, bf16::from_f64, bf16::to_f64);

const N: usize = 6;
const NEG: f64 = f64::NEG_INFINITY;
const POS: f64 = f64::INFINITY;

fn rows() -> Vec<[f64; N]> {
    vec![
        [NEG, NEG, 0.5, 1.0, NEG, 0.25],         // leading -inf prefix
        [0.125, NEG, 0.5, NEG, NEG, 0.75],       // mixed -inf and finite
        [0.125, 0.25, f64::NAN, 0.5, 0.75, 1.0], // NaN in x
        [0.125, POS, 0.5, 0.25, 0.125, 0.0],     // +inf
        [NEG; N],                                // everything -inf
        [0.5, 0.25, 0.75, 1.0, 0.125, 0.5],      // control: finite
    ]
}
const DY: [f64; N] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

/// Stable log-sum-exp of a slice in f64 with the -inf / +inf / NaN rules of the forward kernel.
fn lse(v: &[f64]) -> f64 {
    if v.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let m = v.iter().cloned().fold(NEG, f64::max);
    if m == NEG || m == POS {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

fn forward(x: &[f64; N], reverse: bool) -> [f64; N] {
    let mut y = [0.0; N];
    for (i, slot) in y.iter_mut().enumerate() {
        *slot = if reverse { lse(&x[i..]) } else { lse(&x[..=i]) };
    }
    y
}

fn expected(x: &[f64; N], y: &[f64; N], reverse: bool) -> [f64; N] {
    let mut dx = [0.0; N];
    for (k, slot) in dx.iter_mut().enumerate() {
        let range: Vec<usize> = if reverse {
            (0..=k).collect()
        } else {
            (k..N).collect()
        };
        let mut acc = 0.0;
        for i in range {
            if x[k] == NEG && y[i] == NEG {
                continue; // zero-mass weight
            }
            acc += DY[i] * (x[k] - y[i]).exp();
        }
        *slot = acc;
    }
    dx
}

macro_rules! tests {
    ($name:ident, $T:ty, $kind:expr, $tol:expr) => {
        mod $name {
            use super::*;

            fn run(reverse: bool) {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let r = rows().len();
                let shape = [r as i32, N as i32];
                let mut hx = Vec::new();
                let mut hy = Vec::new();
                let mut hdy = Vec::new();
                let mut want = Vec::new();
                for row in rows() {
                    // The reference sees the values the device sees (rounded to the element type).
                    let xr: [f64; N] = row.map(|v| <$T>::from64(v).to64());
                    let y = forward(&xr, reverse);
                    hx.extend(xr.iter().map(|v| <$T>::from64(*v)));
                    hy.extend(y.iter().map(|v| <$T>::from64(*v)));
                    hdy.extend(DY.iter().map(|v| <$T>::from64(*v)));
                    want.extend(expected(&xr, &y.map(|v| <$T>::from64(v).to64()), reverse));
                }
                let dx_in = DeviceBuffer::from_slice(&ctx, &hx).unwrap();
                let y_in = DeviceBuffer::from_slice(&ctx, &hy).unwrap();
                let ddy = DeviceBuffer::from_slice(&ctx, &hdy).unwrap();
                let mut ddx: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, r * N).unwrap();
                let desc = ScanBackwardDescriptor {
                    kind: ScanKind::LogCumsumExp,
                    input_shape: shape,
                    scan_axis: 1,
                    reverse,
                    element: $kind,
                };
                let plan =
                    ScanBackwardPlan::<$T, 2>::select(&stream, &desc, PlanPreference::default())
                        .expect("select");
                plan.run(
                    &stream,
                    Workspace::None,
                    ScanBackwardArgs::<$T, 2> {
                        dy: TensorRef::new(ddy.as_slice(), shape, contiguous_stride(shape))
                            .unwrap(),
                        dx: TensorMut::new(ddx.as_slice_mut(), shape, contiguous_stride(shape))
                            .unwrap(),
                        x: Some(
                            TensorRef::new(dx_in.as_slice(), shape, contiguous_stride(shape))
                                .unwrap(),
                        ),
                        y: Some(
                            TensorRef::new(y_in.as_slice(), shape, contiguous_stride(shape))
                                .unwrap(),
                        ),
                    },
                )
                .expect("run");
                stream.synchronize().expect("sync");
                let mut got = vec![<$T>::from64(0.0); r * N];
                ddx.copy_to_host(&mut got).unwrap();
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let g = g.to64();
                    let ok = if w.is_nan() {
                        g.is_nan()
                    } else if w.is_infinite() {
                        g == *w
                    } else {
                        (g - w).abs() <= $tol * w.abs().max(1.0)
                    };
                    assert!(
                        ok,
                        "reverse={reverse} row {} col {}: got {g}, want {w}",
                        i / N,
                        i % N
                    );
                }
            }

            #[test]
            fn forward_scan() {
                run(false);
            }

            #[test]
            fn reverse_scan() {
                run(true);
            }
        }
    };
}

tests!(f32_, f32, ElementKind::F32, 1e-4);
tests!(f64_, f64, ElementKind::F64, 1e-10);
tests!(f16_, f16, ElementKind::F16, 3e-2);
tests!(bf16_, bf16, ElementKind::Bf16, 5e-2);
