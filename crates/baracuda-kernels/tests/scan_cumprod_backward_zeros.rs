// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `ScanBackwardPlan + ScanKind::Cumprod` gives the exact gradient when `x` holds zeros (torch convention).
//!
//! `y[i] = prod_{k<=i} x[k]` (forward) or `prod_{k>=i} x[k]` (reverse). The gradient is
//! `dx[j] = sum_{i in suffix(j)} dy[i] * prod_{k in range(i), k != j} x[k]`, which is finite and exact at `x[j] == 0`. The kernel used to
//! divide `dy * y / x[j]`, giving inf / NaN at every zero (and, through `y == 0`, a wrong 0 at the other positions of the row).
//! A NaN anywhere in the row still reaches `dx` (the reference below propagates it the same way).
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, ScanBackwardArgs, ScanBackwardDescriptor, ScanBackwardPlan,
    ScanKind, TensorMut, TensorRef, Workspace, contiguous_stride,
};

const R: usize = 5;
const C: usize = 4;

fn rows() -> Vec<[f64; C]> {
    vec![
        [2.0, 0.0, 3.0, 4.0],
        [0.0, 0.0, 3.0, 4.0],
        [1.5, 2.0, 0.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [2.0, f64::NAN, 3.0, 4.0],
    ]
}
const DY: [f64; C] = [1.0, 2.0, 3.0, 4.0];

/// Reference gradient of one row, in f64. Index loops mirror the formula in the module doc.
#[allow(clippy::needless_range_loop)]
fn expected_row(x: &[f64; C], reverse: bool) -> [f64; C] {
    let mut dx = [0.0; C];
    for j in 0..C {
        let (lo, hi) = if reverse { (0, j) } else { (j, C - 1) };
        let mut acc = 0.0;
        for i in lo..=hi {
            let (klo, khi) = if reverse { (i, C - 1) } else { (0, i) };
            let mut p = 1.0;
            for k in klo..=khi {
                if k != j {
                    p *= x[k];
                }
            }
            acc += DY[i] * p;
        }
        // torch divides by `x[j]` when the row has no zero, so a NaN `x[j]` makes `dx[j]` NaN (its `y[i >= j]` are NaN too).
        dx[j] = if x[j].is_nan() { f64::NAN } else { acc };
    }
    dx
}

#[allow(clippy::needless_range_loop)]
fn forward_row(x: &[f64; C], reverse: bool) -> [f64; C] {
    let mut y = [0.0; C];
    for i in 0..C {
        let (klo, khi) = if reverse { (i, C - 1) } else { (0, i) };
        y[i] = (klo..=khi).map(|k| x[k]).product();
    }
    y
}

macro_rules! tests {
    ($name:ident, $T:ty, $kind:expr, $tol:expr) => {
        mod $name {
            use super::*;

            fn run(reverse: bool) {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let shape = [R as i32, C as i32];
                let mut hx = Vec::new();
                let mut hy = Vec::new();
                let mut hdy = Vec::new();
                let mut want = Vec::new();
                for row in rows() {
                    hx.extend(row.iter().map(|v| *v as $T));
                    hy.extend(forward_row(&row, reverse).iter().map(|v| *v as $T));
                    hdy.extend(DY.iter().map(|v| *v as $T));
                    want.extend(expected_row(&row, reverse));
                }
                let dx_in = DeviceBuffer::from_slice(&ctx, &hx).unwrap();
                let dy_in = DeviceBuffer::from_slice(&ctx, &hy).unwrap();
                let ddy = DeviceBuffer::from_slice(&ctx, &hdy).unwrap();
                let mut ddx: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, R * C).unwrap();
                let desc = ScanBackwardDescriptor {
                    kind: ScanKind::Cumprod,
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
                            TensorRef::new(dy_in.as_slice(), shape, contiguous_stride(shape))
                                .unwrap(),
                        ),
                    },
                )
                .expect("run");
                stream.synchronize().expect("sync");
                let mut got = vec![0 as $T; R * C];
                ddx.copy_to_host(&mut got).unwrap();
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let g = *g as f64;
                    let ok = if w.is_nan() {
                        g.is_nan()
                    } else {
                        (g - w).abs() <= $tol * w.abs().max(1.0)
                    };
                    assert!(
                        ok,
                        "reverse={reverse} row {} col {}: got {g}, want {w}",
                        i / C,
                        i % C
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

tests!(f32_, f32, ElementKind::F32, 1e-5);
tests!(f64_, f64, ElementKind::F64, 1e-12);
