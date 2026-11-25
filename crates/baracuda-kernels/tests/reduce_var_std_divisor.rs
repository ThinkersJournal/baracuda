// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `Var` / `Std` use the torch divisor `max(0, n - correction)` (forward and the Var backward).
//!
//! `extent == correction` gives NaN (0/0), `extent < correction` gives inf (NaN when M2 is 0), never 0 and never a clamped divisor of 1.
//! Before ROADMAP E0i the kernels returned 0 (and the backward divided by 1) whenever `n - correction <= 0`.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, ReduceArgs, ReduceBackwardArgs, ReduceBackwardDescriptor,
    ReduceBackwardPlan, ReduceDescriptor, ReduceKind, ReducePlan, TensorMut, TensorRef, Workspace,
    contiguous_stride,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn same(got: f64, want: f64, rel: f64) -> bool {
    if want.is_nan() {
        got.is_nan()
    } else if want.is_infinite() {
        got == want
    } else {
        (got - want).abs() <= rel * want.abs().max(1.0)
    }
}

macro_rules! tests {
    ($fwd:ident, $bwd:ident, $T:ty, $elem:expr, $to:expr, $from:expr, $rel:expr) => {
        #[test]
        fn $fwd() {
            let (ctx, stream) = setup();
            let cases: [(&[f64], i32, f64, f64); 6] = [
                (&[5.0], 1, f64::NAN, f64::NAN),
                (&[1.0, 3.0], 2, f64::INFINITY, f64::INFINITY),
                (&[5.0], 2, f64::NAN, f64::NAN),
                (&[1.0, 3.0], 1, 2.0, 2f64.sqrt()),
                (&[5.0], 0, 0.0, 0.0),
                (&[1.0, 2.0, 4.0, 8.0], 1, 9.583333333333334, 9.583333333333334f64.sqrt()),
            ];
            for (row, correction, want_var, want_std) in cases {
                let x: Vec<$T> = row.iter().map(|v| $to(*v)).collect();
                let shape = [row.len() as i32];
                let o = [1i32];
                for (kind, want) in [(ReduceKind::Var, want_var), (ReduceKind::Std, want_std)] {
                    let dx = DeviceBuffer::from_slice(&ctx, &x).unwrap();
                    let mut dy: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, 1).unwrap();
                    let desc = ReduceDescriptor::<1> {
                        kind,
                        input_shape: shape,
                        reduce_axis: 0,
                        element: $elem,
                        correction,
                    };
                    let plan =
                        ReducePlan::<$T, 1>::select(&stream, &desc, PlanPreference::default())
                            .unwrap();
                    plan.run(
                        &stream,
                        Workspace::None,
                        ReduceArgs::<$T, 1> {
                            x: TensorRef::new(dx.as_slice(), shape, contiguous_stride(shape))
                                .unwrap(),
                            y: TensorMut::new(dy.as_slice_mut(), o, contiguous_stride(o)).unwrap(),
                        },
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                    let mut got = vec![$to(0.0); 1];
                    dy.copy_to_host(&mut got).unwrap();
                    let g = $from(got[0]);
                    assert!(
                        same(g, want, $rel),
                        "{kind:?} of {row:?} at correction {correction}: got {g}, want {want}"
                    );
                }
            }
        }

        #[test]
        fn $bwd() {
            let (ctx, stream) = setup();
            // Var backward: dx = dy * 2 * (x - mean) / max(0, n - correction), dy = 1.
            let cases: [(&[f64], i32, &[f64]); 4] = [
                (&[5.0], 1, &[f64::NAN]),
                (&[1.0, 3.0], 2, &[f64::NEG_INFINITY, f64::INFINITY]),
                (&[1.0, 3.0], 1, &[-2.0, 2.0]),
                (&[5.0], 0, &[0.0]),
            ];
            for (row, correction, want) in cases {
                let x: Vec<$T> = row.iter().map(|v| $to(*v)).collect();
                let shape = [row.len() as i32];
                let o = [1i32];
                let dev_x = DeviceBuffer::from_slice(&ctx, &x).unwrap();
                let dev_y = DeviceBuffer::from_slice(&ctx, &[$to(0.0)]).unwrap();
                let dev_dy = DeviceBuffer::from_slice(&ctx, &[$to(1.0)]).unwrap();
                let mut dev_dx: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, row.len()).unwrap();
                let desc = ReduceBackwardDescriptor::<1> {
                    kind: ReduceKind::Var,
                    input_shape: shape,
                    reduce_axis: 0,
                    element: $elem,
                    correction,
                };
                let plan = ReduceBackwardPlan::<$T, 1>::select(
                    &stream,
                    &desc,
                    PlanPreference::default(),
                )
                .unwrap();
                plan.run(
                    &stream,
                    Workspace::None,
                    ReduceBackwardArgs::<$T, 1> {
                        dy: TensorRef::new(dev_dy.as_slice(), o, contiguous_stride(o)).unwrap(),
                        x: Some(
                            TensorRef::new(dev_x.as_slice(), shape, contiguous_stride(shape))
                                .unwrap(),
                        ),
                        y: Some(
                            TensorRef::new(dev_y.as_slice(), o, contiguous_stride(o)).unwrap(),
                        ),
                        dx: TensorMut::new(dev_dx.as_slice_mut(), shape, contiguous_stride(shape))
                            .unwrap(),
                    },
                )
                .unwrap();
                stream.synchronize().unwrap();
                let mut got = vec![$to(0.0); row.len()];
                dev_dx.copy_to_host(&mut got).unwrap();
                for (i, w) in want.iter().enumerate() {
                    let g = $from(got[i]);
                    assert!(
                        same(g, *w, $rel),
                        "Var backward of {row:?} at correction {correction}, cell {i}: got {g}, want {w}"
                    );
                }
            }
        }
    };
}

mod f32_ {
    use super::*;
    tests!(
        var_std_f32,
        var_backward_f32,
        f32,
        ElementKind::F32,
        |v: f64| v as f32,
        |v: f32| v as f64,
        1e-5f64
    );
}
mod f64_ {
    use super::*;
    tests!(
        var_std_f64,
        var_backward_f64,
        f64,
        ElementKind::F64,
        |v: f64| v,
        |v: f64| v,
        1e-12f64
    );
}
mod f16_ {
    use super::*;
    tests!(
        var_std_f16,
        var_backward_f16,
        f16,
        ElementKind::F16,
        f16::from_f64,
        |v: f16| v.to_f64(),
        4e-3f64
    );
}
mod bf16_ {
    use super::*;
    tests!(
        var_std_bf16,
        var_backward_bf16,
        bf16,
        ElementKind::Bf16,
        bf16::from_f64,
        |v: bf16| v.to_f64(),
        3e-2f64
    );
}
