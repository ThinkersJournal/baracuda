// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `ReduceBackwardPlan + ReduceKind::Norm2` at a zero norm.
//!
//! `y = sqrt(sum(x^2))`; `dx = dy * x / y`. When every `x` in a group is zero, `y == 0` and the kernel computed `0 / 0 = NaN`. torch's
//! `norm_backward` masks that case to 0 (the subgradient at the origin). A NaN in a group makes `y` NaN and still gives NaN.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, ReduceBackwardArgs, ReduceBackwardDescriptor, ReduceBackwardPlan,
    ReduceKind, TensorMut, TensorRef, Workspace, contiguous_stride,
};

macro_rules! tests {
    ($name:ident, $T:ty, $kind:expr, $tol:expr) => {
        mod $name {
            use super::*;

            #[test]
            fn zero_group_gives_zero_and_nan_group_gives_nan() {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let input_shape = [3i32, 4];
                let dy_shape = [3i32, 1];
                // Row 0 all zero; row 1 ordinary; row 2 holds a NaN.
                let nan = <$T>::NAN;
                let hx: Vec<$T> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0, 1.0, nan, 2.0, 3.0];
                let hy: Vec<$T> = vec![0.0, 5.0, nan];
                let hdy: Vec<$T> = vec![1.0, 2.0, 3.0];
                let dx_in = DeviceBuffer::from_slice(&ctx, &hx).unwrap();
                let dy_in = DeviceBuffer::from_slice(&ctx, &hy).unwrap();
                let ddy = DeviceBuffer::from_slice(&ctx, &hdy).unwrap();
                let mut ddx: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, 12).unwrap();
                let desc = ReduceBackwardDescriptor {
                    kind: ReduceKind::Norm2,
                    input_shape,
                    reduce_axis: 1,
                    element: $kind,
                    correction: 1,
                };
                let plan =
                    ReduceBackwardPlan::<$T, 2>::select(&stream, &desc, PlanPreference::default())
                        .expect("select");
                plan.run(
                    &stream,
                    Workspace::None,
                    ReduceBackwardArgs::<$T, 2> {
                        dy: TensorRef::new(ddy.as_slice(), dy_shape, contiguous_stride(dy_shape))
                            .unwrap(),
                        x: Some(
                            TensorRef::new(
                                dx_in.as_slice(),
                                input_shape,
                                contiguous_stride(input_shape),
                            )
                            .unwrap(),
                        ),
                        y: Some(
                            TensorRef::new(dy_in.as_slice(), dy_shape, contiguous_stride(dy_shape))
                                .unwrap(),
                        ),
                        dx: TensorMut::new(
                            ddx.as_slice_mut(),
                            input_shape,
                            contiguous_stride(input_shape),
                        )
                        .unwrap(),
                    },
                )
                .expect("run");
                stream.synchronize().expect("sync");
                let mut got = vec![0 as $T; 12];
                ddx.copy_to_host(&mut got).unwrap();
                // Row 0: exactly zero (not NaN).
                assert!(
                    got[0..4].iter().all(|v| *v == 0.0),
                    "zero group: {:?}",
                    &got[0..4]
                );
                // Row 1: dy * x / y = 2 * [1, 2, 2, 4] / 5.
                for (k, x) in [1.0f64, 2.0, 2.0, 4.0].iter().enumerate() {
                    let want = 2.0 * x / 5.0;
                    assert!(
                        (got[4 + k] as f64 - want).abs() <= $tol,
                        "ordinary group col {k}: {} vs {want}",
                        got[4 + k]
                    );
                }
                // Row 2: a NaN norm gives NaN everywhere in the group.
                assert!(
                    got[8..12].iter().all(|v| v.is_nan()),
                    "NaN group: {:?}",
                    &got[8..12]
                );
            }
        }
    };
}

tests!(f32_, f32, ElementKind::F32, 1e-5);
tests!(f64_, f64, ElementKind::F64, 1e-12);
