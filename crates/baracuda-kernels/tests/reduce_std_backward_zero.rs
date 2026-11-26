// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `ReduceBackwardPlan + ReduceKind::Std` at a zero standard deviation.
//!
//! `y = std`; `dx = dy * (x - mean) / (m * y)`, `m = max(n - correction, 0)`. A constant group has `y == 0` and the kernel computed `0 / 0 = NaN`;
//! torch's `std_backward` masks `result == 0` to a zero gradient. A NaN in a group makes `y` NaN and still gives NaN.
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
            fn constant_group_gives_zero_and_nan_group_gives_nan() {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let input_shape = [3i32, 4];
                let dy_shape = [3i32, 1];
                let nan = <$T>::NAN;
                // Row 0 constant (std 0); row 1 ordinary ([1, 2, 3, 4], correction 1: var 5/3); row 2 holds a NaN.
                let hx: Vec<$T> = vec![2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0, nan, 3.0, 4.0];
                let std1 = (5.0f64 / 3.0).sqrt();
                let hy: Vec<$T> = vec![0.0, std1 as $T, nan];
                let hdy: Vec<$T> = vec![1.0, 2.0, 3.0];
                let dx_in = DeviceBuffer::from_slice(&ctx, &hx).unwrap();
                let dy_in = DeviceBuffer::from_slice(&ctx, &hy).unwrap();
                let ddy = DeviceBuffer::from_slice(&ctx, &hdy).unwrap();
                let mut ddx: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, 12).unwrap();
                let desc = ReduceBackwardDescriptor {
                    kind: ReduceKind::Std,
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
                assert!(
                    got[0..4].iter().all(|v| *v == 0.0),
                    "constant group: {:?}",
                    &got[0..4]
                );
                // Row 1: dy * (x - mean) / (m * std), m = 3, mean 2.5.
                for (k, x) in [1.0f64, 2.0, 3.0, 4.0].iter().enumerate() {
                    let want = 2.0 * (x - 2.5) / (3.0 * std1);
                    assert!(
                        (got[4 + k] as f64 - want).abs() <= $tol,
                        "ordinary group col {k}: {} vs {want}",
                        got[4 + k]
                    );
                }
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
