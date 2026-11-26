// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `SortBackwardPlan` routes a gradient through the permutation of a sort whose input holds NaN and +-inf, and carries NaN, +inf
//! and -inf gradients unchanged (NaN sweep row, ROADMAP E0). The sort puts NaN last ascending and first descending (E0a total order); the
//! backward is `dx[row][indices[i]] = dy[row][i]`, a scatter, so a NaN input position receives the gradient of the slot it was sorted into
//! and a NaN gradient lands on its source position, nowhere else. f32 and f64.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, SortArgs, SortBackwardArgs, SortBackwardDescriptor,
    SortBackwardPlan, SortDescriptor, SortPlan, TensorMut, TensorRef, Workspace, contiguous_stride,
};

macro_rules! tests {
    ($name:ident, $T:ty, $kind:expr) => {
        mod $name {
            use super::*;

            fn run(descending: bool) {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let nan = <$T>::NAN;
                let inf = <$T>::INFINITY;
                let (batch, row_len) = (2i32, 8usize);
                let shape = [batch, row_len as i32];
                let input: Vec<$T> = vec![
                    3.0, nan, 1.0, inf, -inf, nan, 2.0, 0.5, // two NaN, both infinities
                    0.0, -0.0, nan, 5.0, 4.0, -inf, 1.0, 2.0,
                ];
                // Gradients with a NaN, +inf and -inf among them: dy[i] identifies the SLOT i.
                let dy: Vec<$T> = vec![
                    1.0, 2.0, nan, 4.0, inf, 6.0, -inf, 8.0, 9.0, nan, 11.0, 12.0, 13.0, 14.0, inf,
                    16.0,
                ];
                let din = DeviceBuffer::from_slice(&ctx, &input).unwrap();
                let mut dvals: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, input.len()).unwrap();
                let mut didx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).unwrap();
                let desc = SortDescriptor {
                    batch,
                    row_len: row_len as i32,
                    descending,
                    element: $kind,
                };
                SortPlan::<$T>::select(&stream, &desc, PlanPreference::default())
                    .expect("select")
                    .run(
                        &stream,
                        Workspace::None,
                        SortArgs::<$T> {
                            input: TensorRef::new(din.as_slice(), shape, contiguous_stride(shape))
                                .unwrap(),
                            values: TensorMut::new(
                                dvals.as_slice_mut(),
                                shape,
                                contiguous_stride(shape),
                            )
                            .unwrap(),
                            indices: TensorMut::new(
                                didx.as_slice_mut(),
                                shape,
                                contiguous_stride(shape),
                            )
                            .unwrap(),
                        },
                    )
                    .expect("fw");
                let ddy = DeviceBuffer::from_slice(&ctx, &dy).unwrap();
                let mut ddx: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, input.len()).unwrap();
                let bw = SortBackwardDescriptor {
                    batch,
                    row_len: row_len as i32,
                    element: $kind,
                };
                SortBackwardPlan::<$T>::select(&stream, &bw, PlanPreference::default())
                    .expect("select bw")
                    .run(
                        &stream,
                        Workspace::None,
                        SortBackwardArgs::<$T> {
                            dy: TensorRef::new(ddy.as_slice(), shape, contiguous_stride(shape))
                                .unwrap(),
                            indices: TensorRef::new(
                                didx.as_slice(),
                                shape,
                                contiguous_stride(shape),
                            )
                            .unwrap(),
                            dx: TensorMut::new(ddx.as_slice_mut(), shape, contiguous_stride(shape))
                                .unwrap(),
                        },
                    )
                    .expect("bw");
                stream.synchronize().expect("sync");
                let mut idx = vec![0i32; input.len()];
                didx.copy_to_host(&mut idx).unwrap();
                let mut vals = vec![0 as $T; input.len()];
                dvals.copy_to_host(&mut vals).unwrap();
                let mut dx = vec![0 as $T; input.len()];
                ddx.copy_to_host(&mut dx).unwrap();
                // The sort itself: NaN last ascending / first descending, and the indices are a permutation of each row.
                for row in 0..batch as usize {
                    let o = row * row_len;
                    let mut seen = vec![false; row_len];
                    for &i in &idx[o..o + row_len] {
                        assert!(!seen[i as usize], "row {row}: index {i} twice");
                        seen[i as usize] = true;
                    }
                    let nans = input[o..o + row_len].iter().filter(|v| v.is_nan()).count();
                    let got_nans = vals[o..o + row_len].iter().filter(|v| v.is_nan()).count();
                    assert_eq!(got_nans, nans, "row {row}: NaN count in the sorted values");
                    let block = if descending {
                        &vals[o..o + nans]
                    } else {
                        &vals[o + row_len - nans..o + row_len]
                    };
                    assert!(
                        block.iter().all(|v| v.is_nan()),
                        "row {row}: NaNs not in the expected block"
                    );
                }
                // The backward: a scatter, NaN / inf gradients carried exactly.
                let mut want = vec![0 as $T; input.len()];
                for row in 0..batch as usize {
                    let o = row * row_len;
                    for i in 0..row_len {
                        want[o + idx[o + i] as usize] = dy[o + i];
                    }
                }
                for (i, (g, w)) in dx.iter().zip(&want).enumerate() {
                    let ok = if w.is_nan() { g.is_nan() } else { g == w };
                    assert!(ok, "descending={descending} @ {i}: got {g}, want {w}");
                }
            }

            #[test]
            fn ascending() {
                run(false);
            }

            #[test]
            fn descending() {
                run(true);
            }
        }
    };
}

tests!(f32_, f32, ElementKind::F32);
tests!(f64_, f64, ElementKind::F64);
