// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `ReduceBackwardPlan + ReduceKind::Prod` is exact at zeros (torch's `prod_backward`).
//!
//! No zero in a group: `dx[j] = dy * y / x[j]`. A zero in the group: `dx[j] = dy * (prod_{k<j} x[k]) * (prod_{k>j} x[k])`, so the one zero
//! slot gets the product of the others, and every other slot gets 0 (two zeros: all 0). The kernel used to compute `dy * y / x` always:
//! 0 / 0 = NaN at the zero slot (and `0 * 0 / x = 0` elsewhere only by luck). A NaN in a group reaches the slots whose product holds it, with
//! or without a zero, exactly as torch's formulas do (the host reference below is that formula).
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, ReduceBackwardArgs, ReduceBackwardDescriptor, ReduceBackwardPlan,
    ReduceKind, TensorMut, TensorRef, Workspace, contiguous_stride,
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

/// Reference gradient of one group, in f64.
fn expected(x: &[f64], dy: f64) -> Vec<f64> {
    let y: f64 = x.iter().product();
    let has_zero = x.contains(&0.0);
    (0..x.len())
        .map(|j| {
            if !has_zero {
                dy * y / x[j]
            } else {
                let left: f64 = x[..j].iter().product();
                let right: f64 = x[j + 1..].iter().product();
                dy * left * right
            }
        })
        .collect()
}

fn small_groups() -> Vec<Vec<f64>> {
    vec![
        vec![2.0, 3.0, 4.0, 5.0],      // control: no zero
        vec![2.0, 0.0, 4.0, 5.0],      // one zero
        vec![0.0, 0.0, 4.0, 5.0],      // two zeros
        vec![0.0, 3.0, f64::NAN, 5.0], // a zero and a NaN
        vec![f64::NAN, 2.0, 3.0, 4.0], // NaN, no zero
        vec![2.0, 3.0, 4.0, 0.0],      // zero in the last slot
    ]
}

fn long_groups() -> Vec<Vec<f64>> {
    let base = |n: usize| {
        (0..n)
            .map(|k| 1.0 + 0.0002 * (k % 7) as f64)
            .collect::<Vec<f64>>()
    };
    let mut a = base(1000);
    a[500] = 0.0; // one zero: chunk boundaries do not matter
    let mut b = base(1000);
    b[3] = 0.0;
    b[999] = 0.0; // two zeros in different chunks
    vec![a, b]
}

macro_rules! tests {
    ($name:ident, $T:ty, $kind:expr, $tol:expr) => {
        mod $name {
            use super::*;

            fn run(groups: &[Vec<f64>]) {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let r = groups.len();
                let c = groups[0].len();
                let input_shape = [r as i32, c as i32];
                let dy_shape = [r as i32, 1];
                let dys: Vec<f64> = (0..r).map(|i| 1.0 + i as f64).collect();
                let hx: Vec<$T> = groups.iter().flatten().map(|v| <$T>::from64(*v)).collect();
                let hy: Vec<$T> = groups
                    .iter()
                    .map(|g| <$T>::from64(g.iter().product::<f64>()))
                    .collect();
                let hdy: Vec<$T> = dys.iter().map(|v| <$T>::from64(*v)).collect();
                let dx_in = DeviceBuffer::from_slice(&ctx, &hx).unwrap();
                let dy_in = DeviceBuffer::from_slice(&ctx, &hy).unwrap();
                let ddy = DeviceBuffer::from_slice(&ctx, &hdy).unwrap();
                let mut ddx: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, r * c).unwrap();
                let desc = ReduceBackwardDescriptor {
                    kind: ReduceKind::Prod,
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
                let mut got = vec![<$T>::from64(0.0); r * c];
                ddx.copy_to_host(&mut got).unwrap();
                for (gi, g) in groups.iter().enumerate() {
                    // The reference sees the values the device saw (rounded to the element type).
                    let xs: Vec<f64> = g.iter().map(|v| <$T>::from64(*v).to64()).collect();
                    let want = expected(&xs, dys[gi]);
                    for (j, w) in want.iter().enumerate() {
                        let v = got[gi * c + j].to64();
                        let ok = if w.is_nan() {
                            v.is_nan()
                        } else {
                            (v - w).abs() <= $tol * w.abs().max(1.0)
                        };
                        assert!(
                            ok,
                            "group {gi} slot {j}: got {v}, want {w} (x = {:?})",
                            &g[..g.len().min(6)]
                        );
                    }
                }
            }

            #[test]
            fn zeros_and_nan_in_short_groups() {
                run(&small_groups());
            }

            /// 1000 elements per group: several per thread, a zero mid-chunk, and zeros in two different chunks.
            #[test]
            fn zeros_in_long_groups() {
                run(&long_groups());
            }
        }
    };
}

tests!(f32_, f32, ElementKind::F32, 1e-4);
tests!(f64_, f64, ElementKind::F64, 1e-10);
tests!(f16_, f16, ElementKind::F16, 3e-2);
tests!(bf16_, bf16, ElementKind::Bf16, 5e-2);

/// Axis inference (the launcher finds the reduced axis from the broadcast `dy` / `y` strides): rank 3, the axis in the middle or last, size-1
/// axes elsewhere, a reduced axis of length 1, and a non-contiguous `x`. A shape whose axis cannot be told apart is refused by the plan
/// (single `reduce_axis`) or by the launcher (typed error 2), never guessed; the cases below are the ones that could have been confused.
mod axes {
    use super::*;

    /// Values drawn from a small set that includes 0 and NaN, deterministic per logical index.
    fn val(i: usize) -> f32 {
        [2.0, 0.0, 3.0, 0.5, 4.0, 0.0, 1.5, f32::NAN][(i * 5 + i / 3) % 8]
    }

    /// `x_strides`: physical strides of the saved input (any layout); `dy`, `y`, `dx` are contiguous.
    fn check(shape: [i32; 3], axis: usize, x_strides: [i64; 3]) {
        init().expect("driver init");
        let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
        let stream = Stream::new(&ctx).expect("stream");
        let n: usize = shape.iter().map(|d| *d as usize).product();
        let mut dy_shape = shape;
        dy_shape[axis] = 1;
        let m: usize = dy_shape.iter().map(|d| *d as usize).product();
        let logical = |lin: usize| -> [usize; 3] {
            let mut c = [0usize; 3];
            let mut r = lin;
            for d in (0..3).rev() {
                c[d] = r % shape[d] as usize;
                r /= shape[d] as usize;
            }
            c
        };
        let phys = |c: [usize; 3]| -> usize { (0..3).map(|d| c[d] * x_strides[d] as usize).sum() };
        let extent_phys = (0..3)
            .map(|d| (shape[d] as usize - 1) * x_strides[d] as usize)
            .sum::<usize>()
            + 1;
        let mut hx = vec![1.0f32; extent_phys];
        let xv: Vec<f32> = (0..n).map(val).collect();
        for (lin, v) in xv.iter().enumerate() {
            hx[phys(logical(lin))] = *v;
        }
        let dy_index = |c: [usize; 3]| -> usize {
            let mut cc = c;
            cc[axis] = 0;
            (cc[0] * dy_shape[1] as usize + cc[1]) * dy_shape[2] as usize + cc[2]
        };
        let mut hy = vec![1.0f32; m];
        for (lin, v) in xv.iter().enumerate() {
            hy[dy_index(logical(lin))] *= *v;
        }
        let hdy: Vec<f32> = (0..m).map(|i| 1.0 + i as f32).collect();
        // Reference per logical element, torch's formulas.
        let mut want = vec![0f64; n];
        for (lin, slot) in want.iter_mut().enumerate() {
            let c = logical(lin);
            let group: Vec<f64> = (0..shape[axis] as usize)
                .map(|k| {
                    let mut cc = c;
                    cc[axis] = k;
                    xv[(cc[0] * shape[1] as usize + cc[1]) * shape[2] as usize + cc[2]] as f64
                })
                .collect();
            *slot = expected(&group, hdy[dy_index(c)] as f64)[c[axis]];
        }
        let dx_in = DeviceBuffer::from_slice(&ctx, &hx).unwrap();
        let y_in = DeviceBuffer::from_slice(&ctx, &hy).unwrap();
        let ddy = DeviceBuffer::from_slice(&ctx, &hdy).unwrap();
        let mut ddx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n).unwrap();
        let desc = ReduceBackwardDescriptor {
            kind: ReduceKind::Prod,
            input_shape: shape,
            reduce_axis: axis as u8,
            element: ElementKind::F32,
            correction: 1,
        };
        let plan = ReduceBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
        let run = plan.run(
            &stream,
            Workspace::None,
            ReduceBackwardArgs::<f32, 3> {
                dy: TensorRef::new(ddy.as_slice(), dy_shape, contiguous_stride(dy_shape)).unwrap(),
                x: Some(TensorRef::new(dx_in.as_slice(), shape, x_strides).unwrap()),
                y: Some(
                    TensorRef::new(y_in.as_slice(), dy_shape, contiguous_stride(dy_shape)).unwrap(),
                ),
                dx: TensorMut::new(ddx.as_slice_mut(), shape, contiguous_stride(shape)).unwrap(),
            },
        );
        if x_strides != contiguous_stride(shape) {
            // The plan takes contiguous saved tensors only: a strided `x` is a typed refusal, never a guessed axis.
            assert!(run.is_err(), "strided x must be refused");
            return;
        }
        run.expect("run");
        stream.synchronize().expect("sync");
        let mut got = vec![0f32; n];
        ddx.copy_to_host(&mut got).unwrap();
        for (lin, w) in want.iter().enumerate() {
            let g = got[lin] as f64;
            let ok = if w.is_nan() {
                g.is_nan()
            } else {
                (g - w).abs() <= 1e-4 * w.abs().max(1.0)
            };
            assert!(
                ok,
                "shape {shape:?} axis {axis} strides {x_strides:?} element {:?}: got {g}, want {w}",
                logical(lin)
            );
        }
    }

    #[test]
    fn middle_axis() {
        check([3, 4, 2], 1, [8, 2, 1]);
    }

    #[test]
    fn last_axis_with_size_one_axis_elsewhere() {
        check([2, 1, 6], 2, [6, 6, 1]);
    }

    #[test]
    fn first_axis() {
        check([5, 2, 3], 0, [6, 3, 1]);
    }

    #[test]
    fn reduced_axis_of_length_one() {
        // One element per group: the gradient of a one-element product is `dy`, zero or not.
        check([3, 1, 2], 1, [2, 2, 1]);
    }

    #[test]
    fn non_contiguous_x_is_refused() {
        // `x` stored transposed (strides [1, 3, 12]): the plan refuses a non-contiguous saved `x`.
        check([3, 4, 2], 1, [1, 3, 12]);
    }
}
