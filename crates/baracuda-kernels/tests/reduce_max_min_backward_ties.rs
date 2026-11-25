// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `ReduceBackwardPlan` Max / Min split the gradient evenly across tied positions.
//!
//! `dx[c] = dy[g] / count[g]` where `count[g]` is the number of positions of the reduced group `g` with `x == y` (count clamped to 1, so a
//! group with none, a NaN row, gets zeros). This is torch.amax / amin (`scale_grad_by_count`), JAX (`_reduce_chooser_jvp_rule`) and Fuel's
//! CPU `reduce_max_to_backward`. Before ROADMAP E0h every tied position received the full `dy`.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, ReduceBackwardArgs, ReduceBackwardDescriptor, ReduceBackwardPlan,
    ReduceKind, TensorMut, TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// Forward max / min of one group with NaN propagation (what the forward kernels return).
fn forward(group: &[f64], is_max: bool) -> f64 {
    let mut acc = if is_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    for &v in group {
        if acc.is_nan() {
            break;
        }
        if v.is_nan() || (is_max && v > acc) || (!is_max && v < acc) {
            acc = v;
        }
    }
    acc
}

macro_rules! tests {
    ($name:ident, $T:ty, $elem:expr, $to:expr, $from:expr, $rel:expr) => {
        #[test]
        fn $name() {
            let (ctx, stream) = setup();
            // [3, 17, 4] reduced over axis 1 (strided groups), 3-D so the reduced axis is not the last one.
            let shape = [3i32, 17, 4];
            let axis = 1usize;
            let mut dshape = shape;
            dshape[axis] = 1;
            let strides = contiguous_stride(shape);
            let dstrides = contiguous_stride(dshape);
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let g = Lcg(0xbeef);
            let mut g = g;
            for is_max in [true, false] {
                // 5-value alphabet (many ties); some groups get a NaN.
                let x: Vec<f64> = (0..n)
                    .map(|_| {
                        if g.next() % 23 == 0 {
                            f64::NAN
                        } else {
                            (g.next() % 5) as f64 - 2.0
                        }
                    })
                    .collect();
                let x: Vec<$T> = x.iter().map(|v| $to(*v)).collect();
                let xf: Vec<f64> = x.iter().map(|v| $from(*v)).collect();
                let dyv: Vec<f64> = (0..(3 * 4)).map(|_| (g.next() % 4 + 1) as f64).collect();
                let dy: Vec<$T> = dyv.iter().map(|v| $to(*v)).collect();
                let mut y = vec![$to(0.0); 3 * 4];
                let mut want = vec![0f64; n];
                for i in 0..3usize {
                    for k in 0..4usize {
                        let idx = |j: usize| {
                            (i as i64 * strides[0] + j as i64 * strides[1] + k as i64 * strides[2])
                                as usize
                        };
                        let group: Vec<f64> = (0..17).map(|j| xf[idx(j)]).collect();
                        let yv = forward(&group, is_max);
                        y[(i as i64 * dstrides[0] + k as i64 * dstrides[2]) as usize] = $to(yv);
                        let yv = $from($to(yv));
                        let count = group.iter().filter(|v| **v == yv).count().max(1) as f64;
                        let dyg =
                            $from(dy[(i as i64 * dstrides[0] + k as i64 * dstrides[2]) as usize]);
                        for j in 0..17 {
                            if group[j] == yv {
                                want[idx(j)] = dyg / count;
                            }
                        }
                    }
                }
                let dev_x = DeviceBuffer::from_slice(&ctx, &x).unwrap();
                let dev_y = DeviceBuffer::from_slice(&ctx, &y).unwrap();
                let dev_dy = DeviceBuffer::from_slice(&ctx, &dy).unwrap();
                let mut dev_dx: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, n).unwrap();
                let desc = ReduceBackwardDescriptor::<3> {
                    kind: if is_max {
                        ReduceKind::Max
                    } else {
                        ReduceKind::Min
                    },
                    input_shape: shape,
                    reduce_axis: axis as u8,
                    element: $elem,
                    correction: 1,
                };
                let plan =
                    ReduceBackwardPlan::<$T, 3>::select(&stream, &desc, PlanPreference::default())
                        .unwrap();
                plan.run(
                    &stream,
                    Workspace::None,
                    ReduceBackwardArgs::<$T, 3> {
                        dy: TensorRef::new(dev_dy.as_slice(), dshape, dstrides).unwrap(),
                        x: Some(TensorRef::new(dev_x.as_slice(), shape, strides).unwrap()),
                        y: Some(TensorRef::new(dev_y.as_slice(), dshape, dstrides).unwrap()),
                        dx: TensorMut::new(dev_dx.as_slice_mut(), shape, strides).unwrap(),
                    },
                )
                .unwrap();
                stream.synchronize().unwrap();
                let mut got = vec![$to(0.0); n];
                dev_dx.copy_to_host(&mut got).unwrap();
                let bad: Vec<String> = (0..n)
                    .filter(|&i| ($from(got[i]) - want[i]).abs() > $rel * want[i].abs().max(1.0))
                    .map(|i| {
                        format!(
                            "is_max={is_max} cell {i}: got {}, want {}",
                            $from(got[i]),
                            want[i]
                        )
                    })
                    .collect();
                assert!(
                    bad.is_empty(),
                    "{} wrong cells; first 3:\n{}",
                    bad.len(),
                    bad.iter().take(3).cloned().collect::<Vec<_>>().join("\n")
                );
            }
        }
    };
}

tests!(
    split_ties_f32,
    f32,
    ElementKind::F32,
    |v: f64| v as f32,
    |v: f32| v as f64,
    1e-6f64
);
tests!(
    split_ties_f64,
    f64,
    ElementKind::F64,
    |v: f64| v,
    |v: f64| v,
    1e-12f64
);
tests!(
    split_ties_f16,
    f16,
    ElementKind::F16,
    f16::from_f64,
    |v: f16| v.to_f64(),
    2e-3f64
);
tests!(
    split_ties_bf16,
    bf16,
    ElementKind::Bf16,
    bf16::from_f64,
    |v: bf16| v.to_f64(),
    1.6e-2f64
);

/// The documented examples: `[3, 3, 1]` max with `dy = 4` gives `[2, 2, 0]`; an all-equal row of 4 gives `dy / 4` each; a row holding a NaN gives zeros.
#[test]
fn split_ties_examples_f32() {
    let (ctx, stream) = setup();
    let rows = [
        vec![3.0f32, 3.0, 1.0],
        vec![2.0, 2.0, 2.0],
        vec![1.0, f32::NAN, 5.0],
    ];
    let want = [
        [2.0f32, 2.0, 0.0],
        [4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0],
        [0.0, 0.0, 0.0],
    ];
    for (row, want) in rows.iter().zip(&want) {
        let shape = [1i32, 3];
        let dshape = [1i32, 1];
        let yv = if row.iter().any(|v| v.is_nan()) {
            f32::NAN
        } else {
            row.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        };
        let dev_x = DeviceBuffer::from_slice(&ctx, row).unwrap();
        let dev_y = DeviceBuffer::from_slice(&ctx, &[yv]).unwrap();
        let dev_dy = DeviceBuffer::from_slice(&ctx, &[4.0f32]).unwrap();
        let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 3).unwrap();
        let desc = ReduceBackwardDescriptor::<2> {
            kind: ReduceKind::Max,
            input_shape: shape,
            reduce_axis: 1,
            element: ElementKind::F32,
            correction: 1,
        };
        let plan = ReduceBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
            .unwrap();
        plan.run(
            &stream,
            Workspace::None,
            ReduceBackwardArgs::<f32, 2> {
                dy: TensorRef::new(dev_dy.as_slice(), dshape, contiguous_stride(dshape)).unwrap(),
                x: Some(TensorRef::new(dev_x.as_slice(), shape, contiguous_stride(shape)).unwrap()),
                y: Some(
                    TensorRef::new(dev_y.as_slice(), dshape, contiguous_stride(dshape)).unwrap(),
                ),
                dx: TensorMut::new(dev_dx.as_slice_mut(), shape, contiguous_stride(shape)).unwrap(),
            },
        )
        .unwrap();
        stream.synchronize().unwrap();
        let mut got = vec![0f32; 3];
        dev_dx.copy_to_host(&mut got).unwrap();
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() < 1e-6,
                "row {row:?}: got {got:?}, want {want:?}"
            );
        }
    }
}
