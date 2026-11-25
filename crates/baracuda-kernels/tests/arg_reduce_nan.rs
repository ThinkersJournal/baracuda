// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `ArgReducePlan` argmax / argmin on rows with NaN, +-inf, -0.0 and ties.
//!
//! Spec (torch.argmax/argmin, and Fuel's CPU kernels at fuel origin/argext-nan@7235453af): the FIRST NaN wins, otherwise the first index
//! of the extremum (ties keep the earlier index; `-0.0` and `+0.0` are equal). Before ROADMAP E0g a NaN counted only when it sat at index 0.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ArgReduceArgs, ArgReduceDescriptor, ArgReduceKind, ArgReducePlan, ElementKind, PlanPreference,
    TensorMut, TensorRef, Workspace, contiguous_stride,
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

/// Closed form: first NaN, else first index of the extremum.
fn reference(row: &[f64], is_max: bool) -> i64 {
    if let Some(i) = row.iter().position(|v| v.is_nan()) {
        return i as i64;
    }
    let mut best = 0usize;
    for (i, &x) in row.iter().enumerate() {
        if (is_max && x > row[best]) || (!is_max && x < row[best]) {
            best = i;
        }
    }
    best as i64
}

const NAN: f64 = f64::NAN;

/// Fuel's spec vectors (argmax, argmin).
fn spec() -> Vec<(Vec<f64>, i64, i64)> {
    vec![
        (vec![1.0, NAN, 0.0, 5.0], 1, 1),
        (vec![1.0, 5.0, NAN], 2, 2),
        (vec![NAN, 1.0, 5.0], 0, 0),
        (vec![3.0, NAN, NAN, 1.0], 1, 1),
        (vec![NAN, NAN, NAN], 0, 0),
        (vec![5.0, 5.0, 2.0, 2.0], 0, 2),
        (vec![-0.0, 0.0, 0.0], 0, 0),
        (vec![0.0, -0.0, -0.0], 0, 0),
    ]
}

macro_rules! tests {
    ($spec:ident, $prop:ident, $T:ty, $elem:expr, $to:expr, $from:expr) => {
        fn run_rows(ctx: &Context, stream: &Stream, data: &[Vec<f64>], is_max: bool) -> Vec<i64> {
            let len = data[0].len();
            let rows = data.len();
            let flat: Vec<$T> = data.iter().flatten().map(|v| $to(*v)).collect();
            let shape = [rows as i32, len as i32];
            let oshape = [rows as i32, 1];
            let dx = DeviceBuffer::from_slice(ctx, &flat).unwrap();
            let mut dy: DeviceBuffer<i64> = DeviceBuffer::zeros(ctx, rows).unwrap();
            let desc = ArgReduceDescriptor {
                kind: if is_max {
                    ArgReduceKind::Argmax
                } else {
                    ArgReduceKind::Argmin
                },
                input_shape: shape,
                reduce_axis: 1,
                element: $elem,
            };
            let plan =
                ArgReducePlan::<$T, 2>::select(stream, &desc, PlanPreference::default()).unwrap();
            plan.run(
                stream,
                Workspace::None,
                ArgReduceArgs::<$T, 2> {
                    x: TensorRef::new(dx.as_slice(), shape, contiguous_stride(shape)).unwrap(),
                    y: TensorMut::new(dy.as_slice_mut(), oshape, contiguous_stride(oshape))
                        .unwrap(),
                },
            )
            .unwrap();
            stream.synchronize().unwrap();
            let mut got = vec![0i64; rows];
            dy.copy_to_host(&mut got).unwrap();
            got
        }

        #[test]
        fn $spec() {
            let (ctx, stream) = setup();
            for (row, want_max, want_min) in spec() {
                let got_max = run_rows(&ctx, &stream, std::slice::from_ref(&row), true)[0];
                let got_min = run_rows(&ctx, &stream, std::slice::from_ref(&row), false)[0];
                assert_eq!(got_max, want_max, "argmax of {row:?}");
                assert_eq!(got_min, want_min, "argmin of {row:?}");
            }
        }

        #[test]
        fn $prop() {
            let (ctx, stream) = setup();
            let mut g = Lcg(0xa66);
            // 400 rows of 37: every third row NaN-free, otherwise a NaN in about 1 of 9 cells, a 5-value alphabet (many ties).
            let data: Vec<Vec<f64>> = (0..400)
                .map(|r| {
                    (0..37)
                        .map(|_| {
                            if r % 3 != 0 && g.next() % 9 == 0 {
                                NAN
                            } else {
                                (g.next() % 5) as f64 - 2.0
                            }
                        })
                        .collect()
                })
                .collect();
            for is_max in [true, false] {
                let got = run_rows(&ctx, &stream, &data, is_max);
                let bad: Vec<String> = data
                    .iter()
                    .zip(&got)
                    .enumerate()
                    .filter(|(_, (row, g))| **g != reference(row, is_max))
                    .map(|(r, (row, g))| {
                        format!(
                            "row {r}: got {g}, want {} (row {row:?})",
                            reference(row, is_max)
                        )
                    })
                    .collect();
                assert!(
                    bad.is_empty(),
                    "is_max={is_max}: {} wrong rows; first 3:\n{}",
                    bad.len(),
                    bad.iter().take(3).cloned().collect::<Vec<_>>().join("\n")
                );
            }
            let _ = $from;
        }
    };
}

mod f32_ {
    use super::*;
    tests!(
        spec_f32,
        property_f32,
        f32,
        ElementKind::F32,
        |v: f64| v as f32,
        |v: f32| v as f64
    );
}
mod f64_ {
    use super::*;
    tests!(
        spec_f64,
        property_f64,
        f64,
        ElementKind::F64,
        |v: f64| v,
        |v: f64| v
    );

    /// Two f64 values equal at f32 precision are NOT a tie: argmax must pick the larger, argmin the smaller.
    #[test]
    fn f64_is_compared_in_f64() {
        let (ctx, stream) = setup();
        let hi = 1.0 + 2f64.powi(-40);
        assert_eq!(
            hi as f32, 1.0f32,
            "the fixture must collapse at f32 precision"
        );
        let rows = vec![vec![1.0, hi], vec![hi, 1.0]];
        assert_eq!(run_rows(&ctx, &stream, &rows, true), vec![1, 0]);
        assert_eq!(run_rows(&ctx, &stream, &rows, false), vec![0, 1]);
    }
}
mod f16_ {
    use super::*;
    tests!(
        spec_f16,
        property_f16,
        f16,
        ElementKind::F16,
        f16::from_f64,
        |v: f16| v.to_f64()
    );
}
mod bf16_ {
    use super::*;
    tests!(
        spec_bf16,
        property_bf16,
        bf16,
        ElementKind::Bf16,
        bf16::from_f64,
        |v: bf16| v.to_f64()
    );
}
