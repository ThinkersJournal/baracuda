// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU property test: `ReducePlan` Max / Min and the `reduce_{max,min}_to` FFI on rows with NaN, +-inf and -0.0.
//!
//! Convention (torch.max/min/amax/amin, and Fuel's CPU Maximum/Minimum fold): a NaN anywhere in the row gives NaN; otherwise the
//! extreme, where the first-seen of `-0.0` / `+0.0` is kept. Before ROADMAP E0f the kernels used `fmax` / `fmin`, which skip NaN.
#![cfg(feature = "device-tests")]

use core::ffi::c_void;

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, ReduceArgs, ReduceDescriptor, ReduceKind, ReducePlan, TensorMut,
    TensorRef, Workspace, contiguous_stride,
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

const ROWS: usize = 40;
const LEN: usize = 23;

fn rows(seed: u64) -> Vec<Vec<f64>> {
    let mut g = Lcg(seed);
    (0..ROWS)
        .map(|r| {
            (0..LEN)
                .map(|i| match r % 5 {
                    0 if i == 0 => f64::NAN,
                    1 if i == LEN - 1 => f64::NAN,
                    2 => f64::NEG_INFINITY,
                    3 => (g.next() % 7) as f64 - 3.0,
                    _ => match g.next() % 14 {
                        0 => f64::NAN,
                        1 => f64::INFINITY,
                        2 => f64::NEG_INFINITY,
                        3 => -0.0,
                        4 => 0.0,
                        k => (k as f64) - 8.0,
                    },
                })
                .collect()
        })
        .collect()
}

/// NaN propagates; otherwise strict improvement keeps the first-seen of equal values (so the first zero's sign survives).
fn reference(row: &[f64], is_max: bool) -> f64 {
    let mut acc = if is_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    for &x in row {
        if acc.is_nan() {
            break;
        }
        if x.is_nan() || (is_max && x > acc) || (!is_max && x < acc) {
            acc = x;
        }
    }
    acc
}

fn same(got: f64, want: f64) -> bool {
    if want.is_nan() {
        got.is_nan()
    } else {
        got == want && got.is_sign_negative() == want.is_sign_negative()
    }
}

macro_rules! tests {
    ($max:ident, $min:ident, $T:ty, $elem:expr, $to:expr, $from:expr) => {
        fn run_axis(is_max: bool) {
            let (ctx, stream) = setup();
            let data = rows(0xfeed ^ is_max as u64);
            let flat: Vec<$T> = data.iter().flatten().map(|v| $to(*v)).collect();
            let shape = [ROWS as i32, LEN as i32];
            let oshape = [ROWS as i32, 1];
            let dx = DeviceBuffer::from_slice(&ctx, &flat).unwrap();
            let mut dy: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, ROWS).unwrap();
            let desc = ReduceDescriptor::<2> {
                kind: if is_max {
                    ReduceKind::Max
                } else {
                    ReduceKind::Min
                },
                input_shape: shape,
                reduce_axis: 1,
                element: $elem,
                correction: 0,
            };
            let plan =
                ReducePlan::<$T, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
            plan.run(
                &stream,
                Workspace::None,
                ReduceArgs::<$T, 2> {
                    x: TensorRef::new(dx.as_slice(), shape, contiguous_stride(shape)).unwrap(),
                    y: TensorMut::new(dy.as_slice_mut(), oshape, contiguous_stride(oshape))
                        .unwrap(),
                },
            )
            .unwrap();
            stream.synchronize().unwrap();
            let mut got = vec![$to(0.0); ROWS];
            dy.copy_to_host(&mut got).unwrap();
            let mut bad = Vec::new();
            for (r, row) in data.iter().enumerate() {
                let inputs: Vec<f64> = row.iter().map(|v| $from($to(*v))).collect();
                let want = reference(&inputs, is_max);
                let g = $from(got[r]);
                if !same(g, want) {
                    bad.push(format!("row {r}: got {g}, want {want} (row {row:?})"));
                }
            }
            assert!(
                bad.is_empty(),
                "{} wrong rows; first 3:\n{}",
                bad.len(),
                bad.iter().take(3).cloned().collect::<Vec<_>>().join("\n")
            );
        }

        #[test]
        fn $max() {
            run_axis(true);
        }

        #[test]
        fn $min() {
            run_axis(false);
        }
    };
}

mod f32_ {
    use super::*;
    tests!(
        max_f32,
        min_f32,
        f32,
        ElementKind::F32,
        |v: f64| v as f32,
        |v: f32| v as f64
    );
}
mod f64_ {
    use super::*;
    tests!(
        max_f64,
        min_f64,
        f64,
        ElementKind::F64,
        |v: f64| v,
        |v: f64| v
    );
}
mod f16_ {
    use super::*;
    tests!(
        max_f16,
        min_f16,
        f16,
        ElementKind::F16,
        f16::from_f64,
        |v: f16| v.to_f64()
    );
}
mod bf16_ {
    use super::*;
    tests!(
        max_bf16,
        min_bf16,
        bf16,
        ElementKind::Bf16,
        bf16::from_f64,
        |v: bf16| v.to_f64()
    );
}

/// `reduce_{max,min}_to_f32` (the `ReduceMaxTo` / `ReduceMinTo` autograd primitives), reducing dim 0 of `[LEN, ROWS]`.
#[test]
fn reduce_to_max_min_f32() {
    let (ctx, stream) = setup();
    let data = rows(0xabc);
    // Column c of the [LEN, ROWS] input is data[c]; reduce over dim 0 -> out [1, ROWS].
    let mut host = vec![0f32; LEN * ROWS];
    for (c, row) in data.iter().enumerate() {
        for (i, v) in row.iter().enumerate() {
            host[i * ROWS + c] = *v as f32;
        }
    }
    let in_shape = [LEN as i32, ROWS as i32];
    let out_shape = [1i32, ROWS as i32];
    let in_stride = [ROWS as i64, 1i64];
    for is_max in [true, false] {
        let dev_src = DeviceBuffer::from_slice(&ctx, &host).unwrap();
        let mut dev_dst: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, ROWS).unwrap();
        let status = unsafe {
            let (src, dst) = (
                dev_src.as_slice().as_raw().0 as *const c_void,
                dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            );
            if is_max {
                baracuda_kernels_sys::baracuda_kernels_reduce_max_to_f32_run(
                    src,
                    dst,
                    in_shape.as_ptr(),
                    in_stride.as_ptr(),
                    2,
                    out_shape.as_ptr(),
                    core::ptr::null_mut(),
                    0,
                    stream.as_raw(),
                )
            } else {
                baracuda_kernels_sys::baracuda_kernels_reduce_min_to_f32_run(
                    src,
                    dst,
                    in_shape.as_ptr(),
                    in_stride.as_ptr(),
                    2,
                    out_shape.as_ptr(),
                    core::ptr::null_mut(),
                    0,
                    stream.as_raw(),
                )
            }
        };
        assert_eq!(status, 0, "reduce_to status {status}");
        stream.synchronize().unwrap();
        let mut got = vec![0f32; ROWS];
        dev_dst.copy_to_host(&mut got).unwrap();
        let mut bad = Vec::new();
        for (c, row) in data.iter().enumerate() {
            let inputs: Vec<f64> = row.iter().map(|v| (*v as f32) as f64).collect();
            // The `_to` path has no first-zero rule: compare values, not the sign of a zero.
            let want = reference(&inputs, is_max);
            let g = got[c] as f64;
            if !(want.is_nan() && g.is_nan() || g == want) {
                bad.push(format!(
                    "is_max={is_max} col {c}: got {g}, want {want} (row {row:?})"
                ));
            }
        }
        assert!(
            bad.is_empty(),
            "{} wrong columns; first 3:\n{}",
            bad.len(),
            bad.iter().take(3).cloned().collect::<Vec<_>>().join("\n")
        );
    }
}
