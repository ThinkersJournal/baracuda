// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU property test: `ReducePlan` LogSumExp and `ScanPlan` LogCumsumExp on rows holding NaN, +inf and -inf.
//!
//! Closed form (torch.logsumexp / torch.logcumsumexp): a NaN in the window gives NaN, otherwise a +inf gives +inf, otherwise an empty
//! finite set (only -inf) gives -inf, otherwise `m + ln(sum(exp(v - m)))` over the finite values. Before ROADMAP E0e the max-shift
//! computed `inf - inf = NaN` for an all -inf or +inf row, and a leading NaN of the running scan was dropped.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, ReduceArgs, ReduceDescriptor, ReduceKind, ReducePlan, ScanArgs,
    ScanDescriptor, ScanKind, ScanPlan, TensorMut, TensorRef, Workspace, contiguous_stride,
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

const ROWS: usize = 36;
const LEN: usize = 21;

fn rows(seed: u64) -> Vec<Vec<f64>> {
    let mut g = Lcg(seed);
    (0..ROWS)
        .map(|r| {
            (0..LEN)
                .map(|i| match r % 6 {
                    0 => f64::NEG_INFINITY,
                    1 if i == LEN / 2 => f64::INFINITY,
                    1 => f64::NEG_INFINITY,
                    2 if i == 0 => f64::NAN,
                    2 | 3 => (g.next() % 9) as f64 - 4.0,
                    _ => match g.next() % 16 {
                        0 => f64::NAN,
                        1 => f64::INFINITY,
                        2..=4 => f64::NEG_INFINITY,
                        k => (k as f64) - 9.0,
                    },
                })
                .collect()
        })
        .collect()
}

fn lse(vals: &[f64]) -> f64 {
    if vals.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    if vals.contains(&f64::INFINITY) {
        return f64::INFINITY;
    }
    let fin: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    if fin.is_empty() {
        return f64::NEG_INFINITY;
    }
    let m = fin.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + fin.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}

fn close(got: f64, want: f64, rel: f64) -> bool {
    if want.is_nan() {
        got.is_nan()
    } else if want.is_infinite() {
        got == want
    } else {
        (got - want).abs() <= rel * want.abs().max(1.0)
    }
}

macro_rules! lse_tests {
    ($red:ident, $scan:ident, $T:ty, $elem:expr, $to:expr, $from:expr, $rel:expr) => {
        #[test]
        fn $red() {
            let (ctx, stream) = setup();
            let data = rows(0x15e ^ $rel.to_bits());
            let flat: Vec<$T> = data.iter().flatten().map(|v| $to(*v)).collect();
            let shape = [ROWS as i32, LEN as i32];
            let oshape = [ROWS as i32, 1];
            let dx = DeviceBuffer::from_slice(&ctx, &flat).unwrap();
            let mut dy: DeviceBuffer<$T> = DeviceBuffer::from_slice(&ctx, &[$to(0.0); ROWS]).unwrap();
            let desc = ReduceDescriptor::<2> {
                kind: ReduceKind::LogSumExp,
                input_shape: shape,
                reduce_axis: 1,
                element: $elem,
                correction: 0,
            };
            let plan = ReducePlan::<$T, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
            plan.run(
                &stream,
                Workspace::None,
                ReduceArgs::<$T, 2> {
                    x: TensorRef::new(dx.as_slice(), shape, contiguous_stride(shape)).unwrap(),
                    y: TensorMut::new(dy.as_slice_mut(), oshape, contiguous_stride(oshape)).unwrap(),
                },
            )
            .unwrap();
            stream.synchronize().unwrap();
            let mut got = vec![$to(0.0); ROWS];
            dy.copy_to_host(&mut got).unwrap();
            let mut bad = Vec::new();
            for (r, row) in data.iter().enumerate() {
                let inputs: Vec<f64> = row.iter().map(|v| $from($to(*v))).collect();
                let want = lse(&inputs);
                let g = $from(got[r]);
                if !close(g, want, $rel) {
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
        fn $scan() {
            let (ctx, stream) = setup();
            let mut bad = Vec::new();
            for reverse in [false, true] {
                let data = rows(0x5ca ^ $rel.to_bits() ^ reverse as u64);
                let flat: Vec<$T> = data.iter().flatten().map(|v| $to(*v)).collect();
                let shape = [ROWS as i32, LEN as i32];
                let dx = DeviceBuffer::from_slice(&ctx, &flat).unwrap();
                let mut dy: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, flat.len()).unwrap();
                let desc = ScanDescriptor {
                    kind: ScanKind::LogCumsumExp,
                    input_shape: shape,
                    scan_axis: 1,
                    reverse,
                    element: $elem,
                };
                let plan = ScanPlan::<$T, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
                plan.run(
                    &stream,
                    Workspace::None,
                    ScanArgs::<$T, 2> {
                        x: TensorRef::new(dx.as_slice(), shape, contiguous_stride(shape)).unwrap(),
                        y: TensorMut::new(dy.as_slice_mut(), shape, contiguous_stride(shape)).unwrap(),
                    },
                )
                .unwrap();
                stream.synchronize().unwrap();
                let mut got = vec![$to(0.0); flat.len()];
                dy.copy_to_host(&mut got).unwrap();
                for (r, row) in data.iter().enumerate() {
                    let inputs: Vec<f64> = row.iter().map(|v| $from($to(*v))).collect();
                    for k in 0..LEN {
                        let window = if reverse { &inputs[k..] } else { &inputs[..=k] };
                        let want = lse(window);
                        let g = $from(got[r * LEN + k]);
                        if !close(g, want, $rel) {
                            bad.push(format!(
                                "reverse={reverse} row {r} pos {k}: got {g}, want {want} (row {row:?})"
                            ));
                        }
                    }
                }
            }
            assert!(
                bad.is_empty(),
                "{} wrong cells; first 3:\n{}",
                bad.len(),
                bad.iter().take(3).cloned().collect::<Vec<_>>().join("\n")
            );
        }
    };
}

lse_tests!(
    logsumexp_f32,
    log_cumsum_exp_f32,
    f32,
    ElementKind::F32,
    |v: f64| v as f32,
    |v: f32| v as f64,
    1e-5f64
);
lse_tests!(
    logsumexp_f64,
    log_cumsum_exp_f64,
    f64,
    ElementKind::F64,
    |v: f64| v,
    |v: f64| v,
    1e-12f64
);
lse_tests!(
    logsumexp_f16,
    log_cumsum_exp_f16,
    f16,
    ElementKind::F16,
    f16::from_f64,
    |v: f16| v.to_f64(),
    4e-3f64
);
lse_tests!(
    logsumexp_bf16,
    log_cumsum_exp_bf16,
    bf16,
    ElementKind::Bf16,
    bf16::from_f64,
    |v: bf16| v.to_f64(),
    3e-2f64
);
