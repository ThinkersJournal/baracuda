// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU property test: `ScanPlan` cummax / cummin and `ScanBackwardPlan` on rows with NaN, +-inf, -0.0 and many ties.
//!
//! Convention (torch.cummax / torch.cummin): NaN is sticky. Once the running winner is NaN it stays NaN, and a NaN input becomes the
//! winner; among non-NaN values the first occurrence wins a tie. Before ROADMAP E0d a leading NaN returned the scan identity
//! (`-inf` / `+inf`) and a later NaN was skipped.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, ScanArgs, ScanBackwardArgs, ScanBackwardDescriptor,
    ScanBackwardPlan, ScanDescriptor, ScanKind, ScanPlan, TensorMut, TensorRef, Workspace,
    contiguous_stride,
};

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

const ROWS: usize = 24;
const LEN: usize = 37;

fn rows(seed: u64) -> Vec<Vec<f64>> {
    let mut g = Lcg(seed);
    (0..ROWS)
        .map(|r| {
            (0..LEN)
                .map(|i| {
                    // Every third row starts with a NaN; every fifth row has one only at the end.
                    if i == 0 && r % 3 == 0 || i == LEN - 1 && r % 5 == 0 {
                        return f64::NAN;
                    }
                    match g.next() % 12 {
                        0 => f64::NAN,
                        1 => f64::INFINITY,
                        2 => f64::NEG_INFINITY,
                        3 => -0.0,
                        k => (k as f64) - 7.0,
                    }
                })
                .collect()
        })
        .collect()
}

/// Walk one row in scan order; returns (value, winner index) per step.
fn walk(row: &[f64], is_max: bool, reverse: bool) -> Vec<(f64, usize)> {
    let order: Vec<usize> = if reverse {
        (0..row.len()).rev().collect()
    } else {
        (0..row.len()).collect()
    };
    let mut out = vec![(0.0, 0); row.len()];
    let mut best: Option<(f64, usize)> = None;
    for &i in &order {
        let x = row[i];
        best = match best {
            None => Some((x, i)),
            Some((b, _)) if b.is_nan() => best,
            Some(_) if x.is_nan() => Some((x, i)),
            Some((b, _)) if (is_max && x > b) || (!is_max && x < b) => Some((x, i)),
            keep => keep,
        };
        out[i] = best.unwrap();
    }
    out
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b
}

macro_rules! cum_tests {
    ($fwd:ident, $bwd:ident, $T:ty, $kind:expr, $elem:expr, $is_max:expr) => {
        #[test]
        fn $fwd() {
            let (ctx, stream) = setup();
            let mut bad = Vec::new();
            for reverse in [false, true] {
                let data = rows(0xc0ffee ^ reverse as u64);
                let flat: Vec<$T> = data.iter().flatten().map(|v| *v as $T).collect();
                let shape = [ROWS as i32, LEN as i32];
                let dx = DeviceBuffer::from_slice(&ctx, &flat).unwrap();
                let mut dy: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, flat.len()).unwrap();
                let desc = ScanDescriptor {
                    kind: $kind,
                    input_shape: shape,
                    scan_axis: 1,
                    reverse,
                    element: $elem,
                };
                let plan =
                    ScanPlan::<$T, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
                plan.run(
                    &stream,
                    Workspace::None,
                    ScanArgs::<$T, 2> {
                        x: TensorRef::new(dx.as_slice(), shape, contiguous_stride(shape)).unwrap(),
                        y: TensorMut::new(dy.as_slice_mut(), shape, contiguous_stride(shape))
                            .unwrap(),
                    },
                )
                .unwrap();
                stream.synchronize().unwrap();
                let mut got = vec![0 as $T; flat.len()];
                dy.copy_to_host(&mut got).unwrap();
                for (r, row) in data.iter().enumerate() {
                    let want = walk(row, $is_max, reverse);
                    for i in 0..LEN {
                        let g = got[r * LEN + i] as f64;
                        if !same(g, want[i].0) {
                            bad.push(format!(
                                "reverse={reverse} row {r} pos {i}: got {g}, want {} (row {row:?})",
                                want[i].0
                            ));
                        }
                    }
                }
            }
            assert!(
                bad.is_empty(),
                "{} mismatches; first 4:\n{}",
                bad.len(),
                bad.iter().take(4).cloned().collect::<Vec<_>>().join("\n")
            );
        }

        #[test]
        fn $bwd() {
            let (ctx, stream) = setup();
            let mut bad = Vec::new();
            for reverse in [false, true] {
                let data = rows(0xbadc0de ^ reverse as u64);
                let flat: Vec<$T> = data.iter().flatten().map(|v| *v as $T).collect();
                let mut g = Lcg(7);
                let dyv: Vec<$T> = (0..flat.len()).map(|_| (g.next() % 9) as $T - 4.0).collect();
                let shape = [ROWS as i32, LEN as i32];
                let dev_x = DeviceBuffer::from_slice(&ctx, &flat).unwrap();
                let dev_dy = DeviceBuffer::from_slice(&ctx, &dyv).unwrap();
                let mut dev_dx: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, flat.len()).unwrap();
                let desc = ScanBackwardDescriptor {
                    kind: $kind,
                    input_shape: shape,
                    scan_axis: 1,
                    reverse,
                    element: $elem,
                };
                let plan =
                    ScanBackwardPlan::<$T, 2>::select(&stream, &desc, PlanPreference::default())
                        .unwrap();
                plan.run(
                    &stream,
                    Workspace::None,
                    ScanBackwardArgs::<$T, 2> {
                        dy: TensorRef::new(dev_dy.as_slice(), shape, contiguous_stride(shape))
                            .unwrap(),
                        dx: TensorMut::new(dev_dx.as_slice_mut(), shape, contiguous_stride(shape))
                            .unwrap(),
                        x: Some(
                            TensorRef::new(dev_x.as_slice(), shape, contiguous_stride(shape))
                                .unwrap(),
                        ),
                        y: None,
                    },
                )
                .unwrap();
                stream.synchronize().unwrap();
                let mut got = vec![0 as $T; flat.len()];
                dev_dx.copy_to_host(&mut got).unwrap();
                for (r, row) in data.iter().enumerate() {
                    let w = walk(row, $is_max, reverse);
                    let mut want = vec![0f64; LEN];
                    for i in 0..LEN {
                        want[w[i].1] += dyv[r * LEN + i] as f64;
                    }
                    for i in 0..LEN {
                        if got[r * LEN + i] as f64 != want[i] {
                            bad.push(format!(
                                "reverse={reverse} row {r} pos {i}: dx got {}, want {} (row {row:?})",
                                got[r * LEN + i], want[i]
                            ));
                        }
                    }
                }
            }
            assert!(
                bad.is_empty(),
                "{} mismatches; first 4:\n{}",
                bad.len(),
                bad.iter().take(4).cloned().collect::<Vec<_>>().join("\n")
            );
        }
    };
}

cum_tests!(
    cummax_f32_matches_reference,
    cummax_backward_f32_matches_reference,
    f32,
    ScanKind::Cummax,
    ElementKind::F32,
    true
);
cum_tests!(
    cummin_f32_matches_reference,
    cummin_backward_f32_matches_reference,
    f32,
    ScanKind::Cummin,
    ElementKind::F32,
    false
);
cum_tests!(
    cummax_f64_matches_reference,
    cummax_backward_f64_matches_reference,
    f64,
    ScanKind::Cummax,
    ElementKind::F64,
    true
);
cum_tests!(
    cummin_f64_matches_reference,
    cummin_backward_f64_matches_reference,
    f64,
    ScanKind::Cummin,
    ElementKind::F64,
    false
);
