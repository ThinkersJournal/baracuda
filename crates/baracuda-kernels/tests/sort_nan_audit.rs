// SPDX-License-Identifier: MIT OR Apache-2.0
//! Audit of `SortPlan`, `ArgsortPlan` and `MsortPlan` for NaN and tie handling (follow-up to the `TopkPlan` defect, baracuda#429).
//!
//! The invariants checked on every row, whatever the tie order of the algorithm:
//! 1. the returned indices are a permutation of `0..row_len`;
//! 2. `values[i] == input[indices[i]]` (NaN compares as "is NaN") for the plans that return values;
//! 3. the sequence is sorted under the total order **NaN is the largest value**: non-decreasing when ascending, non-increasing when
//!    descending (so NaN is last ascending and first descending);
//! 4. `MsortPlan` only: it is **stable**, so equal values appear in ascending index order (also when descending).
//!
//! Gated behind `device-tests`; run on the device under the machine-wide lock (`scripts/gpu-run.ps1`).
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ArgsortArgs, ArgsortDescriptor, ArgsortPlan, ElementKind, MsortArgs, MsortDescriptor,
    MsortPlan, PlanPreference, SortArgs, SortDescriptor, SortPlan, TensorMut, TensorRef, Workspace,
    contiguous_stride,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// The three plans, one result shape: per row `(values_if_returned, indices)`.
type Rows = Vec<(Option<Vec<f64>>, Vec<i32>)>;

macro_rules! run_plan {
    ($name:ident, $t:ty, $kind:expr, sort) => {
        run_plan!(@values $name, $t, $kind, SortPlan, SortDescriptor, SortArgs);
    };
    ($name:ident, $t:ty, $kind:expr, msort) => {
        run_plan!(@values $name, $t, $kind, MsortPlan, MsortDescriptor, MsortArgs);
    };
    (@values $name:ident, $t:ty, $kind:expr, $plan:ident, $desc:ident, $args:ident) => {
        fn $name(ctx: &Context, stream: &Stream, rows: &[Vec<$t>], descending: bool) -> Rows {
            let batch = rows.len() as i32;
            let n = rows[0].len() as i32;
            let flat: Vec<$t> = rows.iter().flatten().copied().collect();
            let dev_in = DeviceBuffer::from_slice(ctx, &flat).expect("upload");
            let mut dv: DeviceBuffer<$t> = DeviceBuffer::zeros(ctx, flat.len()).expect("alloc");
            let mut di: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, flat.len()).expect("alloc");
            let desc = $desc { batch, row_len: n, descending, element: $kind };
            let plan = $plan::<$t>::select(stream, &desc, PlanPreference::default()).expect("select");
            let sh = contiguous_stride([batch, n]);
            plan.run(
                stream,
                Workspace::None,
                $args::<$t> {
                    input: TensorRef::new(dev_in.as_slice(), [batch, n], sh).unwrap(),
                    values: TensorMut::new(dv.as_slice_mut(), [batch, n], sh).unwrap(),
                    indices: TensorMut::new(di.as_slice_mut(), [batch, n], sh).unwrap(),
                },
            )
            .expect("run");
            stream.synchronize().expect("sync");
            let mut v = vec![<$t>::default(); flat.len()];
            let mut i = vec![0i32; flat.len()];
            dv.copy_to_host(&mut v).expect("download values");
            di.copy_to_host(&mut i).expect("download indices");
            (0..rows.len())
                .map(|r| {
                    let s = r * n as usize;
                    let e = s + n as usize;
                    (Some(v[s..e].iter().map(|x| f64::from(*x)).collect()), i[s..e].to_vec())
                })
                .collect()
        }
    };
}

macro_rules! run_argsort {
    ($name:ident, $t:ty, $kind:expr) => {
        fn $name(ctx: &Context, stream: &Stream, rows: &[Vec<$t>], descending: bool) -> Rows {
            let batch = rows.len() as i32;
            let n = rows[0].len() as i32;
            let flat: Vec<$t> = rows.iter().flatten().copied().collect();
            let dev_in = DeviceBuffer::from_slice(ctx, &flat).expect("upload");
            let mut di: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, flat.len()).expect("alloc");
            let desc = ArgsortDescriptor {
                batch,
                row_len: n,
                descending,
                element: $kind,
            };
            let plan = ArgsortPlan::<$t>::select(stream, &desc, PlanPreference::default())
                .expect("select");
            let sh = contiguous_stride([batch, n]);
            let ws_bytes = plan.workspace_size();
            let mut ws_buf: DeviceBuffer<u8> =
                DeviceBuffer::zeros(ctx, ws_bytes.max(1)).expect("ws");
            let ws = if ws_bytes == 0 {
                Workspace::None
            } else {
                Workspace::Borrowed(ws_buf.as_slice_mut())
            };
            plan.run(
                stream,
                ws,
                ArgsortArgs::<$t> {
                    input: TensorRef::new(dev_in.as_slice(), [batch, n], sh).unwrap(),
                    indices: TensorMut::new(di.as_slice_mut(), [batch, n], sh).unwrap(),
                },
            )
            .expect("run");
            stream.synchronize().expect("sync");
            let mut i = vec![0i32; flat.len()];
            di.copy_to_host(&mut i).expect("download indices");
            (0..rows.len())
                .map(|r| {
                    let s = r * n as usize;
                    (None, i[s..s + n as usize].to_vec())
                })
                .collect()
        }
    };
}

run_plan!(sort_f32, f32, ElementKind::F32, sort);
run_plan!(sort_f64, f64, ElementKind::F64, sort);
run_plan!(msort_f32, f32, ElementKind::F32, msort);
run_plan!(msort_f64, f64, ElementKind::F64, msort);
run_argsort!(argsort_f32, f32, ElementKind::F32);
run_argsort!(argsort_f64, f64, ElementKind::F64);

/// NaN is the largest value.
fn cmp_nan_largest(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        _ => a.partial_cmp(&b).expect("no NaN here"),
    }
}

/// Check one row's output against the invariants; returns a description of the first violation.
fn violation(
    row: &[f64],
    got: &(Option<Vec<f64>>, Vec<i32>),
    descending: bool,
    stable: bool,
) -> Option<String> {
    let n = row.len();
    let idx = &got.1;
    let mut seen = vec![false; n];
    for &i in idx {
        if i < 0 || i as usize >= n || seen[i as usize] {
            return Some(format!("indices are not a permutation: {idx:?}"));
        }
        seen[i as usize] = true;
    }
    if let Some(v) = &got.0 {
        for (p, (&val, &i)) in v.iter().zip(idx).enumerate() {
            let src = row[i as usize];
            if !((val.is_nan() && src.is_nan()) || val == src) {
                return Some(format!(
                    "values[{p}] = {val} but input[indices[{p}]] = {src}"
                ));
            }
        }
    }
    let key = |p: usize| row[idx[p] as usize];
    for p in 1..n {
        let ord = cmp_nan_largest(key(p - 1), key(p));
        let bad = if descending {
            ord == std::cmp::Ordering::Less
        } else {
            ord == std::cmp::Ordering::Greater
        };
        if bad {
            return Some(format!(
                "not sorted at {p}: {} then {} ({})",
                key(p - 1),
                key(p),
                if descending {
                    "descending"
                } else {
                    "ascending"
                }
            ));
        }
        if stable && ord == std::cmp::Ordering::Equal && idx[p - 1] > idx[p] {
            return Some(format!(
                "not stable at {p}: equal keys at indices {} then {}",
                idx[p - 1],
                idx[p]
            ));
        }
    }
    None
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

fn random_rows(seed: u64, batch: usize, n: usize) -> Vec<Vec<f64>> {
    let mut g = Lcg(seed);
    (0..batch)
        .map(|_| {
            (0..n)
                .map(|_| match g.next() % 12 {
                    0 => f64::NAN,
                    1 => f64::INFINITY,
                    2 => f64::NEG_INFINITY,
                    k => (k as f64) - 6.0,
                })
                .collect()
        })
        .collect()
}

fn vectors() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, f64::NAN, 9.0, 2.0],
        vec![3.0, 5.0, 5.0, 1.0, 5.0, 2.0, 4.0, 5.0],
    ]
}

/// Audit one plan over the fixed vectors and the random rows; returns the violations found.
fn audit(
    name: &str,
    stable: bool,
    run32: &dyn Fn(&[Vec<f32>], bool) -> Rows,
    run64: &dyn Fn(&[Vec<f64>], bool) -> Rows,
    include_big: bool,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut shapes: Vec<(usize, Vec<Vec<f64>>)> = vec![
        (4, vec![vectors()[0].clone()]),
        (8, vec![vectors()[1].clone()]),
    ];
    let mut lens = vec![8usize, 33, 100, 257, 1000];
    if include_big {
        lens.push(2000);
    }
    for n in lens {
        shapes.push((n, random_rows(0xa0d17 ^ n as u64, 16, n)));
    }
    for (n, rows) in shapes {
        for descending in [false, true] {
            let rows32: Vec<Vec<f32>> = rows
                .iter()
                .map(|r| r.iter().map(|v| *v as f32).collect())
                .collect();
            let g32 = run32(&rows32, descending);
            let g64 = run64(&rows, descending);
            for (r, row) in rows.iter().enumerate() {
                for (tag, got) in [("f32", &g32[r]), ("f64", &g64[r])] {
                    if let Some(v) = violation(row, got, descending, stable) {
                        out.push(format!(
                            "{name} {tag} n={n} descending={descending} row {r}: {v}"
                        ));
                    }
                }
            }
        }
    }
    out
}

fn report(name: &str, v: Vec<String>) {
    eprintln!("sort_nan_audit: {name}: {} violation(s)", v.len());
    assert!(
        v.is_empty(),
        "{name}: {} violation(s) (NaN largest; stable only for msort); first 6:\n{}",
        v.len(),
        v.iter().take(6).cloned().collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn sort_is_sorted_with_nan_largest() {
    let (ctx, stream) = setup();
    let v = audit(
        "sort",
        false,
        &|r, d| sort_f32(&ctx, &stream, r, d),
        &|r, d| sort_f64(&ctx, &stream, r, d),
        false,
    );
    report("sort", v);
}

#[test]
fn msort_is_sorted_and_stable_with_nan_largest() {
    let (ctx, stream) = setup();
    let v = audit(
        "msort",
        true,
        &|r, d| msort_f32(&ctx, &stream, r, d),
        &|r, d| msort_f64(&ctx, &stream, r, d),
        false,
    );
    report("msort", v);
}

#[test]
fn argsort_is_sorted_with_nan_largest() {
    let (ctx, stream) = setup();
    let v = audit(
        "argsort",
        false,
        &|r, d| argsort_f32(&ctx, &stream, r, d),
        &|r, d| argsort_f64(&ctx, &stream, r, d),
        true,
    );
    report("argsort", v);
}
