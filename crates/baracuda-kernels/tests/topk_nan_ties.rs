// SPDX-License-Identifier: MIT OR Apache-2.0
//! Correctness of `TopkPlan` with NaNs and with ties (a published defect, fixed in the alpha after 186).
//!
//! Contract under test (the same as torch.topk and Fuel's CPU kernel): **NaN sorts largest** (so it is first for `largest` and last for
//! smallest), and **equal values come out in ascending index order**. Before the fix the block-bitonic comparator treated a NaN as
//! "equal to everything" (neither `<` nor `>`), which is not a strict weak order, so the network could lose a larger finite value:
//! `[1, NaN, 9, 2]`, `k = 2`, largest returned values `[2, NaN]` (the 9 was dropped). Ties were also unordered.
//!
//! Gated behind `device-tests`; run on the device under the machine-wide lock (`scripts/gpu-run.ps1`).
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, TensorMut, TensorRef, TopkArgs, TopkDescriptor, TopkPlan,
    Workspace, contiguous_stride,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// The CPU reference: NaN is the largest value, ties break to the lower index.
fn reference_f64(row: &[f64], k: usize, largest: bool) -> (Vec<u64>, Vec<i32>) {
    let key = |v: f64| -> (u8, f64) { if v.is_nan() { (1, 0.0) } else { (0, v) } };
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| {
        let (ka, kb) = (key(row[a]), key(row[b]));
        let ord =
            ka.0.cmp(&kb.0)
                .then(ka.1.partial_cmp(&kb.1).expect("no NaN in the key"));
        let ord = if largest { ord.reverse() } else { ord };
        ord.then(a.cmp(&b))
    });
    let top = &idx[..k];
    (
        top.iter().map(|&i| row[i].to_bits()).collect(),
        top.iter().map(|&i| i as i32).collect(),
    )
}

macro_rules! run_topk {
    ($name:ident, $t:ty, $kind:expr) => {
        /// Run `TopkPlan` on `rows` (all of `row_len`), returning each row's values (as bits) and indices.
        fn $name(
            ctx: &Context,
            stream: &Stream,
            rows: &[Vec<$t>],
            k: usize,
            largest: bool,
        ) -> Vec<(Vec<u64>, Vec<i32>)> {
            let batch = rows.len() as i32;
            let row_len = rows[0].len() as i32;
            let k_i = k as i32;
            let flat: Vec<$t> = rows.iter().flatten().copied().collect();
            let dev_in = DeviceBuffer::from_slice(ctx, &flat).expect("upload");
            let mut dev_vals: DeviceBuffer<$t> =
                DeviceBuffer::zeros(ctx, rows.len() * k).expect("alloc values");
            let mut dev_idx: DeviceBuffer<i32> =
                DeviceBuffer::zeros(ctx, rows.len() * k).expect("alloc indices");
            let desc = TopkDescriptor {
                batch,
                row_len,
                k: k_i,
                largest,
                element: $kind,
            };
            let plan =
                TopkPlan::<$t>::select(stream, &desc, PlanPreference::default()).expect("select");
            plan.run(
                stream,
                Workspace::None,
                TopkArgs::<$t> {
                    input: TensorRef::new(
                        dev_in.as_slice(),
                        [batch, row_len],
                        contiguous_stride([batch, row_len]),
                    )
                    .unwrap(),
                    values: TensorMut::new(
                        dev_vals.as_slice_mut(),
                        [batch, k_i],
                        contiguous_stride([batch, k_i]),
                    )
                    .unwrap(),
                    indices: TensorMut::new(
                        dev_idx.as_slice_mut(),
                        [batch, k_i],
                        contiguous_stride([batch, k_i]),
                    )
                    .unwrap(),
                },
            )
            .expect("run");
            stream.synchronize().expect("sync");
            let mut v = vec![<$t>::default(); rows.len() * k];
            let mut i = vec![0i32; rows.len() * k];
            dev_vals.copy_to_host(&mut v).expect("download values");
            dev_idx.copy_to_host(&mut i).expect("download indices");
            (0..rows.len())
                .map(|r| {
                    (
                        v[r * k..(r + 1) * k]
                            .iter()
                            .map(|x| f64::from(*x).to_bits())
                            .collect(),
                        i[r * k..(r + 1) * k].to_vec(),
                    )
                })
                .collect()
        }
    };
}

run_topk!(run_f32, f32, ElementKind::F32);
run_topk!(run_f64, f64, ElementKind::F64);

/// Compare the device result of one row with the reference; `NaN` payloads compare as "is NaN".
fn same(got: &(Vec<u64>, Vec<i32>), want: &(Vec<u64>, Vec<i32>)) -> bool {
    got.1 == want.1
        && got.0.iter().zip(&want.0).all(|(a, b)| {
            let (fa, fb) = (f64::from_bits(*a), f64::from_bits(*b));
            (fa.is_nan() && fb.is_nan()) || a == b
        })
}

#[test]
fn nan_does_not_make_topk_drop_a_larger_finite_value() {
    let (ctx, stream) = setup();
    // Fuel's vector 1: the 9 must survive. NaN sorts largest, so the answer is [NaN, 9] at indices [1, 2].
    let row = vec![1.0f32, f32::NAN, 9.0, 2.0];
    let got = run_f32(&ctx, &stream, std::slice::from_ref(&row), 2, true).remove(0);
    let want = reference_f64(
        &row.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
        2,
        true,
    );
    assert!(same(&got, &want), "got {got:?}, want {want:?}");
    // smallest: NaN is last, so [1, 2] at [0, 3].
    let got = run_f32(&ctx, &stream, std::slice::from_ref(&row), 2, false).remove(0);
    let want = reference_f64(
        &row.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
        2,
        false,
    );
    assert!(same(&got, &want), "smallest: got {got:?}, want {want:?}");
}

#[test]
fn ties_come_out_in_ascending_index_order() {
    let (ctx, stream) = setup();
    // Fuel's vector 2: the three largest are 5, 5, 5 at indices [1, 2, 4] (the 5 at index 7 is the fourth).
    let row = vec![3.0f32, 5.0, 5.0, 1.0, 5.0, 2.0, 4.0, 5.0];
    let got = run_f32(&ctx, &stream, std::slice::from_ref(&row), 3, true).remove(0);
    let want = reference_f64(
        &row.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
        3,
        true,
    );
    assert_eq!(want.1, vec![1, 2, 4], "the reference itself");
    assert!(same(&got, &want), "got {got:?}, want {want:?}");
}

/// A small deterministic generator (no dev-dependency): values from a small pool so ties are common, plus NaNs and infinities.
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

fn random_rows(seed: u64, batch: usize, row_len: usize) -> Vec<Vec<f64>> {
    let mut g = Lcg(seed);
    (0..batch)
        .map(|_| {
            (0..row_len)
                .map(|_| match g.next() % 12 {
                    0 => f64::NAN,
                    1 => f64::INFINITY,
                    2 => f64::NEG_INFINITY,
                    n => (n as f64) - 6.0, // 3..=11 -> -3..=5: many ties
                })
                .collect()
        })
        .collect()
}

#[test]
fn random_rows_with_nans_and_ties_match_the_reference_f32_and_f64() {
    let (ctx, stream) = setup();
    let mut disagreements = Vec::new();
    for (row_len, k) in [
        (8usize, 3usize),
        (33, 5),
        (64, 64),
        (100, 17),
        (257, 64),
        (1000, 40),
    ] {
        for largest in [true, false] {
            let rows = random_rows(0x5eed ^ (row_len as u64), 24, row_len);
            let want: Vec<_> = rows.iter().map(|r| reference_f64(r, k, largest)).collect();
            let rows32: Vec<Vec<f32>> = rows
                .iter()
                .map(|r| r.iter().map(|v| *v as f32).collect())
                .collect();
            let got32 = run_f32(&ctx, &stream, &rows32, k, largest);
            let got64 = run_f64(&ctx, &stream, &rows, k, largest);
            for (r, w) in want.iter().enumerate() {
                if !same(&got32[r], w) {
                    disagreements.push(format!(
                        "f32 row_len {row_len} k {k} largest {largest} row {r}: got idx {:?}, want {:?}",
                        got32[r].1, w.1
                    ));
                }
                if !same(&got64[r], w) {
                    disagreements.push(format!(
                        "f64 row_len {row_len} k {k} largest {largest} row {r}: got idx {:?}, want {:?}",
                        got64[r].1, w.1
                    ));
                }
            }
        }
    }
    assert!(
        disagreements.is_empty(),
        "{} row(s) disagree with the CPU reference (NaN largest, ties lowest index); first 5:\n{}",
        disagreements.len(),
        disagreements
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
