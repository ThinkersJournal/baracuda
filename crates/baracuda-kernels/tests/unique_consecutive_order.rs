// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `UniqueConsecutivePlan` is `torch.unique_consecutive` (return_counts): runs in ROW ORDER with RUN-LENGTH counts.
//!
//! Up to 0.1.0-alpha.210 the output slot of a run came from an atomic counter (block-race order) and `counts` was always 1. The oracle here
//! is a plain host loop. Rules pinned: every NaN is its own run (NaN != NaN), `-0.0` and `+0.0` are one run (first element kept), `counter`
//! is the true run count (it can exceed `max_unique`; extra runs are dropped, cells past the counter untouched), input need not be sorted,
//! rows are independent, and a row longer than one block (1024) keeps its runs in order across the chunk boundary. Each case is run many
//! times, so a slot race (which depends on warp scheduling) would show.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    Element, ElementKind, PlanPreference, TensorMut, TensorRef, UniqueConsecutiveArgs,
    UniqueConsecutiveDescriptor, UniqueConsecutivePlan, Workspace, contiguous_stride,
};

trait Bits: Copy {
    fn same_bits_or_eq(a: Self, b: Self) -> bool;
    /// Equality as the kernel defines it (`==`).
    fn eqv(a: Self, b: Self) -> bool;
}
impl Bits for f32 {
    fn same_bits_or_eq(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }
    fn eqv(a: f32, b: f32) -> bool {
        a == b
    }
}
impl Bits for f64 {
    fn same_bits_or_eq(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }
    fn eqv(a: f64, b: f64) -> bool {
        a == b
    }
}
impl Bits for i32 {
    fn same_bits_or_eq(a: i32, b: i32) -> bool {
        a == b
    }
    fn eqv(a: i32, b: i32) -> bool {
        a == b
    }
}
impl Bits for i64 {
    fn same_bits_or_eq(a: i64, b: i64) -> bool {
        a == b
    }
    fn eqv(a: i64, b: i64) -> bool {
        a == b
    }
}

/// Host oracle for one row: (values, counts) of the runs of `==`-equal neighbours.
fn oracle<T: Bits>(row: &[T]) -> (Vec<T>, Vec<i32>) {
    let mut v: Vec<T> = Vec::new();
    let mut c: Vec<i32> = Vec::new();
    for (i, &a) in row.iter().enumerate() {
        if i == 0 || !T::eqv(a, row[i - 1]) {
            v.push(a);
            c.push(1);
        } else {
            *c.last_mut().unwrap() += 1;
        }
    }
    (v, c)
}

fn run_once<T: Element + Bits>(
    ctx: &Context,
    stream: &Stream,
    x: &[T],
    batch: usize,
    row_len: usize,
    max_unique: usize,
    sentinel: T,
    kind: ElementKind,
) -> (Vec<T>, Vec<i32>, Vec<i32>) {
    let (b, n, m) = (batch as i32, row_len as i32, max_unique as i32);
    let in_sh = [b, n];
    let out_sh = [b, m];
    let dx = DeviceBuffer::from_slice(ctx, x).unwrap();
    let mut dv = DeviceBuffer::from_slice(ctx, &vec![sentinel; batch * max_unique]).unwrap();
    let mut dc = DeviceBuffer::from_slice(ctx, &vec![-7i32; batch * max_unique]).unwrap();
    let mut dn = DeviceBuffer::from_slice(ctx, &vec![-77i32; batch]).unwrap();
    let desc = UniqueConsecutiveDescriptor {
        batch: b,
        row_len: n,
        max_unique: m,
        return_counts: true,
        element: kind,
    };
    let plan =
        UniqueConsecutivePlan::<T>::select(stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        stream,
        Workspace::None,
        UniqueConsecutiveArgs::<T> {
            input: TensorRef::new(dx.as_slice(), in_sh, contiguous_stride(in_sh)).unwrap(),
            values: TensorMut::new(dv.as_slice_mut(), out_sh, contiguous_stride(out_sh)).unwrap(),
            counts: TensorMut::new(dc.as_slice_mut(), out_sh, contiguous_stride(out_sh)).unwrap(),
            counter: TensorMut::new(dn.as_slice_mut(), [b], contiguous_stride([b])).unwrap(),
        },
    )
    .expect("run");
    stream.synchronize().unwrap();
    let mut v = vec![sentinel; batch * max_unique];
    let mut c = vec![0i32; batch * max_unique];
    let mut k = vec![0i32; batch];
    dv.copy_to_host(&mut v).unwrap();
    dc.copy_to_host(&mut c).unwrap();
    dn.copy_to_host(&mut k).unwrap();
    (v, c, k)
}

/// Checks `reps` runs of the whole batch against the host oracle, row by row.
fn check<T: Element + Bits + core::fmt::Debug>(
    name: &str,
    x: &[T],
    batch: usize,
    row_len: usize,
    max_unique: usize,
    sentinel: T,
    kind: ElementKind,
    reps: usize,
) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    for rep in 0..reps {
        let (v, c, k) = run_once(&ctx, &stream, x, batch, row_len, max_unique, sentinel, kind);
        for r in 0..batch {
            let (ov, oc) = oracle(&x[r * row_len..(r + 1) * row_len]);
            assert_eq!(
                k[r] as usize,
                ov.len(),
                "{name} rep {rep} row {r}: run count"
            );
            let kept = ov.len().min(max_unique);
            for j in 0..kept {
                assert!(
                    T::same_bits_or_eq(v[r * max_unique + j], ov[j]),
                    "{name} rep {rep} row {r} slot {j}: value {:?} != oracle {:?} (order)",
                    v[r * max_unique + j],
                    ov[j]
                );
                assert_eq!(
                    c[r * max_unique + j],
                    oc[j],
                    "{name} rep {rep} row {r} slot {j}: run length"
                );
            }
            for j in kept..max_unique {
                assert_eq!(
                    c[r * max_unique + j],
                    -7,
                    "{name}: cell past the counter touched"
                );
            }
        }
    }
}

#[test]
fn runs_in_row_order_with_run_lengths_f32_i32_i64() {
    let x = [1.0f32, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0];
    check("f32", &x, 1, 8, 8, -7.0, ElementKind::F32, 50);
    let x = [1i32, 1, 2, 2, 2, 3, 1, 1];
    check("i32", &x, 1, 8, 8, -7, ElementKind::I32, 50);
    let x = [9i64, 9, -(1 << 40), -(1 << 40), -(1 << 40), 3, 9, 9];
    check("i64", &x, 1, 8, 8, -7, ElementKind::I64, 50);
    let x = [2.5f64, 2.5, 2.5, 0.0, 7.0, 7.0];
    check("f64", &x, 1, 6, 6, -7.0, ElementKind::F64, 50);
}

#[test]
fn unsorted_input_and_independent_rows() {
    let x = [3i32, 1, 3, 3, 1, 1, 7, 7, 7, 3, 3, 1];
    check("i32 2 rows", &x, 2, 6, 6, -7, ElementKind::I32, 30);
}

#[test]
fn nan_is_never_equal_and_signed_zeros_are_one_run() {
    let nan = f32::NAN;
    let x = [nan, nan, 1.0, 1.0, nan, -0.0, 0.0, 0.0, 1.0];
    check("f32 nan and zeros", &x, 1, 9, 9, -7.0, ElementKind::F32, 30);
}

#[test]
fn a_row_longer_than_one_block_keeps_order_across_chunks() {
    // 3000 elements: runs of length 1, 700, 2 and a long 1500 run that straddles the 1024 and 2048 chunk boundaries.
    let mut x: Vec<i32> = Vec::new();
    for (v, len) in [1usize, 700, 2, 1500, 3, 1, 1, 792].into_iter().enumerate() {
        for _ in 0..len {
            x.push(v as i32);
        }
    }
    assert_eq!(x.len(), 3000);
    check("i32 3000", &x, 1, 3000, 3000, -7, ElementKind::I32, 20);
    // Many short runs across chunks (every other element differs): 2500 runs.
    let y: Vec<i32> = (0..5000).map(|i| i / 2).collect();
    check(
        "i32 5000 pairs",
        &y,
        1,
        5000,
        5000,
        -7,
        ElementKind::I32,
        10,
    );
    // A row that is one run for its whole length (3 chunks, the run never ends until the row does).
    check(
        "i32 one run",
        &vec![4i32; 3000],
        1,
        3000,
        4,
        -7,
        ElementKind::I32,
        10,
    );
}

#[test]
fn counter_is_the_true_run_count_when_it_exceeds_max_unique() {
    let x: Vec<i32> = (0..40).map(|i| i / 2).collect(); // 20 runs of 2
    check("i32 overflow", &x, 1, 40, 5, -7, ElementKind::I32, 20);
}

#[test]
fn zero_width_rows_count_zero() {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let (_, _, k) = run_once::<f32>(&ctx, &stream, &[], 2, 0, 1, -7.0, ElementKind::F32);
    assert_eq!(k, vec![0, 0]);
}
