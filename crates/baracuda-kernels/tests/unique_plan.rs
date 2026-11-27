// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `UniquePlan` is the sort-then-dedup composition (ROADMAP E0b), torch `unique(sorted=True, return_counts=True)`.
//!
//! Up to 0.1.0-alpha.189 `run` sorted and returned `Ok` without writing anything; alpha.191 made it decline. It now returns, per row, the
//! distinct values ascending with the NaN-largest order of `SortPlan`, run-length counts, and the actual distinct count in `counter`:
//! - every NaN is its own value (NaN != NaN) and NaN sorts last;
//! - `-0.0` and `+0.0` are one value, and the value kept is the first of the run in sort order;
//! - `counter` is the TRUE distinct count even when it exceeds `max_unique` (the extra values are dropped, cells past `max_unique` untouched);
//! - output slot order is the sorted order (no atomic slot race), counts are run lengths (not 1).
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    Element, ElementKind, PlanPreference, TensorMut, TensorRef, UniqueArgs, UniqueDescriptor,
    UniquePlan, Workspace, contiguous_stride,
};

/// Runs `UniquePlan` on `batch` rows of `row_len` and returns (values, counts, counter); values/counts are pre-filled with `sentinel`.
fn run<T: Element + Copy>(
    x: &[T],
    batch: usize,
    row_len: usize,
    max_unique: usize,
    sentinel: T,
    kind: ElementKind,
) -> (Vec<T>, Vec<i32>, Vec<i32>) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let (b, n, m) = (batch as i32, row_len as i32, max_unique as i32);
    let in_sh = [b, n];
    let out_sh = [b, m];
    let dx = DeviceBuffer::from_slice(&ctx, x).unwrap();
    let mut ss = DeviceBuffer::from_slice(&ctx, &vec![sentinel; batch * row_len]).unwrap();
    let mut si: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, batch * row_len).unwrap();
    let mut dv = DeviceBuffer::from_slice(&ctx, &vec![sentinel; batch * max_unique]).unwrap();
    let mut dc = DeviceBuffer::from_slice(&ctx, &vec![-7i32; batch * max_unique]).unwrap();
    let mut dn = DeviceBuffer::from_slice(&ctx, &vec![-77i32; batch]).unwrap();
    let desc = UniqueDescriptor {
        batch: b,
        row_len: n,
        max_unique: m,
        element: kind,
    };
    let plan = UniquePlan::<T>::select(&stream, &desc, PlanPreference::default()).unwrap();
    plan.run(
        &stream,
        Workspace::None,
        UniqueArgs::<T> {
            input: TensorRef::new(dx.as_slice(), in_sh, contiguous_stride(in_sh)).unwrap(),
            sorted_scratch: TensorMut::new(ss.as_slice_mut(), in_sh, contiguous_stride(in_sh))
                .unwrap(),
            sorted_idx_scratch: TensorMut::new(si.as_slice_mut(), in_sh, contiguous_stride(in_sh))
                .unwrap(),
            values: TensorMut::new(dv.as_slice_mut(), out_sh, contiguous_stride(out_sh)).unwrap(),
            counts: TensorMut::new(dc.as_slice_mut(), out_sh, contiguous_stride(out_sh)).unwrap(),
            counter: TensorMut::new(dn.as_slice_mut(), [b], contiguous_stride([b])).unwrap(),
        },
    )
    .expect("UniquePlan::run must succeed");
    stream.synchronize().unwrap();
    let mut v = vec![sentinel; batch * max_unique];
    let mut c = vec![0i32; batch * max_unique];
    let mut k = vec![0i32; batch];
    dv.copy_to_host(&mut v).unwrap();
    dc.copy_to_host(&mut c).unwrap();
    dn.copy_to_host(&mut k).unwrap();
    (v, c, k)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|a| a.to_bits()).collect()
}

#[test]
fn basic_values_ascending_with_run_length_counts() {
    let (v, c, k) = run(&[3f32, 1.0, 3.0, 2.0, 1.0], 1, 5, 5, -7.0, ElementKind::F32);
    assert_eq!(k, vec![3]);
    assert_eq!(&v[..3], &[1.0, 2.0, 3.0]);
    assert_eq!(&c[..3], &[2, 1, 2]);
    assert_eq!(
        &v[3..],
        &[-7.0, -7.0],
        "cells past the counter are untouched"
    );
    assert_eq!(&c[3..], &[-7, -7], "cells past the counter are untouched");
}

#[test]
fn every_nan_is_its_own_value_and_sorts_last() {
    let nan = f32::NAN;
    let x = [2.0, nan, 1.0, nan, 2.0, f32::NEG_INFINITY, f32::INFINITY];
    let (v, c, k) = run(&x, 1, 7, 7, -7.0, ElementKind::F32);
    assert_eq!(k, vec![6], "-inf, 1, 2, inf and two NaNs");
    assert_eq!(&v[..4], &[f32::NEG_INFINITY, 1.0, 2.0, f32::INFINITY]);
    assert!(v[4].is_nan() && v[5].is_nan(), "{v:?}");
    assert_eq!(
        &c[..6],
        &[1, 1, 2, 1, 1, 1],
        "each NaN is a run of length 1"
    );
}

#[test]
fn signed_zeros_are_one_value() {
    let x = [0.0f32, -0.0, 1.0, 0.0, -0.0];
    let (v, c, k) = run(&x, 1, 5, 5, -7.0, ElementKind::F32);
    assert_eq!(k, vec![2], "+0 and -0 are one value");
    assert_eq!(v[0], 0.0);
    assert_eq!(c[0], 4);
    assert_eq!((v[1], c[1]), (1.0, 1));
    // The value kept for a run is its first element in sort order. The sort is stable and compares -0 == +0, so that is the first zero in the
    // row: a row that starts with -0.0 keeps -0.0, one that starts with +0.0 keeps +0.0 (pinned both ways so a change is noticed).
    assert_eq!(bits(&v[..1]), vec![0.0f32.to_bits()]);
    let (v, _, _) = run(&[-0.0f32, 0.0, 1.0, 0.0], 1, 4, 4, -7.0, ElementKind::F32);
    assert_eq!(bits(&v[..1]), vec![(-0.0f32).to_bits()]);
}

#[test]
fn counter_is_the_true_distinct_count_when_it_exceeds_max_unique() {
    let x: Vec<f32> = (0..8).rev().map(|i| i as f32).collect(); // 7..0, eight distinct
    let (v, c, k) = run(&x, 1, 8, 3, -7.0, ElementKind::F32);
    assert_eq!(k, vec![8], "the counter reports the actual count");
    assert_eq!(v, vec![0.0, 1.0, 2.0]);
    assert_eq!(c, vec![1, 1, 1]);
}

#[test]
fn rows_are_independent_and_all_equal_and_full_width_rows_work() {
    // batch of three rows of 6: all equal, all distinct descending, two groups.
    let x = [
        5.0f32, 5.0, 5.0, 5.0, 5.0, 5.0, //
        6.0, 5.0, 4.0, 3.0, 2.0, 1.0, //
        2.0, 1.0, 2.0, 1.0, 2.0, 1.0,
    ];
    let (v, c, k) = run(&x, 3, 6, 6, -7.0, ElementKind::F32);
    assert_eq!(k, vec![1, 6, 2]);
    assert_eq!((v[0], c[0]), (5.0, 6));
    assert_eq!(&v[6..12], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(&c[6..12], &[1; 6]);
    assert_eq!((&v[12..14], &c[12..14]), (&[1.0f32, 2.0][..], &[3, 3][..]));

    // The full sort width (1024): 1024 values, 100 distinct, descending ramp.
    let wide: Vec<f32> = (0..1024)
        .map(|i| (1023 - i) as f32 / 10.0)
        .map(|a| a.floor())
        .collect();
    let (v, c, k) = run(&wide, 1, 1024, 1024, -7.0, ElementKind::F32);
    let mut distinct: Vec<f32> = wide.clone();
    distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distinct.dedup();
    assert_eq!(k, vec![distinct.len() as i32]);
    assert_eq!(&v[..distinct.len()], &distinct[..]);
    assert_eq!(
        c[..distinct.len()].iter().sum::<i32>(),
        1024,
        "run lengths cover the row"
    );
}

#[test]
fn i32_and_f64_rows() {
    let (v, c, k) = run(&[9i32, -4, 9, 0, -4, -4], 1, 6, 6, -77, ElementKind::I32);
    assert_eq!(
        (k, &v[..3], &c[..3]),
        (vec![3], &[-4, 0, 9][..], &[3, 1, 2][..])
    );
    let (v, c, k) = run(
        &[2.5f64, 2.5, f64::NAN, -1.0, f64::NAN],
        1,
        5,
        5,
        -7.0,
        ElementKind::F64,
    );
    assert_eq!(k, vec![4]);
    assert_eq!((&v[..2], &c[..4]), (&[-1.0f64, 2.5][..], &[1, 2, 1, 1][..]));
    assert!(v[2].is_nan() && v[3].is_nan());
}

#[test]
fn an_empty_batch_row_set_is_a_no_op_and_zero_width_rows_count_zero() {
    let (_, _, k) = run::<f32>(&[], 2, 0, 1, -7.0, ElementKind::F32);
    assert_eq!(k, vec![0, 0], "a zero-width row has no distinct value");
}
