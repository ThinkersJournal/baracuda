// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `IndexBoundsPlan`, the device-side bounds check (ROADMAP E4).
//!
//! The indexing plans skip an out-of-range index and cannot raise; `IndexBoundsPlan` counts such
//! values into a device word and `check` turns a non-zero count into `Error::IndexOutOfRange`.
//! Pinned here: the exact count (negative, `== bound`, huge, i32 and i64), zero for an in-range
//! buffer, `clear = false` accumulating across buffers, `clear = true` resetting, an empty buffer,
//! a buffer past one grid stride, and the end-to-end pairing with `IndexSelectPlan` (the select
//! still skips the cell; the check reports it).
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, Error, IndexBoundsArgs, IndexBoundsPlan, IndexElement, IndexSelectArgs,
    IndexSelectDescriptor, IndexSelectPlan, PlanPreference, TensorMut, TensorRef, Workspace,
    contiguous_stride,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Enqueue one check of `idx` against `bound` onto `flag`.
fn enqueue<I: IndexElement + Copy>(
    ctx: &Context,
    stream: &Stream,
    flag: &mut DeviceBuffer<u32>,
    bound: i64,
    idx: &[I],
    clear: bool,
) {
    let didx = DeviceBuffer::from_slice(ctx, idx).unwrap();
    let n = idx.len() as i32;
    IndexBoundsPlan::new(bound)
        .unwrap()
        .run(
            stream,
            flag,
            IndexBoundsArgs::<I> {
                index: TensorRef::new(didx.as_slice(), [n], contiguous_stride([n])).unwrap(),
                clear,
            },
        )
        .expect("run");
    stream.synchronize().unwrap();
}

fn count(stream: &Stream, flag: &DeviceBuffer<u32>) -> u32 {
    match IndexBoundsPlan::check(stream, flag) {
        Ok(()) => 0,
        Err(Error::IndexOutOfRange { count }) => count,
        Err(e) => panic!("unexpected error {e:?}"),
    }
}

#[test]
fn counts_exact_i32() {
    let (ctx, stream) = setup();
    let mut flag = DeviceBuffer::from_slice(&ctx, &[7u32]).unwrap();
    // bound 5: -1, 5, 6, i32::MAX, i32::MIN are out; 0..=4 are in.
    let idx = [0i32, -1, 4, 5, 2, 6, i32::MAX, 1, i32::MIN, 3];
    enqueue(&ctx, &stream, &mut flag, 5, &idx, true);
    assert_eq!(count(&stream, &flag), 5);
}

#[test]
fn counts_exact_i64() {
    let (ctx, stream) = setup();
    let mut flag = DeviceBuffer::from_slice(&ctx, &[0u32]).unwrap();
    // an i64 value that wraps to a valid i32 must still be out of range.
    let idx = [0i64, 4, (1i64 << 32) + 1, -1, i64::MAX, i64::MIN, 3, 5];
    enqueue(&ctx, &stream, &mut flag, 5, &idx, true);
    assert_eq!(count(&stream, &flag), 5);
}

#[test]
fn in_range_is_ok_and_empty_is_ok() {
    let (ctx, stream) = setup();
    let mut flag = DeviceBuffer::from_slice(&ctx, &[9u32]).unwrap();
    enqueue(&ctx, &stream, &mut flag, 4, &[0i32, 1, 2, 3, 3, 0], true);
    IndexBoundsPlan::check(&stream, &flag).expect("all in range");
    enqueue::<i32>(&ctx, &stream, &mut flag, 4, &[], true);
    IndexBoundsPlan::check(&stream, &flag).expect("empty");
}

#[test]
fn accumulates_without_clear_and_resets_with_clear() {
    let (ctx, stream) = setup();
    let mut flag = DeviceBuffer::from_slice(&ctx, &[0u32]).unwrap();
    enqueue(&ctx, &stream, &mut flag, 3, &[0i32, 3, 9], true);
    enqueue(&ctx, &stream, &mut flag, 3, &[-2i64, 1], false);
    assert_eq!(count(&stream, &flag), 3);
    enqueue(&ctx, &stream, &mut flag, 3, &[0i32, 1], true);
    assert_eq!(count(&stream, &flag), 0);
}

#[test]
fn large_buffer_past_one_grid_stride() {
    let (ctx, stream) = setup();
    let mut flag = DeviceBuffer::from_slice(&ctx, &[0u32]).unwrap();
    let n = 2_000_000usize;
    let mut idx: Vec<i32> = (0..n).map(|i| (i % 100) as i32).collect();
    for k in (0..n).step_by(7919) {
        idx[k] = 100; // == bound
    }
    let expect = (0..n).step_by(7919).count() as u32;
    enqueue(&ctx, &stream, &mut flag, 100, &idx, true);
    assert_eq!(count(&stream, &flag), expect);
}

#[test]
fn pairs_with_index_select() {
    let (ctx, stream) = setup();
    let src: Vec<f32> = (0..12).map(|i| i as f32).collect(); // [4, 3]
    let idx = [1i32, 9, 3, -1];
    let dsrc = DeviceBuffer::from_slice(&ctx, &src).unwrap();
    let didx = DeviceBuffer::from_slice(&ctx, &idx).unwrap();
    let mut dout = DeviceBuffer::from_slice(&ctx, &[-7.0f32; 12]).unwrap();
    let out_shape = [4, 3];
    let desc = IndexSelectDescriptor {
        out_shape,
        select_dim: 0,
        src_dim_size: 4,
        element: ElementKind::F32,
    };
    let plan = IndexSelectPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    plan.run(
        &stream,
        Workspace::None,
        IndexSelectArgs::<f32, 2, i32> {
            src: TensorRef::new(dsrc.as_slice(), [4, 3], contiguous_stride([4, 3])).unwrap(),
            idx: TensorRef::new(didx.as_slice(), [4], contiguous_stride([4])).unwrap(),
            out: TensorMut::new(dout.as_slice_mut(), out_shape, contiguous_stride(out_shape))
                .unwrap(),
        },
    )
    .expect("run");
    let mut flag = DeviceBuffer::from_slice(&ctx, &[0u32]).unwrap();
    enqueue(&ctx, &stream, &mut flag, 4, &idx, true);
    assert_eq!(count(&stream, &flag), 2, "the 9 and the -1");
    let mut got = vec![0f32; 12];
    dout.copy_to_host(&mut got).unwrap();
    assert_eq!(&got[0..3], &[3.0, 4.0, 5.0]); // row 1
    assert_eq!(&got[3..6], &[-7.0; 3], "skipped cell untouched");
    assert_eq!(&got[6..9], &[9.0, 10.0, 11.0]); // row 3
    assert_eq!(&got[9..12], &[-7.0; 3], "skipped cell untouched");
}
