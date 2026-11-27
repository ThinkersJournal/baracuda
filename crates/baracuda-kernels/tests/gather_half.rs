// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `GatherPlan` for f16 and bf16 (routed onto the 2-byte bit-copy kernel `gather_u16`).
//!
//! `out[..., j, ...] = src[..., index[..., j, ...], ...]` along `gather_dim`; the index tensor has the output's shape. A gather has no
//! arithmetic, so every element must come back bit-for-bit: NaN payloads (quiet and signalling, both signs), `-0`, `+-inf`, the smallest
//! and largest subnormals and the largest finite values. i32 and i64 indices, gather dim first / middle / last, rank 2 and 3, duplicate
//! indices, and the index policy shared with every indexing plan: an out-of-range or negative index leaves that output cell untouched (the kernel
//! cannot raise an error), pinned with a sentinel pre-fill.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, GatherArgs, GatherDescriptor, GatherPlan, PlanPreference, TensorMut, TensorRef,
    contiguous_stride,
};
use half::{bf16, f16};

trait Bits: Copy {
    fn from_bits16(b: u16) -> Self;
    fn bits16(self) -> u16;
}
impl Bits for f16 {
    fn from_bits16(b: u16) -> Self {
        f16::from_bits(b)
    }
    fn bits16(self) -> u16 {
        self.to_bits()
    }
}
impl Bits for bf16 {
    fn from_bits16(b: u16) -> Self {
        bf16::from_bits(b)
    }
    fn bits16(self) -> u16 {
        self.to_bits()
    }
}

const F16_SPECIAL: [u16; 12] = [
    0x7e00, 0x7c01, 0xfe00, 0x7c00, 0xfc00, 0x0000, 0x8000, 0x0001, 0x03ff, 0xfbff, 0x7bff, 0x3c00,
];
const BF16_SPECIAL: [u16; 12] = [
    0x7fc0, 0x7f81, 0xffc0, 0x7f80, 0xff80, 0x0000, 0x8000, 0x0001, 0x007f, 0xff7f, 0x7f7f, 0x3f80,
];
const SENTINEL: u16 = 0x1234;

fn src_bits(special: &[u16; 12], n: usize) -> Vec<u16> {
    (0..n).map(|i| special[(i * 5 + i / 7) % 12]).collect()
}

/// Row-major offset of `coord` in `shape`.
fn off<const N: usize>(shape: [usize; N], coord: [usize; N]) -> usize {
    coord.iter().zip(shape).fold(0, |a, (c, s)| a * s + c)
}

/// Host reference: `out[c] = src[c with c[dim] = index[c]]`, an out-of-range index leaves the sentinel.
fn reference<const N: usize>(
    special: &[u16; 12],
    src_shape: [usize; N],
    idx_shape: [usize; N],
    dim: usize,
    idx: &[i64],
) -> Vec<u16> {
    let src = src_bits(special, src_shape.iter().product());
    let n: usize = idx_shape.iter().product();
    let mut out = vec![SENTINEL; n];
    for (lin, slot) in out.iter_mut().enumerate() {
        let mut coord = [0usize; N];
        let mut r = lin;
        for d in (0..N).rev() {
            coord[d] = r % idx_shape[d];
            r /= idx_shape[d];
        }
        let k = idx[lin];
        if k < 0 || k as usize >= src_shape[dim] {
            continue;
        }
        let mut sc = coord;
        sc[dim] = k as usize;
        *slot = src[off(src_shape, sc)];
    }
    out
}

macro_rules! suite {
    ($m:ident, $T:ty, $kind:expr, $special:expr) => {
        mod $m {
            use super::*;

            fn run<const N: usize, I>(
                src_shape: [usize; N],
                idx_shape: [usize; N],
                dim: usize,
                idx: &[I],
            ) -> Vec<u16>
            where
                I: baracuda_kernels::IndexElement + Copy,
            {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let s_i: [i32; N] = src_shape.map(|d| d as i32);
                let i_i: [i32; N] = idx_shape.map(|d| d as i32);
                let host_src: Vec<$T> = src_bits(&$special, src_shape.iter().product())
                    .into_iter()
                    .map(<$T>::from_bits16)
                    .collect();
                let out_n: usize = idx_shape.iter().product();
                let dsrc = DeviceBuffer::from_slice(&ctx, &host_src).unwrap();
                let didx = DeviceBuffer::from_slice(&ctx, idx).unwrap();
                let fill: Vec<$T> = vec![<$T>::from_bits16(SENTINEL); out_n];
                let mut dout = DeviceBuffer::from_slice(&ctx, &fill).unwrap();
                let desc = GatherDescriptor {
                    out_shape: i_i,
                    gather_dim: dim as i32,
                    src_dim_size: s_i[dim],
                    element: $kind,
                };
                let plan = GatherPlan::<$T, N>::select(&stream, &desc, PlanPreference::default())
                    .expect("select");
                plan.run(
                    &stream,
                    baracuda_kernels::Workspace::None,
                    GatherArgs::<$T, N, I> {
                        src: TensorRef::new(dsrc.as_slice(), s_i, contiguous_stride(s_i)).unwrap(),
                        index: TensorRef::new(didx.as_slice(), i_i, contiguous_stride(i_i))
                            .unwrap(),
                        out: TensorMut::new(dout.as_slice_mut(), i_i, contiguous_stride(i_i))
                            .unwrap(),
                    },
                )
                .expect("run");
                stream.synchronize().expect("sync");
                let mut got = vec![<$T>::from_bits16(0); out_n];
                dout.copy_to_host(&mut got).unwrap();
                got.iter().map(|v| v.bits16()).collect()
            }

            fn check(got: &[u16], want: &[u16], what: &str) {
                let bad: Vec<_> = (0..want.len()).filter(|i| got[*i] != want[*i]).collect();
                assert!(
                    bad.is_empty(),
                    "{what}: {} of {} cells differ, first at {:?}: got {:#06x}, want {:#06x}",
                    bad.len(),
                    want.len(),
                    bad.first(),
                    bad.first().map(|i| got[*i]).unwrap_or(0),
                    bad.first().map(|i| want[*i]).unwrap_or(0)
                );
            }

            #[test]
            fn dim0_dim1_i32_and_i64() {
                // dim 0: src [5, 6], index [4, 6]: each column picks its own rows.
                let idx: Vec<i64> = (0..24).map(|i| (i * 3 + i / 6) % 5).collect();
                let want = reference(&$special, [5, 6], [4, 6], 0, &idx);
                let i32s: Vec<i32> = idx.iter().map(|v| *v as i32).collect();
                check(&run([5, 6], [4, 6], 0, &i32s), &want, "dim0 i32");
                check(&run([5, 6], [4, 6], 0, &idx), &want, "dim0 i64");
                // dim 1 (the last dim): src [5, 6], index [5, 9], with duplicates.
                let idx: Vec<i64> = (0..45).map(|i| (i * 5 + i / 9) % 6).collect();
                let want = reference(&$special, [5, 6], [5, 9], 1, &idx);
                let i32s: Vec<i32> = idx.iter().map(|v| *v as i32).collect();
                check(&run([5, 6], [5, 9], 1, &i32s), &want, "dim1 i32");
                check(&run([5, 6], [5, 9], 1, &idx), &want, "dim1 i64");
            }

            #[test]
            fn middle_dim_rank3() {
                let idx: Vec<i64> = (0..3 * 4 * 5).map(|i| (i * 7 + i / 5) % 3).collect();
                let want = reference(&$special, [3, 3, 5], [3, 4, 5], 1, &idx);
                let i32s: Vec<i32> = idx.iter().map(|v| *v as i32).collect();
                check(&run([3, 3, 5], [3, 4, 5], 1, &i32s), &want, "dim1 rank3");
            }

            #[test]
            fn size_one_source() {
                let idx: Vec<i32> = vec![0; 8];
                let want = reference(&$special, [1, 8], [1, 8], 0, &[0i64; 8]);
                check(&run([1, 8], [1, 8], 0, &idx), &want, "size-1 source");
            }

            #[test]
            fn out_of_range_and_negative_indices_leave_the_output_untouched() {
                let idx: Vec<i32> = vec![1, -1, 4, 0, 99, 2, 3, 3, -5, 0, 1, 2];
                let idx64: Vec<i64> = idx.iter().map(|v| *v as i64).collect();
                let want = reference(&$special, [5, 3], [4, 3], 0, &idx64);
                assert!(
                    want.contains(&SENTINEL),
                    "the reference must leave sentinels"
                );
                check(&run([5, 3], [4, 3], 0, &idx), &want, "OOB i32");
                check(&run([5, 3], [4, 3], 0, &idx64), &want, "OOB i64");
            }
        }
    };
}

suite!(f16_, f16, ElementKind::F16, F16_SPECIAL);
suite!(bf16_, bf16, ElementKind::Bf16, BF16_SPECIAL);
