// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `IndexSelectPlan` for f16 and bf16 (routed onto the 2-byte bit-copy kernels).
//!
//! `out[..., j, ...] = src[..., idx[j], ...]`. There is no arithmetic, so every element must come back bit-for-bit: NaN payloads (quiet and
//! signalling, both signs), `-0`, `+-inf`, the smallest and largest subnormals and the largest finite values. i32 and i64 indices, select
//! dim first / middle / last, duplicate indices, a size-1 source, and the index policy shared with f32: an out-of-range or negative index
//! leaves that output cell untouched (the kernel cannot raise an error); the test pins that by pre-filling `out` with a sentinel.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, IndexSelectArgs, IndexSelectDescriptor, IndexSelectPlan, PlanPreference,
    TensorMut, TensorRef, Workspace, contiguous_stride,
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

/// Bit patterns that must survive a copy: quiet NaN, signalling NaN with a payload, negative NaN, +-inf, +-0, subnormals, extremes.
const F16_SPECIAL: [u16; 12] = [
    0x7e00, 0x7c01, 0xfe00, 0x7c00, 0xfc00, 0x0000, 0x8000, 0x0001, 0x03ff, 0xfbff, 0x7bff, 0x3c00,
];
const BF16_SPECIAL: [u16; 12] = [
    0x7fc0, 0x7f81, 0xffc0, 0x7f80, 0xff80, 0x0000, 0x8000, 0x0001, 0x007f, 0xff7f, 0x7f7f, 0x3f80,
];

const SENTINEL: u16 = 0x1234;

/// A deterministic source: element `i` is `special[(i * 5 + i / 7) % 12]`, so every pattern appears in every row mix.
fn src_bits(special: &[u16; 12], n: usize) -> Vec<u16> {
    (0..n).map(|i| special[(i * 5 + i / 7) % 12]).collect()
}

macro_rules! suite {
    ($m:ident, $T:ty, $kind:expr, $special:expr) => {
        mod $m {
            use super::*;

            /// Runs one select and returns the output bits. `src_shape`, `select_dim`, `idx`; `out` pre-filled with the sentinel.
            fn run<const N: usize, I>(src_shape: [i32; N], select_dim: usize, idx: &[I]) -> Vec<u16>
            where
                I: baracuda_kernels::IndexElement + Copy,
            {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let mut out_shape = src_shape;
                out_shape[select_dim] = idx.len() as i32;
                let src_n: usize = src_shape.iter().map(|d| *d as usize).product();
                let out_n: usize = out_shape.iter().map(|d| *d as usize).product();
                let host_src: Vec<$T> = src_bits(&$special, src_n)
                    .into_iter()
                    .map(<$T>::from_bits16)
                    .collect();
                let dsrc = DeviceBuffer::from_slice(&ctx, &host_src).unwrap();
                let didx = DeviceBuffer::from_slice(&ctx, idx).unwrap();
                let fill: Vec<$T> = vec![<$T>::from_bits16(SENTINEL); out_n];
                let mut dout = DeviceBuffer::from_slice(&ctx, &fill).unwrap();
                let desc = IndexSelectDescriptor {
                    out_shape,
                    select_dim: select_dim as i32,
                    src_dim_size: src_shape[select_dim],
                    element: $kind,
                };
                let plan =
                    IndexSelectPlan::<$T, N>::select(&stream, &desc, PlanPreference::default())
                        .expect("select");
                plan.run(
                    &stream,
                    Workspace::None,
                    IndexSelectArgs::<$T, N, I> {
                        src: TensorRef::new(
                            dsrc.as_slice(),
                            src_shape,
                            contiguous_stride(src_shape),
                        )
                        .unwrap(),
                        idx: TensorRef::new(
                            didx.as_slice(),
                            [idx.len() as i32],
                            contiguous_stride([idx.len() as i32]),
                        )
                        .unwrap(),
                        out: TensorMut::new(
                            dout.as_slice_mut(),
                            out_shape,
                            contiguous_stride(out_shape),
                        )
                        .unwrap(),
                    },
                )
                .expect("run");
                stream.synchronize().expect("sync");
                let mut got = vec![<$T>::from_bits16(0); out_n];
                dout.copy_to_host(&mut got).unwrap();
                got.iter().map(|v| v.bits16()).collect()
            }

            /// Host reference over a rank-3 view `[outer, dim, inner]`; an out-of-range index leaves the sentinel.
            fn reference(outer: usize, dim: usize, inner: usize, idx: &[i64]) -> Vec<u16> {
                let src = src_bits(&$special, outer * dim * inner);
                let mut out = vec![SENTINEL; outer * idx.len() * inner];
                for o in 0..outer {
                    for (j, &k) in idx.iter().enumerate() {
                        if k < 0 || k as usize >= dim {
                            continue;
                        }
                        for i in 0..inner {
                            out[(o * idx.len() + j) * inner + i] =
                                src[(o * dim + k as usize) * inner + i];
                        }
                    }
                }
                out
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
            fn rows_dim0_i32_and_i64() {
                let idx: Vec<i64> = vec![4, 1, 1, 0, 3];
                let want = reference(1, 5, 12, &idx);
                let i32s: Vec<i32> = idx.iter().map(|v| *v as i32).collect();
                check(&run([5, 12], 0, &i32s), &want, "dim0 i32 idx");
                check(&run([5, 12], 0, &idx), &want, "dim0 i64 idx");
            }

            #[test]
            fn middle_dim_rank3() {
                let idx: Vec<i32> = vec![2, 0, 2, 1];
                let want = reference(3, 3, 6, &idx.iter().map(|v| *v as i64).collect::<Vec<_>>());
                check(&run([3, 3, 6], 1, &idx), &want, "dim1 rank3");
            }

            #[test]
            fn last_dim() {
                let idx: Vec<i32> = vec![5, 5, 0, 3];
                let want = reference(8, 6, 1, &idx.iter().map(|v| *v as i64).collect::<Vec<_>>());
                check(&run([8, 6], 1, &idx), &want, "last dim");
            }

            #[test]
            fn size_one_source() {
                let idx: Vec<i32> = vec![0, 0, 0];
                let want = reference(1, 1, 12, &idx.iter().map(|v| *v as i64).collect::<Vec<_>>());
                check(&run([1, 12], 0, &idx), &want, "size-1 source");
            }

            #[test]
            fn out_of_range_and_negative_indices_leave_the_output_untouched() {
                let idx: Vec<i32> = vec![1, -1, 5, 0, 99, 4];
                let want = reference(1, 5, 12, &idx.iter().map(|v| *v as i64).collect::<Vec<_>>());
                // Sentinel cells prove "untouched": the reference leaves them at the sentinel.
                assert!(want.iter().any(|b| *b == SENTINEL));
                check(&run([5, 12], 0, &idx), &want, "OOB i32");
                let idx64: Vec<i64> = vec![1, -1, 5, 0, 99, 4];
                check(&run([5, 12], 0, &idx64), &want, "OOB i64");
            }
        }
    };
}

suite!(f16_, f16, ElementKind::F16, F16_SPECIAL);
suite!(bf16_, bf16, ElementKind::Bf16, BF16_SPECIAL);
