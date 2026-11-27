// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `MaskedFillPlan` and `MaskedFillBackwardPlan` for f16 and bf16 (a new 2-byte kernel; the fill value travels as a BIT PATTERN).
//!
//! `out[i] = mask[i] != 0 ? fill : src[i]` and `dsrc[i] = mask[i] != 0 ? +0 : dout[i]`: a select, no arithmetic, so every element is checked
//! bit for bit. The fills that matter for masked attention scores are all distinct and all exact: `-inf`, the most negative finite value
//! (`-65504` for f16), a NaN with a payload, `-0` and `+0`. The source holds NaN payloads, infinities, subnormals and `-0`; the mask holds `0`,
//! `1` and other non-zero bytes (non-zero means true). Shapes: size 1, an odd length, 2-D and 3-D.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, MaskedFillArgs, MaskedFillBackwardArgs, MaskedFillBackwardDescriptor,
    MaskedFillBackwardPlan, MaskedFillDescriptor, MaskedFillPlan, PlanPreference, TensorMut,
    TensorRef, Workspace, contiguous_stride,
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

/// quiet NaN, NaN with a payload, negative NaN, +inf, -inf, +0, -0, smallest subnormal, largest subnormal, most negative finite, largest finite, 1.0
const F16_SPECIAL: [u16; 12] = [
    0x7e00, 0x7c01, 0xfe00, 0x7c00, 0xfc00, 0x0000, 0x8000, 0x0001, 0x03ff, 0xfbff, 0x7bff, 0x3c00,
];
const BF16_SPECIAL: [u16; 12] = [
    0x7fc0, 0x7f81, 0xffc0, 0x7f80, 0xff80, 0x0000, 0x8000, 0x0001, 0x007f, 0xff7f, 0x7f7f, 0x3f80,
];

fn src_bits(special: &[u16; 12], n: usize) -> Vec<u16> {
    (0..n).map(|i| special[(i * 5 + i / 7) % 12]).collect()
}

/// 0, 1 and a few other non-zero bytes (any non-zero byte is true).
fn mask_bytes(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| [0u8, 1, 0, 2, 255, 0, 7][(i * 3 + i / 5) % 7])
        .collect()
}

macro_rules! suite {
    ($m:ident, $T:ty, $kind:expr, $special:expr) => {
        mod $m {
            use super::*;

            fn setup() -> (Context, Stream) {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                (ctx, stream)
            }

            fn fill<const N: usize>(shape: [i32; N], fill_bits: u16) -> Vec<u16> {
                let (ctx, stream) = setup();
                let n: usize = shape.iter().map(|d| *d as usize).product();
                let src: Vec<$T> = src_bits(&$special, n)
                    .into_iter()
                    .map(<$T>::from_bits16)
                    .collect();
                let mask = mask_bytes(n);
                let dsrc = DeviceBuffer::from_slice(&ctx, &src).unwrap();
                let dmask = DeviceBuffer::from_slice(&ctx, &mask).unwrap();
                let mut dout: DeviceBuffer<$T> = DeviceBuffer::zeros(&ctx, n).unwrap();
                let desc = MaskedFillDescriptor::<N> {
                    shape,
                    fill_bits: i64::from(fill_bits),
                    element: $kind,
                };
                let plan =
                    MaskedFillPlan::<$T, N>::select(&stream, &desc, PlanPreference::default())
                        .expect("select");
                plan.run(
                    &stream,
                    Workspace::None,
                    MaskedFillArgs::<$T, N> {
                        src: TensorRef::new(dsrc.as_slice(), shape, contiguous_stride(shape))
                            .unwrap(),
                        mask: TensorRef::new(dmask.as_slice(), shape, contiguous_stride(shape))
                            .unwrap(),
                        out: TensorMut::new(dout.as_slice_mut(), shape, contiguous_stride(shape))
                            .unwrap(),
                    },
                )
                .expect("run");
                stream.synchronize().expect("sync");
                let mut got = vec![<$T>::from_bits16(0); n];
                dout.copy_to_host(&mut got).unwrap();
                got.iter().map(|v| v.bits16()).collect()
            }

            fn want_fill(n: usize, fill_bits: u16) -> Vec<u16> {
                let src = src_bits(&$special, n);
                let mask = mask_bytes(n);
                (0..n)
                    .map(|i| if mask[i] != 0 { fill_bits } else { src[i] })
                    .collect()
            }

            fn check(got: &[u16], want: &[u16], what: &str) {
                let bad: Vec<_> = (0..want.len()).filter(|i| got[*i] != want[*i]).collect();
                assert!(
                    bad.is_empty(),
                    "{what}: {} of {} cells differ, first {:?}: got {:#06x}, want {:#06x}",
                    bad.len(),
                    want.len(),
                    bad.first(),
                    bad.first().map(|i| got[*i]).unwrap_or(0),
                    bad.first().map(|i| want[*i]).unwrap_or(0)
                );
            }

            /// The fills that must stay distinct: -inf, the most negative finite value, a NaN with a payload, -0, +0.
            fn fills() -> [(u16, &'static str); 5] {
                [
                    (<$T>::from_bits16($special[4]).bits16(), "-inf"),
                    (
                        <$T>::from_bits16($special[9]).bits16(),
                        "most negative finite",
                    ),
                    (<$T>::from_bits16($special[1]).bits16(), "NaN with payload"),
                    (<$T>::from_bits16($special[6]).bits16(), "-0"),
                    (<$T>::from_bits16($special[5]).bits16(), "+0"),
                ]
            }

            #[test]
            fn every_fill_value_is_exact_and_distinct() {
                let mut seen = Vec::new();
                for (bits, name) in fills() {
                    let got = fill([4, 9], bits);
                    check(&got, &want_fill(36, bits), name);
                    seen.push(bits);
                }
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(
                    seen.len(),
                    5,
                    "the five fills must be five distinct bit patterns"
                );
            }

            #[test]
            fn size_one_and_odd_and_rank3() {
                let (bits, _) = fills()[0];
                check(&fill([1], bits), &want_fill(1, bits), "size 1");
                check(&fill([7], bits), &want_fill(7, bits), "odd length");
                check(&fill([3, 5, 2], bits), &want_fill(30, bits), "rank 3");
            }

            #[test]
            fn backward_zeroes_masked_positions_and_copies_the_rest() {
                let (ctx, stream) = setup();
                let shape = [5i32, 8];
                let n = 40usize;
                let dout_bits = src_bits(&$special, n);
                let dout: Vec<$T> = dout_bits.iter().copied().map(<$T>::from_bits16).collect();
                let mask = mask_bytes(n);
                let ddout = DeviceBuffer::from_slice(&ctx, &dout).unwrap();
                let dmask = DeviceBuffer::from_slice(&ctx, &mask).unwrap();
                // Sentinel in dsrc: a masked position must be WRITTEN with +0, not left alone.
                let mut ddsrc =
                    DeviceBuffer::from_slice(&ctx, &vec![<$T>::from_bits16(0x1234); n]).unwrap();
                let desc = MaskedFillBackwardDescriptor {
                    shape,
                    element: $kind,
                };
                let plan = MaskedFillBackwardPlan::<$T, 2>::select(
                    &stream,
                    &desc,
                    PlanPreference::default(),
                )
                .expect("select");
                plan.run(
                    &stream,
                    Workspace::None,
                    MaskedFillBackwardArgs::<$T, 2> {
                        dout: TensorRef::new(ddout.as_slice(), shape, contiguous_stride(shape))
                            .unwrap(),
                        mask: TensorRef::new(dmask.as_slice(), shape, contiguous_stride(shape))
                            .unwrap(),
                        dsrc: TensorMut::new(ddsrc.as_slice_mut(), shape, contiguous_stride(shape))
                            .unwrap(),
                    },
                )
                .expect("run");
                stream.synchronize().expect("sync");
                let mut got = vec![<$T>::from_bits16(0); n];
                ddsrc.copy_to_host(&mut got).unwrap();
                let got: Vec<u16> = got.iter().map(|v| v.bits16()).collect();
                let want: Vec<u16> = (0..n)
                    .map(|i| if mask[i] != 0 { 0x0000 } else { dout_bits[i] })
                    .collect();
                check(&got, &want, "masked_fill backward");
            }
        }
    };
}

suite!(f16_, f16, ElementKind::F16, F16_SPECIAL);
suite!(bf16_, bf16, ElementKind::Bf16, BF16_SPECIAL);
