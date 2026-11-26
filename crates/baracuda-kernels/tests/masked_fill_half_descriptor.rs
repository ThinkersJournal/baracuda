// SPDX-License-Identifier: MIT OR Apache-2.0
//! The typed half constructors of `MaskedFillDescriptor` carry the 16-bit pattern of the value (host-only; no device needed).

use baracuda_kernels::{ElementKind, MaskedFillDescriptor};
use half::{bf16, f16};

/// The typed constructors carry the same bit patterns the tests above pass by hand.
#[test]
fn descriptor_constructors_carry_the_bit_pattern() {
    let d = MaskedFillDescriptor::<1>::new_f16([4], f16::NEG_INFINITY);
    assert_eq!((d.fill_bits, d.element), (0xfc00, ElementKind::F16));
    let d = MaskedFillDescriptor::<1>::new_f16([4], f16::MIN);
    assert_eq!(d.fill_bits, 0xfbff);
    let d = MaskedFillDescriptor::<1>::new_bf16([4], bf16::NEG_INFINITY);
    assert_eq!((d.fill_bits, d.element), (0xff80, ElementKind::Bf16));
    let d = MaskedFillDescriptor::<1>::new_bf16([4], bf16::MIN);
    assert_eq!(d.fill_bits, 0xff7f);
}
