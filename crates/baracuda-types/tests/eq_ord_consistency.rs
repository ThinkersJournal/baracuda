// SPDX-License-Identifier: MIT OR Apache-2.0
//! `PartialEq` and `PartialOrd` must agree: `a == b` iff `partial_cmp(a, b) == Some(Equal)`.
use baracuda_types::{BFloat16, Half};
use core::cmp::Ordering;

#[test]
fn half_nan_eq_vs_partial_cmp() {
    let n = Half::NAN;
    println!(
        "Half NAN==NAN: {}, partial_cmp: {:?}",
        n == n,
        n.partial_cmp(&n)
    );
    assert_eq!(n == n, n.partial_cmp(&n) == Some(Ordering::Equal));
}

#[test]
fn half_signed_zero_eq_vs_partial_cmp() {
    let (p, m) = (Half::ZERO, Half::NEG_ZERO);
    println!(
        "Half +0==-0: {}, partial_cmp: {:?}",
        p == m,
        p.partial_cmp(&m)
    );
    assert_eq!(p == m, p.partial_cmp(&m) == Some(Ordering::Equal));
}

#[test]
fn bf16_nan_eq_vs_partial_cmp() {
    let n = BFloat16::NAN;
    println!(
        "BF16 NAN==NAN: {}, partial_cmp: {:?}",
        n == n,
        n.partial_cmp(&n)
    );
    assert_eq!(n == n, n.partial_cmp(&n) == Some(Ordering::Equal));
}

#[test]
fn bf16_signed_zero_eq_vs_partial_cmp() {
    let (p, m) = (BFloat16::ZERO, BFloat16::NEG_ZERO);
    println!(
        "BF16 +0==-0: {}, partial_cmp: {:?}",
        p == m,
        p.partial_cmp(&m)
    );
    assert_eq!(p == m, p.partial_cmp(&m) == Some(Ordering::Equal));
}

#[test]
fn hash_agrees_with_eq_for_signed_zero() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    fn h<T: Hash>(t: T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }
    assert_eq!(h(Half::ZERO), h(Half::NEG_ZERO));
    assert_eq!(h(BFloat16::ZERO), h(BFloat16::NEG_ZERO));
}
