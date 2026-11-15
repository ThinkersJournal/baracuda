kind: **BREAKING**: `Half` / `BFloat16` equality is now IEEE (`==` via `f32`), `Eq` removed
---
`baracuda-types`: `Half` and `BFloat16` derived `PartialEq`, `Eq` and `Hash` on the raw bits while `PartialOrd` compared as `f32`, so `NAN == NAN` was true yet `partial_cmp` was `None`, and `+0.0 == -0.0` was false yet `partial_cmp` was `Some(Equal)` (the `PartialOrd` contract was broken). `==` is now IEEE 754 equality through `f32`, like `f32` and the `half` crate: `NaN != NaN`, `+0.0 == -0.0`. `Eq` is removed (NaN is not equal to itself); `Hash` stays, normalised so `-0.0` hashes like `+0.0`.

**Affected:** code that compares `Half`/`BFloat16` with `==` and relied on `NAN == NAN` or on `+0.0 != -0.0`; code that uses either type where `Eq` is required (a `HashMap`/`HashSet` key, `Eq`-bounded generics, `#[derive(Eq)]` on a struct holding one). No use of that kind exists in this workspace.
**Fix:** compare bits with `a.to_bits() == b.to_bits()` where you need bit identity; for a map key, key on `to_bits()`; drop `Eq` from derives on containing structs.

New `tests/eq_ord_consistency.rs` (5 tests) asserts `==` and `partial_cmp` agree for NaN and signed zero, and that `Hash` agrees with `==` for signed zero.
