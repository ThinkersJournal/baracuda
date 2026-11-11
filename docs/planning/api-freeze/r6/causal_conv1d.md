# R6 review: `baracuda-kernels` `causal_conv1d` family

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `a27161e9`. Items: the 40 `stable` rows of
`crates/baracuda-kernels/src/causal_conv1d.rs` in the §7.4 census (6 structs, 23 fields, 11 methods). Method: read the
descriptor, args and plan signatures and the `can_implement` / `run` bodies, and compared with `scan`
(`r6/scan.md`). A verdict on the public shape, not on kernel numerics.

**Feature gate.** The module and its `pub use` are `#[cfg(feature = "mamba")]` (`lib.rs:601-608`); the census does not
apply feature gating (§1), so these 40 items exist only with `mamba` on. The freeze promise is conditional on that
feature.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| `CausalConv1dDescriptor`, `CausalConv1dBackwardDescriptor` + 12 fields | 14 | stable | six `i32`/`bool`/`ElementKind` fields each, all documented; `r2-growth.tsv` rows: 6 fields at first and now, 4 revisions, 0 grew, so R2(b) leaves them literal |
| `CausalConv1dArgs`, `CausalConv1dBackwardArgs` + 11 fields | 13 | stable | tensor borrows with documented shapes; the backward args document that `dw`/`db` must be zero-initialised (atomic accumulation), which is part of the contract |
| `CausalConv1dPlan<T>`, `CausalConv1dBackwardPlan<T>` | 2 | stable | private fields |
| forward plan: `select`, `can_implement`, `workspace_size`, `sku`, `precision_guarantee`, `run` | 6 | stable | the six-method plan shape |
| backward plan: `select`, `workspace_size`, `sku`, `precision_guarantee`, `run` | 5 | stable, **with finding 1** | five methods, no `can_implement` |

No item is proposed for `#[doc(hidden)]` or removal.

## Findings

1. **`CausalConv1dBackwardPlan` has no `can_implement`, and its `run` checks no argument.** The forward plan's
   `can_implement` (`causal_conv1d.rs:180-220`) holds 7 `shape()`/`check_fits` lines and `run` calls it (line 244).
   The backward `run` (lines 465 to 560) has 0 `shape()`, `check_fits` or `Error::` matches; `select` validates only
   `width` in `{2,3,4}` and the dtype. Every other backward plan compared (`scan`, `softmax`, `reduce`, and four of the
   five `norm` ones) carries `can_implement`. So a caller with a wrong-shape `dy`/`dx`/`dw` reaches the kernel launch
   unchecked. Adding `can_implement` is an additive method and not breaking, but it changes behaviour (a bad call
   returns `Error` instead of launching), so it is a fix PR, not part of the freeze review. Not fixed here.
2. **`instance_norm_backward` has the same gap** (0 `can_implement` mentions, against 2 in the other four `norm`
   backward files); noted for the `norm` review.
3. **Width is a runtime `i32` limited to 2..=4.** Stable as a field; widening the range is a kernel change, not a surface
   one.

## Not checked

Numerics; whether Fuel or another consumer calls the backward plan (a consumer would be unaffected by an added
check); the `mamba`-off build (nothing in this family compiles there); the backward `run` was checked by `grep` over
lines 465 to 560, not read line by line.
