# R6 review: `baracuda-kernels` `elementwise` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10, 2026-11-13). Ref: `origin/main` `b8c1b377`. Items: the 353 `stable`
rows under `crates/baracuda-kernels/src/elementwise/` in the §7.4 census (20 `mod`s, 20 re-exports, 60 structs, 133 fields,
120 methods; counted from `items.tsv` by `kind` at this ref, and 20 + 20 + 60 + 133 + 120 = 353). Twenty files, about 19,500
lines, one of them (`unary.rs`) 5,829.
Method: per file, grepped `pub struct .*Plan`, `pub fn can_implement`, `pub fn run`, `self.can_implement`,
`unwrap/expect/panic`, `non_exhaustive`, `aux_element`; read `prelu.rs` `select` and `run`, the `expect` sites in
`unary_backward.rs` and `binary_backward.rs`, the `unary.rs` head and descriptor, and the `unary_backward` support check.
The other bodies were **not** read. A verdict on the public shape, not on kernel numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 20 `mod`s, 20 re-exports | 40 | stable | one module per plan, re-exported from `elementwise/mod.rs` |
| 60 structs (20 descriptors, 20 args, 20 plans) and 133 fields | 193 | stable with a note | findings 2, 4 |
| 120 methods | 120 | stable with a note | findings 1, 3 |

## Findings

1. **`PReluPlan` and `PReluBackwardPlan` have no public `can_implement`.** 18 of the 20 plans have one and their `run`
   calls it (grep: each of the 18 files has one `pub fn can_implement` and one `self.can_implement`); the two PReLU files
   have neither. `PReluPlan::run` (`prelu.rs:151`) validates inline instead (dense views, `x`/`y` shape, weight shape).
   Same finding as `norm` 1, `attention` 1 and `conv` 1, here for one op pair.
2. **No descriptor is `#[non_exhaustive]`.** `grep '#\[non_exhaustive'` finds none in the 20 files, so each descriptor
   (for example `UnaryDescriptor { kind, shape, element }`) is a plain struct with public fields: a new field is a
   breaking change after the freeze. `attention` and `conv` mark some descriptors; `elementwise` marks none. One crate
   rule would settle all three.
3. **The panic sites are guarded, by reading.** Five `expect` calls in non-test code: `binary_backward.rs:870, 874`
   (saved `a` and `b`), `unary_backward.rs:234` (`save_shape_for`) and `:357, :365` (saved `x` and `y`). Each follows a
   `can_implement` or `select` check that returns an error first (read for `unary_backward.rs:205-234` and the
   `saved_ptrs` comment). They are safe only while `run` keeps calling `can_implement`; `PRelu*` has none to call, but has no
   `expect` either.
4. **The op set is a runtime support check, not a type.** `UnaryDescriptor.kind` is a `UnaryKind` (defined outside this
   directory and not located in the crates of this tree, so not reviewed here). `UnaryBackwardPlan::select` accepts a fixed list
   of saved-x and saved-y kinds (about 44 named in its error text) and rejects the rest with `Error::Unsupported` ("other
   (kind, dtype) pairs land in later fanout"), so a valid `UnaryKind` can still fail `select`. That is the design;
   the freeze should say the supported set is not part of the semver promise, or publish it.
5. **`unary.rs` module doc is stale.** Its head says "Today only `UnaryKind::Neg` on `f32` is wired" and that "the other
   unary variants ... land in fanout sessions" (`unary.rs:7-11`), in a file of 5,829 lines whose sibling backward plan
   lists dozens of ops. I did not check which forward ops are wired; the doc line is wrong or the file is dead weight
   until someone does.
6. **`aux_element` is set only where it should be.** `Some` in `cast.rs` and `cast_subbyte.rs` (one each, counted by
   grep: the second dtype), `None` in the other 18. `PRelu` reports `op: UnaryKind::PReLU` in
   `OpCategory::UnaryElementwise` (`prelu.rs:119-128`), so its identity is shared with the unary family.

## Not checked

Kernel numerics; every `can_implement` body; the `select` bodies other than PReLU's and the two backward support checks; the
forward op set of `unary.rs`; the `UnaryKind` enum (outside this tree as far as grep reaches); whether the two
`*_param` plan families and `where_op` follow the same `Plan` shape beyond grep presence; `Send`/`Sync`. Nothing was executed.
