# R6 review: `baracuda-kernels` `segment` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10, 2026-11-13). Ref: `origin/main` `b2f45cd9`. Items: the 368 `stable` rows under
`crates/baracuda-kernels/src/segment/` in the §7.4 census: 20 `mod`s, 20 re-exports, 60 structs, 148 fields, 120 methods
(20 + 20 + 60 + 148 + 120 = 368, counted from `items.tsv` by `kind` at this ref; all `public`, all `stable`). Twenty-one files, 4,324
lines: `segment_{sum,mean,max,min,prod}` and `unsorted_segment_{sum,mean,max,min,prod}`, each with a `_backward`, plus `mod.rs`; 20 plans.
Method: grepped every file for `pub fn can_implement`, `pub fn select`, `pub fn run`, `pub struct .*Plan`, the `#[non_exhaustive]`
attribute line, `aux_element`, `Cell<`, `doc(hidden)`, `unreachable!`, `pub const`, `impl Drop`, `fn workspace_size`,
`panic!`/`unwrap`/`expect`; read the whole public surface of `segment_sum.rs` (`pub` item list and `can_implement`), the `pub` item
lists of `segment_sum_backward.rs` and `segment_max.rs`, and the id arguments of the two sum plans. The other bodies were **not**
read. A verdict on the public shape, not on the CUDA kernels or numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 20 `mod`s, 20 re-exports | 40 | stable | one module per op, re-exported from `segment/mod.rs` |
| 20 plans | 20 | stable | `select` + `can_implement` + `run` + `workspace_size` + `sku` on all (finding 1) |
| 40 descriptors and args, 148 fields | about 188 | **freeze decision** | none is `#[non_exhaustive]` (finding 2) |
| 120 methods | 120 | stable with a note | findings 3, 4 |

## Findings

1. **Every plan has `select`, `can_implement`, `run`: 20 of 20, and `run` calls it.** Grep: 20 files each for `pub struct .*Plan`,
   `pub fn select`, `pub fn can_implement`, `pub fn run`, and `fn workspace_size`. `self.can_implement(&args)?` is the first
   statement in `run` for `segment_sum.rs:139` and `unsorted_segment_sum.rs:150` (read); the other 18 were not checked. Positive
   control: the same `can_implement` grep finds 0 in `pool`, `linalg`, `loss` and `conv`. With `quantize`, one of two families where
   the Plan/Descriptor/Args shape is complete.
2. **No type is `#[non_exhaustive]`.** 0 hits for the attribute line in all 21 files (the word appears in no `Debug` helper either).
   40 descriptor and args structs are `pub` with `pub` fields (`SegmentSumDescriptor`, `SegmentSumArgs` at `segment_sum.rs:25, 39`).
   A new descriptor field is a breaking change. The families with the attribute (`quantize` 5, `pool` 11, `loss` 2) do not agree
   on a rule; this one is the all-or-nothing case.
3. **`aux_element` is set in 3 files only, all `Some(I32)`.** `segment_sum.rs:271`, `unsorted_segment_sum.rs:213`,
   `unsorted_segment_max_backward.rs:221`. The segment ids are `i32` (`TensorRef<'a, i32, 1>`, `segment_sum.rs:44`,
   `unsorted_segment_sum.rs:58`, read), so `I32` is right where it is set. The other 17 plans were not read for how they build
   their `sku`: they may share a builder (26 `KernelSku` mentions in 20 files, of which 20 are the `sku()` signatures) or leave
   the slot empty. Same open question as `pool` 6 and `loss` 5, and the `indexing` finding (`aux_element` always `I32`).
4. **Ids are fixed at `i32`.** Both read plans take `segment_ids: TensorRef<'_, i32, 1>`. A call with `i64` ids is a type error, not
   a run-time one, and widening later is a new plan or a new generic parameter. Whether the other 18 also fix `i32` was not
   checked.
5. **No `Cell`, `Drop`, `doc(hidden)`, `unreachable!`, public const or panicking site.** 0 hits each (positive control: `pub fn select`
   finds 20 files, `Cell<` finds 43 hits in `pool`). So none of the cuDNN-state findings apply; this family is plain launches.
6. **Backward plans are separate types.** 10 `*_backward` modules each have their own `*BackwardPlan`, `*BackwardDescriptor`
   and `*BackwardArgs` (`segment_sum_backward.rs:26, 51, 80`). Same split as `elementwise` and `loss`, unlike `pool`'s `run_fw` /
   `run_bw` on one plan. Consistent within the family.

## Not decided here

- Whether the 40 descriptor and args structs get `#[non_exhaustive]` before the freeze (finding 2).
- Whether id width stays `i32` (finding 4).
- How the other 17 plans build `aux_element` (finding 3).
