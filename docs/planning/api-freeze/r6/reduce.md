# R6 review: `baracuda-kernels` `reduce` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `ca12f302`. Items: the 126 `stable`
rows under `crates/baracuda-kernels/src/reduce/` in the §7.4 census (7 `mod`s, 7 re-exports, 21 structs, 44 fields,
47 methods; the 126 count was re-read from `items.tsv` at this ref and the parts sum to it). Method: read the
descriptor, args and plan signatures in all seven files, grepped each `run` for `self.can_implement`, and read the
`bool_axis` `can_implement` body in full; compared with `scan`, `softmax`, `embedding`. A verdict on the public shape,
not on kernel numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 7 `mod`s, 7 re-exports | 14 | stable | one module per plan, each re-exported |
| 7 descriptors + 27 fields (`arg_axis`, `axis`, `axis_backward`, `bool_axis`, `count_axis`; `reduce_to`; `trace`) | 34 | stable | small `Copy` structs, all fields documented; see finding 2 for `reduce_axis` |
| 7 `*Args` + 17 fields | 24 | stable | tensor borrows with documented shapes; `ArgReduceArgs` carries `I: IndexOutputElement = i64` so existing callers stay source-compatible |
| 7 plans | 7 | stable | private fields |
| 47 methods: 7 plans x the six-method shape, plus `output_shape` (4 descriptors) and `dy_shape` (1) | 47 | stable | all seven `run` bodies call `self.can_implement` |

No item is proposed for `#[doc(hidden)]` or removal.

## Findings

1. **`reduce/mod.rs` doc is stale on the arg-reduce index type.** It says `ArgReducePlan` returns `i64` indices.
   `arg_axis.rs` is generic over `IndexOutputElement` (default `i64`) and its `run` has `IndexOutputKind::U32` arms
   (9 matching lines) beside the `I64` ones. Doc fix only; not fixed here.
2. **The `u8` axis field is in five of seven descriptors** (`reduce_axis: u8` in `arg_axis`, `axis`, `axis_backward`,
   `bool_axis`, `count_axis`; `reduce_to` has none, it reduces every broadcast dim, and `trace` reduces both). This is
   the `u8` side of the portfolio `u8`-vs-`i32` axis finding (`scan` and `softmax` also use `u8`; six `indexing` files
   use `i32`). Each `select` rejects `reduce_axis >= N` (`bool_axis.rs:75`). Stable; the choice only matters for the
   cross-family comparison in the `indexing` review.
3. **Reduction kinds live outside this crate.** `ReduceKind`, `ArgReduceKind`, `ReduceToOp` and `IndexOutputKind` come
   from `baracuda_kernels_types` (re-exported at `lib.rs:67-75`), so adding a kind is a vocab-crate change, as with
   `SoftmaxKind` in `softmax`. Two descriptors (`bool_axis`, `count_axis`) take the shared `ReduceKind` and `select`
   narrows it at runtime rather than by type; not a freeze problem, noted for the vocab review.
4. **Rank is capped at 8 at `can_implement`, not at `select`** (`bool_axis.rs:154`): a `N > 8` const generic compiles
   and fails only when the plan is checked. Not checked whether the other six plans check it at the same point.

## Not checked

Numerics; the `mod.rs` claim that all reductions are deterministic with no atomic-add (a kernel property, not a
surface one); the rank cap in the other six plans; the `can_implement` bodies of the six plans other than `bool_axis`
(their presence and use in `run` were checked by grep, not their contents).
