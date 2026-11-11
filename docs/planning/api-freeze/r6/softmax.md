# R6 review: `baracuda-kernels` `softmax` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `74e5ac19`. Items: the 106 `stable`
rows under `crates/baracuda-kernels/src/softmax/` in the §7.4 census (6 `mod`s, 6 descriptors + 24 fields,
6 `*Args` + 15 fields, 6 plans, 36 `Plan` methods, 1 `const`, 6 re-exports; the 106 count was re-read from
`items.tsv` at this ref). Method: read the descriptor/args/plan signatures in all six files, compared them with `scan`, and
read `r2-growth.tsv`. This is a verdict on the public shape, not on kernel correctness.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 6 `pub mod` (`axis`, `axis_backward`, `gumbel`, `gumbel_backward`, `sparsemax`, `sparsemax_backward`) | 6 | stable | the paths downstream code names; each is re-exported below |
| 6 `*Descriptor<N>` + their fields | 30 | stable | `r2-growth.tsv` `grew_events` is 0 for all six (4 or 5 history revisions each), so R2(b) leaves them literal |
| 6 `*Args` + their fields | 21 | stable | `*Args` stay literal (§7); every field is a `TensorRef` / `TensorMut` borrow |
| 6 `*Plan<T,N>` | 6 | stable | private fields, so no layout promise |
| 36 `Plan` methods (`select`, `can_implement`, `workspace_size`, `sku`, `precision_guarantee`, `run` on each) | 36 | stable | the same six-method shape as `scan` and `reduce` |
| `SPARSEMAX_MAX_EXTENT` | 1 | stable | the documented sparsemax row-extent cap (1024); a value change is a behaviour change, so it needs a version line |
| 6 `pub use` lines | 6 | stable | |

No item is proposed for `#[doc(hidden)]` or removal. Total 106 of 106 stable.

## Findings (none blocks the freeze for `softmax`)

1. **`softmax/mod.rs` module doc is stale on the sparsemax limit.** It says "Row extent limited to 64 in the
   trailblazer (per-thread serial sort in local memory)"; `sparsemax.rs` and `SPARSEMAX_MAX_EXTENT` say 1024 since
   Phase 11.6, and `select` enforces 1024. Doc-only; not fixed here (this PR is docs under `docs/`).
2. **`GumbelSoftmaxPlan` is `!Sync`** (it holds a `Cell<curandGenerator_t>`; documented in its doc comment as the
   same model as `RandomPlan`). That is a threading promise downstream will lean on, so freezing the type freezes
   `!Sync`; the other five plans are plain data. Worth stating in the §7 notes, no change proposed.
3. **`SoftmaxKind` is not ours.** `SoftmaxDescriptor.kind` and `SoftmaxBackwardDescriptor.kind` carry
   `unpopped_vocab::SoftmaxKind` (re-exported by `baracuda-kernels-types`), so the stability of that enum is
   Unpopped's, not this review's.
4. **Axis type is `u8`** (`softmax_axis` in all six descriptors), matching `scan` and `reduce`; the `i32` axes are in
   `indexing` (see the `scan` review, finding 1). No change in `softmax`.
5. **`sparsemax` and `gumbel` descriptors have no `kind`** while `SoftmaxDescriptor` does; they are separate ops, not
   `SoftmaxKind` variants, so the asymmetry is by design.

## Not checked

Kernel numerics (the `axis` plan docs claim bit-stable on the same hardware; not re-measured); whether anything
outside this repository names the `softmax` plans (the Fuel measurement in §7.1 covered `*Descriptor`s only);
the `SoftmaxKind` variant list in `unpopped_vocab`.

— lane: baracuda
