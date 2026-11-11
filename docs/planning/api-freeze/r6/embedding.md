# R6 review: `baracuda-kernels` `embedding` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `fa4c7ac4`. Items: the 125 `stable`
rows under `crates/baracuda-kernels/src/embedding/` in the §7.4 census (6 `mod`s, 6 re-exports, 18 structs, 1 enum +
2 variants, 56 fields, 36 `Plan` methods; the 125 count was re-read from `items.tsv` at this ref and the parts sum to
it). Method: read the descriptor, args and plan signatures in all six files, grepped each `run` for `can_implement`,
and compared with `scan`, `softmax` and `causal_conv1d`. A verdict on the public shape, not on kernel numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 6 `mod`s, 6 re-exports | 12 | stable | the `embedding::embedding` module path is kept deliberately (`mod.rs` comment: renaming is API-breaking) |
| 6 descriptors + 32 fields | 38 | stable | `i32` sizes plus `Option<i32>` `padding_idx`; `r2-growth.tsv` rows: no descriptor grew (0 in every row) |
| 6 `*Args` + 24 fields | 30 | stable | tensor borrows with documented shapes; five carry `I: IndexElement = i32`, so the default keeps `i32` callers source-compatible |
| `EmbeddingBagMode` + `Sum`, `Mean` | 3 | stable | deliberately not `#[non_exhaustive]` (documented: Max has its own plan), so a new variant is a declared break |
| 6 plans | 6 | stable | private fields |
| 36 plan methods | 36 | stable | all six plans have the full six-method shape (`select`, `can_implement`, `workspace_size`, `sku`, `precision_guarantee`, `run`), and all six `run` bodies call `self.can_implement` |

No item is proposed for `#[doc(hidden)]` or removal.

## Findings

1. **`embedding/mod.rs` doc is stale on the index dtype.** It says "Index dtype is `i32` only". The code has
   `IndexElement` generic over `i32` and `i64` in five of six plans (`embedding.rs`, `embedding_backward.rs`,
   `embedding_bag.rs`, `embedding_bag_backward.rs`, `embedding_bag_max.rs`: each `run` matches `IndexElementKind::I64`),
   and `embedding_bag_max.rs` says "Indices: `i32` + `i64`" itself. `embedding_bag_max_backward` takes no index tensor
   (it reads `out_index`, always `i32`), so it is correctly `i32`. Doc fix only; not fixed here.
2. **The `u8`-vs-`i32` index axis does not apply here.** `scan`, `softmax` and `reduce` take `u8` flags and 6
   `indexing` files take `i32` indices; this family is the one that already generalises (`i32` and `i64`). The
   `indexing` review should be compared against this family, not the other way round.
3. **Dtype coverage is asymmetric by design, and documented.** Forward `f32, f64, f16, bf16`; backward (three plans)
   `f32, f64` only, because `atomicAdd` is native-FP. `select` returns `Error::Unsupported` for the rest. Stable.
4. **Sizes are `i32`** (`num_embeddings`, `embedding_dim`, `num_indices`, `total_indices`). A table over `i32::MAX`
   elements per dimension is unrepresentable; widening is a breaking field change, so it belongs to a pre-freeze
   decision if wanted. Raised, not proposed.

## Not checked

Numerics and the `.cu` kernels; whether Fuel or another consumer calls the `i64` paths; the `padding_idx` sentinel
(`i32::MIN` mapped from `None`) was read from `mod.rs` but not exercised, so an `Option<i32>` of `Some(i32::MIN)`
colliding with the sentinel was not tested.
