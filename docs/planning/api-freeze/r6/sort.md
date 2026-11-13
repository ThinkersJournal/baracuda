# R6 review: `baracuda-kernels` `sort` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `8a3414f0`. Items: the 268 `stable` rows
under `crates/baracuda-kernels/src/sort/` in the §7.4 census (14 `mod`s, 14 re-exports, 2 `const`s, 45 structs, 103
fields, 90 methods; counted from `items.tsv` at this ref by `kind`, all 268 rows are `stable`). Method: read the
descriptor, args and plan signatures in all fifteen plans (fourteen files; `msort.rs` holds two), the `select` bodies for
every cap and dtype check, and `mod.rs` in full; grepped each `run` for `self.can_implement`; compared with `indexing`,
`norm`, `gemm`. A verdict on the public shape, not on kernel numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 14 `mod`s, 14 re-exports | 28 | stable | one module per file, re-exported from `sort/mod.rs`; the `sort::sort` path is pinned by a comment and `allow` |
| 2 `const`s (`SORT_MAX_ROW`, `TOPK_MAX_K`) | 2 | **change** | see findings 2 and 3 |
| 15 descriptors + 58 fields | 73 | stable, except `histogramdd` (finding 1) | small `Copy` structs, documented fields; see findings 4, 5, 6 |
| 15 `*Args` + 45 fields | 60 | stable, except `histogramdd` | tensor borrows; see findings 4, 7 |
| 15 plans | 15 | stable, except `histogramdd` | private fields |
| 90 methods: 15 plans x the six-method shape | 90 | stable, except `histogramdd` | all fourteen real plans' `run` call `self.can_implement` |

Row sums: 28 + 2 + 73 + 60 + 15 + 90 = 268. The 58 / 45 field split is by `items.tsv` name, `*Descriptor.` against `*Args.`,
counted at this ref.

## Findings

1. **`HistogramddPlan` cannot be constructed, and its 16 rows are marked `stable`.** `select`
   (`histogramdd.rs:57-79`) returns `Err(Unsupported)` on every path: `ndim != 1` is rejected, and the `ndim == 1` path
   falls through to an unconditional `Err` ("reserved API surface"). So no `HistogramddPlan` value exists; its
   `can_implement`, `run`, `sku`, `workspace_size` and `precision_guarantee` are unreachable, and `can_implement` and
   `run` return `Unsupported` unconditionally (`can_implement` only runs `check_fits` first). The 16 rows are the `mod`,
   the re-export, 3 descriptor fields + the struct, 2 args fields + the struct, the plan, and 6 methods. The
   `ndim > 1` text in the error is also wrong for `ndim == 1`. Freezing this freezes a shape with no implementation
   behind it; the freeze should reclassify the 16 rows (not stable) or drop them until the N-d kernel exists. Not
   changed here.
2. **`SORT_MAX_ROW` is not the family's limit.** `pub const SORT_MAX_ROW: i32 = 1024` is documented "Maximum supported
   `row_len` ... must match `MAX_ROW` in `baracuda_sort.cuh`", and `mod.rs` says larger rows return `Unsupported`. That
   holds for `sort`, `sort_backward`, `msort`, `msort_backward`, `topk`, `topk_backward`, `kthvalue_backward` (all
   check `> SORT_MAX_ROW`), but **`ArgsortPlan` accepts `row_len > 1024`** through a multi-block radix path with a
   non-zero `workspace_size()` (`argsort.rs:88`, `:142`, `:206`). The `mod.rs` doc and the const doc are stale for
   argsort, and a public const named `MAX` that one plan exceeds is a poor thing to freeze. Same drift kind as `gemm`
   and `indexing`.
3. **`TOPK_MAX_K = 64` means two things.** `TopkPlan` and `TopkBackwardPlan` reject `k > 64`; `KthvaluePlan` rejects
   `k + 1 > 64` (`kthvalue.rs:107`), so its largest `k` is 63, and its descriptor doc says `k < 64`. Also `kthvalue`'s
   `k` is 0-indexed (`k < row_len`), which differs from PyTorch's 1-indexed `kthvalue` and from `topk`'s `k` (a count).
   Worth a doc line before the freeze.
4. **Rank and index dtype are fixed in the types.** Every `*Args` uses rank-2 `TensorRef<T, 2>` (`[batch, row_len]`)
   and the indices are always `TensorMut<i32, _>`; `mod.rs` states the `i32` indices as a deliberate contract. The
   `indexing` family is generic over `I: IndexElement`, and `norm`/`softmax` over `const N`. Freezing `i32` indices
   freezes the lack of an `i64` path (and with `row_len: i32`, of rows past 2^31 - 1), and a caller with a rank-3
   tensor must reshape.
5. **Extent types are mixed inside the family.** `batch`, `row_len`, `k`, `max_unique`, `num_bins`, `len_sorted` are
   `i32`; `numel` (`histogram`, `bincount`, `histogramdd`) and `num_queries` (`searchsorted`) are `i64`. So
   `searchsorted` has an `i64` count next to an `i32` length. The portfolio extent-type choice applies; here it is
   inconsistent within one module.
6. **Dtype coverage is narrow and uneven.** `sort`, `msort`, `argsort` (and their backwards) accept `f32, f64, i32,
   i64` only (`validate_sort_desc`, `argsort.rs:100`); no `f16` or `bf16`. `kthvalue` is `f32, f64` ("TopkPlan limit",
   `kthvalue.rs:111`). The `mod.rs` doc does not list dtypes. Not an API-shape defect, but the freeze should know that
   half-precision callers (LLM `topk` is the stated use) have no path.
7. **Two scratch conventions.** `ArgsortPlan` reports a real `workspace_size()` for its big path; `UniquePlan`
   returns `workspace_size() == 0` and instead requires the caller to pass `sorted_scratch` and `sorted_idx_scratch`
   as `Args` fields (`unique.rs:45-52`, "caller-allocated; overwritten"). The `mod.rs` doc says `unique` "chains sort +
   consecutive-dedup at the plan layer", which hides that the caller owns the intermediate buffers. Flag names also
   differ by op for the same idea: `descending` (`sort`, `argsort`, `msort`), `largest` (`topk`), `right`
   (`searchsorted`).
8. **`sku()` index kind is consistently `I32`.** `aux_element: Some(ElementKind::I32)` appears in `sort.rs`'s
   `build_sku` (shared by the sort family), `histogram.rs`, `searchsorted.rs` and `unique_consecutive.rs`. Since the
   indices are `i32` by contract (finding 4), the `indexing` problem (sku says `I32` while `run` accepts `I64`) does not
   arise here.

## Not checked

Kernel numerics and stability/tie semantics; the `can_implement` bodies beyond presence and use in `run` (buffer sizes,
strides); whether `bincount`, `topk`, `unique` and `kthvalue_backward` set `aux_element` themselves or inherit it (the
grep finds four literal sites); whether `topk` accepts `i32`/`i64` (only `kthvalue`'s `f32`/`f64` limit was read);
`Send`/`Sync`; `non_exhaustive` (none appears in the family by grep).
