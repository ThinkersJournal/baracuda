# R6 review: `baracuda-kernels` `indexing` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `5294463c`. Items: the 196 `stable`
rows under `crates/baracuda-kernels/src/indexing/` in the §7.4 census (11 `mod`s, 11 re-exports, 33 structs, 71 fields,
70 methods; the 196 count was re-read from `items.tsv` at this ref, all 196 rows are `stable`, and the parts sum to it).
Method: read the descriptor, args and plan signatures in all eleven files, grepped each `run` for `self.can_implement`,
read the `can_implement` and `select` bodies of `gather` in full and grepped the rest for the same checks; compared with
`scan`, `softmax`, `embedding`, `reduce`. A verdict on the public shape, not on kernel numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 11 `mod`s, 11 re-exports | 22 | stable | one module per plan, each re-exported from `indexing/mod.rs` |
| 11 descriptors + 39 fields | 50 | stable | small `Copy` structs, fields documented; see findings 2 and 5 |
| 11 `*Args` + 32 fields | 43 | stable | tensor borrows; 9 of 11 carry `I: IndexElement = i32`, so existing callers stay source-compatible |
| 11 plans | 11 | stable | private fields |
| 70 methods: 11 plans x the six-method shape (66), plus 4 `MaskedFillDescriptor::new_{f32,f64,i32,bool}` | 70 | stable | all eleven `run` bodies call `self.can_implement` |

The 33 structs are 11 descriptors, 11 args, 11 plans; the 71 fields are 39 on descriptors and 32 on args (split by
`items.tsv` name). Row sums: 22 + 50 + 43 + 11 + 70 = 196. No item is proposed for `#[doc(hidden)]` or removal.

## Findings

1. **`indexing/mod.rs` doc is stale on the index dtype and on dtype coverage.** It says "Index dtype is `i32` only (i64
   deferred)". Nine of the eleven plans (`gather`, `gather_backward`, `index_add`, `index_select`,
   `index_select_backward`, `nonzero`, `one_hot`, `scatter`, `scatter_add`) are generic over `I: IndexElement` and their
   `run` has `IndexElementKind::I64` arms; `gather.rs:240` says so ("Phase 11.5"). `OneHotPlan` is described as "input is
   always `i32` class indices", but `OneHotArgs` is generic too. The dtype list is also off: it gives `ScatterAddPlan`
   and `IndexSelectPlan` `f32, f64, i32`, but `scatter_add.rs:125` and `index_select_backward.rs:99` accept `f32, f64`
   only; `index_add.rs:124` and `scatter.rs:133` additionally accept `f16` and `bf16` and are not mentioned in the doc at
   all. Doc fix only; not fixed here. This is the same kind of drift as `embedding` and `reduce`.
2. **The axis and extent fields are `i32` in seven descriptors** (`gather_dim`/`src_dim_size`, `add_dim`/`dst_dim_size`,
   `select_dim`/`src_dim_size`, `scatter_dim`/`out_dim_size`, and the `gather_backward` and `index_select_backward`
   counterparts), with `max_nz: i32` (`nonzero`) and `num_classes: i32` (`one_hot`). This is the `i32` side of the
   portfolio `u8`-vs-`i32` axis finding: `scan`, `softmax` and `reduce` use `u8`. The carried note said six `indexing`
   files; re-counted at this ref it is **seven** by `pub *_dim: i32`. Each `select` rejects a dim outside `0..N`
   (`gather.rs:114`), so a negative value is an error, not PyTorch-style wrap (the module doc says this too). `embedding`
   already generalises its index to `i32`/`i64`; the axis type is the open choice for the freeze.
3. **Every `sku()` reports `aux_element: Some(ElementKind::I32)` regardless of the index type** (nine files:
   `gather.rs:153`, `gather_backward.rs:134`, `index_add.rs:147`, `index_select.rs:133`, `index_select_backward.rs:121`,
   `nonzero.rs:163`, `one_hot.rs:146`, `scatter.rs:153`, `scatter_add.rs:147`). The sku is built in `select`, before an
   `I` is known, and `run` then accepts an `i64` index buffer, so the identity of the kernel this plan picked is wrong for
   `i64`. Freezing `KernelSku` as it stands freezes that. Not fixed here; the fix is a design choice (move the index
   kind into the descriptor, or make `sku()` depend on `run`).
4. **The rank cap is checked at `can_implement`, not at `select`, in all eleven plans** (`N > 8`, for example
   `gather.rs:184`); `N == 0` is checked at `select` in nine plans (all but `masked_fill`, `masked_fill_backward`). This answers
   the open question from the `reduce` review for this family. A `N > 8` const generic compiles and fails only when the
   plan is checked.
5. **`MaskedFillDescriptor` carries both `T` and an `element` field.** `MaskedFillPlan<T, N>` is generic over `T` and the
   descriptor also stores `element: ElementKind`; the four `new_*` constructors set the two consistently, but nothing
   stops a hand-built descriptor from disagreeing with `T` (the `select` bodies check `T::KIND`, `masked_fill.rs:144`).
   `fill_bits` is a raw `i64` the caller must encode by hand for `f32`/`f64`/`i32`/`bool` (documented on the field).
   Stable as is; the freeze fixes both the redundancy and the encoding.
6. **`can_implement` is not a full validation.** `gather.rs:173-212` checks out/index shape, rank, buffer lengths and
   `src numel`, but not that `src_dim_size` matches the `src` tensor's shape along the gather axis (the comment at
   `gather.rs:206` says it checks "enough to address `src_dim_size`", and the code only checks for overflow). Out-of-range
   index values are skipped by the kernel by design (module doc). Not checked whether the other nine do better.

## Not checked

Kernel numerics and the skipped-on-out-of-bounds semantics; the `can_implement` and `select` bodies of the ten plans
other than `gather` (presence and use in `run` were checked by grep, and the dim, rank and dtype checks by grep, not the
buffer-size checks); whether `masked_fill`, `masked_fill_backward` and `nonzero` have a per-dtype `run`; `Send`/`Sync`
(no explicit bounds or `non_exhaustive` appear in the family; auto traits were not checked by compiling).
