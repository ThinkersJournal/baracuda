# R6 review: `baracuda-kernels` `scan` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `316bb006`. Items: the 38 `stable`
rows under `crates/baracuda-kernels/src/scan/` in the §7.4 census (6 structs, 16 fields, 12 methods, 2 re-exports,
2 `mod`s). Method: read `scan/mod.rs`, `axis.rs`, `axis_backward.rs` signatures and compared them with `reduce`,
`softmax` and `indexing` descriptors by `grep`. This is a verdict on the public shape, not on kernel correctness.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| `pub mod axis`, `pub mod axis_backward` | 2 | stable | the two paths downstream code names; both re-exported below |
| `ScanDescriptor<N>`, `ScanBackwardDescriptor<N>` + 10 fields | 12 | stable | field set did not grow in 5 and 6 history revisions (`r2-growth.tsv`: `grew_events` 0), so R2(b) leaves them literal; `kind`/`input_shape`/`scan_axis`/`reverse`/`element` |
| `ScanArgs`, `ScanBackwardArgs` + 6 fields | 8 | stable | `*Args` stay literal (§7); every field is a tensor borrow |
| `ScanPlan<T,N>`, `ScanBackwardPlan<T,N>` | 2 | stable | private fields, so no layout promise |
| 12 `Plan` methods (`select`, `can_implement`, `workspace_size`, `sku`, `precision_guarantee`, `run` on each) | 12 | stable | the same six-method shape as `softmax` and `reduce` plans |
| the two `pub use` lines | 2 | stable | |

No item is proposed for `#[doc(hidden)]` or removal. Total 38 of 38 stable.

## Findings (none blocks the freeze for `scan`)

1. **Axis type is `u8` here and `i32` in `indexing`.** `scan_axis`, `softmax_axis`, `reduce_axis`, `channel_axis` and
   `axis` are `u8` (23 fields by `grep`), while `gather_dim`, `select_dim` and `scatter_dim` are `i32` (6 files:
   `indexing/{gather,gather_backward,index_select,index_select_backward,scatter,scatter_add}.rs`). `scan` follows the
   majority, so nothing changes in `scan`; the question belongs to the `indexing` review.
2. **`ScanBackwardPlan::can_implement` is 68 lines against `ScanPlan::can_implement`'s 28** (lines 170 to 238 against
   157 to 185). Not a surface issue; noted only because the extra checks (saved `x` / `y` presence per kind) are the
   documented backward contract.
3. **Module docs say "Phase 4 Category F"**, a build-history label; harmless, left alone.

## Not checked

Kernel numerics (the `scan` plan docs claim bit-stable on the same hardware; not re-measured here); whether anything
outside this repository names `ScanPlan` (the Fuel `e6cbf782` measurement in §7.1 covered `*Descriptor`s only).
