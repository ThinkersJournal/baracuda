# R6 review: `baracuda-kernels` `norm` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `3713ab29`. Items: the 235 `stable` rows
under `crates/baracuda-kernels/src/norm/` in the §7.4 census (10 `mod`s, 10 re-exports, 30 structs, 115 fields, 70
methods; counted from `items.tsv` at this ref by `kind`, all 235 rows are `stable`, and 10 + 10 + 30 + 115 + 70 = 235).
Method: read the descriptor, args and plan signatures in all ten files and every `select` body up to the sku;
grepped each `run` for `self.can_implement`; read the `InstanceNorm*` wrappers in full; compared with `indexing`,
`gemm`, `scan`, `softmax`, `reduce`. A verdict on the public shape, not on kernel numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 10 `mod`s, 10 re-exports | 20 | stable | one module per plan, each re-exported from `norm/mod.rs` |
| 10 descriptors + 49 fields | 59 | stable | small `Copy` structs; see findings 2, 3, 4 |
| 10 `*Args` + 66 fields | 76 | stable | tensor borrows; see finding 5 for the stat-buffer naming |
| 10 plans | 10 | stable | private fields |
| 70 methods | 70 | stable | 8 plans x six-method shape (48), 2 instance plans x five (10), 12 descriptor helpers; see finding 1 |

(The 49 / 66 split of the 115 fields is by `items.tsv` name, `*Descriptor.` against `*Args.`, counted at this ref. Row sums: 20 + 59 + 76 + 10 + 70 = 235.)

## Findings

1. **`InstanceNorm{,Backward}Plan` have no public `can_implement`, and `select` can panic.** Both are wrappers over the
   `GroupNorm` plans and expose `select`, `workspace_size`, `sku`, `precision_guarantee`, `run` only. `run` does call the
   inner plan's `run`, which calls `can_implement` (checked: all eight other plans' `run` call `self.can_implement`), so
   the args are validated, but a caller cannot pre-validate as with every other plan in the crate. Worse,
   `InstanceNormPlan::select` (`instance_norm.rs:71`) and the backward twin read `desc.input_shape[desc.channel_axis as
   usize]` **before** the inner `select` rejects `channel_axis != 1`; a `channel_axis >= N` (a `u8`, so up to 255)
   indexes out of bounds and panics instead of returning `Error`. By reading; not run. The same unguarded index is in
   the public `num_channels()` of the four `BatchNorm`/`GroupNorm` descriptors (`batch_norm.rs:58`,
   `batch_norm_backward.rs:40`, `group_norm_backward.rs:43`, `group_norm.rs`), safe inside `select` only because the
   `channel_axis != 1` check comes first. Fix is small; not made here.
2. **`channel_axis` is a field that accepts exactly one value.** Six descriptors (`BatchNorm`, `GroupNorm`,
   `InstanceNorm`, forward and backward) carry `channel_axis: u8` and every `select` rejects anything but `1`
   ("this trailblazer"). The freeze either drops the field (channel-first only) or implements it. Freezing a field
   that has one legal value makes a later second value a behaviour change on a frozen type.
3. **Rank bounds disagree.** `BatchNorm`, `GroupNorm` and (through it) `InstanceNorm` take `N` in `2..=8`; `LayerNorm`
   and `RMSNorm` take `1..=8`. All are checked in `select` (`N == 0 || N > 8`), unlike `indexing` where the cap is
   checked at `can_implement`. Consistent within the family; this answers the open `reduce` question for `norm`.
4. **Axis and extent types.** `norm_axes_mask: u8` (a bitmask, rank <= 8 fits), `channel_axis: u8`, `num_groups: u32`,
   `input_shape: [i32; N]`, `num_channels() -> i32`. The `u8` axis agrees with `scan`/`softmax`/`reduce`; the `i32`
   shape is the portfolio's open extent-type choice, not specific to `norm`. `num_groups: u32` is the one unsigned
   extent, and `select` casts `num_channels()` to `u32` to test divisibility.
5. **Saved-statistic naming and shape differ across the family.** `BatchNorm`/`GroupNorm`/`InstanceNorm` save
   `saved_mean`, `saved_rstd` as rank-1 `[C]` (or `[N*G]`) buffers; `LayerNorm` saves `mean`, `inv_std` as rank-`N`
   buffers shaped `save_shape()`; `RMSNorm` saves `rms` (the RMS itself, not its reciprocal) at `rms_shape()`. Three
   spellings for "what forward hands backward". Same `T` as the data (so `f16` stats for `f16` data) although the
   kernels accumulate in `f32`. Stable as a shape; the naming is the freeze's call.
6. **Affine flags differ.** `BatchNorm`/`GroupNorm`/`InstanceNorm` use one `has_affine` for `gamma` and `beta`
   together; `LayerNorm` uses `has_gamma` and `has_beta`; `RMSNorm` has only `has_gamma`. Each `*Args` also carries
   `Option` tensors, so the flag and the `Option` can disagree; `can_implement` is where that is checked.
7. **`InstanceNorm*Plan::sku()` reports the `GroupNorm` kernel.** It returns `self.inner.sku()`, so an instance-norm
   plan's identity is `NormalizationKind::GroupNorm`. True to the "same kernel symbols" doc, but a caller keying on the
   sku cannot tell the two ops apart. All ten `sku()` have `aux_element: None`, so the `indexing` index-dtype problem
   does not arise here.
8. **`norm/mod.rs` doc is mostly current.** Its op list and deferred list (`WeightNorm`, `LocalResponseNorm`,
   BatchNorm inference mode) match the code; only "Design notes" mention the naive `O(extent^2)` per-row cost without
   saying whether that is still true. Not re-measured.

## Not checked

Kernel numerics and the stated `f32` accumulation; the `can_implement` bodies beyond presence and use in `run` (buffer
sizes, strides); the `NormalizationKind` enum (lives outside this crate); `Send`/`Sync`; finding 1's panic was read,
not executed (a `select` needs a `Stream`, so a device-free test was not attempted).
