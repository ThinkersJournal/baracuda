# R6 review: `baracuda-kernels` `pool` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10, 2026-11-13). Ref: `origin/main` `b2f45cd9`. Items: the 390 rows under
`crates/baracuda-kernels/src/pool/` in the §7.4 census: 384 `stable` and 6 `candidate-for-removal` (all `public`). By `kind`: 18 `mod`s,
19 re-exports, 48 structs, 161 fields, 140 methods, 1 enum, 3 variants; 18 + 19 + 48 + 161 + 140 + 1 + 3 = 390 (counted from
`items.tsv` at this ref). Twenty files, 6,352 lines: `adaptive_{avg,max}_pool{1,2,3}d`, `avg_pool{1,2,3}d`, `max_pool{1,2,3}d`,
`fractional_max_pool{2,3}d`, `lp_pool{1,2}d`, `lp_pool{1,2}d_backward`, plus `mod.rs` and `pool_nd.rs`; 18 plans (grep).
Method: grepped the whole directory for `fn can_implement`, `non_exhaustive`, `Cell<`, `fn select`, `fn aux_element`, `doc(hidden)`,
`unreachable!`, `pub const`, `fn workspace`, `impl Drop`, `fn run_fw`/`run_bw`; read `max_pool2d.rs` (`PoolMode`, the descriptor head,
`sku`, `unreachable!` site), the `AvgPool1dPlan` struct and its `select` head, the `mod.rs` re-export list and the head of
`adaptive_avg_pool1d::run_fw`. The other bodies were **not** read. A verdict on the public shape, not on cuDNN or numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 18 `mod`s, 19 re-exports | 37 | stable | one module per plan family, re-exported from `pool/mod.rs` |
| 6 `derived_kernel_stride` rows | 6 | **census is stale** | finding 1 |
| 48 structs, 161 fields, 3 variants, 1 enum | about 213 | stable with a note | finding 4 |
| 18 plans | 18 | **freeze decision** | no `can_implement`, `Cell` cuDNN state, two entry points (findings 2, 3, 5) |
| 134 other methods | 134 | stable with a note | findings 2, 3 |

## Findings

1. **The census still lists six methods that are already removed.** The 6 `candidate-for-removal` rows are
   `Adaptive{Avg,Max}Pool{1,2,3}dPlan::derived_kernel_stride`. `CHANGELOG.md:865` records them removed in R5 ("it only ever
   returned a poison value"), and `git grep derived_kernel_stride` outside the CHANGELOG and the R6 README finds only those two
   documents; the same grep in `adaptive_avg_pool1d.rs` is empty. The census was generated before R5, so the family's real
   count is 384, not 390. Not checked: whether the other families' censuses carry removed rows.
2. **No plan has a public `can_implement`.** 0 of 18 (grep over every file). `select` validates the descriptor and `run_fw` /
   `run_bw` call a private `check_fw_args` (read for `adaptive_avg_pool1d.rs:157-165`). A caller cannot pre-validate a call.
   Same family-level gap as `conv` 1 and `linalg` 1.
3. **cuDNN plans hold `Cell` state and a `Drop`.** `AvgPool{1,2,3}dPlan` and `MaxPool{1,2,3}dPlan` each have an `impl Drop` (grep,
   6 files) and keep a `Cell<cudnnHandle_t>` plus three descriptor `Cell`s (read for `avg_pool1d.rs:35-38`; the other five
   by grep, 43 `Cell<` hits in 8 files). By reading, `Cell` state over raw handles is neither `Send` nor `Sync`; not compiled.
   The adaptive, fractional and LP plans have no `Drop` (grep).
4. **`#[non_exhaustive]` is on 10 of 20 files, not uniformly.** By grep: the three adaptive-avg, both fractional, `lp_pool1d`,
   `lp_pool2d`, `max_pool{1,2,3}d` carry it (`max_pool2d.rs` twice, on `PoolMode` and `Pool2dDescriptor`). The adaptive-max,
   `avg_pool{1,2,3}d` and the two `*_backward` files carry none. Which of the 48 structs each attribute sits on was not traced
   per type, so the split by struct is unknown. `avg_pool*` reuse the `max_pool*` descriptors (`mod.rs` re-exports), so
   their descriptors are covered while their args may not be.
5. **Two entry points per plan, except LP.** `run_fw` and `run_bw` in 14 of 18 files (2 hits each);
   the four LP files have 1 `run_*` hit each (by file name, forward in `lp_pool*d` and backward in `*_backward`; bodies not read), so LP forward and
   backward are different plan types, while every other family has them on one plan. `workspace_size` exists in all 18
   (grep); its value was read only for `max_pool2d.rs:260`, not the others.
6. **`aux_element` is `None` where it appears, and appears in 5 files.** `adaptive_avg_pool1d`, `fractional_max_pool2d`,
   `lp_pool1d`, `lp_pool2d`, `max_pool2d` set `aux_element: None` (grep). The max-pool and fractional-pool families take or
   produce index tensors; whether the index element type is recorded in the sku elsewhere was not read, so this is
   a question, not a defect. Same open question as `loss` 5.
7. **One bare `unreachable!`.** `max_pool2d.rs:466`: "`Pool2dPlan::select` gates on F32/F64/F16/Bf16". The gate was not
   traced; the dtype set is a run-time check, not a type. Same shape as `cholesky.rs:227` in `linalg`.
8. **No public consts, no `#[doc(hidden)]`.** 0 hits each for `pub const` and `doc(hidden)` in the directory (positive control:
   `fn select` finds 18 files, so the pattern set reaches the sources). The family is free of kernel-limit consts, unlike
   `linalg`'s `WY_NB`.

## Not decided here

- Whether `can_implement` joins the freeze (finding 2); a family-level choice for conv, linalg, loss, pool together.
- Whether the six `Cell` plans become `Send`/`Sync` or stay documented as single-thread (finding 3).
- Whether to regenerate the census at the current ref (finding 1).
