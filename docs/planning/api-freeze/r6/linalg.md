# R6 review: `baracuda-kernels` `linalg` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10, 2026-11-13). Ref: `origin/main` `3e04bc56`. Items: the 295 `stable`
rows under `crates/baracuda-kernels/src/linalg/` in the §7.4 census (15 `mod`s, 15 re-exports, 45 structs, 121 fields, 91
methods, 1 const, 2 enums, 5 variants; counted from `items.tsv` by `kind` at this ref, and 15 + 15 + 45 + 121 + 91 + 1 + 2 + 5
= 295; all `public` and `stable`). Seventeen files, 7,242 lines: `cholesky`, `eig`, `eigh`, `inverse`, `lstsq`, `lu`, `qr`,
`solve`, `svd` (cuSOLVER-backed), `qr_batched`, `qr_batched_materialize`, `svd_batched`, `svda_batched`, `ormqr_batched`,
`ormqr_batched_wy`, plus `handle_init`; 15 plans.
Method: per file, grepped `pub struct .*Plan`, `pub fn can_implement`, `pub fn run`, `self.can_implement`,
`unwrap/expect/panic/unreachable`, `non_exhaustive`, `aux_element`, `^impl`; read `handle_init.rs` in full, `cholesky.rs`
(`workspace_size`, `query_workspace_size`, the head of `run`), the two enums and `WY_NB`, and the `mod.rs` re-export list. The
other bodies were **not** read. A verdict on the public shape, not on cuSOLVER or numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 15 `mod`s, 15 re-exports | 30 | stable | one module per plan, re-exported from `linalg/mod.rs` |
| 2 enums, 5 variants, 1 const | 8 | stable with a note | findings 5, 6 |
| 30 descriptors and args, 121 fields | about 150 | stable with a note | finding 5 |
| 15 plans | 15 | **freeze decision** | no `can_implement`, `Cell` state, per-dtype `run` impls (findings 1 to 3) |
| 91 methods | 91 | stable with a note | findings 2, 3 |

## Findings

1. **No plan in the family has a public `can_implement`.** 0 of 15 (grep over every file). Validation is a private
   `check_args` called from `run` (read for `cholesky.rs:311-322`: `check_dense` on `a` and `info`, then
   `self.check_args(&args)`). A caller cannot pre-validate a linalg call. Same family-level gap as `conv` 1.
2. **Cached workspace size and a `_stream` that is ignored.** `workspace_size()` returns a `Cell` that is `0` before the
   first query or run and, for `batch_size > 1`, stays `0` because cuSOLVER allocates internally (documented,
   `cholesky.rs:186-193`). `query_workspace_size(&self, _stream: &Stream)` takes a stream it does not use (`cholesky.rs:197`;
   the same signature in `eigh`, `svd`, `qr_batched_materialize`, by grep). The plan holds a lazily created handle and a
   `Drop` impl (`cholesky.rs:473`). By reading, `Cell` state over a raw handle is neither `Send` nor `Sync`; not compiled.
   Same shape as `conv` 2 and 4.
3. **`run` is defined per concrete dtype, not on the generic plan.** `CholeskyPlan<T: Element>` has `select` and the
   accessors generically but `run` in `impl CholeskyPlan<f32>` and `impl CholeskyPlan<f64>` (`cholesky.rs:309, 396`), each
   taking `CholeskyArgs<'_, f32>` or `<'_, f64>`. `run` is defined twice in `cholesky`, `lu`, `svd`, four times in `eigh`, once
   elsewhere (grep). So `CholeskyPlan<f16>` is nameable and `select` rejects it at run time ("select() gates on F32 / F64", the
   `unreachable!` message, `cholesky.rs:227`). Adding a dtype is a new `impl`, not a break; the supported set is
   not visible in the type.
4. **`handle_init` is a public module that the freeze census does not list.** `#[doc(hidden)] pub mod handle_init;`
   (`mod.rs:104-105`) with two `pub fn`: `create_cublas_handle`, `create_cusolver_dn_handle` (`handle_init.rs`, read). 0 rows
   in `items.tsv` mention it (15 `mod` rows, 16 modules), so it is public and unfrozen by accident. Callers inside this crate:
   the linalg plans and `loss/fused_linear_cross_entropy.rs` (grep). Whether any other crate calls these two by path was
   **not** established: the `kernels-sys` facades matched the broader pattern and were not inspected. This is the HANDOFF
   backlog item ("-> `pub(crate)`"), now with the census gap.
5. **`#[non_exhaustive]` on none of the 15 descriptors or the two enums.** `BatchedOrmqrSide { Left, Right }` is closed by
   nature; `BatchedOrmqrOp { N, T, C }` is not purely so, because `T` is real-only and `C` is complex-only (documented
   at `ormqr_batched.rs:75-80`), so the op validity depends on the dtype at run time. Same split as `attention` 2,
   `conv` 5, `elementwise` 2, `image` 1.
6. **`WY_NB` is a public const that must equal a CUDA header's value.** `pub const WY_NB: i32 = 32;` with the doc "Must
   match the value in `baracuda_batched_ormqr_wy.cuh`" (`ormqr_batched_wy.rs:93`). A frozen public number that two files must
   keep equal; keep `pub(crate)`, or generate one from the other.
7. **`aux_element` is `Some(I32)` in 12 of 15 plans** (the `info` buffer); `None` in `ormqr_batched`, `ormqr_batched_wy`,
   `qr_batched_materialize` (grep). Consistent as "the int output"; not checked against each `Args` tensor type.
8. **The panics are guarded by `select`/`check_args`, by reading of the messages only.** Non-test `unreachable!`/`expect`
   sites: the dtype `match` arms ("select() gates on F32 / F64", `cholesky.rs:227`, `lstsq.rs:329`, `svda_batched.rs:298`,
   `eig.rs:511, 523`), three with no message (`lstsq.rs:370, 409`, `qr_batched_materialize.rs:334`), and two `expect`s that cite
   `check_args` (`svda_batched.rs:414, 423`). The bare ones were not traced to their gate.

## Not checked

cuSOLVER or numerical behaviour; every `check_args`, `select` and `run` body beyond the lines quoted; whether the three
bare `unreachable!()` are gated; `Send`/`Sync` by compilation; the `kernels-sys` facades' use of the `handle_init` names
(finding 4). Nothing was executed.
