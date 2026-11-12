# R6 review: `baracuda-kernels` `gemm` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `3713ab29`. Items: the 217 rows under
`crates/baracuda-kernels/src/gemm/` in `items.tsv` at this ref (210 `stable`, 2 `sealed-trait-pattern`, 1 with the class
cell typed `stabl` (`mod.rs:54`, a census typo, treated as `stable`), 4 with an empty class cell (the `mod.rs` re-exports at
lines 49, 57, 61, 62, where the TSV split on the comma inside `{...}`)). By kind: 10 `mod`, 14 `reexport`, 27 `struct`, 1 `enum`,
3 `variant`, 93 `field`, 62 `method`, 4 `trait`, 2 `const`, 1 `fn`; 10+14+27+1+3+93+62+4+2+1 = 217.
Method: read every descriptor, args and plan signature in the eleven files, grepped each `run` for `self.can_implement`,
read the `select` zero/negative-extent checks, the `cfg(feature)` gates and the `sku()` bodies; compared with
`indexing`, `scan`, `softmax`, `reduce`. A verdict on the public shape, not on kernel numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 10 `mod`s, 14 re-exports | 24 | stable (finding 1) | one module per plan, each re-exported; `mod.rs` doc is stale |
| 9 descriptors + their fields | see below | stable, but see 2, 3, 5 | `Copy` structs with `pub` fields; only two are `#[non_exhaustive]` |
| 9 `*Args` structs + fields | see below | stable | tensor borrows; no builder |
| 9 plans, 62 methods | 9 + 62 | stable | every `run` calls `self.can_implement` (grep: all nine, `int_gemm.rs:209` inside its match arm) |
| `Sealed` x2, `AwqActivation`, `MarlinActivation` | 4 | stable | sealed to `f16` (Marlin) / AWQ dtypes; the sealing is the contract |
| `gptq_to_marlin` (2 structs, 9 fields, 2 consts, 1 fn) | 14 | **needs attention** (finding 6) | different error type and style from the plans |
| `DenseGemmLayout` + 3 variants | 4 | stable | `Copy, Eq, Hash`, not `non_exhaustive` |

Nine plans: bin, dense, fp8, gemv_dense, int (dispatcher), int4, int4_awq, int4_marlin, sparse24. Nothing is proposed for
removal. The two items proposed for `#[doc(hidden)]` or a type change are in findings 4 and 6.

## Findings

1. **`gemm/mod.rs` module doc is stale.** It says "Today this module hosts the integer GEMM dispatcher `IntGemmPlan`".
   The module now exports nine plans (dense FP, GEMV, FP8, bin, int4, sparse 2:4, Marlin, AWQ) plus a host repack
   utility. Doc fix only, not fixed here. Same drift as `indexing/mod.rs` and `embedding`.
2. **Only two of nine descriptors are `#[non_exhaustive]`** (`Int4AwqGemmDescriptor` `int4_awq.rs:96`,
   `Int4MarlinGemmDescriptor` `int4_marlin.rs:93`, both with `new` + `with_*` builders). The other seven
   (`BinGemm`, `DenseGemm`, `Fp8Gemm`, `GemvDense`, `Int4Gemm`, `IntGemm`, `GemmSparse24`) are plain structs with `pub`
   fields built by struct literal, so adding any field after the freeze is a breaking change. The freeze should choose one
   convention for the whole family; the AWQ/Marlin builder shape is the only one that lets a descriptor grow.
3. **Extent and dim types are `i32` everywhere** (`m`, `n`, `k`, `batch`, `ic`, `oc`, `group_size`, `max_par`,
   `split_k_iters`), while the batch strides are `i64` (`dense_gemm.rs:138-146`, `gemv_dense.rs:89-97`) and the host
   repack uses `usize` (`gptq_to_marlin.rs:83-87`). The `i32` side of the portfolio axis finding (see `indexing` finding 2),
   plus a `usize`-vs-`i32` split inside one family. A product `m*k` is not checked here for `i32` overflow; the buffer-fit
   checks are `view_check::check_fits` on the tensors.
4. **Field naming and order disagree across plans.** `GemmSparse24Descriptor` declares `n` before `m` (`sparse24.rs:81-86`)
   where every other plan is `m, n, k`; AWQ names the GEMM extents `m, ic, oc` while Marlin names the same problem
   `m, n, k` (`int4_awq.rs:99-106` vs `int4_marlin.rs:96-102`). Both are struct-literal-visible. `IntGemmArgs`,
   `IntGemmDescriptor` and `GemmSku` are not defined here: `int_gemm.rs:35` re-exports them from `baracuda_cutlass`, so
   freezing `baracuda-kernels` also freezes three types owned by another crate (a cross-crate coupling the freeze should
   either accept explicitly or break with local wrappers).
5. **`m == 0` is accepted or rejected depending on the plan.** `select` rejects `m <= 0` in `bin`, `fp8`, `int4`, `int`
   and accepts `m == 0` in `dense`, `gemv_dense`, `int4_awq`, `int4_marlin`, `sparse24` (`< 0` rejected only).
   For AWQ the acceptance is hollow: `can_implement` rejects `m <= 0` (`int4_awq.rs:312`) and `run` calls it first, so
   the `m == 0` early return at `int4_awq.rs:340` is unreachable. Marlin's `can_implement` checks shapes against
   `[m, k]`; whether `m == 0` reaches its early return at `int4_marlin.rs:334` was not run to ground. The empty-problem
   contract (no-op `Ok` vs error) should be one rule for the family before it is frozen.
6. **`gptq_to_marlin::repack` returns `Result<_, &'static str>`**, not the crate `Error` used by every plan
   (`gptq_to_marlin.rs:126`), and its `GptqWeights` / `MarlinWeights` have `pub` fields with no `non_exhaustive`.
   `MARLIN_PERM_LEN` / `MARLIN_SCALE_PERM_LEN` are public consts that are layout internals of vendored Marlin. Proposed:
   convert the error to the crate type before freeze, and keep the two consts `pub` only if a caller needs them (none
   checked).
7. **Feature-gated `run` bodies keep the signature, change the result.** `bin`, `fp8`, `int4` need `sm89`, `sparse24`
   needs `xformers_sparse24`, Marlin `marlin`, AWQ `awq`; without the feature `run` compiles and returns
   `Error::Unsupported` (`bin_gemm.rs:233-237`), so the public surface is stable and the capability is a runtime error.
   That is the intended design (stated in `mod.rs`); the freeze should say so, because `select` succeeding does not imply
   `run` can. Checked for `bin_gemm`; the other gates were read by grep for `cfg(feature` only.
8. **`dense_gemm` and `gemv_dense` overlap by design** (m == 1 capture-safe companion; both docs say so). Both accept `m == 0`
   and `batch == 0` and skip launch (`dense_gemm.rs:272`, `gemv_dense.rs:222`), consistent with each other.

## Not checked

Kernel numerics; `can_implement` bodies beyond the `m/n/k` and shape checks quoted (alignment and stride rules for
`int4`, `fp8`, `bin`, `sparse24` were not compared); whether `Int4MarlinGemmPlan` with `m == 0` returns `Ok`;
`Send`/`Sync` of the plans (auto traits not compiled); the `sku()` bodies of `dense_gemm`, `gemv_dense` (grep found only
the signature at `dense_gemm.rs:392`; `gemv_dense` has no `sku()` at all, only `backend()`, which is itself an
inconsistency with the other eight plans); per-field counts per descriptor (the 93 fields were split by kind, not by owner).
