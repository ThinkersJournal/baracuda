# R6 review: `baracuda-cublas`

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `fdfb1715`. Items: the 176 rows for `baracuda-cublas` in the §7.4 census:
139 `stable`, 28 `doc-hidden`, 9 `sealed-trait-pattern`; by `reach`, 167 `public` and 9 `private-mod`. By `kind`: 38 fns,
33 methods, 29 trait-members, 21 variants, 18 traits, 11 mods, 9 re-exports, 7 structs, 5 enums, 3 fields, 2 types
(38 + 33 + 29 + 21 + 18 + 11 + 9 + 7 + 5 + 3 + 2 = 176, counted from `items.tsv` at this ref). Twelve files, 4,891 lines in
`crates/baracuda-cublas/src/`.
Method: grepped the directory for `non_exhaustive`, `unsafe impl`, `pub unsafe fn`, `# Safety`, `doc(hidden)`, `Sealed`, `pub const`,
`can_implement`, `pub fn as_raw`; read `handle.rs` (head), `lt.rs` (`Activation`, `MatmulHeuristic`, `unsafe impl Send`),
`xt.rs` head, the head of `blas_scalar.rs` and `ex.rs`, and `lib.rs` re-exports. Method bodies were **not** read. A verdict on
the public shape, not on cuBLAS behaviour.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 11 mods, 9 re-exports | 20 | stable | one module per level plus `lt`, `xt`, `ex`; re-exports match the census lines |
| 9 public scalar traits, 9 `Sealed` | 18 | stable with a note | findings 3, 4 |
| 29 trait-members (28 `doc(hidden)`) | 29 | **freeze decision** | finding 3 |
| 5 enums, 21 variants | 26 | **freeze decision** | finding 1 |
| `Handle`, `LtHandle`, `XtHandle` and 4 Lt descriptors | 7 structs, 3 fields | stable with a note | findings 2, 5 |
| 71 fns and methods | 71 | stable with a note | findings 5, 6 |

## Findings

1. **0 `#[non_exhaustive]` in the crate** (grep for the attribute line). The 5 enums are `Op`, `Fill`, `Side`, `Diag` (the fixed
   BLAS option sets, closed by the BLAS standard) and `lt::Activation` (`#[repr(i32)]`, the cuBLASLt epilogue list, which
   NVIDIA extends). `MatmulHeuristic` has 3 public fields, one of a sys type, and is not `#[non_exhaustive]` either.
   Decision: `Activation` and `MatmulHeuristic` yes; the four BLAS enums can stay closed.
2. **The documented `Send` on `Handle` looks wrong by reading.** `handle.rs:10-14` says handles are `Send` but not `Sync`;
   `Handle` is `Arc<HandleInner>` and only `HandleInner` has `unsafe impl Send` (`handle.rs:27`). `Arc<T>: Send` needs
   `T: Send + Sync`, and `HandleInner` is not `Sync` (raw pointer), so `Handle` is, by reading, neither `Send` nor `Sync` while
   the doc says `Send`. Not compiled. `LtHandle` and `XtHandle` carry `unsafe impl Send` directly (`lt.rs:142`, `xt.rs:21`), so they
   differ from `Handle`. Decision: compile a `fn assert_send<T: Send>()` check, then fix the doc or the type.
   The `SAFETY` comment exists on `HandleInner` and a plain comment on `LtHandle`; `XtHandle` has none.
3. **28 `#[doc(hidden)]` trait methods on `pub` traits.** `BlasScalar::gemm_raw`, `axpy_raw`, `BatchedGemmScalar::gemm_batched_raw`
   and the L1/L2/L3/direct/xt equivalents are `unsafe fn`s taking `cublasHandle_t`. The traits are sealed (9 `Sealed` traits in
   private mods; no downstream impls), but the methods are callable by anyone, hidden from rustdoc, and still part of the semver
   surface. Decision: make them `pub(crate)` via a private supertrait, or declare them frozen.
4. **Every scalar trait is sealed**, 9 public traits and 9 private `Sealed` markers (the census class
   `sealed-trait-pattern`); `L2Real` and `HermitianScalar` add `SealedReal` and `SealedComplex`. Consistent. The `blas_scalar.rs` doc
   says the seal is "at v0.1 ... while we work out the f16/bf16/complex stories", so widening the impl set later is compatible
   and narrowing is not.
5. **16 `pub unsafe fn`, with 17 `# Safety` headings in the crate** (grep). The `ex` module (6, type-erased `*const c_void`) shares
   names with the safe crate-root `axpy`, `dot`, `nrm2`, `scal`; `xt::gemm` shares one with the root `gemm`. Callers disambiguate by
   module path. `MatmulDesc::set_bias_pointer(*const c_void)` and `bias_pointer()` are safe fns over a raw pointer; the dereference
   is behind the `unsafe fn matmul` (`lt.rs:450`), so this is a documented-contract question, not unsoundness by reading.
6. **No `can_implement` anywhere** (0 hits) and no public consts (0 `pub const`). `#![warn(missing_debug_implementations)]` is set
   (`lib.rs:75`).
7. **The sys crate is part of the public API.** `lib.rs:89-90` re-exports `cublasComputeType_t`, `cudaDataType_t`, `cublasMath_t`,
   `cublasPointerMode_t`; 6 `as_raw` methods (`Handle` 1, `lt` 4, `xt` 1, by `pub fn as_raw` grep) and `MatmulHeuristic.algo` expose
   `cublas*_t`. Freezing them freezes the sys crate's enum shapes (finding 1 from the sys side).
8. **The census is current for this crate.** The 9 re-export rows sit at `lib.rs:89-97`, which is where the grep finds them, and
   all 71 census fn and method names resolve to `fn <name>` in the source (loop over the TSV; positive control: `fn heuristics_search`
   finds 1). 71 matches the 55 `pub fn` + 16 `pub unsafe fn` count.

## Not decided here

- Whether `Activation` and `MatmulHeuristic` become `#[non_exhaustive]` (finding 1).
- Whether `Handle`'s `Send` doc or type changes (finding 2; needs a compile check).
- Whether the 28 hidden trait methods are frozen or made crate-private (finding 3).
- Not checked: method bodies, doc-test coverage, the `Cell`/`Drop` ordering of the Lt descriptors.
