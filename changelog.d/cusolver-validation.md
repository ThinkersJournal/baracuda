kind: **BREAKING**: baracuda-cusolver dense functions validate matrix arguments; SolverScalar members are uncallable; Op is non_exhaustive
---
baracuda-cusolver, from the R6 review (docs/planning/api-freeze/r6/cusolver.md, findings 1, 2, 5, 8, 9). Behaviour that changes:

- `getrf`, `getrs`, `geqrf`, `potrf`, `potrs`, `gesvd`, `syevd`, `orgqr`, `ormqr` and `potri` check each matrix argument against LAPACK's rule for it (`lda >= max(1, rows)`, `len >= lda * (cols - 1) + rows`; `gesvd`'s `U`/`V^T` by `jobu`/`jobvt`) and return `INVALID_VALUE` before any cuSOLVER call. Nothing else is checked (`ipiv`, `tau`, `w`, `s`, `info`, Jacobi, batched, `gels`, multi-GPU): that stays the caller's contract until the NVIDIA documentation can be read.
- The 12 `#[doc(hidden)]` members of `SolverScalar` take a token that cannot be built outside the crate, so `<f32 as SolverScalar>::getrf(..)` no longer compiles outside it.
- `Op` is `#[non_exhaustive]`.
- README and crate docs no longer claim `Xgeqrf`, `Xpotrf`, `Xsyevd` or a "full Rf API"; the `Cargo.toml` description no longer says "dense LU factorization at v0.1"; `gesvd` documents `rwork` correctly.
- Removed the unused dependencies `thiserror` and `baracuda-cuda-sys`.
- `sgetrf` / `sgetrs` kept as convenience wrappers.
