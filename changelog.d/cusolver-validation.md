kind: **BREAKING**: baracuda-cusolver safe dense functions validate buffer sizes; SolverScalar members are uncallable; Op is non_exhaustive
---
baracuda-cusolver, from the R6 review (docs/planning/api-freeze/r6/cusolver.md, findings 1, 2, 5, 8, 9). Behaviour that changes:

- The 15 safe dense functions that take `m`, `n`, leading dimensions and device buffers (`getrf`, `getrs`, `geqrf`, `potrf`, `potrs`, `gesvd`, `syevd`, `syevj`, `gesvdj`, `orgqr`, `ormqr`, `gels`, `potri`, `syevj_batched`, `gesvdj_batched`) check `lda >= max(1, rows)`, `len >= lda * (cols - 1) + rows` and the `ipiv`/`tau`/`s`/`w`/`info` lengths, and return `INVALID_VALUE` before any cuSOLVER call. They used to pass any size to cuSOLVER. `mg::DeviceGrid::new` checks `devices.len() >= rows * cols`.
- The 12 `#[doc(hidden)]` members of `SolverScalar` take a token that cannot be built outside the crate, so `<f32 as SolverScalar>::getrf(..)` no longer compiles outside it. (The PM ruled "private sub-trait"; a private supertrait cannot name `Self::Real`, so the same effect comes from the token.)
- `Op` is `#[non_exhaustive]`.
- Removed the claims the code does not earn: README and crate docs no longer list `Xgeqrf`, `Xpotrf`, `Xsyevd` or a "full Rf API"; the `Cargo.toml` description no longer says "dense LU factorization at v0.1"; `gesvd` documents the `rwork` buffer correctly.
- Removed the unused dependencies `thiserror` and `baracuda-cuda-sys` (`cargo check --all-targets` passes without them; `Cargo.lock` loses the two edges).
- `sgetrf` / `sgetrs` are kept as documented convenience wrappers over the generic functions.

Verified: 4 unit tests for the size checks and the compile_fail doctest for the sealing run without a GPU; the doctest fails on the old trait (control); 3 device tests in `tests/validation.rs` were compiled, not run (GPU off until the Defender fix). The extents are the standard LAPACK/cuSOLVER sizes as I know them; I did not re-read the cuSOLVER documentation.
