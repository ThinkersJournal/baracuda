# baracuda-cusolver

Safe Rust wrappers for **NVIDIA cuSOLVER** — dense and sparse linear-system
solvers, eigensolvers, and matrix factorizations on the GPU. Covers
`cuSolverDn`, `cuSolverSp`, `cuSolverRf`, and `cuSolverMg` (see Coverage for what
each part has).

Generic over scalar type (`f32` / `f64` / `Complex32` / `Complex64`)
where the math allows.

## Coverage

- **Dense (`cuSolverDn`)**:
  - LU: `getrf` + `getrs`
  - QR: `geqrf` + `orgqr` + `ormqr`
  - Cholesky: `potrf` + `potrs` + `potri`
  - SVD (`gesvd`, `gesvdj`, batched `gesvdj`)
  - Eigensolvers: `syevd`, `syevj`, batched `syevj`
  - Least-squares: `gels`
  - Generic 64-bit `X*` family, as raw `unsafe` functions in `xapi`:
    `Xgetrf` (+ buffer size), `Xgetrs`, `Xpotrs`.
- **Sparse (`cuSolverSp`)**: `scsrlsvchol`, `scsrlsvqr` (f32 only).
- **Refactor (`cuSolverRf`)**: create, `analyze`, `refactor`, destroy only (no matrix setup yet).
- **Multi-GPU (`cuSolverMg`)**: `getrf`, `potrf`, `syevd`.

## Workspace handling

Like cuBLASLt and cuSPARSE, cuSOLVER requires the caller to provide
device + host scratch sized via a `bufferSize` query. baracuda-cusolver
exposes both queries and the actual operation as separate calls so you
can amortize the allocation across many problems of the same shape.

Pairs with [`baracuda-cusolver-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cusolver-sys`]: https://docs.rs/baracuda-cusolver-sys
