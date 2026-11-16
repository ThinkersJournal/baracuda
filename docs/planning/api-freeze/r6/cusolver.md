# R6 review: `baracuda-cusolver`

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `71547fb82`. Items: the 95 rows for `baracuda-cusolver` in the §7.4 census (`items.tsv`: 82 `stable`, 12 `doc-hidden`, 1 `sealed-trait-pattern`), all in
`src/lib.rs` (3,004 lines). By `kind`: 29 fn, 27 methods, 13 trait-members, 9 structs, 5 variants, 4 mods, 2 enums, 2 traits, 2 types, 2 reexports (29 + 27 + 13 + 9 + 5 + 4 + 2 + 2 + 2 + 2 = 95,
counted from `items.tsv` at this ref). The census omits the `real_impl!`/`complex_impl!` macro bodies; they are `impl SolverScalar for` blocks and add no new public name. Method: read `lib.rs` in full except
the macro bodies (lines 321-800) and the bodies of the three mg `*_buffer_size` functions, read `Cargo.toml`, `README.md`, `tests/gpu_smoke.rs`, `examples/qr.rs`, and counted with greps over `git ls-files`.
Nothing was built or run: the one test is gated behind `device-tests` and needs a device. A verdict on the public shape, not on numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| `SolverScalar`, `Real`, 12 hidden members, `Sealed` | 15 | **freeze decision** | finding 2 |
| 17 safe dense free functions (`getrf` .. `gesvdj_batched`, `sgetrf`, `sgetrs`) | 17 | **freeze decision** | findings 1, 8 |
| 12 `pub unsafe fn` (mg 6, `xapi` 4, `sparse` 2) | 12 | stable with a note | finding 4 |
| 2 re-exports, `Error`, `Result` | 4 | stable with a note | finding 3 |
| `Op`, `Side` and their 5 variants | 7 | **freeze decision** (`Op`) | finding 5 |
| 9 structs, their 27 methods, 4 `pub mod` | 40 | stable with a note | findings 3, 6, 7 |

Rows sum: 15 + 17 + 12 + 4 + 7 + 40 = 95.

## Findings

1. **Safe functions take unchecked dimensions over device pointers.** All 17 dense free functions are safe `pub fn`s that take `m`, `n`, `lda`, `nrhs`, `batch_size` as `i32` and pass
   `DeviceBuffer::as_raw()` pointers to cuSOLVER. `src/lib.rs` has **0** `assert`, `panic`, `expect` and `unwrap` and one `.len()` (`lib.rs:2225`, the `devices.len()` cast in `mg::Handle::device_select`;
   the same grep finds that line, so it can see them). So `getrf::<f32>(h, 1000, 1000, &mut a, 1000, ...)` with a 4-element `a` compiles, is not `unsafe`, and can make cuSOLVER read and write
   past the allocation on the device. `info`, `ipiv`, `tau` and `rwork` lengths are equally unchecked, and `gesvd`/`gesvdj` take `jobu: u8` and `jobvt: u8` as raw LAPACK bytes. `mg::DeviceGrid::new`
   (`lib.rs:2254`) takes `num_row_devices`, `num_col_devices` and `devices: &[i32]` separately and passes `devices.as_ptr()`; if cuSOLVER reads `rows * cols` entries (I believe so, from the
   `cusolverMgCreateDeviceGrid` signature; **not checked against NVIDIA's documentation**), a short slice is a host over-read from a safe function. Decision before the freeze: validate lengths and return
   `Err` (observable, so it must precede the freeze), or make the functions `unsafe`. Either way is breaking after the freeze.
2. **`SolverScalar` is sealed, but its 12 `#[doc(hidden)]` members are public callable API.** The trait is `pub`, its supertrait `sealed::Sealed` is in a private module, so nobody outside can implement it
   (`lib.rs:146`, `lib.rs:312`). The 12 members (`getrf_buf`, `getrf`, `getrs`, `geqrf_buf`, `geqrf`, `potrf_buf`, `potrf`, `potrs`, `gesvd_buf`, `gesvd`, `syevd_buf`, `syevd`) are `pub unsafe fn`
   trait items taking a raw `cusolverDnHandle_t` and raw pointers; `#[doc(hidden)]` removes them from rustdoc, not from the language, so `<f32 as SolverScalar>::getrf(..)` compiles for any caller and
   is semver-visible. The census marks them `doc-hidden`; that is a statement about docs. Their signatures freeze the `-sys` handle type and the exact argument order of 12 cuSOLVER calls. Decision:
   move them to a private sub-trait (keeping `Real` and the bound public) so the hidden surface is genuinely private, or accept them as frozen. The census's `doc-hidden` class should not be read as "free to change"
   for these rows. Separately, 8 free functions (`syevj`, `gesvdj`, `orgqr`, `ormqr`, `gels`, `potri`, `syevj_batched`, `gesvdj_batched`) do not use the trait: each dispatches on
   `mem::size_of::<T>() == mem::size_of::<f32>() && mem::size_of::<T::Real>() == 4` (8 occurrences, grep) and falls into the `Complex64` `else` branch for anything unmatched. For the four sealed
   types the branches are right (read, not run); a fifth impl of `SolverScalar` would silently take the complex-double path, so adding a scalar type is not a compile-checked change.
3. **Vendor types are in the public signatures.** `pub use baracuda_cusolver_sys::{cublasFillMode_t as Fill, cusolverEigMode_t as EigMode}` (`lib.rs:27`) and `{gesvdjInfo_t as GesvdjInfoRaw, syevjInfo_t as
   SyevjInfoRaw}` (`lib.rs:1252`) make four `-sys` names part of this crate's public API; `Fill` or `EigMode` appears in 24 lines of parameter lists (grep, approximate) and `cudaDataType` is a parameter of `mg::MatrixDesc::new`
   and the 4 `xapi` functions. `pub type Error = baracuda_core::Error<cusolverStatus_t>` (`lib.rs:30`) fixes the identity of `baracuda_core::Error` and of the status type, as in cutensor finding 2 and
   cuvs-sys finding 9. Nine `as_raw` methods return raw `-sys` handles ("Use with care"). Whether `Fill`/`EigMode` are `#[non_exhaustive]` is a property of the `-sys` definitions, which I did not read.
4. **12 `pub unsafe fn`, each with a `# Safety` section (12 sections, grep).** mg: `getrf_buffer_size`, `getrf`, `potrf_buffer_size`, `potrf`, `syevd_buffer_size`, `syevd` take `*mut *mut c_void` and
   `*mut *mut c_int` host-side arrays of device pointers; `xapi`: `xgetrf_buffer_size`, `xgetrf`, `xgetrs`, `xpotrs` take type-erased `*mut c_void` plus a `cudaDataType` tag, so the element type is not
   checked against the pointer; `sparse`: `scsrlsvchol`, `scsrlsvqr` take `descr_a: *mut c_void` (a cuSPARSE descriptor with the type erased). The `xapi` and `mg` functions are the only route to cuSOLVER
   on very large or multi-GPU problems, so their shape is the interface. Stable with the note that the dense safe API (finding 1) and this raw API sit side by side and a user cannot tell from the types which is which.
5. **`Op` and `Side` are closed enums.** `Op` has `N`, `T`, `C` (3 variants) and mirrors `cublasOperation_t`; `Side` has `Left`, `Right`. Neither is `#[non_exhaustive]`. Both are complete for the
   operations that take them today. Recommendation: `#[non_exhaustive]` on `Op` (a precaution: I did not check whether `cublasOperation_t` has more members than these three),
   leave `Side` closed (left/right cannot grow). `Op` derives `Default`, `Side` does not.
6. **Thread-safety is inconsistent.** `unsafe impl Send for DnHandle {}` (`lib.rs:53`) with no `Sync`; `SpHandle` and `RfHandle` carry an explicit `_not_send: PhantomData<*mut ()>`; the other six handle types
   (`SyevjInfo`, `GesvdjInfo`, `mg::Handle`, `DeviceGrid`, `MatrixDesc`, `xapi::Params`) say nothing and are `!Send` because they hold a raw pointer (auto-trait rule, not compiled). One `unsafe impl` in the file
   (grep). Whether `DnHandle: Send` is correct (a cuSOLVER handle is created against the current device and context) is **not checked**; there is no `assert_send`/`assert_not_send` test.
7. **Two modules are far smaller than their names.** `refactor::RfHandle` has `new`, `as_raw`, `analyze`, `refactor` and nothing that sets up a matrix or a sparsity pattern, so `analyze()` on a fresh handle has
   nothing to analyze (I did not run it). `sparse` has two functions, both `f32` (`scsrlsvchol`, `scsrlsvqr`), no `f64`/complex. A `Dn`-style generic API for these would replace the f32-named
   functions, which is a breaking change after the freeze; the names `scsrlsv*` encode a precision, as `sgetrf` does (finding 8).
8. **Documentation drift, 5 places.** (a) The crate header (`lib.rs:10`) says `xgeqrf`, `xpotrf` are exposed under `xapi`; `xapi` has `xgetrf`, `xgetrf_buffer_size`, `xgetrs`, `xpotrs` only (`grep "fn xgeqrf|fn xpotrf|fn xsyevd"`
   finds 0; the same grep finds `fn xgetrf`). (b) `README.md` lists `Xgeqrf`, `Xpotrf`, `Xsyevd` and "full Rf API" (finding 7). (c) `Cargo.toml` describes the crate as "dense LU factorization at v0.1".
   (d) `gesvd` says to pass "an empty buffer" for `rwork`; the example allocates a 1-element buffer. (e) `sgetrf`/`sgetrs` are labelled "Back-compat: single-precision shortcuts" for a crate with
   no earlier release line to be compatible with; they are 2 of the 17 safe functions and the only two the one test calls. Decision: delete them, or keep them and call them convenience.
9. **Two unused dependencies.** `thiserror` and `baracuda-cuda-sys` are in `[dependencies]`; neither `thiserror` nor `baracuda_cuda_sys` appears in `src/`, `tests/` or `examples/` (grep; control: `baracuda_core`
   in the same files is found at `lib.rs:30`). Removing a dependency is not an API change but changes the published manifest.
10. **Census lines are current; consumers and tests.** Loop over the 48 struct/enum/fn/type/trait/mod rows: 48 checked, 0 mismatched (each row's line must contain `struct|enum|fn|type|trait|mod <name>`);
    methods, variants and trait-members were not line-checked. The only in-repo dependents are the `baracuda` facade (`cusolver` feature, `pub use baracuda_cusolver as cusolver`, `crates/baracuda/src/lib.rs:114`),
    so the facade re-publishes this whole surface, and `math = [... "cusolver"]` includes it; `examples/qr.rs` and `tests/gpu_smoke.rs` are the only callers. Tests: 1 `#[test]` (`dense_lu_solve_3x3`, `device-tests`
    feature, covers `sgetrf` and `sgetrs`), 4 `no_run` doctests; no other public function has an executed test (1 `#[test]` in the crate, grep).

## Not decided here

- Finding 1: validate lengths (Err) or make the functions `unsafe`.
- Finding 2: move the hidden members to a private trait, or freeze them.
- Finding 5: `#[non_exhaustive]` on `Op`.
- Finding 8(e): keep or delete `sgetrf`/`sgetrs`.
- Not checked: macro bodies, `Send` soundness, cuSOLVER documentation, `-sys` enum definitions, numeric behaviour, and whether the cuSOLVER symbols named in the wrappers exist.
