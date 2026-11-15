# R6 review: `baracuda-kernels-sys`

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `c8df7e3e`. Items: the 159 rows for `baracuda-kernels-sys` in the §7.4 census:
158 `stable`, 1 `doc-hidden`; by `reach`, 73 `public` and 86 `private-mod`. By `kind`: 54 consts, 34 `ffi-import-fn`, 28 types,
15 `fn-extern`, 8 fns, 8 fields, 8 re-exports, 4 structs (54 + 34 + 28 + 15 + 8 + 8 + 8 + 4 = 159, counted from `items.tsv` at this ref).
The crate is 92,123 lines of Rust, 79,842 of them in `lib.rs` (`#![no_std]`, raw `extern "C"` declarations plus facades).
Method: grep counts over `src/*.rs`, read `lib.rs` head, `handle_init.rs`, the census rows by kind. Declaration bodies and the C side were
**not** read. A verdict on the public shape, not on kernel behaviour.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| ABI constant, `AbiMismatch`, `check_abi` | 3 | stable | version integer is independent of the crate version, asserted by `scripts/check-abi-manifest.py` |
| `#[repr(C)]` structs (`cuComplex`, `cuDoubleComplex`, `cufftComplex`, `cufftDoubleComplex`) + 8 fields | 12 | stable with a note | finding 4 |
| 28 vendor types, 54 vendor consts | 82 | **freeze decision** | finding 3 |
| ~5,460 `baracuda_kernels_*` extern fns | not in census | **freeze decision** | finding 1 |
| 8 re-exports | 8 | stable with a note | finding 5 |
| `handle_init` | 2 fns | stable | serialises cuBLAS/cuSOLVER handle creation, plain `Result<_, i32>` |

## Findings

1. **The census covers about 1% of the public function surface.** Grep over `src/*.rs` finds 5,593 `pub fn` lines (5,460 named
   `baracuda_kernels_*`, 61 `cusolver*`, 34 `cudnn*`, 16 `cublas*`, 10 `cufft*`, 8 `curand*`, 4 helpers); the census has 57 fn-like rows
   (8 `fn`, 34 `ffi-import-fn`, 15 `fn-extern`). The 161 `extern "C"` blocks in `lib.rs` are therefore mostly unrowed. Every one of them is part
   of the C ABI the crate exports. Decision: either widen the census extractor to the extern blocks or declare ABI-manifest
   coverage (`check-abi-manifest.py`) the freeze mechanism for them. Not decided here which.
2. **The census line numbers are stale.** `baracuda_kernels_gather_rows_f32_run` is census line 77222 and sits at `lib.rs:76387`
   (about 835 lines earlier). Names are current: all 57 fn-like census names resolve to `fn <name>` (positive control: the gather-rows
   name above finds 1) and all 54 const names resolve to `const <name>`.
3. **Vendor types and constants are frozen by being public.** 28 types and 54 consts mirror cuDNN, cuBLAS, cuFFT, cuSOLVER, cuRAND and the CUDA
   runtime (22 `CUDNN_*`, 12 `CUBLAS_*`, 9 `CUFFT_*`, 6 `CUDA_*`, 3 `CUSOLVER_*`, 2 `CURAND_*` by name prefix). A vendor
   release that adds a value is not a change here, but removing or renumbering one is a breaking change this crate would have to follow.
   `baracuda-cudnn` and `baracuda-cublas` re-export sys types (see their R6 docs), so freezing here freezes there.
4. **`#[repr(C)]` layout is public API.** The four complex structs have `pub x`, `pub y` fields and no `#[non_exhaustive]`; the layout
   is the vendor's, so it cannot change. 0 `#[non_exhaustive]` anywhere in the crate and 0 `pub enum` (vendor enums are modelled as
   type aliases plus consts), so the question that dominates the other crates does not arise.
5. **8 `pub use <facade>::*` glob re-exports.** Facade modules are private (`mod x_facade; pub use x_facade::*;`), so the surface of
   each facade is whatever it declares `pub`, with no explicit list. A new `pub fn` in a facade is a new public item with no signal in the
   diff of `lib.rs`. Decision: keep (cheap) or switch to named re-exports (reviewable).
6. **Documentation shape.** 1,837 `# Safety` headings across the crate for 8 `pub unsafe fn` (the extern declarations carry their
   contract in the doc comment, and the crate docs list the common preconditions and the 0..5 status codes). 0 `#[doc(hidden)]` by grep; the
   one census row classed `doc-hidden` (`gather_rows_f32_run`) is listed that way by the extractor, but grep finds no `doc(hidden)`
   in the crate, so the class looks wrong. Not compiled; not checked in rustdoc output.
7. **The handle lock is a public contract.** `handle_init::create_cublas` and `create_cusolver_dn` are the supported way to create handles (a
   process-wide `Mutex<()>`, poison ignored). Returning a raw `i32` status keeps the crate free of an error type. A caller that creates a
   handle by calling `cublasCreate_v2` directly bypasses the lock; nothing prevents it.

## Not decided here

- How the ~5,460 extern fns are frozen (finding 1).
- Whether the facade glob re-exports become named (finding 5).
- Whether census row 77222's `doc-hidden` class is an extractor error (finding 6).
- Not checked: C-side signatures against the Rust declarations (the ABI manifest script does that), rustdoc output, the `links` key behaviour.
