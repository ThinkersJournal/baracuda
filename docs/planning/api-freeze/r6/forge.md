# R6 review: `baracuda-forge`

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `25d4baeb`. Items: the 126 rows for `baracuda-forge` in the §7.4 census, all `stable`;
by `reach`, 8 `public` (the re-exports) and 118 `private-mod`. By `kind`: 80 methods, 12 fields, 10 structs, 10 variants, 8 re-exports,
4 fns, 1 enum, 1 type (80 + 12 + 10 + 10 + 8 + 4 + 1 + 1 = 126, counted from `items.tsv` at this ref). Methods by type: `KernelBuilder` 28,
`ParallelConfig` 9, `ComputeCapability` 8, `SourceSelector` 8, `DependencyManager` 7, `GpuArch` 7, `BuildCache` 5, `ExternalDependency` 4,
`CudaToolkit` 3, `PtxOutput` 1 (sums to 80). Nine files, 3,145 lines in `src/`. A vendored fork of `cudaforge` (provenance in the crate `NOTICE`).
Method: grep counts over `src/*.rs`, read `lib.rs`, `error.rs`, the struct heads of `compute_cap.rs` and `toolkit.rs`, the census rows. Method bodies
were **not** read, nothing was built. A verdict on the public shape, not on nvcc behaviour.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 8 re-exports | 8 | stable | `lib.rs:93-100` re-export every public type; the 9 modules are private |
| `Error`, 10 variants, `Result` | 12 | **freeze decision** | finding 1 |
| `KernelBuilder`, `PtxOutput` and 29 methods | 31 | stable with a note | finding 3 |
| `GpuArch`, `ExternalDependency`, `CudaToolkit` and 12 pub fields | 15 | **freeze decision** | finding 2 |
| other structs and 51 methods | 61 | stable | no hazard found by reading heads and grep |
| 4 free fns | 4 | stable | `detect_compute_cap`, `get_gpu_arch_string`, `resolve_cutlass_from_cargo_checkouts`, `collect_headers`, each resolves to one `pub fn` |

## Findings

1. **`Error` is closed and its variants carry stringly payloads.** 0 `#[non_exhaustive]` in the crate (grep). `Error` has 10 variants, so a new
   failure mode (a new nvcc diagnostic class, a cache format) is a breaking change as the crate stands. Eight variants wrap a `String` or
   `PathBuf`; only `CompilationFailed` is a struct variant. `Io(#[from] std::io::Error)` makes `std::io::Error` part of the API, and
   `thiserror` derives the impls (not re-exported). Decision: `#[non_exhaustive]` on `Error`. Variants stay as they are.
2. **12 public fields on three structs, none `#[non_exhaustive]`.** `GpuArch { base, suffix }`, `CudaToolkit { nvcc_path, include_dir, lib_dir,
   version }` and `ExternalDependency { name, repo_url, commit, include_paths, extra_paths, recurse_submodules }`. Struct-literal construction
   is possible downstream, so adding a field (a cached `nvcc` version string, a `sparse` flag) breaks every literal. `GpuArch` is also
   `PartialEq + Eq + Hash`, so its fields are part of equality. Decision: `#[non_exhaustive]` on `CudaToolkit` and `ExternalDependency`
   (the likeliest to grow), keep `GpuArch` closed or add it too.
3. **`KernelBuilder` is a by-value builder with one odd receiver.** By grep over `builder.rs`, 16 `pub fn`s take `self` or `mut self`, 1 takes
   `&mut self` (`set_compute_cap`, `builder.rs:199`, a setter that returns nothing) and 5 take `&self`. `Default` is implemented (`builder.rs:76`).
   The struct derives only `Debug` (no `Clone`), so a half-configured builder cannot be reused. Neither is a freeze blocker; the receiver
   mix is worth a decision before it is frozen. Not checked by compiling.
4. **Census `reach` is `private-mod` for 118 of 126 rows.** Every public type sits in a private module and is re-exported at `lib.rs:93-100`, so the
   items are reachable from outside the crate. The extractor's `reach` column marks the defining module, not the path a consumer sees; read the
   re-export rows to get the reachable set. A reader who filters on `reach = public` sees 8 rows and misses the API.
5. **0 `unsafe`, 0 `doc(hidden)`.** 0 `pub unsafe fn`, 0 `unsafe impl`, 0 `# Safety` headings, 0 `pub static` in `src/` (grep). The two
   `pub const` grep hits in `builder.rs` sit in a doc comment and a generated-code string template, not items. `#![deny(missing_docs)]` is set (`lib.rs`).
6. **Serde is on `BuildCache`.** `hash.rs:17,24` derive `Serialize, Deserialize` for the cache types, so their field set is an on-disk format
   as well as an API. A format change silently invalidates every consumer's incremental-build cache (`BuildCache` is `pub`). Decision: version the
   cache file or document that a format change forces a full rebuild.
7. **Two in-repo consumers.** `baracuda-cutlass-kernels-sys` (`Cargo.toml:42`) and `baracuda-kernels-bench` (`:103`) depend on forge; which surface
   they exercise was not read, so the other types may have no in-repo caller to keep them honest.
8. **Census names are current for the 4 free fns.** Each resolves to one `pub fn` in `src/` (grep count 1 each, so the query finds present items).
   Line numbers were not re-checked per row.

## Not decided here

- Whether `Error`, `CudaToolkit` and `ExternalDependency` become `#[non_exhaustive]` (findings 1, 2).
- Whether `KernelBuilder`'s receivers are unified (finding 3).
- Whether the cache file is versioned (finding 6).
- Not checked: method bodies, doc-test coverage, the `NOTICE` provenance claims.
