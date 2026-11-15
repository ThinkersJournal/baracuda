# R6 review: `baracuda-driver`

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `8fae2fb9`. Items: the 626 `public` rows for `baracuda-driver`
in the §7.4 census `items.tsv`: 361 methods, 94 fns, 41 structs, 39 mods, 38 variants, 16 re-exports, 10 fields, 9 enums, 6 trait members,
5 consts, 3 traits, 2 types, 2 `macro_export` (361 + 94 + 41 + 39 + 38 + 16 + 10 + 9 + 6 + 5 + 3 + 2 + 2 = 626, counted from the file by `kind`;
624 `stable`, 2 `doc-hidden`). Thirty-one source files, 9,680 lines; the largest by rows are `memory` 92, `graph` 65, `graphics` 49, `array` 44.
Method: grepped every file for `^\s*#\[non_exhaustive\]`, `pub enum`, `pub unsafe fn`, `fn as_raw`/`from_raw`, `impl Drop for`,
`doc(hidden)`, `sealed`, `cfg(feature`, `unsafe impl Send`, `# Safety` (per function, by scanning each doc block); read `lib.rs`, `error.rs`, `pointer.rs` (`MemoryType`), the `Dim3`, `TextureDesc` and
`UpdateResult` definitions, and the `[features]` table. The other method bodies were **not** read. A verdict on the public shape, not on
driver behaviour.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 39 `mod`s, 16 re-exports | 55 | stable | 30 top-level modules plus 9 inline (`graphics` 7, `pinned::flags`, `graph::instantiate_flags`) |
| 9 enums, 38 variants | 47 | **freeze decision** | none `#[non_exhaustive]` (finding 1) |
| 41 structs, 10 fields | 51 | stable with a note | 3 types have `pub` fields (finding 2) |
| 361 methods, 94 fns, 5 consts | 460 | stable with a note | raw-handle surface (finding 3), 72 `unsafe fn` (finding 4) |
| `DevicePtr`, `DevicePtrMut`, `Present` + 6 members | 9 | **freeze decision** | public traits (finding 5) |
| `test_support`, 2 macros | 5 | stable, feature-gated | finding 6 |
| `Error`, `Result` | 2 | stable | aliases of `baracuda_core::Error<CUresult>` (finding 7) |

## Findings

1. **No enum is `#[non_exhaustive]`.** 0 hits for the attribute line in 31 files, and `items.tsv` has `0` in its `non_exhaustive` column for all
   626 rows. Nine public enums: `ArrayFormat`, `TextureAddressMode`, `TextureFilterMode` (`array.rs`), `CaptureMode` (`graph.rs`), `ManagedAttach`,
   `MemAdvise`, `PrefetchTarget` (`memory.rs`), `MemoryType` (`pointer.rs`), `AccessFlags` (`vmm.rs`). Each mirrors a CUDA enum, and CUDA adds
   members across releases. Exactly one, `MemoryType`, has a forward-compatible arm (`Unknown(u32)`, `pointer.rs:44`); the other eight would need a
   breaking change to grow. Decision for the PM or CireSnave: add the attribute now (free before the freeze, a breaking change after), or give the
   others an `Unknown`-style arm like `MemoryType`.
2. **Three types have `pub` fields.** `Dim3` (`x`, `y`, `z`, `launch.rs:42-46`), `TextureDesc` (4 fields, `array.rs:334-342`) and `UpdateResult`
   (3 fields, `graph.rs:1065-1072`); none is `#[non_exhaustive]`, so one new field breaks every struct-literal caller. `Dim3` is a plain triple that is
   unlikely to grow. `TextureDesc` mirrors a driver descriptor and `UpdateResult` mirrors `cuGraphExecUpdate`'s result, which can.
   `UpdateResult.result` is a raw `c_int` that the doc tells the caller to compare against `baracuda_cuda_sys` constants, so a public field is
   typed by the sys crate (finding 3).
3. **The safe API is typed by `baracuda-cuda-sys` handle types.** `as_raw` and `from_raw` (names matched by grep: `pub fn as_raw`,
   `pub fn from_raw`, `pub unsafe fn from_raw`, `pub fn raw(`) occur in 20 files; for example `Context::as_raw() -> CUcontext`
   (`context.rs:302`, `:401`) and `Stream::as_raw() -> CUstream` (`stream.rs:250`). So the driver crate's freeze is only as stable as
   `baracuda-cuda-sys`'s `CU*` types: freeze them together, or document the handles as opaque. The census does not record signature types, so it
   cannot show this coupling. Not counted per item.
4. **72 `pub unsafe fn` in 17 files** (`graphics` 26, `memory` 10, `graph` 7, `pointer` 5, `coredump` 4, `stream` 4, 11 more files with 1-2 each; the same
   count for all `unsafe fn`, so none is private). Each one's preceding doc block was scanned: 72 of 72 have a `# Safety` heading (0 without;
   positive control: the same scan counts all 72). That the heading exists says nothing about whether the contract is right. Their contracts are part
   of the freeze, because tightening one later is a breaking change in effect (it was done once: #208, alpha.136, made the async host copies unsafe).
5. **`DevicePtr` and `DevicePtrMut` are `pub unsafe trait`s** (`memory.rs:1270`, `:1304`; 5 members), so implementing them is `unsafe` and carries a
   documented `# Safety` contract (`memory.rs:1265-1269`). They are not sealed (0 hits for `sealed`/`Sealed` in the crate), so outside crates may
   implement them and a new required method is breaking. Decision: freeze the method set (4 + 1), or seal. `test_support::Present<T>` (`:36`) is a
   plain public trait with one method and the same question.
6. **`test_support` is a public module behind the `test-support` feature** (`lib.rs:67`; 5 census rows: `Present`, `Present::present`, `require!`,
   `require_optional!`, `__declined`). The census lists it with no feature qualifier, so its row count includes items a default build does not
   have. `__declined` is `#[doc(hidden)]` but `pub`, because the macros call it through `$crate`. It is not a contract and should be listed as
   excluded from the freeze. The other `doc(hidden)` row is `AccessFlags::raw` (`vmm.rs:243`, public method).
7. **`Error` is `baracuda_core::Error<CUresult>`** (`error.rs:7`), so the error type belongs to `baracuda-core`; a change to its variants moves
   every crate that aliases it. Review it once, in the `baracuda-core` doc. `error_name` and `error_string` return `Result<&'static str>`:
   the lookup itself can fail, so a caller formatting an error can get a second error.
8. **A declared feature with nothing behind it: `async`.** `Cargo.toml` has `async = ["dep:futures-core"]`; `cfg(.*feature` over `src` and `tests`
   finds `test-support` (`lib.rs:67`) and `device-tests` (test files) but 0 hits for `"async"`, and 0 hits for `futures_core`. The feature gates
   no public item, so it enables an unused dependency. Decision: remove it before the freeze (it is a feature name the freeze would otherwise
   carry), or implement what it promises.

## Where the census is stale or silent

- **Stale in places.** Dated 2026-10-14; the crate is at 0.1.0-alpha.141. Line numbers have drifted: the census puts `DevicePtr` at `memory.rs:1226` and it is at `:1270` (44 lines later; #208 touched `memory.rs` after the census). The census also has no `unsafe` column, so it cannot show finding 4. This review recounted by `kind` at `8fae2fb9`; the 30 top-level `pub mod` lines of `lib.rs`
  match the 30 top-level `mod` rows (positive control: the nine inline modules are the other nine rows). No item was found missing, only moved.
- Whether rows were added or removed since 2026-10-14 was **not** checked beyond the module count (no second census run): rerun `items.tsv` before the freeze.
- Feature gating is not a census column (finding 6), nor is the sys-crate coupling (finding 3).

## Not reviewed

Method bodies, whether the 72 `# Safety` contracts are correct, the per-module doc coverage, and every `graphics` API (49 rows, most are interop with GL, D3D, EGL, VDPAU
and NvSci, which are not exercised on this box; `docs/UNVERIFIED-HARDWARE.md`). A verdict on the public shape only.
