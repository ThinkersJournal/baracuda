# R6 review: `baracuda-runtime`

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `b931ce59`. Items: the 431 `public` rows for `baracuda-runtime` in the
§7.4 census `items.tsv`: 216 methods, 77 fns, 36 fields, 34 structs, 33 mods, 12 variants, 11 re-exports, 4 enums, 2 types, 2 consts, 2 traits,
2 trait members (216 + 77 + 36 + 34 + 33 + 12 + 11 + 4 + 2 + 2 + 2 + 2 = 431, counted from the file by `kind`; all `public`, all `stable`).
Twenty-six source files, 5,287 lines; the largest by rows are `memory` 50, `graphics` 48, `graph` 37, `query` 35.
Method: grepped every file for `^\s*#\[non_exhaustive\]`, `pub enum`, `pub unsafe fn` (with a scan of each one's preceding doc block for `# Safety`),
`pub trait`, `fn as_raw`/`from_raw`, `unsafe impl Send`, `impl Drop for`, `cfg(feature`, `sealed`; read `lib.rs`, `error.rs`, `interop.rs`, the
`[features]` table, the `Device` definition and the `PointerAttributes`/`DeviceProperties` field lists; cross-matched struct and enum names against the
`baracuda-driver` census rows. The other method bodies were **not** read. A verdict on the public shape, not on runtime behaviour.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 33 `mod`s, 11 re-exports | 44 | stable | 21 top-level `pub mod` lines plus `interop` behind a feature (finding 6); the other 11 `mod` rows are nested or inline |
| 34 structs, 36 fields | 70 | **freeze decision** | 6 structs carry the `pub` fields (36 census rows), none `#[non_exhaustive]` (finding 2) |
| 4 enums, 12 variants | 16 | **freeze decision** | none `#[non_exhaustive]` (finding 1) |
| 216 methods, 77 fns, 2 consts | 295 | stable with a note | 59 `unsafe fn` (finding 3); the same names as `baracuda-driver` (finding 4) |
| 2 traits, 2 members | 4 | stable | `driver-interop` only (finding 6) |
| `Error`, `Result` | 2 | stable | aliases of `baracuda_core::Error<cudaError_t>` (finding 5) |

## Findings

1. **No enum is `#[non_exhaustive]`.** 0 hits for the attribute line in 26 files, and `0` in the census `non_exhaustive` column for all 431 rows.
   Four enums: `CaptureMode` (`graph.rs:27`), `PrefetchTarget` (`memory.rs:237`), `AccessFlags` (`mempool.rs:23`), `MemoryType` (`query.rs:15`).
   Each mirrors a CUDA enum that grows between releases. Same decision as `driver.md` finding 1, and the same four type names exist there.
2. **Six structs have `pub` fields and no `#[non_exhaustive]`** (census `field` rows): `DeviceProperties` (22, `query.rs:87-127`), `PointerAttributes`
   (4, `query.rs:42-49`), `MulticastProp` (4, `multicast.rs:26-33`), `Dim3` (3, `launch.rs:15-19`), `DriverEntryPoint` (2, `driver_entry.rs:22-24`) and
   `UpdateResult` (2, `graph.rs:603-607`): 37 fields counted by name, against 36 `field` rows, so one of these rows is not a field row in the census
   (not traced). `DeviceProperties` mirrors `cudaDeviceProp`, which gains members every toolkit; one new field breaks every struct literal and every
   exhaustive destructure downstream. Decision: `#[non_exhaustive]` on the descriptors the runtime fills (`DeviceProperties`, `PointerAttributes`,
   `MulticastProp`, `DriverEntryPoint`, `UpdateResult`), so callers never need to build them; `Dim3` stays a plain triple.
3. **59 `pub unsafe fn` in 12 files** (`graphics` 27, `stream` 7, `memcpy3d` 5, `graph` 4, 8 more files). Scan of each preceding doc block: 59 of 59
   have a `# Safety` heading, 0 without (positive control: the scan counts the same 59). That the heading exists says nothing about whether the
   contract is right. `Stream::from_raw` (`stream.rs:59`) and `as_raw` (`:123`) expose `cudaStream_t`: the safe API is typed by
   `baracuda-cuda-sys` handles, as in `driver.md` finding 3. `as_raw`/`from_raw` occur in 13 files.
4. **30 of the crate's 38 public struct and enum names are also public types in `baracuda-driver`** (matched by name between the two census files):
   `Device`, `Stream`, `Event`, `Graph`, `GraphExec`, `GraphNode`, `DeviceBuffer`, `ManagedBuffer`, `Array`, `Kernel`... `Dim3`, `CaptureMode`,
   `UpdateResult`, `PrefetchTarget`, `AccessFlags`, `MemoryType` and more. Only the names were compared, not the shapes: they are two parallel API
   surfaces over the Driver and Runtime APIs, and a user of both crates (the `driver-interop` bundle) imports two `Dim3`s. The pair freezes together
   or not at all: a field added to one without the other splits them. Decision: say in the docs that they are intentionally separate types, or
   re-export the shared ones from one crate.
5. **`Error` is `baracuda_core::Error<cudaError_t>`** (`error.rs:7`): same ownership question as `driver.md` finding 7; review once in the
   `baracuda-core` doc.
6. **One feature gates public surface: `driver-interop`** (`Cargo.toml`, `lib.rs:107`; module `interop` with 2 traits, 2 members and 2 impls
   on `baracuda-driver` types, 6 census rows). The census lists them with no feature qualifier. Both traits are unsealed (0 hits for `sealed`). It pulls
   in an optional dependency on `baracuda-driver`, so the freeze of this crate depends on the driver crate's freeze when the feature is on. Unlike
   `driver`'s `async`, the feature gates real items.
7. **`Device` is a `pub(crate)` ordinal** (`device.rs:14`, `Device(pub(crate) i32)`): construction goes through the crate. Good for the freeze (no
   public representation to keep); contrast with `Dim3`.

## Where the census is stale or silent

- Dated 2026-10-14; the crate is at 0.1.0-alpha.141. Not re-run. The header counts come from `items.tsv` itself; only the module, enum, trait and `unsafe`
  counts were re-derived from the source at `b931ce59`, and they match. Rerun `items.tsv` before the freeze.
- No column for feature gating (finding 6), `unsafe` (finding 3) or signature types from other crates (finding 3).

## Not reviewed

Method bodies, whether the 59 `# Safety` contracts are right, per-module doc coverage, and the `graphics` API (48 rows of GL, D3D, EGL, VDPAU and
NvSci interop, not exercised on this box; `docs/UNVERIFIED-HARDWARE.md`).
