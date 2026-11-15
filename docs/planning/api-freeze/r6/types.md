# R6 review: `baracuda-types`

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `edd7ae72`. Items: the 125 rows for `baracuda-types` in the §7.4 census, all `stable` and all
`reach = public`. By `kind`: 36 assoc-consts, 28 methods, 15 variants, 10 re-exports, 9 mods, 9 trait-members, 6 fields, 5 structs, 4 traits,
2 enums, 1 fn (36 + 28 + 15 + 10 + 9 + 9 + 6 + 5 + 4 + 2 + 1 = 125, counted from `items.tsv` at this ref). Ten files, 1,339 lines in `src/` plus 260
in `tests/`. Method: grep counts over `src/*.rs`, read `lib.rs`, `kernel_arg.rs`, `zero_bits.rs` (head), `stream_mode.rs`, `version.rs` (enum head),
`Half`'s `PartialOrd`, and the census rows. Method bodies were **not** read, nothing was built. A verdict on the public shape.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 9 mods, 10 re-exports | 19 | stable with a note | finding 6 |
| `Half`, `BFloat16`, `Complex32`, `Complex64`, 28 consts, 6 fields | 38+ | **freeze decision** | findings 2, 3 |
| `KernelArg`, `ValidAsZeroBits` (both `unsafe trait`) | 2 + 1 member | **freeze decision** | finding 4 |
| `HostSlice`, `CudaStatus` and 8 members | 10 | stable | plain traits, doc comments present |
| `Feature` (13 variants), `supports`, `CudaVersion` and 8 consts | 23 | stable | `Feature` is `#[non_exhaustive]`; `CudaVersion(u32)` is private |
| `StreamMode` (2 variants) | 3 | stable with a note | finding 1 |

## Findings

1. **1 `#[non_exhaustive]` in the crate, on `Feature`** (`version.rs:82`, 13 variants, the right place: it grows with every CUDA release). The other
   enum, `StreamMode` (`Legacy`, `PerThread`), is closed with `Default` derived. It mirrors the two CUDA default-stream modes, so closed is
   defensible; a third mode would be a breaking change. Decision: keep closed or add the attribute.
2. **Public fields and closed layout on the numeric types.** `Half(pub u16)`, `BFloat16(pub u16)` (`#[repr(transparent)]`) and `Complex32 { pub re, pub im }`,
   `Complex64 { pub re, pub im }` (`#[repr(C)]`). The layout is the CUDA one (`__half`, `float2`, `double2`), so freezing it costs nothing. The
   public tuple field means any `u16` is constructible directly, bypassing `from_bits`; that is intended, not unsound.
3. **`Half` and `BFloat16` derive `PartialEq, Eq, Hash` on the bits but implement `PartialOrd` via `f32`.** `numeric.rs:19` and `:216` derive; `:188`
   and `:310` implement `partial_cmp` through `to_f32()`. By reading, `Half::NAN == Half::NAN` is true (same bits) while `partial_cmp` of the same
   pair is `None`, and `+0.0` and `-0.0` are unequal under `==` but `Some(Equal)` under `partial_cmp`. The `PartialOrd` contract (`a == b` iff `partial_cmp`
   is `Some(Equal)`) is broken; confirmed by a compiled test (`tests/eq_ord_consistency.rs`, 4 of 4 failed on `main`), fixed in #391 (IEEE equality, `Eq` dropped). `Complex32`/`Complex64` derive `PartialEq` only (no `Eq`). Decision: pick bit
   equality or IEEE equality before freezing, and make `PartialOrd` match.
4. **Two public `unsafe trait`s, each with a `# Safety` section.** `KernelArg` (`kernel_arg.rs:28`) and `ValidAsZeroBits` (`zero_bits.rs:60`); 21
   `unsafe impl` lines in `src/` (grep, macro-generated ones counted once). Neither is sealed, so a downstream `unsafe impl` becomes part of the semver
   surface, and any new required method is a breaking change. `KernelArg::as_kernel_arg_ptr(&self) -> *mut c_void` returns a mutable pointer from a
   shared reference, and the `&T` impl does `*self as *const T as *mut c_void`: sound only because the launch API reads the bytes, which the trait
   contract states (item 3). Decision: freeze the `*mut` return or change it to `*const` (breaking, so before freeze).
5. **`DeviceRepr` is not owned here.** `device_repr.rs:14` re-exports `unpopped_vocab::DeviceRepr` (non-optional dependency, `Cargo.toml:26-29`), and
   the `derive` feature re-exports the derive macro of the same name. Freezing `baracuda-types` freezes the trait's identity to the `unpopped-vocab`
   major, so an `unpopped-vocab` breaking release is a breaking release here. The 4 `unsafe impl DeviceRepr` for the numeric types are the local part.
6. **Feature-gated impls expose third-party types.** Five cargo features (`derive`, `half-crate`, `num-complex-crate`, `bytemuck`, `f8-crate`) add
   impls for `half::f16`/`bf16`, `num_complex::Complex`, `bytemuck::Pod`/`Zeroable` and `float8::F8E4M3`/`F8E5M2`. Those crates' version
   requirements are therefore public API. `external_impls` and `zero_bits` are `pub mod`, so their paths are frozen too, though the types are
   reachable from the crate root only for `ValidAsZeroBits`.
7. **0 `pub unsafe fn`, 0 `doc(hidden)`, 0 sealed traits** (grep over `src/*.rs`). `#![warn(missing_debug_implementations)]` is set (`lib.rs`).
   57 `pub const` hits by grep versus 36 census assoc-const rows; the difference is consts the census puts under other kinds or macro-expanded
   ones, not checked.
8. **The census is current for this crate.** For the 12 struct, enum, trait and fn rows, the line named in the census contains the item's name
   (loop over the TSV, 0 mismatches; positive control: the same loop checked 12 rows and found 12 names).

## Not decided here

- Equality semantics of `Half`/`BFloat16` (finding 3).
- `*mut` vs `*const` in `KernelArg` (finding 4); whether either `unsafe trait` is sealed.
- Whether `StreamMode` becomes `#[non_exhaustive]` (finding 1).
- Not checked: method bodies, doc-test coverage, whether the 57-vs-36 const difference hides an unrowed public item.
