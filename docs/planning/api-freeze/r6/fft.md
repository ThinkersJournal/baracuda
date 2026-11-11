# R6 review: `baracuda-kernels` `fft` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `1286d682` for the source; the census
(`items.tsv`) is from alpha.97. Items: the 139 `stable` rows under `crates/baracuda-kernels/src/fft/` (counts below). Method: read the descriptor, args and
plan signatures in all seven files, grepped `can_implement`, `Error::` and `run` per file, read `mod.rs` in full, read
`check_args` in `fft.rs`; compared with `scan`, `softmax`, `embedding`, `reduce`. A verdict on the public shape, not
on numerics.

Count check, from `items.tsv` rows per file: `fft.rs` 15, `fftn.rs` 17, `fftshift.rs` 14, `fftshift_nd.rs` 19,
`rfft.rs` 28, `rfftn.rs` 34, `mod.rs` 12; 15+17+14+19+28+34+12 = 139. By kind: 12 `mod`/`reexport`, 24 structs
(8 plans, 8 descriptors, 8 args), 2 consts, 49 fields (6+7+6+8+10+12), 52 methods (6+7+5+6+12+16).

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 6 `mod`s, 6 re-exports | 12 | stable | `fft::fft` is kept deliberately (`mod.rs` comment: renaming is API-breaking) |
| 8 descriptors | 8 | stable | `Copy` structs; `batch`, `n` are `i32`, ND ones carry `dims: [i32; 4]` and `rank: u8` (finding 4) |
| 8 `*Args` | 8 | stable, **with finding 3** | 1-D plans take `TensorRef`/`TensorMut<_, 2>`; the ND C2C/R2C plans take flat `DeviceSlice` |
| 49 fields (descriptors and args together) | 49 | stable | all documented (`deny(missing_docs)`); the per-struct split was not tallied |
| 8 plans | 8 | stable | private fields; each holds a lazily created `cufftHandle` in a `Cell`, destroyed in `Drop` |
| `FFTSHIFT_ND_MAX_RANK` = 8, `FFTSHIFT_ND_MAX_SHIFT_AXES` = 4 | 2 | stable | documented capacity constants; raising them is additive, lowering breaks |
| 52 methods | 52 | stable, **with findings 1 and 2** | see below |

No item is proposed for `#[doc(hidden)]` or removal.

## Findings

1. **Only `FftShiftNdPlan` has `can_implement`; the other seven plans do not.** `grep` finds `can_implement` once in
   the family (`fftshift_nd.rs:248`, and its `run` calls it at line 297). The other seven validate inside `run`
   (`Error::` counts per file: 17, 11, 8, 13, 32, 26 for `fft`, `fftn`, `fftshift`, `fftshift_nd`, `rfft`, `rfftn`),
   through a private `check_args` (`fft.rs:229`), so a bad call is rejected, but a caller cannot pre-check. `scan`,
   `softmax`, `embedding` and `reduce` plans all carry the six-method shape. Adding the method is additive; it is a fix PR.
2. **`run` is not generic.** `FftPlan<T>` has `select`, `sku`, `precision_guarantee`, `workspace_size` on
   `impl<T: Element>`, but `run` is defined separately on `impl FftPlan<Complex64>` and `impl FftPlan<Complex128>`
   (`fft.rs:266`, `:316`), and likewise for the other ND/real plans. So 6 methods on `FftPlan` are 4 generic + 2
   type-specific. A caller generic over `T` cannot call `run`. Stable as written; recorded because the freeze fixes it.
3. **Args style is inconsistent inside the family.** `FftArgs`, `RfftArgs`, `IrfftArgs`, `FftShiftArgs` and
   `FftShiftNdArgs` carry shaped `TensorRef`/`TensorMut` (the last names them `input`/`output`, the rest `x`/`y`), while
   `FftNdArgs`, `RfftNdArgs` and `IrfftNdArgs` carry flat `DeviceSlice`/`DeviceSliceMut` with the shape in the
   descriptor (`fftn.rs:99-103`, `rfftn.rs:100-102`). Changing either is breaking; if a unification is wanted it is a
   pre-freeze decision. Not proposed here.
4. **ND rank is documented as `1..=3` but the arrays are 4 long.** `mod.rs` says "rank `1..=3`", while `dims: [i32; MAX_RANK]`
   with a private `MAX_RANK = 4` (`fftn.rs:52`, `rfftn.rs:43`); the public type `[i32; 4]` is therefore part of the surface.
   Whether rank 4 is accepted by `select` was not read.
5. **Thread-safety doc over-claims.** `mod.rs:56` and six struct docs say the plan is `!Sync` / `!Send` "via the
   `Cell<cufftHandle>`". `cufftHandle` is `c_int` (`baracuda-cufft-sys/src/lib.rs:23`), and `Cell<c_int>` is `Send` but
   not `Sync`, and nothing in the family has an `unsafe impl`. So by reading, the plans are `Send` and `!Sync`. Not
   compiled to confirm (a `static_assertions` check would settle it); doc fix only, not fixed here.

## Not checked

Numerics and cuFFT behaviour; the `Send` claim in finding 5 (derived from reading, not from a compile check); whether
`select` accepts rank 4 (finding 4); the validation bodies in the seven `check_args`-style paths (counted by `grep`, not
read); the `mod.rs` normalisation statement (`1/N` after inverse).
