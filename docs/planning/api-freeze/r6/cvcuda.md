# R6 review: `baracuda-cvcuda`

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `f7662ec1`. Items: the 118 rows for `baracuda-cvcuda` in the §7.4 census, all `stable`, all
`reach = public`, all `non_exhaustive = 0`. By `kind`: 85 methods (62 `submit`, 21 `new`, `from_raw`, `as_raw`), 21 structs, 6 variants, 2 types, 2 re-exports,
1 enum, 1 fn (85 + 21 + 6 + 2 + 2 + 1 + 1 = 118, counted from `items.tsv` at this ref). One source file, `src/lib.rs`, 2,655 lines (`git ls-files`). Method: grep
counts over `lib.rs`, read the head, `Tensor`, `Resize`, the `simple_op!` macro and one invocation, the census rows. Method bodies were **not** read, nothing was
built. A verdict on the public shape.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| ~42 `simple_op!` operator structs and their `new` | not in census | **freeze decision** | finding 1 |
| 21 hand-written operator structs, 62 `submit`, 21 `new` | 104 | **freeze decision** | findings 2, 3 |
| `Tensor` (`new`, `from_raw`, `as_raw`), `Error`, `Result`, `probe`, `Interpolation` | 10 | stable with a note | findings 3, 4, 5 |
| 2 re-export lines | 2 | stable with a note | finding 6 |

## Findings

1. **About 42 public operator structs are missing from the census.** `simple_op!` (`lib.rs:~400`) is a `macro_rules!` that expands to
   `pub struct $Name { op: OpHandle }` plus `pub fn new() -> Result<Self>`; it is invoked 42 times (`grep -c '^simple_op! {'`, names such as `PillowResize`, `WarpAffine`,
   `WarpPerspective`). None of the 42 names is a census row (loop over the invocation names, 0 hits; positive control: the hand-written `Resize` is 1 hit). The census extractor
   does not expand macros, but it does see the `impl $Name { pub unsafe fn submit ... }` blocks written outside the macro, so the 62 `submit` rows belong to
   structs the census does not list, and the 42 macro-generated `new` fns are absent while the 21 hand-written ones are present. The real public struct count is
   about 63, not 21. Decision: expand macros in the extractor (or add the macro invocations as rows) before the freeze.
2. **62 `pub unsafe fn submit` with a raw `stream: *mut c_void`.** Each operator's `submit` takes the stream as a raw pointer (`lib.rs`, `Resize::submit`) with a one-line
   `# Safety` (`stream` must be a live `cudaStream_t`), 63 `# Safety` headings for 63 `pub unsafe fn` (grep; the 63rd is `Tensor::from_raw`). `baracuda-driver` is only a dev-dependency
   (`Cargo.toml:29`), so a typed `Stream` is not in the public signatures; changing `submit`'s first parameter later would be breaking in 62 places. Decision: freeze the raw pointer or
   introduce a stream trait before the freeze.
3. **Untyped integers where the sys crate has types.** `Tensor::new(shape: &[i64], dtype: i32, layout: i32)` takes opaque `i32` codes (the doc points at
   `NVCVTensorLayout.h`), and `Threshold::new(threshold_type: u32, max_batch_size: i32)` takes `u32` while `NVCVThresholdType` is re-exported at `lib.rs:378`. Freezing the integer
   signatures freezes the lack of type safety; a typed alternative can be added additively but the old signature would have to stay.
4. **No `Send`/`Sync` statements.** 0 `unsafe impl` and 0 mentions of `Send`/`Sync` in `lib.rs` (grep). `Tensor` and `OpHandle` hold raw pointers, so by reading every
   operator is neither `Send` nor `Sync`. Not compiled. Whether NVCV handles may move between threads is a vendor question the crate does not answer; fixing it later
   (adding `unsafe impl Send`) is additive, removing it is not.
5. **`Error` is `baracuda_core::Error<NVCVStatus>`** (`lib.rs:25-26`), a type alias into another crate, so `baracuda-core`'s error shape and its `NVCVStatus` payload are part of this
   API. `Interpolation` (6 variants, `Eq`, no `Hash`) is closed and has no `#[non_exhaustive]`; the NVCV interpolation list has grown before. Decision: `#[non_exhaustive]` on
   `Interpolation`.
6. **The sys crate is part of the public API.** `lib.rs:23` re-exports `NVCVColorConversionCode`; `lib.rs:378`'s braced list re-exports `NVCVAdaptiveThresholdType`, `NVCVBorderType`, `NVCVFloat2`,
   `NVCVMorphologyType`, `NVCVSize2D`, `NVCVThresholdType`. `Tensor::as_raw`/`from_raw` expose `NVCVTensorHandle`. Freezing them freezes `baracuda-cvcuda-sys`' type shapes.
7. **The census line numbers are stale for the hand-written operators.** Of the 23 struct, enum and fn rows, 15 do not name their item on the census line (`Threshold` at
   census 1186 sits at 1175, 11 lines earlier; the other 14 rows, `AdaptiveThreshold` through `PairwiseMatcher`, also mismatch; their drift was not measured); the other 8 rows match. Positive control: the
   loop confirms the 8 matching rows. Names resolve; lines drifted.

## Not decided here

- How the extractor handles `simple_op!` (finding 1).
- Raw stream pointer vs typed stream (finding 2).
- `Interpolation` becoming `#[non_exhaustive]` (finding 5).
- Not checked: method bodies, the `dtype`/`layout` code lists, `Send`/`Sync` by compiling, doc-test coverage.
