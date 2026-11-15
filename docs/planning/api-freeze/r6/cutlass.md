# R6 review: `baracuda-cutlass`

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `daa4c795`. Items: the 118 rows for `baracuda-cutlass` in the §7.4 census, all `stable`, all
`reach = public`; 117 with `non_exhaustive = 0` and 1 with `1` (the `Error` row). By `kind`: 56 fields, 29 methods, 14 structs, 8 variants, 5 re-exports,
3 mods, 2 enums, 1 type (56 + 29 + 14 + 8 + 5 + 3 + 2 + 1 = 118, counted from `items.tsv` at this ref). Four source files, 6,593 lines in `src/`, 6,110 of
them in `plan.rs`; eleven smoke-test files in `tests/`. Method: grep counts over `git ls-files` of `src/*.rs`, read `lib.rs`, `error.rs`, `types.rs`
(`GroupedScheduleMode`, `GemmSku`), the `# Safety` section of `PreparedGroupedGemm::run`, and the census rows. Method bodies were **not** read, nothing
was built. A verdict on the public shape.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 5 plan structs (`GemmPlan`, `BatchedGemmPlan`, `GroupedGemmPlan`, `PreparedGroupedGemm`, `IntGemmPlan`) + 29 methods | 34 | stable with a note | finding 4 |
| 9 descriptor/args/preference structs, 56 pub fields | 65 | **freeze decision** | finding 1 |
| `GroupedScheduleMode`, `GroupedPlanPreference` | 3 | **freeze decision** | finding 2 |
| `Error`, `Result`, 31 re-exported vocabulary types | 33+ | stable with a note | finding 3 |
| `GemmSku` and `precision_guarantee` | 6 | stable with a note | finding 5 |

## Findings

1. **56 public fields on 9 structs, none `#[non_exhaustive]`.** `GemmDescriptor`, `GemmArgs`, `BatchedGemmDescriptor`, `BatchedGemmArgs`, `GroupedProblem`,
   `IntGemmDescriptor`, `IntGemmArgs`, `GroupedPlanPreference` and `GemmSku`. The crate docs construct them by struct literal (`GemmDescriptor { m, n, k, layout,
   epilogue }`, `GemmArgs { a, b, c, d, bias, alpha, beta }`), so every new field, for example a second bias or a batch stride, breaks every caller. 0
   `#[non_exhaustive]` in `src/` (grep). Decision: `#[non_exhaustive]` plus constructors on the descriptor and args structs, or declare them frozen.
2. **`GroupedScheduleMode` has one variant and the doc says another is planned.** `types.rs:242-247` (`DeviceOnly`, `#[default]`), doc: CUTLASS also offers
   `HostPrecompute` "we'll add it later". The enum is closed and derives `Eq`, so adding the variant is breaking. Decision: `#[non_exhaustive]` now.
3. **`Error` and 22 of the vocabulary types are not defined here.** `error.rs:5` re-exports `Error`, `Result` from `baracuda-kernels-types` (the
   census row `enum Error` at `error.rs:23` is therefore stale: the line is a comment, the enum is `#[non_exhaustive]` in
   `baracuda-kernels-types/src/error.rs:24-25`). `types.rs:22` re-exports 21 names and `:37` aliases `Element` as `CutlassElement`; `lib.rs:60` re-exports 31 names
   from `types`. Freezing this crate freezes those names' identity to the `baracuda-kernels-types` version, the same coupling as `baracuda-types` finding 5.
4. **1 `pub unsafe fn`, with a `# Safety` section.** `PreparedGroupedGemm::run` (`plan.rs:5483`): an asynchronous H2D copy out of a pinned buffer plus raw
   device addresses, so the caller keeps the plan and buffers alive until the stream completes and as long as any captured graph can replay. The contract
   is written down. Everything else is safe: 28 `pub fn` plus that one makes the 29 census methods. The plan types document `Send + Sync` (`plan.rs:3505`,
   `:4709`, `:5086`); not compiled, no `assert_send` test was looked for.
5. **`GemmSku` is public identity.** `Copy, Eq, Hash` with five public fields, and the doc promises that "a stable SKU implies a stable precision guarantee"
   (`types.rs`, `precision_guarantee`). Adding a SKU field breaks literals and changes `Hash`; adding an `ArchSku` or `ElementKind` variant is covered only if
   those re-exported enums are `#[non_exhaustive]` in `baracuda-kernels-types` (not checked here).
6. **No sealed traits, no consts, no `doc(hidden)`.** 0 `pub trait`, 0 `pub const`, 0 `doc(hidden)`, 0 `unsafe impl` in `src/` (grep). `#![deny(missing_docs)]`
   is set (`lib.rs`).
7. **The census line numbers are stale for `plan.rs` and `error.rs`.** The 5 plan structs are census lines 3502, 4690, 5069, 5411, 5576 and sit at 3521, 4713,
   5092, 5434, 5611 (19 to 35 lines later); `error.rs:23` is covered in finding 3. Loop over the struct and enum rows: 16 checked, 6 mismatched, the other 10
   (all in `types.rs`) match, so the loop does find present names. Names resolve; only lines drifted.
8. **Two in-repo consumers.** `baracuda-kernels` and `baracuda-kernels-bench` depend on it (`Cargo.toml`); which methods they call was not read.

## Not decided here

- Whether the descriptor/args structs and `GroupedScheduleMode` become `#[non_exhaustive]` (findings 1, 2).
- Whether `baracuda-kernels-types`' `ArchSku`/`ElementKind` are non-exhaustive (finding 5).
- Whether the plan types are `Send + Sync` (finding 4; needs a compile check).
- Not checked: method bodies, the smoke tests, doc-test coverage.
