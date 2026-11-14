# R6 review: `baracuda-kernels` `shape_layout` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10, 2026-11-13). Ref: `origin/main` `c6a598e5`. Items: the 324 `stable` rows under
`crates/baracuda-kernels/src/shape_layout/` in the §7.4 census: 19 `mod`s, 19 re-exports, 57 structs, 105 fields, 124 methods
(19 + 19 + 57 + 105 + 124 = 324, counted from `items.tsv` by `kind` at this ref; all `public`, all `stable`). Twenty files, 6,847 lines:
`concat`, `flip`, `pad`, `permute`, `repeat`, `roll`, `tril`, `triu` (each with a `_backward`), `contiguize`, `fill`, `write_slice`, plus
`mod.rs`; 19 plans (8 forward/backward pairs and 3 single plans, 8 x 2 + 3 = 19).
Method: grepped every file for `pub fn can_implement`, `pub fn select`, `pub fn run`, `self.can_implement`, `pub struct .*Plan`, the
`#[non_exhaustive]` attribute line, `aux_element`, `Cell<`, `doc(hidden)`, `unreachable!`, `pub const`, `pub trait`, `impl Drop`,
`fn workspace_size`, `panic!`/`unwrap`/`expect`, `const N`, `Nibble`; read `concat.rs` `run` head (`:228-236`), the `unreachable!` site and
`FastPath` in `write_slice.rs` (`:330-352`), and the nibble notes in `contiguize.rs`. The other bodies were **not** read. A verdict on
the public shape, not on the CUDA kernels or numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 19 `mod`s, 19 re-exports | 38 | stable | one module per op, re-exported from `shape_layout/mod.rs` |
| 19 plans | 19 | stable | `select` + `can_implement` + `run` + `workspace_size` on all (finding 1) |
| 38 descriptors and args, 105 fields | about 143 | **freeze decision** | no `#[non_exhaustive]`; rank is a const generic (findings 2, 3) |
| 124 methods | 124 | stable with a note | findings 4, 5 |

## Findings

1. **Every plan has `select`, `can_implement`, `run`: 19 of 19, and `run` calls it.** Grep: 19 files each for `pub struct .*Plan`, `pub fn
   select`, `pub fn can_implement`, `pub fn run`, `fn workspace_size`; `self.can_implement` appears in all 19 files. `concat.rs:234` is the
   first statement of `run` (read); in the other 18 the call was found but its position in `run` was not checked. Positive control: the
   same `can_implement` grep finds 0 in `pool`, `linalg`, `loss` and `conv`. With `quantize` and `segment`, the third family with the
   complete Plan/Descriptor/Args shape.
2. **No type is `#[non_exhaustive]`.** 0 hits for the attribute line in 20 files. 38 descriptor and args structs are `pub` with `pub`
   fields (`ConcatDescriptor`, `ConcatArgs` at `concat.rs:29, 52`). A new field is a breaking change. Same all-or-nothing case as `segment` 2.
3. **Rank is a const generic: 54 of 57 structs carry `const N: usize`.** `ConcatPlan<T, const N: usize>`, `ConcatDescriptor<const N: usize>`,
   `WriteSliceDescriptor<const N: usize>` (`concat.rs:80, 29`, `write_slice.rs:64`, by grep). Freezing the structs freezes the rank-as-type
   shape; a caller picks `N` at compile time. The three structs without it are `FillDescriptor`, `FillArgs` and `FillPlan` (`fill.rs:31, 43, 68`, by
   grep on the signature lines; the file body was not read). The supported rank range is not a public constant: 0 hits for `MAX_RANK` or `MAX_DIMS` in the directory (control: `pool`
   has none either, `quantize` has `MAX_RANK`); what `select` does for a large `N` was not read.
4. **`aux_element` is `None` in all 19 plans.** One hit per file, each `None` (`concat.rs:141` to `write_slice.rs:229`). Unlike `segment`,
   `quantize` and `pool` there are no index tensors here, so a uniform `None` is expected; nothing to ask.
5. **Sub-byte element types are guarded by documented constraints, not by the type.** `contiguize.rs:92-95` documents that `S4` / `U4`
   need an innermost-stride constraint (the nibble constraint, 3 mentions) and `write_slice.rs` carries 5 `Nibble` mentions, including that
   the innermost-axis alignment is "guaranteed even by select-time guard" (`:350`, comment). Whether `can_implement` or only `select`
   enforces it, and what error a violation returns, was not read.
6. **One bare `unreachable!()`, and it is internal.** `write_slice.rs:348`, `FastPath::Generic => unreachable!()`, inside a `match
   self.fast_path` arm whose pattern is `WholeDest | ContiguousChunk { .. }` (read, `:338-348`). The inner `match` repeats the outer test, so
   `Generic` cannot reach it today; `FastPath` is a private enum. A refactoring hazard, not a public-API one. No other panicking site by grep.
7. **No `Cell`, `Drop`, `doc(hidden)`, public const or public trait.** 0 hits each (positive control: `pub fn select` finds 19 files,
   `pub fn` finds 124 hits, matching the 124 methods). None of the cuDNN-state findings apply.
8. **Backward plans are separate types, as in `segment`.** 8 `*_backward` modules (`concat`, `flip`, `pad`, `permute`, `repeat`, `roll`,
   `tril`, `triu`) each with their own plan; `contiguize`, `fill` and `write_slice` have none.

## Not decided here

- Whether the 38 descriptor and args structs get `#[non_exhaustive]` before the freeze (finding 2).
- Whether rank stays a const generic or the rank limit becomes a named constant (finding 3).
- Which of `select` / `can_implement` carries the sub-byte guard (finding 5).
