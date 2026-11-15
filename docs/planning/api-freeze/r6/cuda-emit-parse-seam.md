# R6 review: `baracuda-cuda-emit`, `baracuda-cuda-parse`, `baracuda-cuda-seam`

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10, 2026-11-14). Ref: `origin/main` `c6a598e5` (the three crates at 0.45.0, as published
2026-11-14). Items: the 25 census rows for the three crates in the §7.4 census (`items.tsv`: emit 21, parse 2, seam 2; all 25 `public` and `stable`,
one of them with `non_exhaustive = 1`). **The census is stale for these crates** (finding 1), so this review reads the
source, not only the rows. Method: read `lib.rs` of emit, parse and seam, `nvrtc.rs` (the enum and `new`), the `pub` lines of
`cfamily_shadow.rs` and `extern_asm.rs`, and `BaracudaSynthesizer` (`seam lib.rs:224-246`); grepped `non_exhaustive`, `cfamily_shadow`
and `use tree_sitter`. The 7,000-line `cuda.rs` and the seam `synthesize` body were **not** read. A verdict on the public shape, not on the
emitted CUDA.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| `Cuda` (unit struct) + `impl Backend` | 1 + trait | stable at the Unpopped line | finding 2 |
| 3 `emit_*_helper` fns | 3 | stable | decided "keep" at alpha.163 (ROADMAP) |
| `cfamily_shadow` (7 fns) | 7 | **freeze decision** | finding 3 |
| `nvrtc` (`NvrtcCompiler`, `NvrtcTargetError`) | 6 | stable | feature-gated; the enum is `#[non_exhaustive]` |
| `extern_asm` (feature `extern_asm_cuda`) | 3 | **not in the census** | finding 1; `ExternError` is not `#[non_exhaustive]` (finding 4) |
| `parse_cuda`, `CUDA` | 2 | stable at the Unpopped line | finding 2 |
| `BaracudaSynthesizer`, `new` | 2 + trait | stable at the FireLine line | finding 2 |

## Findings

1. **The census predates `extern_asm`.** `extern_asm.rs` arrived in `cefd50ae` (2026-11-05) and `items.tsv` was last changed in `454de1d3`
   (2026-10-14). `grep extern_asm items.tsv` finds 0 rows, while `lib.rs:29-31` makes `pub mod extern_asm` and re-exports
   `ExternError` and `extern_asm_cuda` behind `extern_asm_cuda`; `CUDA_PTX_TARGET` is a fourth public item. Positive control: the
   same file has rows for `nvrtc`, which is also feature-gated, so gating is not what hides them. Regenerate the census before the freeze.
   (Same kind of gap as `pool` 1.)
2. **Every public signature is built from types of a 0.x crate we do not own.** `Cuda` implements `unpopped::backend::Backend`;
   `NvrtcCompiler` implements `unpopped::Compiler` and takes an `unpopped_vocab::TargetId`; `parse_cuda` returns
   `Option<tree_sitter::Tree>` (`parse lib.rs:33,45`) and `CUDA` is an `unpopped` `Frontend`; `BaracudaSynthesizer` implements
   `fireline_synth::Synthesizer` (`~0.2`). So a minor of any of them is a breaking change here, which is why the three crates' MAJOR.MINOR
   follow Unpopped (the CHANGELOG versioning exception, 0.45.0). "Frozen" for these crates can only mean frozen against one Unpopped
   line; it cannot mean a 1.0 of their own while the line moves.
3. **`cfamily_shadow` is a public module that only `cuda.rs` uses.** 7 `pub fn`s (`scalar_ctype`, `promote_load_f32`, `demote_store_f32`,
   `cast_scalar`, `param_ctype`, `out_ctype_of`, `store_expr_of`); three take `&unpopped::plan::KernelPlan`. Its own doc calls it
   "Baracuda's own closed-set copy" of spellers moved out of Unpopped. `git grep cfamily_shadow` outside its file finds 2 hits, both
   inside `baracuda-cuda-emit` (`cuda.rs:9` and the `lib.rs:25` declaration); none in `crates/` or `tools/` elsewhere (control: the one
   `cuda.rs` use is found). Whether it is public API or an internal that was left `pub` is the decision; `pub(crate)` would remove 7 items from the freeze.
4. **`#[non_exhaustive]` is on one type.** `NvrtcTargetError` (`nvrtc.rs:21`). `ExternError` (`extern_asm.rs:14`, a public enum with
   `Clone, Debug, PartialEq, Eq`) is not, and `Cuda` and `BaracudaSynthesizer` are a unit struct and a struct with private fields.
5. **`BaracudaSynthesizer` hides its state.** Private `max_compile_ms` and a `Mutex<HashMap<..>>` registry (`:230-235`); `new(u32)` is the
   only constructor; the doc promises it never panics and declines as a typed `SeamResponse::Declined`. A reviewer's note: the
   `Mutex` makes it `Send + Sync` by construction (not compiled here).
6. **`Cuda::lower_kernel` is a growing surface that is not a census item.** It is a trait method, so no row; #378 (stage 1) gives it a
   body and stages 2 and 3 widen what it accepts. Declines are typed `UnsupportedKernel` by name, so a widening is not a signature change,
   but the set of accepted kernels is observable behaviour.
7. **Feature-gated public items.** `nvrtc` and `extern_asm_cuda` are default-off; a consumer without the feature sees neither. The census
   does not apply gating (README), so the freeze counts both.

## Not decided here

- Whether `cfamily_shadow` stays public (finding 3).
- Whether the emitter crates are "frozen" per Unpopped line, with a policy for the next Unpopped minor (finding 2).
- Whether `ExternError` gets `#[non_exhaustive]` before the freeze (finding 4).
