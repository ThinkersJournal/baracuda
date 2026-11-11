# R6 review: `baracuda-kernels` crate root (`src/lib.rs`)

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `17e049a8` for the source; the census (`items.tsv`) is from alpha.97 and `lib.rs` has not changed since alpha.80 (`git log` on the file), so it applies. Items: the `stable` rows of
`crates/baracuda-kernels/src/lib.rs` in the §7.4 census: 25 `mod`s and 46 `reexport`s, 71 in all. (The HANDOFF said 68,
a count taken at `316bb006`; `items.tsv` has 71, which is the number reviewed.) Method: read all of
`lib.rs`, listed every `#[cfg]` and `pub mod`, and read the `Cargo.toml` feature table. The root has no types of its
own, so this is a verdict on what it exposes and under what conditions, not on any plan's shape (the per-family
reviews cover those).

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 22 `pub mod`s of op families | 22 | stable | each module's contents are covered by its own R6 review (`scan`, `softmax`, `embedding`, `reduce`, `causal_conv1d` done; the rest queued) |
| re-exports of family modules (`gemm`, `elementwise`, ..., `sort`, the FlashInfer and sampler blocks) | 41 | stable | mirror module items at the root; each name must exist in its module, so a re-export cannot drift without failing to compile |
| re-exports of `baracuda_kernels_types` and `baracuda_cutlass` | 2 | stable, **with finding 1** | the root's `Error`, `Result` and the whole vocabulary live in other crates |
| `optim`, `transformer_engine`, `megatron` modules | 3 | stable, **with finding 2** | feature-gated facades over three other crates |
| their 3 `pub use` re-exports | 3 | stable, **with finding 2** | include the `Error as X` / `Result as X` renames |

22 + 41 + 2 + 3 + 3 = 71. No item is proposed for `#[doc(hidden)]` or removal.

## Findings

1. **`baracuda_kernels::Error` and `Result` are `baracuda_cutlass`'s, and the whole vocabulary (about 67 names:
   `ElementKind`, `ArchSku`, `KernelSku`, `TensorRef`, every `*Kind`, `Bool`, `U8`, `S4`, ...) is `baracuda_kernels_types`'s.**
   The root re-exports them, so a change to either crate's public items is a change to this crate's surface. Freezing
   `baracuda-kernels` therefore does not freeze it alone: `baracuda-cutlass` and `baracuda-kernels-types` must be
   reviewed in the same R6 pass (they are in the non-kernels list). Same coupling as `SoftmaxKind` in `softmax`.
2. **Three modules rename on re-export**: `optim::{OptimError, OptimResult}`, `transformer_engine::{TransformerEngineError,
   TransformerEngineResult}`, `megatron::{MegatronError, MegatronResult}` (`Error as X`, `Result as X`). The census tool
   records them as one mangled name each (for example `ErrorasOptimError`), so a name-keyed diff against these rows
   would show a false add/remove if the tool's parser changes. Stable as written; flagged for the tooling.
3. **The freeze promise is conditional on features for 16 of the 71 rows**: 6 `mod`s (`conv`, `pool`, `causal_conv1d`,
   `optim`, `transformer_engine`, `megatron`) and 10 re-exports (`CtcLossCudnn*`, `Conv*`, `Pool*` under `cudnn`; the
   three Mamba blocks under `mamba`; `FlashSdpaSm89*` under `sm89`; and the three facade re-exports). `r6/causal_conv1d.md` already said this for
   `mamba`; the census does not apply gating (§1). Counted by reading `#[cfg]` lines in `lib.rs` (`linalg` is not gated; the `sm89` cfg is on the
   `FlashSdpaSm89*` re-export only), not by building each configuration.
4. **`#![deny(missing_docs)]` is on (`lib.rs:64`)**, so every item reachable from the root is documented at compile
   time; that is the one root-level guarantee the other 70 rows rely on.

## Not checked

The 16-row count of gated rows was matched by hand against the `#[cfg]` lines, not computed per feature (the census `line` column does not match `lib.rs` line numbers, so it was not used to place rows); a `--no-default-features`
and an all-features build were not run; `window_axis` and `view_check` are private modules and were not reviewed.
