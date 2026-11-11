# R6 review: `baracuda-kernels` `random` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10). Ref: `origin/main` `5294463c` for the source; the census
(`items.tsv`) is from alpha.97. Items: the 146 `stable` rows under `crates/baracuda-kernels/src/random/`. Count, from
`items.tsv` rows per file: `dropout.rs` 31, `perrow_spec_sampling.rs` 47, `plan.rs` 19, `token_penalty.rs` 16,
`topk_topp_sampling.rs` 23, `mod.rs` 10; 31+47+19+16+23+10 = 146. By kind: 10 `mod`/`reexport`, 22 structs (7 plans,
7 descriptors, 8 args), 2 enums + 8 variants, 61 fields, 43 methods. Method: read the descriptor, args and plan
signatures in all six files, grepped `can_implement`, `Error::`, `non_exhaustive` and `cfg`, read `mod.rs` in full;
compared with `fft`, `reduce`, `embedding`. A verdict on the public shape, not on numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 5 `mod`s, 5 re-exports | 10 | stable | one module per op group, each re-exported |
| 7 descriptors, 8 args (`RandomBoolArgs` is the extra) | 15 | stable | `Copy` descriptors; `RandomArgs` and `RandomBoolArgs` split by output dtype instead of one generic |
| 61 fields (descriptors and args together) | 61 | stable | all documented (`deny(missing_docs)`); the per-struct split was not tallied |
| `SamplerKind`, `PerRowSampler` + 4 variants each | 10 | stable | both `#[non_exhaustive]` (`topk_topp_sampling.rs:60`, `perrow_spec_sampling.rs:27`), so adding a variant is not a break |
| 7 plans | 7 | stable | private fields |
| 43 methods | 43 | stable, **with findings 1 and 2** | see below |

22 structs = 7 + 7 + 8; 10 + 15 + 61 + 10 + 7 + 43 = 146. No item is proposed for `#[doc(hidden)]` or removal.

## Findings

1. **Three of seven plans have no `can_implement`: `DropoutPlan`, `DropoutBackwardPlan`, `RandomPlan`.** The four
   sampling plans (`TopKTopPSamplingPlan`, `PerRowSamplingPlan`, `SpeculativeSamplingPlan`, `TokenPenaltyPlan`) have it
   and call it from `run`. The other three validate inside `run` (`Error::` counts: 29 in `dropout.rs`, 28 in
   `plan.rs`), so a bad call is rejected but cannot be pre-checked. Same shape as `fft` (7 of 8 plans without).
   Additive fix, a separate PR.
2. **`run` is per-dtype, not generic, in the same three plans.** `DropoutPlan<f32, N>` and `<f64, N>` each define
   `run` (`dropout.rs:208`, `:285`); `RandomPlan` has three (`f32`, `f64`, `Bool`, `plan.rs:287`, `:352`, `:408`).
   So `DropoutPlan` shows 6 methods as 4 generic + 2 per-dtype, `RandomPlan` 7 as 4 + 3. A caller generic over `T`
   cannot call `run`. Same finding as `fft`.
3. **`mod.rs` doc is stale.** It lists only `RandomPlan`, `DropoutPlan` and `DropoutBackwardPlan`; the module also
   exports `TopKTopPSamplingPlan`, `PerRowSamplingPlan`, `SpeculativeSamplingPlan` and `TokenPenaltyPlan` (and says "Each
   `*Plan` creates one [cuRAND generator]", which is false for the four sampling plans, which do not use cuRAND). It
   also names a local path (`~/.claude/plans/warm-prancing-comet.md`) that does not exist for a reader of the crate.
   Doc fix only; not fixed here.
4. **Three sampling plans' `run` depends on the `flashinfer` feature**: `TopKTopPSamplingPlan`, `PerRowSamplingPlan`
   and `SpeculativeSamplingPlan` compile without it (the `#[cfg(feature = "flashinfer")]` lines are on imports and the
   launch body), and `TopKTopPSamplingPlan::run` returns `Error::Unsupported` when it is off
   (`topk_topp_sampling.rs:256-260`). `TokenPenaltyPlan` is a bespoke kernel and always runs (`token_penalty.rs:6-7`).
   The `PerRowSamplingPlan` and `SpeculativeSamplingPlan` off-branch was not read (their module doc says they require
   the feature). Same conditional promise as `causal_conv1d` under `mamba`.
5. **Thread-safety is understated, not wrong.** `mod.rs` says the plan is `!Sync` through `Cell<curandGenerator_t>`;
   `curandGenerator_t` is `*mut c_void` (`baracuda-curand-sys/src/lib.rs:21`), so the plan is also `!Send`, which the
   struct doc (`plan.rs:76`) says. Unlike `fft`, where `cufftHandle` is `c_int`.

## Not checked

Numerics and cuRAND behaviour; what the `PerRowSamplingPlan` and `SpeculativeSamplingPlan` `run` returns without `flashinfer` (finding 4); the validation bodies in
`dropout.rs` and `plan.rs` (counted by `grep`, not read); that the plans are `!Send` (derived from the pointer type by
reading, not from a compile check).
