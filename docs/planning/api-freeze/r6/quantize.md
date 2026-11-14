# R6 review: `baracuda-kernels` `quantize` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10, 2026-11-13). Ref: `origin/main` `b2f45cd9`. Items: the 628 rows under
`crates/baracuda-kernels/src/quantize/` in the §7.4 census: 625 `stable` and 3 `sealed-trait-pattern`. By `kind`: 28 `mod`s, 30
re-exports, 94 structs, 281 fields, 173 methods, 3 enums, 8 variants, 6 traits, 3 free `fn`s, 2 consts; 28 + 30 + 94 + 281 + 173 + 3 + 8 +
6 + 3 + 2 = 628 (counted from `items.tsv` at this ref). Twenty-eight files (23 top-level modules plus `mod.rs`, and the 4
`gguf/` files plus `gguf/mod.rs`); 27 plans.
Method: grepped every file, `gguf/` included, for `pub fn can_implement`, `pub fn select`, `pub fn run`, `pub struct .*Plan`,
`#[non_exhaustive]` (the attribute line, not the word), `aux_element`, `Cell<`, `doc(hidden)`, `unreachable!`, `impl Drop`, `pub const`;
read the three public free functions and two consts, the `sealed` module in `gguf/mmvq.rs`, the `bnb_nf4` stub in `nf4.rs`. The
plan bodies were **not** read. A verdict on the public shape, not on numerics or the CUDA kernels.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 28 `mod`s, 30 re-exports | 58 | stable | one module per op, `gguf/` nested |
| 27 plans | 27 | stable | `select` + `can_implement` + `run` on every one (finding 1) |
| 94 structs, 281 fields, 3 enums, 8 variants | about 386 | stable with a note | findings 2, 3 |
| 6 traits (3 sealed) | 6 | stable | closed dtype sets (finding 4) |
| 3 free fns, 2 consts | 5 | **freeze decision** | host helpers and a CUDA-mirroring const (finding 5) |
| 173 methods | 173 | stable with a note | findings 3, 6 |

## Findings

1. **Every plan has `select`, `can_implement` and `run`: 27 of 27.** `pub struct .*Plan` 27, `pub fn select` 27, `pub fn can_implement`
   27, `pub fn run` 27 (grep over the 28 files; the three plans in `nf4.rs` and one each in `gguf/{dequantize,mmvq,mmvq_batched}.rs`
   included). The same `can_implement` grep finds 0 in `pool`, `linalg`, `loss` and `conv`, so the query is not empty-handed by
   construction. This is the shape the freeze would want elsewhere. Not checked: that each `can_implement` is the check `run` uses,
   and not a second copy (`per_tensor.rs:144` and `nf4.rs:211,331,470` were located, not read).
2. **`#[non_exhaustive]` is on 5 of about 97 types.** Attribute lines (not `finish_non_exhaustive`, which is a `Debug` helper in 6
   files): `DynamicRangeMode`, `DynamicRangeScope` (`dynamic_range.rs:54, 68`), `SmoothQuantLinearDescriptor` (`smoothquant.rs:82`),
   `GgufMmvqDescriptor` (`gguf/mmvq.rs:57`), `GgufMmvqBatchedFormat` (`gguf/mmvq_batched.rs:87`). The other per-tensor,
   per-channel, per-group, per-token, fake-quantize and NF4 descriptors carry none. Same inconsistency as `norm`, `loss`, `pool`.
3. **`aux_element` records the integer output kind, with one `None`.** 11 `KernelSku` constructions set it: `Some(TOut::KIND)` in
   `dynamic_range`, `per_group`, `per_tensor`, `per_token`; `Some(TWQ::KIND)` in `quantized_linear`, `smoothquant`; `Some(U8)` in `nf4`
   and the three GGUF files; `None` in `fake_quantize.rs:253`. `per_token_backward.rs:77,105` and `per_tensor_backward.rs:30`
   say the backward kernel is `TOut`-agnostic but keeps the forward's `aux_element` for consistency (comments; the code was not
   read). The per-channel and dequantize files set none directly; they share `build_sku` (`per_channel.rs` imports it), not read.
4. **The dtype sets are closed by sealed traits, unlike `linalg`.** `GgufMmvqActivation`, `GgufMmvqBatchedActivation` and
   `Nf4Activation` each extend a `Sealed` trait (3 `sealed-trait-pattern` rows). Read for `gguf/mmvq.rs:41-53`: private `mod sealed`
   with impls for `f32`, `f16`, `bf16`; the other two were not read. So an unsupported `T` fails at compile time there, where
   `CholeskyPlan<f16>` fails at run time. Other plans bound `TIn`/`TOut` by `Element`; whether `select` rejects the rest was not read.
5. **Five free items are public host helpers, not plans.** `default_q_range` (`mod.rs:223`, `I8`->`(-128,127)`, `U8`->`(0,255)`,
   else `None`), `nf4_quantize_value`, `nf4_pack_weight` (`nf4.rs:844, 867`; documented as test and weight-prep host references),
   `NF4_CODEBOOK` (`pub const [f32; 16]`, documented as bit-identical to bitsandbytes `kernels.cu`) and `MAX_RANK = 4`
   (`per_channel.rs:30`, documented as mirroring `baracuda_quantize.cuh`). Freezing them freezes a codebook table and a CUDA-header
   value. Same kind of decision as `linalg` 6 (`WY_NB`).
6. **The `bnb_nf4` feature gates the launch, not the types.** `nf4.rs:622` is `#[cfg(feature = "bnb_nf4")]` and `:653` the
   `not(...)` stub, which returns the literal `3` (read, `nf4.rs:808-811`); the file notes "the Rust types compile but launching"
   is unavailable (`:670`). So `Nf4*Plan` is in the public surface with the feature off. What error `3` becomes was not traced.
   Whether `select` or `can_implement` reports the feature as missing before `run` does is the open question.
7. **No `Cell`, no `Drop`, no `doc(hidden)`, no `unreachable!`.** 0 hits each (positive control: `Cell<` finds 43 hits in 8 files in
   `pool`, and `fn select` finds 22 files here). So none of the interior-mutability findings from `conv`, `linalg` or `pool` apply.
   Workspace-size shape was not probed.

## Not decided here

- Whether the 3 free `fn`s and 2 consts join the frozen surface or move behind a module (finding 5).
- Whether `bnb_nf4` off should fail at `select` (finding 6).
- Whether the five `#[non_exhaustive]` types set the rule for the other descriptors (finding 2).
