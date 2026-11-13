# R6 review: `baracuda-kernels` `attention` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10, 2026-11-13). Ref: `origin/main` `4081d9ba`. Items: the 716 `stable`
rows under `crates/baracuda-kernels/src/attention/` in the §7.4 census (24 `mod`s, 24 re-exports, 82 structs, 386 fields,
187 methods, 6 consts, 2 enums, 5 variants; counted from `items.tsv` by `kind` at this ref; all 716 are `public` and
`stable`, and 24 + 24 + 82 + 386 + 187 + 6 + 2 + 5 = 716). 25 files, about 14,000 lines.
Method: per file, grepped `pub struct .*Plan`, `pub fn can_implement`, `pub fn run`, `self.can_implement`,
`unwrap/expect/panic`, `non_exhaustive`, `aux_element`; read the two scan families' backward plans, the hyper-connection
`run_backward`, and the flash backward `lse_f32` use in full. This is a probe-based review, **not** a line read of all 25
files. A verdict on the public shape, not on kernel numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 24 `mod`s, 24 re-exports | 48 | stable | one module per plan family, re-exported from `attention/mod.rs` |
| 6 consts, 2 enums, 5 variants | 13 | stable with a note | see findings 3 and 4 |
| descriptors, args and their 386 fields | about 400 | stable with a note | see findings 2 and 5 |
| 27 plans (`*Plan`) | 27 | stable | private fields; 25 of 27 have a public `can_implement` (finding 1) |
| 187 methods | 187 | stable with a note | finding 1 for the three backward plans, finding 6 for hyper-connection |

(The per-group row counts other than the first row are by kind, not by a name split of fields between descriptors and args, which this review did not redo.)

## Findings

1. **Three backward plans have no public `can_implement`.** `SelectiveScanBackwardPlan` (`selective_scan.rs:516`) and
   `SsdChunkScanBackwardPlan` (`ssd_chunk_scan.rs:422`) expose `select`, `workspace_size`, `sku`, `precision_guarantee`,
   `run` only, and validate inside `run` (the d_skip/d_d, z/dz and delta_bias/d_delta_bias both-or-none checks are in
   `run`, read at `selective_scan.rs:614-650`). `HyperConnectionPlan::run_backward` calls a private `check_backward_args`.
   Every forward plan, and 25 of the 27 plan structs (counted by `pub struct .*Plan`), has the public pre-validation; the hyper-connection backward is a mode of a plan that has it for forward only. A caller cannot
   pre-validate these three backward paths. Same shape of finding as `norm` 1.
2. **`#[non_exhaustive]` is on 7 descriptors and enums and absent on the rest.** By attribute: `FlashSdpaDescriptor`,
   `FlashSdpaBackwardDescriptor`, `FlashSdpaVarlenDescriptor`, `BatchPagedPrefillDescriptor`, `FlashDecodingDescriptor`,
   `RopeScaling`, `Fp8KvDtype`. The other descriptors (for example the scan, ring, block-sparse, cascade and rope ones)
   are plain structs with public fields, so adding a field to them is a breaking change after the freeze. The census
   `non_exhaustive` column marks only 5 rows; it misses `BatchPagedPrefillDescriptor` and `FlashDecodingDescriptor`
   (source has the attribute at `batch_paged_prefill.rs:64` and `flash_decoding.rs:79`). Freeze decision: all
   descriptors `non_exhaustive` with a `new`, or none. The census column should be re-derived.
3. **Public limits as consts.** `FLASH_SDPA_MAX_D`, `FLASH_DECODING_MAX_D`, `RING_ATTENTION_HEAD_DIM` (all 128),
   `SDPA_BLOCK_SPARSE_MAX_D` (128), `SDPA_BLOCK_SPARSE_MAX_BLOCK` (64) and `ROPE_DEFAULT_BASE` (10000.0) are public and
   therefore frozen. The first five are kernel limits that a new kernel would raise; raising a `pub const` is a
   behaviour change. Keep them `pub(crate)`, or document them as "current limit, may rise".
4. **`RopeScaling` and `Fp8KvDtype` are `non_exhaustive` enums** (good), and `Fp8KvDtype` feeds `sku().aux_element`
   (`batch_paged_decode_fp8.rs:132`), so a new variant must extend that `match`. Fine as a shape.
5. **`aux_element` is set in two places only**: `Fp8KvDtype` (fp8 decode) and a constant `ElementKind::Bf16` in the
   hyper-connection `sku()` (`hyper_connection.rs:491`, commented "rmsnorm_weight dtype"). Every other `sku()` has
   `aux_element: None`. Whether the hard-coded `Bf16` is right for the half-precision path was not checked.
6. **`HyperConnectionPlan` is one plan with two modes.** `select` and `select_training` build the same type; `run_backward`
   returns `Unsupported` at run time on a plan from `select`. A mode a plan can be in only by construction is a runtime
   error rather than a type. Also public: the associated fn `scratch_bytes(desc, training)`. Freeze decision: keep, or
   split a backward plan.
7. **Both-or-none `Option` pairs are checked at run time.** The scan backward `Args` carry `Option` tensors that must
   come in pairs (finding 1); an invalid combination is an `Error::InvalidProblem`, not unrepresentable. Stable as a
   shape; noted because the pre-validation of finding 1 would be where a caller learns it.
8. **No panic in the public path found.** The only `unwrap` in non-test code of this family is
   `flash_sdpa_backward.rs:556` and `:789` on `args.lse_f32`, guarded by an `is_none()` early return in `can_implement`
   (read, `:545-550`) which `run` calls; `rope_scaling.rs` has nine `expect` calls, all after its `#[cfg(test)]` at
   `:391`. Safe only while `run` keeps calling `can_implement`.

## Not checked

Kernel numerics; `can_implement` bodies beyond presence and use in `run` for the 25 plans that have one; each file's
field docs; the unread bodies of `ring_attention`, `sdpa_block_sparse`, `cascade_attn`, the paged plans and
`kv_cache`; `Send`/`Sync`; finding 5's `Bf16` against the half path. Nothing was executed.
