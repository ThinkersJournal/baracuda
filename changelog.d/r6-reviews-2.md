kind: docs: R6 reviews for the remaining 13 crates, U19 flattened FlashInfer decode kernel
---
R6 review records for the 13 remaining crates (cuda-vocab, cuvs, megatron, transformer-engine, cudf, npp, ozimmu, cutlass-kernels-sys, facade, nvjitlink, build, flashinfer, types-derive); ROADMAP Mamba-1 stale-row fix. Adds `docs/planning/u19/`: a hand-flattened FlashInfer paged decode kernel (f16, d128, group 1) with buffer params plus one scalar struct, and its bitwise harness against the vendored original (sm89, 30 cases, diff_o=0 diff_lse=0). Docs only; no crate code changes.
