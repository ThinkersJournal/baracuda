kind: docs: R6 review of the kernels gemm family (217 rows, 8 findings)
---
Adds `docs/planning/api-freeze/r6/gemm.md`: the R6 public-shape review of the 217 `gemm` census rows at origin/main 3713ab29. Findings: stale `mod.rs` doc, only 2 of 9 descriptors `non_exhaustive`, i32/usize extent types, sparse24 field order and AWQ/Marlin naming, cutlass types re-exported through `IntGemmPlan`, inconsistent `m == 0` handling, `repack` error type, feature-gated `run`. No code change. Verified by reading the sources and grepping each `run` for `can_implement`.
