kind: docs: R6 review of the kernels sort family (268 rows, 8 findings)
---
Adds `docs/planning/api-freeze/r6/sort.md`: the R6 public-shape review of the 268 `sort` census rows at origin/main 8a3414f0. Findings: `HistogramddPlan` can never be constructed yet its 16 rows are `stable`, `SORT_MAX_ROW` is exceeded by `ArgsortPlan`, `TOPK_MAX_K` means two things, rank-2 and `i32` indices fixed in the types, mixed `i32`/`i64` extents, narrow dtypes, two scratch conventions. No code change. Verified by reading the sources and grepping each `run` for `can_implement`.
