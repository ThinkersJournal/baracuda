kind: docs: R6 review of `baracuda-cutlass` (118 census rows, 8 findings)
---
Adds `docs/planning/api-freeze/r6/cutlass.md`: the R6 public-API-freeze review of `baracuda-cutlass` at `daa4c795`. Eight findings, among them 56 public fields on 9 struct-literal-constructed descriptor and args structs with no `#[non_exhaustive]`, a single-variant `GroupedScheduleMode` whose doc plans a second variant, and `Error` re-exported from `baracuda-kernels-types` so the census `enum Error` row is stale. Docs only; no code, version or lockfile change. Verified by grep and read, no build.
