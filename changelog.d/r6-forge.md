kind: docs: R6 review of `baracuda-forge` (126 census rows, 8 findings)
---
Adds `docs/planning/api-freeze/r6/forge.md`: the R6 public-API-freeze review of `baracuda-forge` at `25d4baeb`. Eight findings, among them 0 `#[non_exhaustive]` on a 10-variant `Error` and three structs with 12 public fields, a serde-derived `BuildCache` whose format is an on-disk contract, and a census `reach` column that reads `private-mod` for 118 of 126 rows although every type is re-exported. Docs only; no code, version or lockfile change. Verified by grep and read, no build.
