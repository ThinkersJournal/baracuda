kind: docs: R6 review of `baracuda-cudnn` (241 rows, 8 findings)
---
Adds `docs/planning/api-freeze/r6/cudnn.md`: the R6 public-API-freeze review of `baracuda-cudnn` at `fdfb1715`. Eight findings, among them 0 `#[non_exhaustive]` on 19 enums, 17 `unsafe impl Send` with no stated invariant, and 17 `as_raw` methods that put the sys crate's types in the public API. Docs only; no code, version or lockfile change. Verified by grep and read, no build.
