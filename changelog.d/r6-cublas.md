kind: docs: R6 review of `baracuda-cublas` (176 rows, 8 findings)
---
Adds `docs/planning/api-freeze/r6/cublas.md`: the R6 public-API-freeze review of `baracuda-cublas` at `fdfb1715`. Eight findings, among them 0 `#[non_exhaustive]` on 5 enums, a documented `Send` on `Handle` that looks wrong by reading (`Arc<HandleInner>` where only `HandleInner` is `Send`), and 28 `#[doc(hidden)]` raw methods on sealed public traits. Docs only; no code, version or lockfile change. Verified by grep and read, no build.
