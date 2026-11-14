kind: docs: R6 review of `baracuda-driver` (626 rows, 8 findings)
---
Adds `docs/planning/api-freeze/r6/driver.md`: the R6 public-API-freeze review of `baracuda-driver` at `8fae2fb9`. Eight findings, among them no `#[non_exhaustive]` on 9 enums, 72 public `unsafe` functions (all with a `# Safety` section), public unsafe traits that are not sealed, and an `async` feature that gates nothing. Docs only; no code, version or lockfile change. Verified by grep and read, no build.
