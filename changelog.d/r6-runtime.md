kind: docs: R6 review of `baracuda-runtime` (431 rows, 7 findings)
---
Adds `docs/planning/api-freeze/r6/runtime.md`: the R6 public-API-freeze review of `baracuda-runtime` at `b931ce59`. Seven findings, among them no `#[non_exhaustive]` on 4 enums or on the 22-field `DeviceProperties`, 59 public `unsafe` functions (all with a `# Safety` section), and 30 of 38 public type names shared with `baracuda-driver`. Docs only; no code, version or lockfile change. Verified by grep and read, no build.
