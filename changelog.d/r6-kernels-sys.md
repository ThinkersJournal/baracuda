kind: docs: R6 review of `baracuda-kernels-sys` (159 census rows, 7 findings)
---
Adds `docs/planning/api-freeze/r6/kernels-sys.md`: the R6 public-API-freeze review of `baracuda-kernels-sys` at `c8df7e3e`. Seven findings, chiefly that the census rows cover about 1% of the public function surface (57 fn-like rows against 5,593 `pub fn` by grep) and that its line numbers are stale. Docs only; no code, version or lockfile change. Verified by grep and read, no build.
