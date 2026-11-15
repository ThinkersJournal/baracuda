kind: docs: R6 review of `baracuda-cvcuda` (118 census rows, 7 findings)
---
Adds `docs/planning/api-freeze/r6/cvcuda.md`: the R6 public-API-freeze review of `baracuda-cvcuda` at `f7662ec1`. Seven findings, chiefly that 42 `simple_op!` macro-generated public operator structs are absent from the census (the extractor does not expand macros), 62 `pub unsafe fn submit` taking a raw `*mut c_void` stream, and untyped `i32`/`u32` codes in `Tensor::new` and `Threshold::new`. Docs only; no code, version or lockfile change. Verified by grep and read, no build.
