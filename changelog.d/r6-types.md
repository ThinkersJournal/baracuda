kind: docs: R6 review of `baracuda-types` (125 census rows, 8 findings)
---
Adds `docs/planning/api-freeze/r6/types.md`: the R6 public-API-freeze review of `baracuda-types` at `edd7ae72`. Eight findings, among them `Half`/`BFloat16` deriving bit-equality `Eq`/`Hash` while `PartialOrd` compares as `f32` (confirmed by a compiled test: `NAN == NAN` but `partial_cmp` is `None`; fixed in #391), two unsealed `unsafe trait`s, `KernelArg` returning `*mut` from `&self`, and `DeviceRepr` owned by `unpopped-vocab`. Docs only; no code, version or lockfile change. Verified by grep and read, no build.
