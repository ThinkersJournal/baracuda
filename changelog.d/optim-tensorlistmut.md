kind: **BREAKING**: baracuda-optim TensorListMut for written tensors, non_exhaustive configs, sealed FFI dispatch
---
baracuda-optim, from the R6 review (findings 3, 6, 7) as ruled by the PM:

- New `TensorListMut`, built with `TensorListMut::from_buffers` from `&mut DeviceBuffer`. Every step takes `&mut TensorListMut` for the tensors it writes (parameters, moments, momentum, LAMB scratch) and `&TensorList` for gradients. `TensorList::new_mut` is removed. A read-only list where a written one is needed, and the same buffer in two lists of one call, are now compile errors (it was possible to write through a shared borrow). `DistributedAdamStepPlan::step` follows.
- `AdamConfig`, `SgdConfig` and `LambConfig` are `#[non_exhaustive]`: build from `Default` and assign fields (a struct expression, even with `..Default::default()`, no longer compiles outside the crate).
- `AdamParamDtype::adam_ffi` and `SgdParamDtype::sgd_ffi` take a token only this crate can build, and the field of `AdamFfi` / `SgdFfi` is crate-private, so the shim entry points are no longer reachable from outside.
- ROADMAP Phase 58 gets a tracked list of the distributed plan's open items (sharding, all-gather send pointer, aliasing, multi-rank run).

Verified: 3 new `compile_fail` doctests (each compiles on the previous API, so each guards the change by reasoning; the old tree was not re-run for these) and the README doctest run without a GPU; the 7 device-test files were migrated and type-checked (`cargo check --tests --features device-tests,sm80,distributed_optim`), not run.
