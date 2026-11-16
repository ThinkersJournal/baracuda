kind: **BREAKING**: baracuda-optim step functions validate sizes, half-precision moments and step index
---
baracuda-optim, from the R6 review (docs/planning/api-freeze/r6/optim.md, findings 1, 2, 4, 5, 8, 9). Calls that used to launch, or return `Ok(())` with wrong results, now return an error:

- `AdamStepPlan::step_with_f32_state`, `SgdStepPlan::step` and `SgdStepPlan::step_with_f32_momentum` compared tensor counts only; they now compare every tensor's element count (`Error::LengthMismatch`), as `AdamStepPlan::step` and `LambStepPlan::step` already did. `LambStepPlan::step` also compares `u_scratch_tensors`.
- `AdamStepPlan::<f16/bf16>::step` and `SgdStepPlan::<f16/bf16>::step` take half-precision moments that the shim reads as `f32`; they now return `Error::InvalidArgument` (use `step_with_f32_state` / `step_with_f32_momentum`).
- With `bias_correction`, a `step_index` below 1 (the shim divided by `1 - beta^0 = 0` and wrote NaN) returns `Error::InvalidArgument`.
- `TensorList::new` / `new_mut` return `Error::InvalidArgument` for a tensor over `i32::MAX` elements instead of wrapping in a release build.
- `DistributedAdamStepPlan` is documented as experimental and outside the API freeze; its ZeRO-1 claims (README, ROADMAP Phase 58, module docs) are corrected: the optimizer state is not sharded.
- The crate README example now compiles (it is a doctest) and uses the real API.

Verified: 4 new unit tests and the README doctest run without a GPU (the doctest fails on the old README); 4 new device tests in `tests/step_validation.rs` guard the step-function behaviour and were compiled, not run (GPU work is off until the Defender fix).
