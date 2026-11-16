# R6 review: `baracuda-optim`

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `71547fb82`. Items: the 85 rows for `baracuda-optim` in the §7.4 census (`items.tsv`: 80 `stable`, 4 `doc-hidden`, 1 `sealed-trait-pattern`): 75 in
`src/lib.rs` (1,072 lines) and 10 in `src/distributed.rs` (475 lines, behind the default-off `distributed_optim` feature; the census counts them anyway). By `kind`: 29 fields, 27 methods, 11 structs,
7 variants, 3 traits, 2 enums, 2 trait-members, 1 each of fn, mod, reexport, type (29 + 27 + 11 + 7 + 3 + 2 + 2 + 1 + 1 + 1 + 1 = 85, counted from `items.tsv` at this ref). Method: read `lib.rs` and
`distributed.rs` in full, read `Cargo.toml`, `README.md`, `vendor/apex/VENDOR.md`, the `adam`/`sgd`/`lamb` entry points of `csrc/baracuda_optim_shim.cu` and one Apex functor
(`multi_tensor_adam.cuh`), and counted with greps over `git ls-files`. Nothing was built or run, and no GPU was used. A verdict on the public shape, not on numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| `Error` (5 variants), `Result` | 7 | stable | `#[non_exhaustive]` already; finding 10 |
| 3 `*StepPlan` structs and their 12 methods | 15 | **freeze decision** | findings 1, 2, 5 |
| `TensorList` and its 6 methods | 7 | **freeze decision** | findings 3, 4 |
| 3 config structs and their 24 fields | 27 | **freeze decision** | finding 6 |
| `Sealed`, `AdamParamDtype`, `SgdParamDtype`, `adam_ffi`, `sgd_ffi`, `AdamFfi`, `SgdFfi`, 2 `.0` fields | 9 | **freeze decision** | finding 7 |
| `AdamMode` and 2 variants, `MultiTensorApplyContext` (3 fields, `fetch`) | 8 | stable | `AdamMode` is `#[non_exhaustive]` |
| `mod distributed`, its re-export, `DistributedAdamStepPlan` and 8 methods, `shard_range` | 12 | **freeze decision** | finding 8 |

Rows sum: 7 + 15 + 7 + 27 + 9 + 8 + 12 = 85.

## Findings

1. **Four of the six safe step functions check only the tensor count, not the sizes.** `validate_quadruple` (the per-tensor element-count comparison) is called by `AdamStepPlan::step` (`lib.rs:648`) and
   `LambStepPlan::step` (`lib.rs:979`) and nowhere else (grep, 2 call sites). `AdamStepPlan::step_with_f32_state`, `SgdStepPlan::step` and `SgdStepPlan::step_with_f32_momentum` compare `len()` only, then pass
   `params.sizes_ptr()` to the shim, which launches over the **param** sizes for all lists. A `grads` or `momentum` tensor shorter than its param is therefore read or written past its end on the device from a safe `pub fn`.
   `LambStepPlan::step` compares counts and the lengths of `w_norm_dev` / `u_norm_dev`, but not the sizes of `u_scratch_tensors` against `params`, and does not (cannot) check that the device array
   `u_scratch_ptr_dev` holds the pointers of those tensors. Decision: one validation helper for all six, plus a documented precondition for the LAMB pointer array (or build that array inside the plan).
2. **The half-precision moment type is not enforced.** The docs say moments "MUST be f32 (the FFI rejects anything else)" (`lib.rs:609`). The shim cannot reject it: its `f16`/`bf16` entry points take `void*` arrays, check
   only for null (`csrc/baracuda_optim_shim.cu:207`) and launch `adam_kernel_launch<__half, __half, float>` and `sgd_kernel_launch<__half, __half, float>` (6 launches; the state type is `float` in every one).
   But `AdamStepPlan<T>::step` takes `exp_avg: &TensorList<T>` and `SgdStepPlan<T>::step` takes `momentum: &TensorList<T>`, so on `AdamStepPlan::<f16>` the signature accepts `f16` moment buffers, which the kernel
   then reads and writes as `f32`: twice the allocation. The safe type signature allows the case the doc calls forbidden. Decision: `step` exists only for `f32` (an inherent method on `AdamStepPlan<f32>`), the half plans use
   only the `_with_f32_*` methods. That removes a public method from the half instantiations, so it is breaking.
3. **Mutability is not in the types.** `TensorList::new(&[&DeviceBuffer<T>])` stores `*mut c_void` made from shared references; `new_mut` exists "for buffers the optimizer step writes" but nothing requires it:
   every `step` takes `&TensorList` for `params`, `exp_avg`, `exp_avg_sq`, `momentum` (all written) and `grads` (read). `DistributedAdamStepPlan::step` itself builds the writable `params` list with `TensorList::new` from
   `&DeviceBuffer` (`distributed.rs`, step 2). Nothing prevents the same buffer being in `params` and `grads`. Decision: two types (`TensorList` for reads, `TensorListMut`) so a `step` signature says what it writes.
4. **Sizes are `i32` and truncate silently.** `TensorList` keeps `sizes: Vec<i32>` and the FFI takes `*const i32`. `new` has a `debug_assert!(n <= i32::MAX)` (`lib.rs:295`), so in a release build an
   element count above 2^31 - 1 wraps; `new_mut` (`lib.rs:319`) has no assertion at all. 2^31 f32 elements is 8 GiB, a plausible single embedding tensor. `total_elements` casts back with `as usize`, and
   `LambStepPlan::workspace_bytes` does unchecked `usize` arithmetic. Decision: return `Error::InvalidArgument` in both constructors (the variant exists).
5. **`step_index` and the configs are not validated.** `step_index: i32` is documented "1-based" and is checked nowhere (in `lib.rs`, `Error::InvalidArgument` is returned from 2 places, both LAMB buffer lengths). With
   `bias_correction` and `step_index == 0` the shim computes `1 - powf(beta, 0) = 0` (`csrc/baracuda_optim_shim.cu:210`) and the functor divides by it (`multi_tensor_adam.cuh:123`), writing inf/NaN into the
   parameters with `Ok(())`. `SgdConfig::nesterov` is documented "Requires `momentum > 0`"; nothing checks it in Rust, and the shim passes it through (grep). `lr`, betas and epsilon are unvalidated. Decision: validate in
   `step` (returns `Err`), which is a behaviour change and so must precede the freeze.
6. **Three all-pub config structs, closed.** `AdamConfig` (7 fields), `SgdConfig` (7), `LambConfig` (10): 24 `pub` fields, no `#[non_exhaustive]`, all `Copy`. A new hyperparameter (e.g. AMSGrad, a second
   epsilon) breaks every struct literal that lacks `..Default::default()`. `AdamMode` and `Error` already carry `#[non_exhaustive]`, so the crate has the pattern; the configs and `MultiTensorApplyContext` (3 pub fields of FFI
   constants) do not use it. Decision: `#[non_exhaustive]` on the three configs (callers then write `Default` plus field assignment).
7. **The sealed traits' hidden members are callable.** `AdamParamDtype` and `SgdParamDtype` are sealed (`sealed::Sealed`, private module, `lib.rs:553`) but each has a `#[doc(hidden)] fn adam_ffi()` / `sgd_ffi()` returning
   a `#[doc(hidden)] pub struct AdamFfi(pub unsafe extern "C" fn(..15 args..))` / `SgdFfi(pub unsafe extern "C" fn(..14 args..))`. The census lists the two structs `doc-hidden` and their two `.0` fields `stable`.
   Hidden is not private: `<f32 as AdamParamDtype>::adam_ffi().0(..)` compiles for any caller, so the argument order of two shim entry points is public API. Same issue as cusolver finding 2; decision: a private
   sub-trait carries the dispatch, the public trait stays a marker. The census should classify `.0` with its parent.
8. **`DistributedAdamStepPlan` does not do what its documentation says.** (a) The docs promise ZeRO-1 memory savings from "sharded" moment buffers, but `step` calls the inner `AdamStepPlan::step`, which runs
   `validate_quadruple`: a sharded (smaller) `exp_avg` fails the per-tensor size comparison with `LengthMismatch`, so the only accepted shape is moments the size of the params, i.e. no saving. (b) Step 3 computes
   `per_rank = p.len() / world` and passes the whole buffer as the send buffer with `per_rank` elements, which contradicts both the "each buffer is this rank's local shard" doc and the in-place all-gather
   condition the code quotes in its own comment (`send_ptr == recv_ptr + rank * count`): it holds for rank 0 only, unless `Communicator::all_gather` offsets the send pointer itself (I did not read it). (c) `all_reduce(send_ref, *g, ..)` passes a `&DeviceBuffer` and a `&mut` to the same buffer
   in one call, derived through a raw pointer, which breaks Rust's aliasing rules as written (not run under Miri; I did not read `baracuda_nccl::all_reduce`). (d) `shard_range` is `pub`, safe, and its guards are
   `debug_assert!`s, so `world_size == 0` panics (divide by zero) and `rank >= world_size` returns an offset past `n` in release. (e) The public signatures name `baracuda_nccl::{Communicator, NcclScalar}`, so
   the freeze of this feature fixes those two types. The test that would exercise the multi-rank path is named `distributed_adam_multi_rank_scaffold` and needs 2 GPUs. `shard_range_for` and `shard_range` are the
   only users of the shard arithmetic; `step` does not call them. Decision: either fix (a)-(c) before the freeze or mark the module experimental and out of the freeze (it is off by default).
9. **Documentation drift, 4 places.** (a) The README example (`README.md:45`) calls `AdamStepPlan::<f32>::new().with_lr(1e-3)...`; `new` takes an `AdamConfig` (`lib.rs:621`) and `with_lr` appears
   0 times in `src/` (the same grep finds `fn new(cfg`). The example does not compile. (b) The crate doc says the Apex sources are "vendored verbatim" (`lib.rs:49`); `vendor/apex/VENDOR.md` says the four `.cuh` files
   are PyTorch-free rewrites sharing at most 9 lines with upstream, and that only `LICENSE` is verbatim. The licence and notices are right (VENDOR.md, G-APEX-2); the crate doc is wrong. (c) `Cargo.toml` `description` is 1 sentence of changelog ("NEW in Phase 49; deliberate scope
   expansion"). (d) The ZeRO-1 text of finding 8(a).
10. **Census lines are current; consumers and tests.** Loop over the 19 struct/enum/fn/type/trait/mod rows: 19 checked, 0 mismatched; fields, methods, variants and trait-members were not line-checked. The only in-repo
    dependent is `baracuda-kernels` (optional `optim` feature, which also turns on `baracuda-optim/sm80`); the two `-sys` and transformer-engine `Cargo.toml` hits for the string are comments. `Error::LaunchFailed(i32)` carries the raw
    shim status and `Error::NcclFailed(String)` a rendered string, so callers cannot match the cause; `Error` has no `PartialEq`. Tests: 10 `#[test]` in `tests/*.rs`, every file behind `device-tests`, plus 4 unit tests of
    `shard_range` (behind `distributed_optim`); no test calls a step function with mismatched tensor sizes, so findings 1, 2 and 5 have no test that would fail on the current behaviour.

## Not decided here

- Finding 1: one validation helper for all six step functions.
- Finding 2: `step` only on `AdamStepPlan<f32>`/`SgdStepPlan<f32>`, or keep and validate.
- Finding 3: a separate mutable list type.
- Finding 5: validate `step_index` and configs (behaviour change).
- Finding 6: `#[non_exhaustive]` on the three configs.
- Finding 7: private dispatch sub-trait.
- Finding 8: fix or declare the `distributed` module experimental.
- Not checked: kernel numerics, `baracuda-nccl`, the LAMB stage-2 kernel and `lamb.cuh`, Miri, whether the shim's `u_scratch` device array is read before it is validated.
