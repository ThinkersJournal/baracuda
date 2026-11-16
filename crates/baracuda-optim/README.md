# baracuda-optim

Fused multi-tensor optimizer kernels (Adam / LAMB / SGD) for the
baracuda CUDA stack.

## What it wraps

The `multi_tensor_apply` idiom vendored from
[NVIDIA Apex](https://github.com/NVIDIA/apex) (BSD-3-Clause). Apex
sources live under `vendor/apex/`; PyTorch-specific glue was replaced by
a PyTorch-free C-ABI shim in `csrc/baracuda_optim_shim.cu`.

## Why this crate

baracuda's main facade (`baracuda-kernels`) ships **zero** optimizers —
it's a kernel substrate, not a training framework. Without
multi-tensor apply, the optimizer step on a 32B-parameter model would
launch ~10,000 kernels per step at ~5µs of launch overhead each =
**50ms of pure launch latency per step**. Multi-tensor apply collapses
that to a single launch (or a small handful when the per-launch
tensor cap of 110 is exceeded) by packing all per-tensor pointers +
sizes into one parameter struct. The crate boundary is deliberate —
inference-only consumers don't pay the FFI surface cost.

## What's exposed

| Type | Update rule | Param dtype |
| --- | --- | --- |
| `AdamStepPlan` | classic Adam or AdamW (mode flag) | f32 / f16 / bf16 |
| `LambStepPlan` | LAMB (Adam + per-layer trust ratio) | f32 (Phase 49) |
| `SgdStepPlan` | SGD + momentum + Nesterov + weight decay | f32 / f16 / bf16 |
| `DistributedAdamStepPlan` | experimental data-parallel Adam over NCCL (behind `distributed_optim`; not ZeRO-1, see below) | f32 / f16 / bf16 |

Mixed-precision wiring: when param/grad is f16/bf16, the moments
(`exp_avg` / `exp_avg_sq` / momentum_buf) **must** stay in f32 —
half-precision moments lose precision catastrophically.

## Quick example

```rust,no_run
# use baracuda_driver::{DeviceBuffer, Stream};
# use baracuda_optim::{AdamConfig, AdamMode, AdamStepPlan, TensorList, TensorListMut};
# fn demo<'a>(
#     stream: &Stream,
#     params: Vec<&'a mut DeviceBuffer<f32>>,
#     grads: &[&DeviceBuffer<f32>],
#     exp_avg: Vec<&'a mut DeviceBuffer<f32>>,
#     exp_avg_sq: Vec<&'a mut DeviceBuffer<f32>>,
# ) -> Result<(), Box<dyn std::error::Error>> {
// `AdamConfig` is `#[non_exhaustive]`: start from the default and assign.
let mut cfg = AdamConfig::default();
cfg.lr = 1e-3;
cfg.weight_decay = 0.01;
cfg.mode = AdamMode::Decoupled;
let plan = AdamStepPlan::<f32>::new(cfg);

// One `step` call updates ALL parameter tensors in a single launch
// (up to the 110-tensor per-launch cap). Step indices start at 1.
// Tensors the step writes go in a `TensorListMut` (borrowed `&mut`);
// the gradients it only reads go in a `TensorList`.
plan.step(
    &mut TensorListMut::from_buffers(params)?,
    &TensorList::new(grads)?,
    &mut TensorListMut::from_buffers(exp_avg)?,
    &mut TensorListMut::from_buffers(exp_avg_sq)?,
    1,
    stream,
)?;
# Ok(()) }
```

On a half-precision plan (`AdamStepPlan::<half::f16>`) the moments stay in
`f32` and go through `step_with_f32_state`; `step` returns
`Error::InvalidArgument` there.

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `sm80` | no | Build the Ampere-baseline kernel set. Forward-compatible to Ada / Hopper. |
| `sm89` | no | Build the Ada Lovelace specializations. |
| `sm90a` | no | Build the Hopper-specialized kernels. |
| `distributed_optim` | no | Enable `DistributedAdamStepPlan` (**experimental, outside the API freeze**: all-reduce of the gradients, a full-tensor Adam step on every rank, then an in-place all-gather; the optimizer state is not sharded). Pulls in `baracuda-nccl` (which dynamically loads `libnccl`; first `nccl()` call returns `LoaderError::LibraryNotFound` on hosts without it). Default OFF — inference-only and single-GPU training consumers don't pay the dep cost. |

Pick exactly one of `sm80` / `sm89` / `sm90a`; the build script picks
`sm90a > sm89 > sm80` when multiple are on.

## Status / scope

- Phase 49 — initial Adam / SGD / LAMB fanout.
- Phase 58 — `DistributedAdamStepPlan`, **experimental**: it does not
  shard the optimizer state, so it gives no ZeRO-1 memory saving, and the
  multi-rank path has not been run (it needs 2 GPUs). Single-rank degenerate
  case reduces to `AdamStepPlan`.

## Related crates

- [`baracuda-kernels`] — re-exports these plans behind the
  `optim` cargo feature when callers want the unified facade.
- [`baracuda-driver`] — `Stream` / `DeviceBuffer`.
- [`baracuda-nccl`] — optional dep behind `distributed_optim`.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

[`baracuda-kernels`]: https://docs.rs/baracuda-kernels
[`baracuda-driver`]: https://docs.rs/baracuda-driver
[`baracuda-nccl`]: https://docs.rs/baracuda-nccl
