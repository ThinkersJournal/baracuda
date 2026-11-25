# baracuda-hip-emit

HIP (AMD) backend for the neutral `unpopped` kernel generator: the `Hip` `Backend`, kernel-layer lowering **stage 1**
(signature, argument block, thread/block builtins, control flow, `__shared__`, the block barrier).

Sibling of `baracuda-cuda-emit`. Anything NVIDIA-specific (`cp.async`, `mbarrier`, `wgmma`, `tcgen05`, `ldmatrix`,
clusters, named barriers) and anything whose AMD semantics have not been read (warp collectives, `Cast`/`Fma`/`Bitcast`) is
**declined by name**, never translated or approximated. The decline text says why.

The crate's version follows the `unpopped` release it works with (one-version exception). Not published yet (`publish = false`).
Not built with `hipcc` on the development box (no ROCm): see the module docs of `src/kernel.rs` for what the tests do and do not show.

Part of [baracuda](https://github.com/ciresnave/baracuda); licensed under MIT OR Apache-2.0 (see `LICENSE-MIT`, `LICENSE-APACHE`).
