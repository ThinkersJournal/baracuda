kind: Changed: baracuda-cuda-emit / -parse / -seam rebuilt against unpopped 0.57.0 (BREAKING: needs unpopped ~0.57.0)
---
The emitter, parser and seam crates move to 0.57.0 to match unpopped 0.57.0. `baracuda-cuda-emit` now lowers `Stmt::ProxyFence { space: Shared }` to `fence.proxy.async.shared::cta` (ptxas accepts it for sm_90 and refuses it for sm_89, as the ISA says); any other space is refused with a reason. The other new 0.57.0 statements and the `Wgmma`/`Tcgen05` operand-layout fields stay refused with an explicit reason ("not lowered by the CUDA backend yet"), never a default.
