# baracuda-cuda-seam

FireLine's kernel-seam `Synthesizer` implementation over the CUDA backend (the one Fuel calls).

`BaracudaSynthesizer` answers the JIT-on-request envelope (`fireline-synth`) using `baracuda-cuda-emit`; without the `nvrtc` feature it falls back to `unpopped::StubCompiler`.
It depends on `fireline-synth` and `fireline-grammar` (`~0.1`) from crates.io, which must resolve to the same `unpopped-vocab` as the rest of the graph. Up to alpha.104 it required `fuel-kernel-seam` and `fuel-kernel-seam-types` `^0.43` instead.

Part of [baracuda](https://github.com/ciresnave/baracuda); licensed under MIT OR Apache-2.0 (see `LICENSE-MIT`, `LICENSE-APACHE`).
