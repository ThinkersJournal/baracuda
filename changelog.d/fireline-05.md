kind: build: repin the seam crates to fireline ~0.5
---
`baracuda-cuda-seam` (`fireline-grammar`, `fireline-synth`) and `baracuda-seam` (`fireline-announce`) move from `~0.4` to `~0.5` (fireline 0.5.0). The crates read no contracts, so nothing else changes. `baracuda-cuda-seam` is 0.47.1 (emitter crates carry their own version). `cargo test --locked -p baracuda-cuda-seam -p baracuda-seam --all-features` passes.
