kind: build: lock fireline-grammar/-synth/-announce at 0.4.2
---
`Cargo.lock` only: the three fireline crates move 0.4.0 -> 0.4.2 (the `~0.4` pins in `baracuda-cuda-seam` and `baracuda-seam` already accept it, so no manifest or published-version change). `cargo test --locked -p baracuda-cuda-seam -p baracuda-seam -p baracuda-cuda-emit --all-features` passes. fishstringer-slang-emit/-parse are already at 0.3.0 (latest).
