kind: build: repin the Slang dev-dependencies to fishstringer 0.3.0
---
`baracuda-cuda-emit` and `baracuda-cuda-parse` dev-dependencies `fishstringer-slang-emit` / `-parse` move from `~0.2.0` to `~0.3.0` (0.3.0 makes `Slang::lower_kernel` buffers strided views with a `params_block`). Tests only; no library code change. `cargo test --locked -p baracuda-cuda-emit -p baracuda-cuda-parse --all-features` passes. Note: `fishstringer-slang-parse` 0.3.0 requires Rust 1.90 (dev-dependency only).
