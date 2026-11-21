kind: provenance: 461 option A remainder (credits, notices, 9-file MIT narrowing, 37 stamps); device-test runners stop using dtolnay
---
Documentation, licence-header and CI-only; no code or ABI change.

- **Credits and notices:** `baracuda-cutlass-sys/build.rs` credits cudaforge (Guoqing Bao); `baracuda_simt_broadcast_epilogue.h` reproduces NVIDIA's CUTLASS BSD-3-Clause block (v4.2.0, compared against upstream); `baracuda-kernels-sys` `LICENSE-thirdparty.md` gains the Liger and candle rows and pointer lines on `build.rs`/`lib.rs`; `n_blocking_smoke.rs` says only the chunking it exercises is ported from RIKEN-RCCS.
- **Retractions and labels:** G-FA-1 and G-Q8-1 retracted in `PROVENANCE.md`/`VENDOR.md`; version labels SHA-first; the llama.cpp correction note and the `mmvq.cu` launcher count (132) match the files.
- **A2:** nine llama.cpp/ggml/cutf-derived files in `baracuda-kernels-sys` change `SPDX-License-Identifier` from `MIT OR Apache-2.0` to `MIT`; holder lines unchanged.
- **Stamps:** 37 baracuda-written files in the cutlass, cutlass-kernels-sys, cutlass-sys, kernels-sys, tensorrt and transformer-engine-sys crates gain our SPDX line (1 line each; each was skipped only by the provenance-crate rule).
- **CI:** `device-tests.yml` and `device-tests-kernel-features.yml` drop `dtolnay/rust-toolchain` (it installs into the shared RUSTUP_HOME; board 429 stage 2) and print CARGO_HOME/RUSTUP_HOME in a probe step.
- **Verified:** `cargo check` with `DOCS_RS=1` on the touched crates (including tests); `tools/add-spdx-headers.py` dry run: 8 skips left, all intentional.
