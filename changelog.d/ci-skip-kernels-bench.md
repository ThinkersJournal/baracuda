kind: CI: skip kernels-bench in the portfolio job
---
ci: `.coderipper.toml` declares `baracuda-kernels-bench` a `[[skip_member]]` while `nvcc` is absent, so the advisory portfolio job no longer fails on OPS-011 (the bench binary cannot link on a runner without nvcc). No code change.
