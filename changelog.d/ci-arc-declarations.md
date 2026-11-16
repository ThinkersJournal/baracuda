kind: ci
---
Declare the ARC-003/004/006/007/012 inputs in `.coderipper.toml` so those five rules run instead of reporting "not run" (which fails the portfolio job). Policy components: baracuda-types, baracuda-cuda-vocab; infrastructure: the 29 `-sys` crates and baracuda-driver; composition root: the `baracuda` facade; API budgets 130 and 50; owned data: the PyTorch baseline JSON of baracuda-kernels-bench.
