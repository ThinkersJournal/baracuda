kind: tools: prototype kernel instantiation list (tools/instantiation-list.py)
---
Adds `tools/instantiation-list.py`, which runs `nvcc -E` on every `.cu` in one kernels-sys directory and prints a TSV of (symbol, launcher, template args, source file, site line), plus a ROADMAP row. Run over `kernels/elementwise` at main 012ec162a: 220 `.cu`, 1515 symbols, 0 duplicates, 12 without a recoverable launcher (affine f16/bf16, prelu); other directories not run. No crate code changes.
