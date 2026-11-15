kind: census `internal` class
---
docs: API-freeze census classes every `publish = false` crate with no published dependent as `internal` (4 of 77 packages, 173 rows: `baracuda-bench-stats` 58, `baracuda-kernels-bench` 115), so `stable` drops 10,486 to 10,313 and the freeze scope is the published surface. Adds `tool/internal-crates.py.txt` and a third `classify.py` argument. No code change.
