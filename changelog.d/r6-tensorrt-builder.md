kind: docs: R6 review of baracuda-tensorrt-builder (99 census rows, 10 findings)
---
`docs/planning/api-freeze/r6/tensorrt-builder.md`, read from source at `origin/main` `5f95bd845`. Headline findings: 16 safe `Layer` setters `static_cast` the layer to a kind with no check (UB from safe code); a network, config or profile from a different builder is accepted; the census omits the 9 `trt_enum!` enums and their 72 variants (99 rows, 180 with macros); all enums closed; the whole API is behind `shim`, so docs.rs renders only `shim_built`. Documentation only: nothing built or run, no code changed.
