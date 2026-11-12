# ARC / MOD programme: declaration, findings and ratchet

PM [TASK] `arc-mod-programme` 2026-11-12 (CireSnave: *"Lets run those checks against our projects and start fixing the issues they find.
Clean code and clean architecture mean less bugs and easier maintainability."*). Advisory. The R6 reviews and the CUDA parser rows stay
first on any conflict.

## What was measured

coderipper 0.43.20 (`target/release` of the coderipper checkout; the PM census used 0.43.59), `--workspace --profile extended`, on
`origin/main` `8a3414f0` plus the docs-only branches, one process per rule, 75 crates in `crates/` plus `examples` and `xtask` (77 members).

| rule | before the declaration | after | note |
|---|---|---|---|
| MOD-002 cycles | 3 (2 Low, 1 Info) | 3 | two in `baracuda-driver`, one in `baracuda-kernels` |
| MOD-004 | 6 (Info) | 6 | the six `conv*` plans |
| MOD-005 size/coupling | 46 (1 Low, 45 Info) | 46 | Low: `baracuda-kernels-sys/src/lib.rs`, 60,427 code lines |
| MOD-006 | 1 (Info) | 1 | `baracuda-cuda-emit/src/cuda.rs` |
| MOD-008 | 27 (Info) | 27 | per-crate `check` forwarders |
| ARC-001 layers | not run | **0 findings**, 278 edges, 0 unplaced | new: needs the declaration |
| ARC-002 cycles between components | not run | **0 cycles**, 77 components | new |
| ARC-005 stability | 1 (Info) | 1 | `baracuda-ozimmu -> baracuda-ozimmu-sys`, margin 0.33 |
| ARC-003, 004, 006, 007, 012 | not run | not run | each needs a further table (below) |

The five MOD rows' totals (3 + 6 + 46 + 1 + 27 = 83) equal the PM census's 83 for baracuda; the census's three Low findings are the
two `baracuda-driver` cycles and the `baracuda-kernels-sys` size outlier.

## The declaration (`.coderipper.toml`, `[architecture]`)

One component per workspace crate, six layers, bottom first (a crate may depend on its own layer or below): `foundation` (build,
forge, types, core, vocab, parse, seam, bench-stats), `sys` (29 raw FFI crates), `wrappers` (27: the driver, the library wrappers,
`baracuda-kernels-types`), `composites` (cutlass, megatron, optim, tensorrt-builder, cuda-emit, cuda-seam), `plan`
(`baracuda-kernels`), `top` (facade, flashinfer, bench, examples, xtask). It restates `ARCHITECTURE.md` "Layered design"; the layers
were derived from `cargo metadata --no-deps` and the declaration holds on the first try (0 edges go up a layer), so the file is
a description of the real graph, not a wish.

Not declared yet, each making one more rule run: `[architecture.volatile]` (ARC-003 and 004: which vendor namespaces stable policy
must not name), `[architecture.api_budget]` (ARC-006: a public-item budget per crate, needs a measured baseline first),
`[architecture.composition]` (ARC-012: the roots; this is a library workspace, so the roots are `examples`, `xtask` and the benches),
`owns_data` (ARC-007: nothing here owns tables). They are rows R4 to R6 below.

## Rows

Non-Info findings first (PM instruction 1). "Waived" means an `[[allow]]` in a `.coderipper.toml` (see the last section for which) with owner and removal condition.

| row | finding | disposition | size |
|---|---|---|---|
| R1 | MOD-005 Low `baracuda-kernels-sys/src/lib.rs` (60,427 code lines) | **waived, scheduled.** Split into one module per op family, re-exported at the root. The ABI-manifest script reads this file and must change in the same PR; do it when no other lane has a `kernels-sys` PR open | one PR + script |
| R2 | MOD-002 Low `baracuda-driver` `library,module` | **waived, scheduled.** Move `Function` and its owner enum into a module both import | small |
| R3 | MOD-002 Low `baracuda-driver` `memory,stream` | **waived until the next breaking release**: moving `Stream::memcpy_dtod` to `DeviceBuffer` breaks a published method | small, breaking |
| R3b | ARC-005 Info `ozimmu -> ozimmu-sys` | waived: a wrapper on its own `-sys` crate is by design | none |
| R4 | declare `volatile` and `composition` | next, once the layers are merged | one PR |
| R5 | declare `api_budget` | needs a measured baseline per crate (ARC-006 counts public items); fold into the R6 reviews' public-surface counts | with R6 |
| R6 | `MOD-005` Info x45 | **scheduled, not fixed.** 21 are in `baracuda-kernels` (the conv files also carry MOD-004), 25 are in other crates (the raw-FFI crates and the library wrappers, mostly `lib.rs`, `functions.rs`, `types.rs`, `generated.rs`); the FFI ones are declaration tables, but that was not read file by file. Fix the kernels ones only when the R6 review of that family says the file mixes concerns; waive the FFI tables with a reason | per family |
| R7 | `MOD-004` Info x6 (`Conv{1,2,3}dPlan`, `ConvTranspose{1,2,3}dPlan`: 3 usage clusters) | scheduled with the `conv` R6 review | with R6 |
| R8 | `MOD-008` Info x27 in 24 crates: mostly `check` only forwarding to `Error::check` (a few other forwarders) | scheduled. One decision: either a single re-exported helper in `baracuda-core`, or accept the per-crate error type. Needs a look at whether each crate's `Error` is its own type (it is, by the `-sys` split) | small, many files |
| R9 | `MOD-006` Info x1, `MOD-002` Info x1 (`attention::flash_sdpa <-> flash_sdpa_sm89`) | scheduled; the cycle is inside one module | small |

## Ratchet (instruction 3)

Decision (PM, 2026-11-12): wait for coderipper's own ratchet key (`ratchet = true` in `[architecture]`, parsed by coderipper's
`src/architecture.rs` but not read yet) and do not write a baseline script in CI. The key is not set here until coderipper reads it:
setting it now would claim a ratchet that nothing enforces. The CI job that will run it is the advisory `portfolio` job, repinned to
coderipper 33cc3b7 (0.43.49) in a separate PR. Until then the baseline in the table above is the number to hold.

## Not checked

Where the waivers live: a `--workspace` run reads each member's own `.coderipper.toml`, not the root's (`src/allowlist.rs`; a first attempt with the MOD waivers in the root file waived 0 of 3 in a `--workspace` MOD-002 run). So the three MOD waivers are in `crates/baracuda-driver/.coderipper.toml` and `crates/baracuda-kernels-sys/.coderipper.toml`, with `file` relative to the member; checked with `-p baracuda-driver` (MOD-002: 0 findings, 2 waived) and `-p baracuda-kernels-sys` (MOD-005: 3 Info left, 1 Low waived). The ARC-005 waiver is in the root file (checked: 0 findings, 1 waived). A whole `--workspace` rerun after the move was not done. `cargo`-feature sets other than the default were not analysed
(`--all-features` would add edges). Kernel numerics and API shape are the R6 reviews' subject, not this one's.
