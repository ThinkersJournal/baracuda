# R6 review: `baracuda-kernels-bench`

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `9b08b4e4c`. Items: the 115 rows for `baracuda-kernels-bench` in the regenerated §7.4 census, all
`class = stable`, `reach = public`, `non_exhaustive = 0`. By `kind`: 53 fields, 20 fns, 18 consts, 10 methods, 9 structs, 2 trait-members, 1 trait, 1 mod,
1 static (53 + 20 + 18 + 10 + 9 + 2 + 1 + 1 + 1 = 115, counted from `items.tsv` at this ref). 111 rows are in `src/lib.rs` (1,605 lines), 4 in
`src/machine_state.rs` (169 lines). Method: counts over `git ls-files` of `src/`, read `Cargo.toml`, the crate header, `LiveScalar`, the
`PytorchBaseline*` loader and the census rows. Function bodies were **not** read and nothing was built. A verdict on the public shape.

## Verdict

**Out of freeze scope; recommend the census class be `internal`, not `stable`.** The crate is `publish = false` (`Cargo.toml`), so no crates.io consumer can
depend on its API. The only in-repo users are its own `benches/` (26 files), `tests/` (12 files) and `src/bin/baracuda-bench.rs` (grep for
`baracuda_kernels_bench` over `git ls-files '*.rs'`). No other crate's `Cargo.toml` lists it (the two hits in `baracuda-bench-stats` are description text and a
comment). The 115 rows are a harness surface, so a freeze buys nothing and costs the ability to reshape sweeps and CSV columns freely.

| group | items | verdict | why |
|---|---|---|---|
| sweep consts and `static` (`GEMM_*`, `FLASH_*`, `CROSS_*`, `LOSS_*`, `POOL_SWEEP`, `CONV2D_SWEEP`, `WARMUP_ITERS`, `DEVICE_TIMING_LOCK`) | 19 | not frozen | finding 2 |
| timing/gate fns, `setup_device`, flops helpers, CSV fns (20 fns minus the 4 in `machine_state`) | 16 | not frozen | finding 3 |
| `Pytorch*` (5 structs, their fields and `load_*`/`lookup*`/`spread`) | 39 | **on-disk format is the contract, not the Rust shape** | finding 4 |
| row/shape structs (`PhaseTwentyNineRow`, `Conv2dShape`, `PoolShape`, `Spread`) and their fields and methods | 33 | not frozen | finding 1 |
| `LiveScalar` + 2 trait-members (impls are not census rows) | 3 | not frozen | finding 5 |
| `machine_state` mod + 4 fns | 5 | not frozen | finding 6 |

## Findings

1. **No `#[non_exhaustive]`, but the structs are crate-internal.** 0 in `src/` (grep, with the census `non_exhaustive = 0` on all 115 rows). `PhaseTwentyNineRow` has
   public fields (`op`, `shape`, `dtype`, `baracuda_ns`, `reference_ns`, `reference`, `pytorch_ns`, ...) that the benches build by literal. Adding a field is breaking
   only inside this repo, where one `cargo check` finds every site.
2. **Names encode project history.** `PhaseTwentyNineRow`, `PHASE29_CSV_HEADER` and the `Phase N` feature comments in `Cargo.toml` name roadmap phases, not
   behaviour. Not a freeze concern here; it would be one if the crate were published.
3. **`setup_device` and the timing fns panic.** `src/lib.rs:109-112` use `expect` on `init`, `Device::get(0)`, `Context::new` and `Stream::new`; `time_with_events*`
   use `expect` on every event call (`:247-317`); `assert_cell_live` panics by design (`:941`). Acceptable for a bench binary, and the reason this is not a
   library API.
4. **The real compatibility surface is the JSON baseline.** `PytorchBaselineMetadata`, `PytorchProvenanceRun`, `PytorchRegenTrigger`, `PytorchBaselineEntry` and
   the private `PytorchBaselineFile` (`#[derive(Deserialize)]`) read `bench-baselines/pytorch_rtx4070_2.14.0_cu126.json`. `load_from` accepts
   `schema_version` 1 or 2 and rejects others (`src/lib.rs:1149-1175`); v2 fields are `#[serde(default)]`. That is a versioned format with a real
   compatibility rule, authored by `tools/refresh_pytorch_baseline.py` and checked by `.github/workflows/pytorch-baseline-liveness.yml`. The Rust structs are
   `Deserialize` only, no `Serialize`.
5. **`LiveScalar` is a public trait with 4 impls** (`f32`, `f64`, `half::f16`, `half::bf16`, `:893-917`). Unsealed, but it is `publish = false`, so nothing
   outside this repo can implement it.
6. **`machine_state` returns `Result<_, String>` and `Option`** (`capture_nvidia_smi_sample`, `capture_other_gpu_process_count`, `capture_power_plan`,
   `capture_machine_state`). Stringly errors; the record types come from `baracuda-bench-stats`, which is the crate whose shape matters (not reviewed here).
7. **CSV header is a public const that the writer also checks.** `PHASE29_CSV_HEADER` (13 columns) is written and verified by `append_csv_row_with_spread`
   (`:663`, `:736`). Changing it is the stale-header path the code already handles. Its readers (any script parsing the CSV) were **not** searched.
8. **Census line numbers.** Not re-checked for this crate; the 4 `doc-hidden` rows reported by the pre-#394 census were extractor false positives and are gone from
   the regenerated one (0 `doc(hidden)` in `src/`, grep).
9. **Safety.** 0 `pub unsafe fn`, 0 `unsafe impl` in `src/` (grep). `#![deny(missing_docs)]` is set.

## Not decided here

- Whether the census should classify `publish = false` crates as `internal`. That is a census-class decision (and would drop these 115 rows from the stable
  count); it is raised, not made, here. The same question applies to every other `publish = false` crate. Not measured: how many there are.
- Not checked: function bodies, doc tests, the CSV's readers, `baracuda-bench-stats` shape.
