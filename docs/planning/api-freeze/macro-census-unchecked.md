# Feature-gated macro census: the five crates item 4/5 left unchecked (origin/main ce2911956)

Method: `git grep` at origin/main for `macro_rules!`, `cfg(feature ...)` and bare `name! {` / `name!(` invocations per crate, then
`expand.sh` (`rustc +nightly -Zunpretty=expanded`, DOCS_RS=1) with the crate's features, counting rows the expansion adds over the default expansion.
Control: `baracuda-cuvs-sys` is gated by `cfg(feature = "cuvs")` at lib.rs:34 and :492 and is the known-positive.

| crate | features | macro_rules! | cfg(feature) in src | non-std macro invocations | status |
|---|---|---|---|---|---|
| baracuda-kernels | 30 lines | 32 | 138 | >0 (control hit) | expanded (see below) |
| baracuda-kernels-sys | 18 lines | 39 | 244 | >0 | TO EXPAND |
| baracuda-cuvs | 4 | 0 | 7 | 0 | no macro to gate (grep) |
| baracuda-flashinfer | 3 | 0 | 1 | 0 | no macro to gate (grep) |
| baracuda (facade) | 28 | 0 | 19 | 0 | no macro to gate (grep); features only forward |

## Results (nightly `-Zunpretty=expanded`, DOCS_RS=1, `--locked`, default features vs features on; items = `pub fn/struct/enum/type/trait`)

| crate | default | all features | added | of which macro-made (name never literal in `src`) |
|---|---|---|---|---|
| baracuda-kernels | 1619 fn / 803 types | 1925 / 912 | 415 | **0** (the syn walk already sees them) |
| baracuda-kernels-sys | 5542 fn / 17 types | 5893 / 35 | 369 | **119** |

Per feature on `baracuda-kernels-sys` (one feature at a time, vs default): `cudnn` +172 (**119 macro-made**), `sm89` +134, `flashinfer` +65, `mamba` +42,
`fa2_backward` +24, `bnb_nf4` +22, `fa2` +14, `xformers_sparse24` +12, `ring_attention` +11, `xformers_blocksparse` +8, `mhc` +6, `awq` +5, `marlin` +2,
`sm80` 0, `sm90a` 0; macro-made 0 for every feature except `cudnn`. All-features macro-made (119) equals the union of the single-feature sets, so no interaction.

**Finding:** the 119 are the `baracuda_kernels_{max,avg}_pool_{1,2,3}d_{fw,bw}_{f16,bf16,f32,f64}_run`-style wrappers that the `*_pool_{1d,2d,3d}_pair!` macros emit in
`crates/baracuda-kernels-sys/src/pool_cudnn_facade.rs` (behind `cudnn`). `items.tsv` has 0 rows for them (control: 472 `baracuda-kernels-sys` rows, so the table is the one we think it is;
`pool_cudnn_facade` has 1 row, the module). Fix in this change: `tool/expand-features.txt` now lists `baracuda-kernels-sys cudnn`, so the next deliberate census run counts them.
The committed tables are NOT regenerated (PM ruling 2026-11-19: only at the next deliberate census run, with denominators restated).
`baracuda-cuvs`, `baracuda-flashinfer` and the `baracuda` facade have no macro to gate (0 `macro_rules!`, 0 non-std macro invocations, controls above).
Not caveats-under-a-claim: `sm89` +134 are literal items the source walk already sees (name appears in `src`), counted as 0 macro-made by the same test.
Limitation: "name never literal in src" is a name test; a macro-made item that shares a name with a literal one would be missed.
