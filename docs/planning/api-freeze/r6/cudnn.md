# R6 review: `baracuda-cudnn`

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `fdfb1715`. Items: the 241 rows for `baracuda-cudnn` in the §7.4 census
(all `public`, all `stable`). By `kind`: 58 fns, 57 methods, 81 variants, 17 structs, 19 enums, 5 re-exports, 2 types, 1 trait,
1 trait-member (58 + 57 + 81 + 17 + 19 + 5 + 2 + 1 + 1 = 241, counted from `items.tsv` at this ref). One file,
`crates/baracuda-cudnn/src/lib.rs`, 4,217 lines.
Method: grepped `lib.rs` for `non_exhaustive`, `unsafe impl`, `impl Drop`, `# Safety`, `pub unsafe fn`, `as_raw`, `doc(hidden)`,
`pub const`, `can_implement`, `Sync`; read `Handle`, `DType`, `CudnnDataType` and the head of the re-export blocks. Method bodies
were **not** read. A verdict on the public shape, not on cuDNN behaviour.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 19 enums, 81 variants | 100 | **freeze decision** | finding 1 |
| 17 descriptor/handle structs | 17 | stable with a note | findings 2, 3 |
| 115 fns and methods | 115 | stable with a note | findings 3, 4 |
| 5 re-exports, 2 types | 7 | **freeze decision** | finding 5 |
| `CudnnDataType` + `DTYPE` | 2 | stable | finding 6 |

## Findings

1. **0 `#[non_exhaustive]` in the crate.** Grep for the attribute line (`^\s*#\[non_exhaustive\]`) finds 0; the 19 enums
   (`DType`, `TensorFormat`, `ActivationMode`, `ConvMode`, `FwdAlgo`, `BwdDataAlgo`, `BwdFilterAlgo`, `PoolingMode`, `SoftmaxAlgo`,
   `SoftmaxMode`, `BatchNormMode`, `OpTensorOp`, `ReduceOp`, `MathType`, `ReorderType`, `NormMode`, `NormAlgo`, `NormOp`, `BnOp`)
   are exhaustive. `DType` has 6 variants and `DType::raw` is a closed `match`; cuDNN adds dtypes and algorithms per release
   (the algo enums most often), so a new variant would break callers' matches. Decision: add the attribute to all 19 before
   the freeze, as the kernels families did.
2. **17 `unsafe impl Send` and no `Sync`.** `Handle` and 16 descriptors carry `unsafe impl Send` (grep, 17 hits); 0 hits for `Sync`.
   By reading, the raw-pointer fields make them `!Sync` by default, which is the right default for a cuDNN handle. No comment
   at the 17 sites says why `Send` is sound (a `Handle` bound to a stream, descriptors read concurrently by the library); not
   checked beyond `Handle`. Decision: document the invariant or drop `Send` from the descriptors.
3. **Raw handles escape through 17 `as_raw` methods** (grep; `Handle`, every descriptor). Each returns the sys `cudnn*_t`
   pointer, so the sys crate's types are part of this crate's public API. Needed for interop; the freeze should say the sys
   version is part of the contract. 18 `pub fn` signatures mention a `cudnn*_t` type (regex, approximate).
4. **11 `pub unsafe fn`, all with a `# Safety` section** (grep on the 12 lines above each: 11 of 11). They are the RNN and
   multi-head-attention entry points plus `set_attribute_raw`, `reorder_filter_and_bias`, `SeqDataDescriptor::set` and
   `DropoutDescriptor::restore`. No safe wrappers exist for the RNN/attention calls, so most of the 241 rows sit behind
   `unsafe`. No `can_implement` anywhere (0 hits), unlike the kernels plans.
5. **Five re-export rows cover eight sys types**, under their own names or `Raw*` aliases: `BackendAttrName`, `BackendAttrType`,
   `BackendDescType`, `BwdDataAlgoPerf`, `BwdFilterAlgoPerf`, `FwdAlgoPerf`, `RawMathType`, `RawReorderType` (two `pub use` blocks and
   three one-line `pub use ... as` rows; `Error` and `Result` are the 2 `type` rows). Freezing them freezes the sys crate's enum layouts and their
   exhaustiveness (finding 1 applies to them too, from the sys side). Decision: keep re-exports or wrap.
6. **`CudnnDataType` is public and open to implementors**: it needs `DeviceRepr + Copy + 'static` and one const. Impls for
   `f32`, `f64`, `Half`, `BFloat16`, `i8`, `i32`, and `half::f16/bf16` behind `half-crate`. Not sealed, so a downstream impl
   with a wrong `DTYPE` is possible but memory-safe at the type level. Decision: seal or leave.
7. **The census is stale in line numbers.** Re-export rows say 2152, 2634, 2636, 2638, 3296; at this ref they are at 2135,
   2610, 2612, 2614, 3267. All 115 census `fn` names still exist (loop over the TSV, grep for `fn <name>`; positive
   control: `fn rnn_forward` finds 1). Row counts match: 104 `pub fn` + 11 `pub unsafe fn` = 115 = 58 + 57. So the
   census is stale in position only.
8. **No public consts, no `#[doc(hidden)]`** (0 hits each).

## Not decided here

- Whether the 19 enums become `#[non_exhaustive]` (finding 1); the recommendation is yes.
- Whether `Send` stays on the 16 descriptors (finding 2).
- Whether sys re-exports are part of the frozen surface (findings 3, 5).
- Not checked: method bodies, doc-test coverage, whether `Handle::drop` ignoring errors is acceptable.
