# R6 review: `baracuda-kernels` `conv` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10, 2026-11-13). Ref: `origin/main` `446aa5d6`. Items: the 371 `stable`
rows under `crates/baracuda-kernels/src/conv/` in the §7.4 census (9 `mod`s, 9 re-exports, 39 structs, 185 fields, 129
methods; counted from `items.tsv` by `kind` at this ref, and 9 + 9 + 39 + 185 + 129 = 371). Nine files: `conv1d/2d/3d`,
`conv_transpose1d/2d/3d` (six cuDNN-backed plans) and `im2col`, `im2col1d`, `col2im1d` (three kernel plans).
Method: per file, grepped `pub fn`, `can_implement`, `check_*_args`, `Cell` fields, `non_exhaustive`, `aux_element`,
`unwrap/expect/panic`; read `conv2d.rs` `select`, `run_fw`, the workspace accessors and the plan struct in full, and read the
public fn lists of `conv1d.rs`, `conv_transpose2d.rs` and the three im2col files. `conv1d/3d` and the other transposes were
**not** read beyond their signatures; they are taken to mirror `conv2d` on that evidence. A verdict on the public shape,
not on cuDNN or kernel numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 9 `mod`s, 9 re-exports | 18 | stable | one module per plan, re-exported from `conv/mod.rs` |
| descriptors, args and their 185 fields | about 190 | stable with a note | findings 5, 6 |
| 9 plans | 9 | **freeze decision** | no `can_implement`, `Cell` state, three run names (findings 1 to 3) |
| 129 methods | 129 | stable with a note | findings 1, 2, 3, 4 |

## Findings

1. **No plan in the family has a public `can_implement`.** Zero of nine (`grep can_implement` over the directory finds
   nothing), where every plan in `attention` has one except the two scan backward plans, and `norm`'s `InstanceNorm`
   pair is the only other gap. The six cuDNN plans validate in private `check_fw_args`, `check_bw_data_args`,
   `check_dw_args` (three per file, counted) called from `run_*`; the three im2col/col2im plans validate in `select` (read for `im2col.rs`: `validate_geometry` at `:126`)
   only. A caller cannot pre-validate a conv call. This is the family-level counterpart of the per-plan gaps in
   `norm` 1 and `attention` 1.
2. **The six conv plans are not `select`-then-`run` plans; they build cuDNN state lazily behind `Cell`.** `Conv2dPlan`
   holds a `cudnnHandle_t`, four descriptors and four size and algorithm slots as `Cell<...>` (`conv2d.rs:247-262`),
   created on first `run_*` or `query_*`. `select` takes a `_stream` and `_pref` it ignores. By reading, a struct of
   `Cell` over raw cuDNN handles is neither `Sync` nor (raw pointers) `Send`; no `unsafe impl` exists in the directory,
   and this was not compiled. Whether a conv plan may cross threads is therefore the freeze's call, and the answer is
   currently "no" by accident of the field types rather than by statement.
3. **Three run entry points, not `run`.** The six conv plans expose `run_fw`, `run_bw_data`, `run_dw` (with `Conv2dArgs`,
   `Conv2dBwArgs`, `Conv2dDwArgs`); every other family has one `run` (or a separate backward plan). The im2col trio has
   `run`. Names agree inside conv (all six, checked by grep), so the freeze decision is whether conv is a deliberate
   exception.
4. **Two workspace spellings, one of which is `0` until used.** `workspace_size_fw/_bw_data/_bw_filter()` return the
   cached cell (documented as `0` before the first query or run, `conv2d.rs:462-478`), and
   `query_fw_workspace_size(&Stream)` and two siblings compute it. The standard `workspace_size()` used by the other
   families, which a caller sizes a buffer from before `run`, does not exist on the six conv plans. A caller following
   the crate's convention gets `0`-byte guidance here.
5. **`#[non_exhaustive]` is on the six conv descriptors (two attribute sites per file, counting the doc mention) and absent on the
   three im2col/col2im descriptors.** The conv descriptors have a `new` plus `with_padding/stride/dilation/groups`
   builders; the im2col descriptors are plain public-field structs. Same split as `attention` 2; one rule across the
   crate would settle both.
6. **`fw_algo` picker exists only on `Conv2dPlan`.** `with_fw_algo_picker` and `fw_algo()` (19 mentions in `conv2d.rs`,
   0 in the other five conv files) are the opt-in picker of SA-49 (see `docs/conv2d-picker-measurements.md`). It is
   a public, frozen method on one of six plans, on a feature that is still opt-in and under measurement. Freeze
   decision: keep, move it behind a feature, or mark it unstable.
7. **`sku()` has `aux_element: None` in all nine; no panic in non-test code.** The only `unwrap` hits are in `#[test]`
   code of `im2col.rs` and `im2col1d.rs` (`assert_eq!(...unwrap()...)`). `Conv*Plan::select` rejects `desc.element !=
   T::KIND`, non-positive extents, strides and dilations as errors (read for `Conv2dPlan` only).

## Not checked

cuDNN or kernel numerics; the `check_*_args` bodies (their presence and call from `run_*` only); `Drop` of the cuDNN
handles (six `impl Drop` exist in the directory, counted; bodies not read, and whether they destroy every handle and descriptor is a leak question, not a shape one);
`Send`/`Sync` by compilation; the `conv1d`, `conv3d` and transpose bodies beyond their signatures. Nothing was executed.
