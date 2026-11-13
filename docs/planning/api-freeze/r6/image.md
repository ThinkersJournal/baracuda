# R6 review: `baracuda-kernels` `image` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10, 2026-11-13). Ref: `origin/main` `df7eed36`. Items: the 276 `stable`
rows under `crates/baracuda-kernels/src/image/` in the §7.4 census (12 `mod`s, 12 re-exports, 36 structs, 129 fields, 80
methods, 1 enum, 6 variants; counted from `items.tsv` by `kind` at this ref, and 12 + 12 + 36 + 129 + 80 + 1 + 6 = 276; all
`public` and `stable`). Twelve files, 3,266 lines: `affine_grid`, `grid_sample(_backward)`, `interpolate(_backward)`, `nms`,
`pixel_shuffle`, `pixel_unshuffle`, `roi_align(_backward)`, `roi_pool(_backward)`; 12 plans.
Method: per file, grepped `pub struct .*Plan`, `pub fn can_implement`, `pub fn run`, `self.can_implement`,
`unwrap/expect/panic`, `non_exhaustive`, `aux_element`; read the `InterpolateMode` enum and `InterpolateDescriptor` in full
and the descriptor field lists of six other files. The `select`, `can_implement` and `run` bodies were not read beyond what
is quoted. A verdict on the public shape, not on kernel numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 12 `mod`s, 12 re-exports | 24 | stable | one module per plan, re-exported from `image/mod.rs` |
| 1 enum + 6 variants | 7 | **freeze decision** | `InterpolateMode`: five of six variants are rejected by `select` (finding 2) |
| 24 descriptors and args, 129 fields | 153 | stable with a note | findings 1, 3, 4 |
| 12 plans | 12 | stable | private fields; all 12 have `can_implement` and `run` calls it |
| 80 methods | 80 | stable | the family's six-method shape; no gap found |

## Findings

1. **`#[non_exhaustive]` only on the `interpolate` pair.** Three items carry it: `InterpolateMode`, `InterpolateDescriptor`,
   `InterpolateBackwardDescriptor` (matches the census `non_exhaustive` column, 3 rows). The other ten descriptors are
   plain structs with public fields, so a new field is a breaking change after the freeze. `GridSampleDescriptor` has
   no mode or padding field at all (`n, c, ih, iw, oh, ow, element`), so adding `align_corners` or a padding mode, which
   `interpolate` already carries, would be exactly that change. Same split as `attention` 2, `conv` 5, `elementwise` 2.
2. **`InterpolateMode` publishes five modes `select` rejects.** Only `Bilinear2d` is wired; `Nearest2d`, `Bicubic2d`,
   `Trilinear3d`, `Linear1d`, `Area2d` are documented "reserved" and `select` returns `Unsupported` for them
   (`interpolate.rs:214`, `if !matches!(desc.mode, InterpolateMode::Bilinear2d)`). The enum is `#[non_exhaustive]`, so
   adding modes is safe; freezing the five reserved names is the choice (keep as a roadmap, or remove until wired).
3. **Extent and scalar types are not uniform.** Every image descriptor uses named `i32` extents (`n, c, ih, iw, oh, ow`,
   `num_boxes`, `num_rois`) where `norm`, `elementwise` and `indexing` use `[i32; N]` shapes. Float parameters are `f32`
   (`NmsDescriptor::iou_threshold`, `RoiAlignDescriptor::spatial_scale`, `RoiPoolDescriptor::spatial_scale`) but
   `InterpolateDescriptor::scale_h/scale_w` are `Option<f64>`. Both are the portfolio's open extent and scalar-type
   choices rather than defects of `image`.
4. **`aux_element` is `Some` in three plans.** `Nms` reports `U8` (the keep mask), `RoiPool` and `RoiPoolBackward`
   report `I32` (the argmax index), by reading (`nms.rs:105`, `roi_pool.rs:131`, `roi_pool_backward.rs:121`); the nine
   others have `None` (grep). `RoiAlign` has no such buffer. Unlike `indexing`'s always-`I32`, these three describe the
   buffer they actually take; not checked against the `Args` tensor types.
5. **No gap in plan shape and no non-test panic.** All 12 plans have a public `can_implement` and `run` calls
   `self.can_implement` (grep per file: each has exactly one of each), unlike `norm`, `attention`, `conv` and `elementwise`,
   and none of the 12 files has an `unwrap`, `expect`, `panic!` or `unreachable!` (grep, 0 in each).

## Not checked

Kernel numerics; every `select`, `can_implement` and `run` body beyond presence and the `InterpolateMode` check; whether
the `Args` tensor dtypes match the `aux_element` values; `Send`/`Sync`; NMS output layout. Nothing was executed.
