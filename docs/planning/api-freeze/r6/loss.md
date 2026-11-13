# R6 review: `baracuda-kernels` `loss` family

Reviewer: baracuda lane (R6 owner, PM [TASK] 2026-11-10, 2026-11-13). Ref: `origin/main` `3e04bc56`. Items: the 726 rows
under `crates/baracuda-kernels/src/loss/` in the §7.4 census (722 `stable`, 4 `doc-hidden`; 21 `mod`s, 21 re-exports, 123
structs, 346 fields, 214 methods, 1 const; counted from `items.tsv` by `kind` and `class` at this ref, and 21 + 21 + 123 + 346 +
214 + 1 = 726). Twenty-two files, 10,170 lines: 20 loss ops each with a forward and a backward plan (`bce`,
`bce_with_logits`, `cosine_embedding`, `cross_entropy`, `ctc`, `fused_linear_cross_entropy`, `gaussian_nll`,
`hinge_embedding`, `huber`, `kl_div`, `l1`, `margin_ranking`, `mse`, `multi_margin`, `multilabel_margin`,
`multilabel_soft_margin`, `nll`, `poisson_nll`, `smooth_l1`, `triplet_margin`), `ctc_loss_cudnn` (one plan), and `common`
(`pub(crate)` helpers only); 41 plans.
Method: per file, grepped `pub struct .*Plan`, `pub fn can_implement`, `pub fn run`, `self.can_implement`,
`unwrap/expect/panic/unreachable`, `non_exhaustive`, `aux_element: Some`; read `mse.rs` (descriptor, `select` head, `run`
head), the descriptor and `select` head of `fused_linear_cross_entropy.rs`, the plan struct and method list of
`ctc_loss_cudnn.rs`, and the `target` field types of five files. The other bodies were **not** read. A verdict on the
public shape, not on numerics.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 21 `mod`s, 21 re-exports | 42 | stable | one module per op, re-exported from `loss/mod.rs` |
| 82 descriptors and args, 346 fields | about 430 | stable with a note | findings 3, 5, 6 |
| 41 plans | 41 | **freeze decision** | none has `can_implement` (finding 1); one cuDNN plan keeps state in `Cell`s (finding 2) |
| 214 methods | 214 | stable with a note | findings 1, 2, 4 |
| 1 const | 1 | stable | `FLCE_DEFAULT_IGNORE_INDEX = -100`, PyTorch's value |
| 4 rows marked `doc-hidden` | 4 | **census check** | finding 7 |

## Findings

1. **None of the 41 plans has a public `can_implement`.** 0 of 41 (grep over every file). `MseLossPlan::run` validates
   inline: `check_dense` on `pred`, `target`, `out`, then the shape against the descriptor (`mse.rs:116-131`, read). A
   caller cannot pre-validate. The family-level gap seen in `conv` 1 and `linalg` 1; `norm`, `image` and most of `elementwise` do
   have it.
2. **`CtcLossCudnnPlan` is the third state-holding plan style.** It keeps `Cell`s over a cuDNN handle, two tensor
   descriptors, a CTC descriptor, `workspace_bytes` and `workspace_queried` (`ctc_loss_cudnn.rs:119-131`) and has a `Drop`
   (`:533`). Its workspace API is three methods: `workspace_size()`, `workspace_size_queried()`, `query_workspace_size(..)`.
   By reading, `Cell` over raw handles is neither `Send` nor `Sync` (not compiled). Compare `conv` 2 and 4 and `linalg` 2; `MseLossPlan` and its backward
   have `workspace_size()` only (read); the other plans' state was not read.
3. **`#[non_exhaustive]` on two of 41 descriptors.** Only `FusedLinearCrossEntropyDescriptor` and its backward twin carry
   it (`fused_linear_cross_entropy.rs:108, 972`), with `new` and `with_reduction`/`with_ignore_index` setters. The other 39
   descriptors, such as `MseLossDescriptor<N> { input_shape, reduction, element }`, are plain structs with public fields,
   so a new field is a breaking change after the freeze. Same split as `attention` 2, `conv` 5, `elementwise` 2, `image` 1,
   `linalg` 5.
4. **`FusedLinearCrossEntropyPlan` has two scratch figures that disagree by design.** `workspace_size()` returns `0`
   and `conceptual_scratch_bytes()` reports what the plan "needs internally (informational; not enforced by `run`)"
   (`fused_linear_cross_entropy.rs:283-292`, read); `chunk_size()` is a public accessor on the plan. A caller that sizes
   from `workspace_size()` gets `0`, which is correct; the second figure is the one to freeze or drop.
5. **Class-index targets disagree on `aux_element`.** `HingeEmbedding`, `MultiMargin` and `MultilabelMargin`, forward and
   backward, report `aux_element: Some(ElementKind::I64)` for an `i64` target; `Nll` (`target: TensorRef<i64, 1>`) and
   `CrossEntropy` (`target: Option<TensorRef<i64, 1>>`) also take `i64` targets and report `None` (grep `aux_element: Some`:
   2 each in the three, 0 in the others). Either the `sku` identity should name the index dtype everywhere or nowhere; the
   `indexing` family has the converse problem (always `I32`).
6. **`ignore_index` exists only on fused linear cross-entropy.** `grep -c ignore_index` gives 14 in
   `fused_linear_cross_entropy.rs` and 0 in `nll.rs` and `cross_entropy.rs`; PyTorch's `nll_loss` and `cross_entropy` carry
   it. By name only: whether the two plans cover it another way (a target value, a mask) was not read. Adding it later is a
   new field on plain descriptors (finding 3).
7. **Four census rows are classed `doc-hidden` and the source has no `#[doc(hidden)]` in `loss/`.** The rows are
   `FusedLinearCrossEntropyDescriptor` (struct) and its fields `bt` and `h`, and `FusedLinearCrossEntropyBackwardDescriptor.bt`
   (`items.tsv` lines 109, 110, 112, 982). `git grep 'doc(hidden)' -- crates/baracuda-kernels/src/loss` finds 0, and the
   README rule is "`#[doc(hidden)]` → `doc-hidden`". The doc comments of those items say "hidden states" or "hidden
   dimension"; I did not read the census tool, so the cause is not established. If the rows are wrong they are public and
   stable, not hidden.
8. **No non-test panic.** Grep finds one `unwrap`/`expect`/`panic!`/`unreachable!` in `ctc_loss_cudnn.rs` and none in the
   other 21 files; the one was not read.

## Not checked

Numerics; every `select` and `run` body other than the quoted heads; `LossReduction` (defined outside this directory);
the single `ctc_loss_cudnn.rs` panic site; the census extractor behind finding 7; `Send`/`Sync` by compilation. Nothing was
executed.
