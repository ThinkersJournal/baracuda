// SPDX-License-Identifier: MIT OR Apache-2.0
//! `unique` plan — sort + consecutive-dedup composition (**not wired: `run` returns `Error::Unsupported`**).
//!
//! Meant to compose [`crate::sort::SortPlan`] (into a caller-supplied scratch buffer) followed by
//! [`crate::sort::UniqueConsecutivePlan`], PyTorch `torch.unique(x, sorted=True)`. Up to 0.1.0-alpha.189 `run` only sorted and returned
//! `Ok` without writing any output; it now declines. Compose the two plans yourself until ROADMAP E0b lands.
//!
//! Trailblazer dtype coverage: `f32, f64, i32`. Set-valued — no BW.
//!
//! Args carry a `sorted` scratch buffer the caller allocates (same
//! shape as `input`) to receive the sort output; the dedup then
//! reads from it. We compose at the plan layer so the kernel side
//! stays simple — no fused sort+dedup kernel ships.

use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SortKind, TensorMut,
    TensorRef, Workspace,
};

use super::unique_consecutive::build_unique_sku;

/// Descriptor for a `unique` op.
#[derive(Copy, Clone, Debug)]
pub struct UniqueDescriptor {
    /// Number of independent rows.
    pub batch: i32,
    /// Length of each input row.
    pub row_len: i32,
    /// Maximum unique values per output row.
    pub max_unique: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `unique` launch.
#[derive(Debug)]
pub struct UniqueArgs<'a, T: Element> {
    /// Input `[batch, row_len]`.
    pub input: TensorRef<'a, T, 2>,
    /// Scratch buffer for sorted input `[batch, row_len]` (caller-
    /// allocated; overwritten).
    pub sorted_scratch: TensorMut<'a, T, 2>,
    /// Scratch buffer for sorted indices `[batch, row_len]` (caller-
    /// allocated; overwritten — unused after the dedup).
    pub sorted_idx_scratch: TensorMut<'a, i32, 2>,
    /// Output values `[batch, max_unique]`.
    pub values: TensorMut<'a, T, 2>,
    /// Output per-cell counts `[batch, max_unique]`.
    pub counts: TensorMut<'a, i32, 2>,
    /// Per-row counter `[batch]`.
    pub counter: TensorMut<'a, i32, 1>,
}

/// `unique` plan.
///
/// Sort-then-dedup composition (PyTorch `torch.unique(x, sorted=True)`).
/// At the plan layer chains [`SortPlan`](crate::SortPlan) into a
/// caller-supplied scratch buffer, then
/// [`UniqueConsecutivePlan`](crate::UniqueConsecutivePlan) to collapse
/// runs.
///
/// **When to use**: per-row distinct-value extraction. Set-valued —
/// no BW (output dimensionality is data-dependent).
///
/// **Dtypes**: `{f32, f64, i32}`.
///
/// **Shape limits**: input `[batch, row_len]`; `row_len ≤ 1024` (sort
/// cap). Outputs `[batch, max_unique]`; the caller's `max_unique`
/// bounds the output; rows with more uniques have the overflow
/// dropped (the `counter[]` reports the actual count).
///
/// **Workspace**: zero in [`Workspace`]; caller supplies
/// `sorted_scratch`, `sorted_idx_scratch`, and `counter` in
/// [`UniqueArgs`].
///
/// **Precision guarantee**: deterministic, bit-stable.
#[derive(Debug)]
pub struct UniquePlan<T: Element> {
    desc: UniqueDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> UniquePlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &UniqueDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::UniquePlan: descriptor element != type parameter T",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::I32
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::UniquePlan: today only f32 / f64 / i32 wired",
            ));
        }
        let sku = build_unique_sku::<T>(SortKind::Unique);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &UniqueArgs<'_, T>) -> Result<()> {
        crate::view_check::check_fits(&args.input)?;
        crate::view_check::check_fits(&args.sorted_scratch)?;
        crate::view_check::check_fits(&args.sorted_idx_scratch)?;
        crate::view_check::check_fits(&args.values)?;
        crate::view_check::check_fits(&args.counts)?;
        crate::view_check::check_fits(&args.counter)?;
        let in_shape = [self.desc.batch, self.desc.row_len];
        let out_shape = [self.desc.batch, self.desc.max_unique];
        if args.input.shape() != in_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UniquePlan: input shape mismatch",
            ));
        }
        if args.sorted_scratch.shape() != in_shape || args.sorted_idx_scratch.shape() != in_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UniquePlan: sorted_scratch / sorted_idx_scratch shape mismatch",
            ));
        }
        if args.values.shape() != out_shape || args.counts.shape() != out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UniquePlan: values / counts shape mismatch",
            ));
        }
        if args.counter.shape() != [self.desc.batch] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UniquePlan: counter shape != [batch]",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes (the sorted-scratch buffers are caller-
    /// supplied as Args fields, so the plan reports 0 here).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch — sort, then dedup.
    pub fn run(
        &self,
        _stream: &Stream,
        _workspace: Workspace<'_>,
        args: UniqueArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        // Until the dedup stage is wired this plan declines rather than "succeed" with an empty result: up to 0.1.0-alpha.189 `run` only
        // sorted into `sorted_scratch` and never wrote `values`, `counts` or `counter` (ROADMAP E0b). To get `unique` today, sort with
        // `SortPlan` and dedup the sorted row with `UniqueConsecutivePlan` (the pair this plan is meant to compose).
        Err(Error::Unsupported(
            "baracuda-kernels::UniquePlan::run: the sort-then-dedup composition is not wired; compose SortPlan and UniqueConsecutivePlan",
        ))
    }
}
