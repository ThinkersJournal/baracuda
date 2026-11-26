// SPDX-License-Identifier: MIT OR Apache-2.0
//! `unique` plan: [`crate::sort::SortPlan`] into a caller-supplied scratch buffer, then an ordered dedup of each sorted row
//! (PyTorch `torch.unique(x, sorted=True, return_counts=True)`). Up to 0.1.0-alpha.189 `run` only sorted and returned `Ok` without writing
//! any output; alpha.191 made it decline; ROADMAP E0b wires it. The second stage is the ordered
//! [`crate::sort::UniqueConsecutivePlan`] kernel (run order, run-length counts), called directly on the sorted scratch.
//!
//! Trailblazer dtype coverage: `f32, f64, i32`. Set-valued — no BW.
//!
//! Args carry a `sorted` scratch buffer the caller allocates (same
//! shape as `input`) to receive the sort output; the dedup then
//! reads from it. We compose at the plan layer so the kernel side
//! stays simple — no fused sort+dedup kernel ships.

use core::ffi::c_void;
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

    /// Launch: sort each row ascending (NaN largest, the [`SortPlan`](crate::SortPlan) order) into `sorted_scratch`, then dedup the sorted rows
    /// in order.
    ///
    /// Convention (torch `unique(sorted=True)`): `values` holds the distinct values ascending; every NaN is its own value (NaN != NaN, and NaN sorts
    /// last), `-0.0` and `+0.0` are one value (the first of the run in sort order is kept); `counts[k]` is the number of times `values[k]` occurs
    /// (the run length); `counter[row]` is the actual number of distinct values, which can exceed `max_unique` (the extra values are dropped).
    /// Cells of `values` / `counts` past the counter are left untouched.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: UniqueArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch == 0 {
            return Ok(());
        }
        if self.desc.row_len > 1024 {
            return Err(Error::Unsupported(
                "baracuda-kernels::UniquePlan: row_len > 1024 (the sort and the dedup are one block per row)",
            ));
        }
        let sorted = args.sorted_scratch.data().as_raw().0 as *const c_void;
        let vals = args.values.data().as_raw().0 as *mut c_void;
        let counts = args.counts.data().as_raw().0 as *mut c_void;
        let counter = args.counter.data().as_raw().0 as *mut c_void;
        if self.desc.row_len > 0 {
            let sort = crate::sort::SortPlan::<T>::select(
                stream,
                &crate::sort::SortDescriptor {
                    batch: self.desc.batch,
                    row_len: self.desc.row_len,
                    descending: false,
                    element: self.desc.element,
                },
                PlanPreference::default(),
            )?;
            sort.run(
                stream,
                Workspace::None,
                crate::sort::SortArgs {
                    input: args.input,
                    values: args.sorted_scratch,
                    indices: args.sorted_idx_scratch,
                },
            )?;
        }
        let (b, n, m) = (self.desc.batch, self.desc.row_len, self.desc.max_unique);
        let sp = stream.as_raw();
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unique_consecutive_f32_run(
                    b,
                    n,
                    m,
                    sorted,
                    vals,
                    counts,
                    counter,
                    core::ptr::null_mut(),
                    0,
                    sp,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unique_consecutive_f64_run(
                    b,
                    n,
                    m,
                    sorted,
                    vals,
                    counts,
                    counter,
                    core::ptr::null_mut(),
                    0,
                    sp,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unique_consecutive_i32_run(
                    b,
                    n,
                    m,
                    sorted,
                    vals,
                    counts,
                    counter,
                    core::ptr::null_mut(),
                    0,
                    sp,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UniquePlan::run reached an unimplemented dtype",
                ));
            }
        };
        crate::indexing::gather::map_status(status)
    }
}
