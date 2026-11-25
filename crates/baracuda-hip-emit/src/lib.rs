// SPDX-License-Identifier: MIT OR Apache-2.0
//! `baracuda-hip-emit`: the HIP (AMD) backend for the neutral `unpopped` kernel generator (ROADMAP H1).
//!
//! [`Hip`] implements [`Backend`]. Stage 1 lowers kernel-layer kernels only ([`Backend::lower_kernel`]); the plan-based
//! `lower` is not taught and says so by a typed decline. See `kernel` for what is lowered and what is declined by name.

mod kernel;

use unpopped::backend::{Backend, DtypeSupport, GeneratedKernel, LowerError};
use unpopped_vocab::{ElementKind, TargetId};

/// The HIP backend.
#[derive(Copy, Clone, Debug, Default)]
pub struct Hip;

impl Backend for Hip {
    fn name(&self) -> &str {
        "hip"
    }

    fn provider(&self) -> &str {
        "baracuda"
    }

    /// Stage 1 teaches the kernel layer only; a plan is declined, never lowered through a default.
    fn lower(&self, plan: &unpopped::plan::KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        let _ = plan;
        Err(LowerError::UnsupportedPlanShape {
            detail: "baracuda-hip-emit stage 1 lowers kernel-layer kernels (`lower_kernel`); the plan-based `lower` is not taught yet"
                .to_string(),
        })
    }

    fn lower_kernel(
        &self,
        kernel: &unpopped::kernel::KernelFn,
    ) -> Result<GeneratedKernel, LowerError> {
        kernel::lower_kernel(kernel)
    }

    /// No dtype is claimed for the plan path (it is not taught); the kernel layer states its own table by name.
    fn supports_dtype(&self, _dtype: ElementKind, _target: TargetId) -> DtypeSupport {
        DtypeSupport::Unsupported
    }
}
