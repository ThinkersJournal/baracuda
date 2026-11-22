// SPDX-License-Identifier: MIT OR Apache-2.0
//! The shared error type for the baracuda kernel facade.
//!
//! Defined here so every per-library wrapper and `baracuda-kernels` share one `Error`;
//! `baracuda_cutlass::Error` and `baracuda_kernels::Error` re-export it.

use thiserror::Error;

/// Crate-local result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors raised by the safe kernel wrappers.
///
/// Re-exported as `baracuda_cutlass::Error` and `baracuda_kernels::Error`; the kernel facade returns the same
/// error surface for every op family (the cuSOLVER / cuDNN / cuFFT
/// facades all map their library-native status codes into one of these
/// variants).
///
/// `#[non_exhaustive]` — error variants have grown every couple of
/// phases as new failure modes surface (Phase 7 added cuDNN-status
/// fallback paths; Phase 22 added the cuSOLVER facade plumbing). Match
/// arms must include a `_ =>` catch-all so adding a new variant
/// doesn't break downstream `match e { ... }` blocks.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The requested SKU isn't available in this build.
    ///
    /// Either no arch feature is enabled (no kernels were compiled), or the
    /// problem requires a SKU that's outside the curated v0 set
    /// (non-RCR layout, unsupported epilogue, etc.).
    #[error("baracuda-cutlass: requested kernel is unavailable: {0}")]
    Unsupported(&'static str),

    /// A problem dimension or stride is invalid (e.g., M, N, or K is non-positive).
    #[error("baracuda-cutlass: invalid problem: {0}")]
    InvalidProblem(&'static str),

    /// A pointer or stride is misaligned for the selected kernel's tensor-op
    /// instructions.
    #[error("baracuda-cutlass: misaligned operand")]
    MisalignedOperand,

    /// The provided workspace is too small for the selected plan or
    /// `Workspace::None` was passed when scratch was required.
    #[error("baracuda-cutlass: workspace too small (need {needed} bytes, got {got})")]
    WorkspaceTooSmall {
        /// Required workspace size in bytes.
        needed: usize,
        /// Provided workspace size in bytes.
        got: usize,
    },

    /// A device buffer is too small for the declared matrix shape and stride.
    #[error(
        "baracuda-cutlass: buffer too small for declared shape (need {needed} elements, got {got})"
    )]
    BufferTooSmall {
        /// Minimum elements required for `(rows, cols, ld)`.
        needed: usize,
        /// Elements available in the supplied slice.
        got: usize,
    },

    /// CUTLASS reported an internal error during launch (typically a
    /// kernel-launch failure surfaced through `cudaGetLastError`).
    ///
    /// A code of `1000 + n` means `cudaFuncSetAttribute` failed with CUDA runtime
    /// error `n`; the message then names it (`1035` -> `cudaErrorInsufficientDriver`).
    #[error("baracuda-cutlass: kernel internal error (status code {0}{detail})", detail = kernel_status_detail(.0))]
    KernelInternal(i32),

    /// Underlying baracuda-driver error (context, stream, etc.).
    #[error("baracuda-cutlass: driver error: {0}")]
    Driver(#[from] baracuda_driver::Error),
}

/// Name of a CUDA runtime `cudaError_t` that `cudaFuncSetAttribute` (the call behind a `1000 + n`
/// kernel status) can return. A deliberately small table, kept in this crate because it must not
/// depend on `baracuda-cuda-sys` (the full table is `cudaError_t::name` there); unknown codes
/// return `None` and are printed as the bare number.
pub(crate) const fn cuda_runtime_error_name(code: i32) -> Option<&'static str> {
    Some(match code {
        1 => "cudaErrorInvalidValue",
        2 => "cudaErrorMemoryAllocation",
        3 => "cudaErrorInitializationError",
        9 => "cudaErrorInvalidConfiguration",
        35 => "cudaErrorInsufficientDriver",
        98 => "cudaErrorInvalidDeviceFunction",
        100 => "cudaErrorNoDevice",
        101 => "cudaErrorInvalidDevice",
        209 => "cudaErrorNoKernelImageForDevice",
        700 => "cudaErrorIllegalAddress",
        701 => "cudaErrorLaunchOutOfResources",
        719 => "cudaErrorLaunchFailure",
        999 => "cudaErrorUnknown",
        _ => return None,
    })
}

/// The `: CUDA runtime error n = name` suffix of a `KernelInternal` message, empty for a code below
/// 1000 (a CUTLASS status, not a wrapped CUDA error).
fn kernel_status_detail(code: &i32) -> String {
    match code.checked_sub(1000) {
        Some(n) if n > 0 => match cuda_runtime_error_name(n) {
            Some(name) => format!(": CUDA runtime error {n} = {name}"),
            None => format!(": CUDA runtime error {n}"),
        },
        _ => String::new(),
    }
}

#[cfg(test)]
mod kernel_status_tests {
    use super::*;

    #[test]
    fn code_1035_names_insufficient_driver() {
        let msg = Error::KernelInternal(1035).to_string();
        assert!(msg.contains("status code 1035"), "{msg}");
        assert!(msg.contains("CUDA runtime error 35 = cudaErrorInsufficientDriver"), "{msg}");
    }

    #[test]
    fn unknown_wrapped_code_prints_the_number_and_bare_status_is_unchanged() {
        assert!(Error::KernelInternal(1999).to_string().ends_with("(status code 1999: CUDA runtime error 999 = cudaErrorUnknown)"));
        assert!(Error::KernelInternal(1888).to_string().ends_with("(status code 1888: CUDA runtime error 888)"));
        // control: a bare CUTLASS status gets no suffix
        assert_eq!(Error::KernelInternal(5).to_string(), "baracuda-cutlass: kernel internal error (status code 5)");
        assert_eq!(Error::KernelInternal(1000).to_string(), "baracuda-cutlass: kernel internal error (status code 1000)");
    }
}
