// SPDX-License-Identifier: MIT OR Apache-2.0
//! Driver initialization helpers.
//!
//! CUDA requires `cuInit(0)` before any other driver call. `baracuda-driver`
//! will call it automatically on first use of [`crate::Device::get`] and
//! friends, but you may also call [`init`] yourself (e.g. at process
//! start-up) to fail fast when CUDA is unavailable.

use core::sync::atomic::{AtomicBool, Ordering};

use baracuda_cuda_sys::driver;

use crate::error::{Result, check};

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Ensure `cuInit(0)` has been called. Idempotent and thread-safe.
pub fn init() -> Result<()> {
    if INITIALIZED.load(Ordering::Acquire) {
        return Ok(());
    }
    let d = driver()?;
    // A device test outside `gpu-run` is refused here, after the driver library loaded (so a machine without CUDA never reaches it) and
    // before `cuInit` (so the refusal does not touch the device). `test-support` is a dev-dependency feature: production builds do not have it.
    #[cfg(feature = "test-support")]
    crate::test_support::require_gpu_run_lock();
    let cu = d.cu_init()?;
    // SAFETY: `cuInit` takes a flags word; NVIDIA reserves all bits (pass 0).
    check(unsafe { cu(0) })?;
    INITIALIZED.store(true, Ordering::Release);
    Ok(())
}

/// Driver version exposed by the installed `libcuda`, e.g. `CudaVersion::CUDA_12_6`.
pub fn version() -> Result<baracuda_types::CudaVersion> {
    init()?;
    let d = driver()?;
    let cu = d.cu_driver_get_version()?;
    let mut raw: core::ffi::c_int = 0;
    check(unsafe { cu(&mut raw) })?;
    Ok(baracuda_types::CudaVersion::from_raw(raw as u32))
}
