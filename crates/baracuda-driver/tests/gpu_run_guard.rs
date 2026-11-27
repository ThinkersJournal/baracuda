// SPDX-License-Identifier: MIT OR Apache-2.0
//! The structural guard of ROADMAP S19: a device test refuses to use the GPU outside `gpu-run` (the machine-wide GPU lock).
//!
//! `baracuda_driver::init()` calls `test_support::require_gpu_run_lock()` after the driver library loads and before `cuInit`, so a test
//! binary started by a bare `cargo test --features device-tests` panics at its first device call, naming `gpu-run`, without having touched
//! the device. This file proves it by running a child test process:
//!
//! * negative control (always, when a CUDA driver library is present): the child, with `GPU_RUN_HELD` and `BARACUDA_ALLOW_UNGATED_GPU`
//!   removed, FAILS and its output names `gpu-run`;
//! * positive control (only when THIS process holds the lock, i.e. runs under `gpu-run` or `cargo gpu-test`): the child, with the marker
//!   passed on, gets through `init()`. It is not run from an ungated parent, because it would be the very ungated GPU use being refused.
//!
//! On a machine with no CUDA driver library `init()` fails before the guard, so the guard is not reached and the test declares its skip.

use baracuda_driver::require_optional;
use std::process::Command;

const CHILD: &str = "guard_child_probe";

/// Runs only in the child process (`BARACUDA_GUARD_CHILD=1`): calls `init()`; elsewhere it does nothing.
#[test]
fn guard_child_probe() {
    if std::env::var_os("BARACUDA_GUARD_CHILD").is_none() {
        return;
    }
    // Reaching `init()` ungated panics (the guard); under the marker it returns.
    let r = baracuda_driver::init();
    println!("INIT-RETURNED ok={}", r.is_ok());
}

fn child(marker: bool) -> std::process::Output {
    let exe = std::env::current_exe().expect("this test binary");
    let mut c = Command::new(exe);
    c.args([CHILD, "--exact", "--nocapture", "--test-threads=1"])
        .env("BARACUDA_GUARD_CHILD", "1")
        .env_remove("BARACUDA_ALLOW_UNGATED_GPU");
    if marker {
        c.env("GPU_RUN_HELD", "1");
    } else {
        c.env_remove("GPU_RUN_HELD");
    }
    c.output().expect("the child test process runs")
}

#[test]
fn a_device_test_outside_gpu_run_is_refused_by_name() {
    require_optional!(
        baracuda_cuda_sys::driver::driver().map(|_| ()),
        "a CUDA driver library (without one init() fails before the guard)"
    );
    let o = child(false);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(
        !o.status.success(),
        "the child must be refused without the marker:\n{text}"
    );
    assert!(
        text.contains("refusing to use the GPU outside gpu-run") && text.contains("gpu-run"),
        "the refusal must name gpu-run:\n{text}"
    );
    assert!(
        !text.contains("INIT-RETURNED"),
        "init() must not have returned:\n{text}"
    );
}

#[test]
fn a_device_test_under_gpu_run_gets_through() {
    require_optional!(
        baracuda_cuda_sys::driver::driver().map(|_| ()),
        "a CUDA driver library"
    );
    // The positive control touches the GPU, so only a parent that holds the lock may run it.
    require_optional!(
        (std::env::var("GPU_RUN_HELD").as_deref() == Ok("1")).then_some(()),
        "this process under gpu-run (GPU_RUN_HELD=1): the positive control would otherwise be an ungated GPU use"
    );
    let o = child(true);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(
        o.status.success(),
        "the child must pass with the marker:\n{text}"
    );
    assert!(
        text.contains("INIT-RETURNED"),
        "init() must return:\n{text}"
    );
}
