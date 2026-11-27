// SPDX-License-Identifier: MIT OR Apache-2.0
//! Test-support: the [`require!`] primitive for on-device tests.
//!
//! Replaces the ad-hoc `if probe().is_err() { return; }` / `let Some(x) = probe()
//! else { return }` idiom that let a `#[test]` clear its gate, *run*, and report
//! `ok` having asserted nothing — the "silent skip" that makes a test that didn't
//! execute indistinguishable from one that ran and agreed.
//!
//! A [`require!`] that can't get its resource does one of two things instead of
//! returning silently:
//!
//! * **Fails loud** when `BARACUDA_GPU_REQUIRED` is set — which `cargo gpu-test`
//!   does on the box where the evidence MUST exist. On that machine a would-be
//!   skip becomes a hard failure: *absence would have failed*, which is the claim
//!   CI needs to make rather than "green, and the device happened to be present."
//! * **Declares the skip** otherwise (a developer's non-GPU laptop): it writes a
//!   machine-discoverable `SKIP-DECLARED` record that `cargo gpu-test` tallies into
//!   "N declared skippable, N skipped" — honest, not a silent `ok`.
//!
//! Gated behind the `test-support` feature (a dev-dependency only), so it costs
//! production builds nothing.
//!
//! ```ignore
//! use baracuda_driver::require;
//! #[test]
//! fn my_gpu_test() {
//!     let stream = require!(setup_stream(), "CUDA device + stream");
//!     // ... assertions that only make sense with a device ...
//! }
//! ```
//! For a resource that may legitimately be absent even on the box (optional
//! runtimes — NVSHMEM, multi-GPU, cuFile/GDS, TensorRT, ...), use
//! [`require_optional!`], which declares-and-skips but never fails loud.

/// Uniform "is the resource present?" over `Result` and `Option`, so [`require!`]
/// accepts either kind of probe.
pub trait Present<T> {
    /// The resource if present, else `None`.
    fn present(self) -> Option<T>;
}

impl<T> Present<T> for Option<T> {
    fn present(self) -> Option<T> {
        self
    }
}

impl<T, E> Present<T> for ::core::result::Result<T, E> {
    fn present(self) -> Option<T> {
        self.ok()
    }
}

/// Backing function for [`require!`] / [`require_optional!`] when a resource is
/// absent. `critical` = the resource must exist where evidence is required (a
/// bare GPU), so a set `BARACUDA_GPU_REQUIRED` turns absence into a hard failure.
/// Not called directly — use the macros.
#[doc(hidden)]
pub fn __declined(reason: &str, critical: bool) {
    // cargo names each test thread with the test's full path — the honest "where".
    let where_ = ::std::thread::current()
        .name()
        .map(str::to_owned)
        .unwrap_or_else(|| "<unknown test>".to_owned());

    if critical && ::std::env::var_os("BARACUDA_GPU_REQUIRED").is_some() {
        panic!(
            "require!: {reason} — resource absent, but BARACUDA_GPU_REQUIRED is set. This test \
             MUST run where the evidence exists (the GPU box); a silent skip here would hide that \
             the resource is missing. [{where_}]"
        );
    }

    // Tool-discoverable declaration. Preferred sink is the file named by
    // BARACUDA_SKIP_LOG (set by `cargo gpu-test`) — robust and parallel-safe via
    // append; falls back to stdout for a human running the test directly.
    let record = format!("SKIP-DECLARED\t{where_}\t{reason}\n");
    if let Some(path) = ::std::env::var_os("BARACUDA_SKIP_LOG") {
        use ::std::io::Write as _;
        if let Ok(mut f) = ::std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = f.write_all(record.as_bytes());
            return;
        }
    }
    print!("{record}");
}

/// Whether this process may use the GPU: it runs under `gpu-run` (which exports `GPU_RUN_HELD=1` into its child, so every test binary
/// that `cargo gpu-test` launches has it) or the caller said so with `BARACUDA_ALLOW_UNGATED_GPU=1` (a machine with no other lane on the
/// GPU). Pure, so the rule is unit-tested without a GPU.
#[must_use]
pub fn gpu_run_marker_ok(held: Option<&str>, allow: Option<&str>) -> bool {
    held == Some("1") || allow == Some("1")
}

/// Refuse to touch the GPU outside `gpu-run`: the GPU is one machine-wide resource shared by every lane (the `gpu-run` lock), and a device
/// test started by a bare `cargo test --features device-tests` used it without the lock (twice, 2026-11-27, disclosed). Called by
/// [`crate::init`] once the driver library has loaded and BEFORE `cuInit`, so the refusal itself never touches the device, and a machine
/// with no CUDA driver (hosted CI, a laptop without NVIDIA) never reaches it. Only compiled with the `test-support` feature, a
/// dev-dependency, so a production build never has it.
#[doc(hidden)]
pub fn require_gpu_run_lock() {
    let held = ::std::env::var("GPU_RUN_HELD").ok();
    let allow = ::std::env::var("BARACUDA_ALLOW_UNGATED_GPU").ok();
    if gpu_run_marker_ok(held.as_deref(), allow.as_deref()) {
        return;
    }
    let where_ = ::std::thread::current()
        .name()
        .map(str::to_owned)
        .unwrap_or_else(|| "<unknown test>".to_owned());
    panic!(
        "refusing to use the GPU outside gpu-run (GPU_RUN_HELD is not 1): the GPU is shared by every lane through one machine-wide \
         lock. Run `cargo gpu-test -p <crate> --features device-tests ...` (it compiles unlocked, then takes the lock per test binary) \
         or `pwsh scripts/gpu-run.ps1 -Project <lane> -- <test binary>` after `cargo test --no-run`. A machine with no other user of \
         the GPU can set BARACUDA_ALLOW_UNGATED_GPU=1. [{where_}]"
    );
}

/// Require a resource for an on-device `#[test]` (which returns `()`). `$probe`
/// is any `Result` or `Option`.
///
/// * present → yields the value; the test continues to its assertions.
/// * absent + `BARACUDA_GPU_REQUIRED` (the box) → **panics** (absence-would-fail).
/// * absent otherwise → declares a tool-discoverable skip and `return`s.
///
/// For a resource that may be absent even on the box, use [`require_optional!`].
#[macro_export]
macro_rules! require {
    ($probe:expr, $reason:expr $(,)?) => {
        match $crate::test_support::Present::present($probe) {
            ::core::option::Option::Some(v) => v,
            ::core::option::Option::None => {
                $crate::test_support::__declined($reason, true);
                return;
            }
        }
    };
}

/// Like [`require!`], but for a resource that may legitimately be absent even
/// where evidence is required (optional runtimes / extra hardware). Always
/// declares-and-skips on absence; never fails loud. Still tool-discoverable, so an
/// off-hardware run reports it rather than a silent `ok`.
#[macro_export]
macro_rules! require_optional {
    ($probe:expr, $reason:expr $(,)?) => {
        match $crate::test_support::Present::present($probe) {
            ::core::option::Option::Some(v) => v,
            ::core::option::Option::None => {
                $crate::test_support::__declined($reason, false);
                return;
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::Present;

    #[test]
    fn present_normalizes_option_and_result() {
        assert_eq!(Some(5).present(), Some(5));
        assert_eq!(None::<i32>.present(), None);
        assert_eq!(Ok::<i32, ()>(5).present(), Some(5));
        assert_eq!(Err::<i32, ()>(()).present(), None);
    }

    #[test]
    fn require_yields_the_value_on_the_present_path() {
        // The present path never touches the env / skip machinery, so this is a
        // pure check that the macro plumbing compiles and unwraps both kinds.
        let v = crate::require!(Some(42u32), "always present");
        assert_eq!(v, 42);
        let w = crate::require!(Ok::<_, ()>(7i32), "always present");
        assert_eq!(w, 7);
        // (The absent paths — declared-skip and fail-loud-under-BARACUDA_GPU_REQUIRED
        // — are validated end-to-end by `cargo gpu-test`, since they read the
        // environment and return from the test; unit-testing them would race on
        // process-global env.)
    }

    #[test]
    fn the_gpu_run_marker_rule() {
        use super::gpu_run_marker_ok as ok;
        assert!(ok(Some("1"), None), "gpu-run exports GPU_RUN_HELD=1");
        assert!(ok(None, Some("1")), "an explicit opt-out");
        assert!(ok(Some("1"), Some("1")));
        // The refusals (so the assertions above can fail): no marker, a wrong value, an empty one.
        assert!(!ok(None, None));
        assert!(!ok(Some("0"), None));
        assert!(!ok(Some(""), Some("0")));
        assert!(
            !ok(Some("true"), None),
            "only the exact `1` gpu-run exports"
        );
    }
}
