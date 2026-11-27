// SPDX-License-Identifier: MIT OR Apache-2.0
//! The published `unpopped::fixtures` (unpopped 0.58.5): ONE checker-valid kernel or module per `Stmt` variant. Every fixture passes
//! `check_kernel` / `check_module`, so a refusal the emitter returns for one carries the EMITTER's own text, never the checker's (the
//! vacuity that made the hand-built `named_barrier` / `proxy_fence` declines prove only the checker, see `unlowered_statements.rs`).
//!
//! Census: each variant either lowers or is refused with a typed `UnsupportedKernel` that is not an `ill-formed` (checker) refusal, and the
//! set of refused variants is pinned below, so a statement that starts or stops lowering is a test change, never a silent default. A new
//! `Stmt` variant breaks the `unpopped` fixtures build until it has a fixture, and then this census until it is classified.
//!
//! Source-shape only (`lower_kernel`); nothing runs. A `Module` fixture is lowered at its entry function (the backend has no module entry
//! point, so a `Call` to a callee is the statement under test there).

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, LowerError};
use unpopped::fixtures::{Fixture, fixtures};
use unpopped::kernel::KernelFn;

/// Where the emitter stops a refused fixture: its own statement arm, the launch shape, or an allocation it cannot spell. Only the first
/// proves the STATEMENT's arm; the other two refuse earlier (the published fixture of a warpgroup / tensor-memory / barrier statement carries the
/// launch or allocation that statement needs), so for those the statement arm is covered by the hand-built kernels of `unlowered_statements.rs`.
const STATEMENT: &str = "is not lowered by the CUDA backend yet";
const LAUNCH: &str = "a subgroup, cluster or multi-device launch is not lowered yet";
const ALLOC: &str = "has no CUDA spelling";

/// Variants the CUDA backend refuses today, with where it stops them (fixtures-list order).
const REFUSED: &[(&str, &str)] = &[
    ("MaskedLoad", STATEMENT),
    ("MaskedStore", STATEMENT),
    ("AtomicRmw", STATEMENT),
    ("AtomicCas", STATEMENT),
    ("MulticastLoadReduce", LAUNCH),
    ("GridDepWait", STATEMENT),
    ("GridDepLaunchDependents", STATEMENT),
    ("Fence", STATEMENT),
    ("Collective", LAUNCH),
    ("Mma", LAUNCH),
    ("MatrixLoad", LAUNCH),
    ("MbarInit", ALLOC),
    ("MbarArrive", ALLOC),
    ("MbarCompleteTx", ALLOC),
    ("MbarWait", ALLOC),
    ("MatrixStore", LAUNCH),
    ("Wgmma", LAUNCH),
    ("WgmmaCommit", LAUNCH),
    ("WgmmaWait", LAUNCH),
    ("Tcgen05Mma", LAUNCH),
    ("Tcgen05Commit", LAUNCH),
    ("Tcgen05Alloc", LAUNCH),
    ("Tcgen05Dealloc", LAUNCH),
    ("Tcgen05RelinquishPermit", LAUNCH),
    ("Tcgen05Ld", LAUNCH),
    ("CopyBulk", ALLOC),
    ("CopyUnpack", ALLOC),
];

/// Fixtures that are modules (a `Call` to a callee, a combining collective): not reachable through `lower_kernel`, see the test.
const MODULE_ONLY: &[&str] = &["CollectiveTuple", "Call", "CallTuple", "ExternCall"];

fn entry(f: &Fixture) -> &KernelFn {
    match f {
        Fixture::Kernel(k) => k,
        Fixture::Module { module, entry } => &module.fns[*entry as usize],
        other => panic!("a new Fixture shape needs a decision here: {other:?}"),
    }
}

#[test]
fn every_statement_variant_lowers_or_is_refused_by_the_emitter() {
    let mut refused = Vec::new();
    let mut lowered = Vec::new();
    let mut module_only = Vec::new();
    for (name, fx) in fixtures() {
        fx.check().unwrap_or_else(|e| {
            panic!("{name}: the published fixture must pass the checker: {e:?}")
        });
        let is_module = matches!(fx, Fixture::Module { .. });
        match Cuda.lower_kernel(entry(&fx)) {
            // A module fixture's entry function on its own is not a well-formed kernel (its callee lives in the module), and the backend has no
            // module entry point: it cannot reach the emitter. Such a fixture must be refused, and is listed apart, never counted as a decline.
            Err(LowerError::UnsupportedKernel { detail }) if is_module => {
                assert!(
                    detail.contains("ill-formed"),
                    "{name}: a module entry is expected to be refused by the checker alone: {detail}"
                );
                module_only.push(name);
            }
            Ok(_) if is_module => panic!("{name}: a module entry lowered without its module"),
            Ok(_) => lowered.push(name),
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(
                    !detail.contains("ill-formed"),
                    "{name}: this is the checker's refusal, not the emitter's: {detail}"
                );
                let class = [STATEMENT, LAUNCH, ALLOC]
                    .into_iter()
                    .find(|c| detail.contains(c))
                    .unwrap_or_else(|| panic!("{name}: an unclassified emitter refusal: {detail}"));
                if class == STATEMENT {
                    assert!(
                        detail.contains(name),
                        "{name}: the refusal must name the statement: {detail}"
                    );
                }
                refused.push((name, class));
            }
            Err(other) => panic!("{name}: an untyped refusal: {other:?}"),
        }
    }
    // Positive control: the census is not "everything refused" (a broken backend would pass a decline-only test).
    assert!(
        lowered.contains(&"Return") && lowered.contains(&"Let"),
        "control: Return and Let must lower: {lowered:?}"
    );
    assert_eq!(
        module_only, MODULE_ONLY,
        "the set of module-only fixtures changed"
    );
    assert_eq!(refused, REFUSED, "the set of refused statements changed");
}
