// SPDX-License-Identifier: MIT OR Apache-2.0
//! A statement the CUDA backend does not lower is refused by the EMITTER, with the statement named, and never by default. Each fixture
//! passes `check_kernel` first (a checker refusal would carry its own text and make the assertion vacuous, as fishstringer found for its
//! Wgmma/Tcgen05 declines), so the text asserted here can only come from the emitter's fall-through arm. Source-shape only; nothing runs.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::kernel::{Block, KernelFn, LocalDecl, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "u",
        ElementKind::I32,
        vec![],
        vec![LocalDecl::scalar(ElementKind::I32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([128, 1, 1], [1, 1, 1]))
}

/// The same kernel with a declared tensor-memory allocation (128 lanes x 32 columns of f32), the one `Tcgen05Alloc` / `Tcgen05Dealloc` name.
fn kernel_with_tmem(body: Vec<Stmt>) -> KernelFn {
    kernel(body).with_allocs(vec![AllocDecl::new(
        "tmem",
        MemorySpace::Tensor,
        ElementKind::F32,
        Extent::Static(128 * 32),
    )])
}

#[test]
fn unlowered_statements_are_refused_by_the_emitter_and_named() {
    let cases = [
        (Stmt::WgmmaCommit, "WgmmaCommit"),
        (Stmt::WgmmaWait { pending: 0 }, "WgmmaWait"),
        (Stmt::GridDepWait, "GridDepWait"),
        // unpopped 0.58.0 (ROADMAP U8c slice 4): `tcgen05.relinquish_alloc_permit` needs no allocation, so it reaches the statement arm.
        (Stmt::Tcgen05RelinquishPermit, "Tcgen05RelinquishPermit"),
    ];
    let mut reached = Vec::new();
    for (stmt, name) in cases {
        let k = kernel(vec![stmt]);
        if check_kernel(&k).is_err() {
            // The checker refuses this fixture itself (e.g. it needs a warpgroup launch): it cannot prove the emitter arm, so it is skipped
            // and counted, never asserted. The test below requires at least one fixture to get past the checker.
            continue;
        }
        let e = Cuda
            .lower_kernel(&k)
            .expect_err("an unlowered statement must be refused");
        let d = format!("{e:?}");
        assert!(
            d.contains("is not lowered by the CUDA backend yet") && d.contains(name),
            "{name}: refusal does not come from the emitter's fall-through arm: {d}"
        );
        reached.push(name);
    }
    assert!(
        !reached.is_empty(),
        "no fixture passed the checker, so the emitter arm was never reached (vacuous)"
    );
    // The 0.58.0 statement must get past the checker, or its refusal text would be the checker's.
    assert!(
        reached.contains(&"Tcgen05RelinquishPermit"),
        "Tcgen05RelinquishPermit: the fixture did not pass `check_kernel`, so the emitter's refusal was not reached"
    );
}

/// Control: the same harness on a statement the emitter DOES lower must not be refused.
#[test]
fn a_lowered_statement_is_not_refused() {
    let k = kernel(vec![Stmt::Return]);
    check_kernel(&k).expect("well-formed");
    Cuda.lower_kernel(&k).expect("Return lowers");
}

/// `Tcgen05Alloc` / `Tcgen05Dealloc` act on a tensor-memory allocation, and the backend does not lower tensor memory (nor `Tcgen05Mma` /
/// `Tcgen05Ld`, the statements that use it): the emitter refuses the kernel at the allocation, naming it, before it reaches the statements.
/// Each fixture passes `check_kernel`, so the text asserted (an `allocation ... is not lowered yet` refusal, not `ill-formed`) is the emitter's.
#[test]
fn tensor_memory_allocations_are_refused_by_the_emitter_and_named() {
    for (stmt, name) in [
        (Stmt::Tcgen05Alloc { d: 0, ncols: 32 }, "Tcgen05Alloc"),
        (Stmt::Tcgen05Dealloc { d: 0, ncols: 32 }, "Tcgen05Dealloc"),
    ] {
        let k = kernel_with_tmem(vec![stmt]);
        check_kernel(&k)
            .unwrap_or_else(|e| panic!("{name}: the fixture must pass the checker: {e:?}"));
        let e = Cuda
            .lower_kernel(&k)
            .expect_err("tensor memory must be refused");
        let d = format!("{e:?}");
        assert!(
            d.contains("allocation `tmem`")
                && d.contains("is not lowered yet")
                && !d.contains("ill-formed"),
            "{name}: refusal does not come from the emitter's allocation arm: {d}"
        );
    }
}
