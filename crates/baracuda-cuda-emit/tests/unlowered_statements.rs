// SPDX-License-Identifier: MIT OR Apache-2.0
//! A statement the CUDA backend does not lower is refused by the EMITTER, with the statement named, and never by default. Each fixture
//! passes `check_kernel` first (a checker refusal would carry its own text and make the assertion vacuous, as fishstringer found for its
//! Wgmma/Tcgen05 declines), so the text asserted here can only come from the emitter's fall-through arm. Source-shape only; nothing runs.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace, SubgroupSize};
use unpopped::ir::ScalarExpr;
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
}

/// Control: the same harness on a statement the emitter DOES lower must not be refused.
#[test]
fn a_lowered_statement_is_not_refused() {
    let k = kernel(vec![Stmt::Return]);
    check_kernel(&k).expect("well-formed");
    Cuda.lower_kernel(&k).expect("Return lowers");
}

/// ROADMAP S12a-1: `Tcgen05Alloc`, `Tcgen05Dealloc` and `Tcgen05RelinquishPermit` and the `Tensor` allocation they act on now LOWER (see
/// `tcgen05_alloc.rs`). This is the control for the refusal below: the same fixtures are accepted.
#[test]
fn the_tensor_memory_management_statements_are_lowered() {
    for (stmt, want) in [
        (Stmt::Tcgen05Alloc { d: 0, ncols: 32 }, "tcgen05.alloc"),
        (Stmt::Tcgen05Dealloc { d: 0, ncols: 32 }, "tcgen05.dealloc"),
        (
            Stmt::Tcgen05RelinquishPermit,
            "tcgen05.relinquish_alloc_permit",
        ),
    ] {
        let k = kernel_with_tmem(vec![stmt]);
        check_kernel(&k).unwrap_or_else(|e| panic!("{want}: the fixture must pass: {e:?}"));
        let g = Cuda
            .lower_kernel(&k)
            .unwrap_or_else(|e| panic!("{want} must lower: {e:?}"));
        assert!(g.source.contains(want), "{want}:\n{}", g.source);
    }
}

/// `Tcgen05Ld` (and `Tcgen05Mma`, `Tcgen05Commit`) are not lowered yet (S12a-3 to S12a-5). The fixture passes `check_kernel`, so the refusal asserted is
/// the emitter's, not an `ill-formed` one.
#[test]
fn the_other_tcgen05_statements_are_refused_by_the_emitter_and_named() {
    let k = kernel_with_tmem(vec![Stmt::Tcgen05Ld {
        src: 0,
        col: ScalarExpr::ConstInt(0, ElementKind::U32),
        count: 4,
        dst: 1,
    }])
    .with_allocs(vec![
        AllocDecl::new(
            "tmem",
            MemorySpace::Tensor,
            ElementKind::F32,
            Extent::Static(128 * 32),
        ),
        AllocDecl::new(
            "regs",
            MemorySpace::Private,
            ElementKind::F32,
            Extent::Static(4),
        ),
    ])
    .with_launch({
        // A tensor-memory load needs a one-dimensional launch of 32-lane subgroups (the checker's rule).
        let mut l = LaunchShape::new([128, 1, 1], [1, 1, 1]);
        l.subgroup = Some(SubgroupSize::new(32, 32));
        l
    });
    check_kernel(&k)
        .unwrap_or_else(|e| panic!("Tcgen05Ld: the fixture must pass the checker: {e:?}"));
    let e = Cuda
        .lower_kernel(&k)
        .expect_err("Tcgen05Ld is not lowered yet");
    let d = format!("{e:?}");
    // The emitter refuses it, by name, before the statement: the checker's rule makes a tensor-memory load need a launch of 32-lane
    // subgroups, and the emitter does not lower a subgroup launch yet (a prerequisite of S12a-4, with the `Private` allocation `regs`).
    assert!(
        !d.contains("ill-formed") && d.contains("subgroup"),
        "refusal does not come from the emitter: {d}"
    );
}
