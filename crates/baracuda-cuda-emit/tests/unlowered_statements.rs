// SPDX-License-Identifier: MIT OR Apache-2.0
//! A statement the CUDA backend does not lower is refused by the EMITTER, with the statement named, and never by default. Each fixture
//! passes `check_kernel` first (a checker refusal would carry its own text and make the assertion vacuous, as fishstringer found for its
//! Wgmma/Tcgen05 declines), so the text asserted here can only come from the emitter's fall-through arm. Source-shape only; nothing runs.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::Backend;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace, SubgroupSize};
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

/// ROADMAP S12a-4: `Tcgen05Ld` lowers (see `tcgen05_alloc.rs`), on a launch of 32-lane subgroups. The emitter still refuses a subgroup range
/// that does not contain 32 (an NVIDIA subgroup is a warp) and a cluster launch, by name; the fixtures pass `check_kernel`, so the text
/// asserted is the emitter's, not an `ill-formed` one.
#[test]
fn a_subgroup_launch_without_32_and_a_cluster_launch_are_refused_by_the_emitter() {
    let k = kernel(vec![Stmt::Return]).with_launch({
        let mut l = LaunchShape::new([128, 1, 1], [1, 1, 1]);
        l.subgroup = Some(SubgroupSize::new(64, 64));
        l
    });
    check_kernel(&k).expect("well-formed");
    let d = format!(
        "{:?}",
        Cuda.lower_kernel(&k).expect_err("a 64-lane subgroup")
    );
    assert!(
        d.contains("subgroup launch of 64..=64") && !d.contains("ill-formed"),
        "{d}"
    );
    // Control: a subgroup range that contains 32 lowers.
    let k = kernel(vec![Stmt::Return]).with_launch({
        let mut l = LaunchShape::new([128, 1, 1], [1, 1, 1]);
        l.subgroup = Some(SubgroupSize::new(32, 32));
        l
    });
    check_kernel(&k).expect("well-formed");
    Cuda.lower_kernel(&k)
        .expect("a 32-lane subgroup launch lowers");
    // A cluster launch is still refused.
    let k = kernel(vec![Stmt::Return]).with_launch({
        let mut l = LaunchShape::new([128, 1, 1], [2, 1, 1]);
        l.cluster = Some([2, 1, 1]);
        l
    });
    if check_kernel(&k).is_ok() {
        let d = format!("{:?}", Cuda.lower_kernel(&k).expect_err("a cluster launch"));
        assert!(d.contains("cluster") && !d.contains("ill-formed"), "{d}");
    }
}
