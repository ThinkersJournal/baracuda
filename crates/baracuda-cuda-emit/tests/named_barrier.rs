// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hotfix: a `Barrier` of a *named* kind (`bar.sync`/`bar.arrive id, count`) was lowered as `__syncthreads()`, because
//! `Stmt::Barrier { scope, .. }` ignored the kind. A named barrier waits for `count` threads, not the whole block, so the
//! kernel synchronised the wrong set (over-sync or deadlock). The lowering is now kind-aware and refuses what it does not lower.
//! Source-shape tests need no GPU; nothing runs here.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::Backend;
use unpopped::exec::{LaunchShape, Scope};
use unpopped::kernel::{BarrierKind, Block, KernelFn, LocalDecl, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "b",
        ElementKind::I32,
        vec![],
        vec![LocalDecl::scalar(ElementKind::I32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([128, 1, 1], [1, 1, 1]))
}

fn barrier(scope: Scope, kind: BarrierKind) -> Stmt {
    Stmt::Barrier { scope, kind }
}

#[test]
fn named_barriers_are_spelled_and_are_not_syncthreads() {
    let k = kernel(vec![
        barrier(Scope::Block, BarrierKind::NamedArrive { id: 2, count: 64 }),
        barrier(Scope::Block, BarrierKind::Named { id: 2, count: 64 }),
    ]);
    check_kernel(&k).expect("the fixture is well-formed");
    let g = Cuda.lower_kernel(&k).expect("named barriers lower");
    for want in [
        "asm volatile(\"bar.arrive %0, %1;\" :: \"n\"(2), \"n\"(64) : \"memory\");",
        "asm volatile(\"bar.sync %0, %1;\" :: \"n\"(2), \"n\"(64) : \"memory\");",
    ] {
        assert!(g.source.contains(want), "missing `{want}`:\n{}", g.source);
    }
    assert!(!g.source.contains("__syncthreads"), "{}", g.source);
}

/// Control (the query can say "no"): the full block barrier is still `__syncthreads()`.
#[test]
fn the_full_block_barrier_is_still_syncthreads() {
    let k = kernel(vec![barrier(Scope::Block, BarrierKind::Full)]);
    check_kernel(&k).expect("well-formed");
    let g = Cuda.lower_kernel(&k).expect("lowers");
    assert!(g.source.contains("__syncthreads();"), "{}", g.source);
    assert!(!g.source.contains("bar.sync"), "{}", g.source);
}

/// A malformed named barrier (id above 15, count not whole warps, count above the block) never reaches a lowering: the IR's own
/// checker refuses it. The emitter repeats the same checks (`named_barrier`) as a second line, because it must never print
/// `__syncthreads()` for any named kind; those arms are unreachable through `check_kernel` and so are not exercised here.
/// Positive control: a well-formed named barrier passes the same checker (see the first test).
#[test]
fn the_checker_refuses_a_malformed_named_barrier() {
    for kind in [
        BarrierKind::Named { id: 16, count: 64 },
        BarrierKind::Named { id: 1, count: 48 },
        BarrierKind::Named { id: 1, count: 256 },
    ] {
        let k = kernel(vec![barrier(Scope::Block, kind)]);
        assert!(
            check_kernel(&k).is_err(),
            "{kind:?} should be refused by the checker"
        );
    }
}

/// `lower_kernel` runs the IR checker first, so a malformed named barrier is refused there (`ill-formed`, naming the barrier rule) and the
/// emitter's own `named_barrier` checks are a defensive second line no public call reaches. Nothing is asserted about their text (it
/// would be vacuous: the checker's text would satisfy it); what is asserted is that none of these kernels ever lowers.
#[test]
fn a_malformed_named_barrier_never_lowers() {
    for kind in [
        BarrierKind::Named { id: 16, count: 64 },
        BarrierKind::Named { id: 1, count: 48 },
        BarrierKind::Named { id: 1, count: 256 },
        BarrierKind::NamedArrive { id: 1, count: 0 },
    ] {
        let k = kernel(vec![barrier(Scope::Block, kind)]);
        let e = Cuda
            .lower_kernel(&k)
            .expect_err("a malformed named barrier must not lower");
        let d = format!("{e:?}");
        assert!(
            d.contains("ill-formed") && d.contains("NamedBarrierUnsupported"),
            "{kind:?}: refused by the checker, got: {d}"
        );
    }
}
