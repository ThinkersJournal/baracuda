// SPDX-License-Identifier: MIT OR Apache-2.0
//! A statement the CUDA backend does not lower is refused by the EMITTER, with the statement named, and never by default. Each fixture
//! passes `check_kernel` first (a checker refusal would carry its own text and make the assertion vacuous, as fishstringer found for its
//! Wgmma/Tcgen05 declines), so the text asserted here can only come from the emitter's fall-through arm. Source-shape only; nothing runs.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::Backend;
use unpopped::exec::LaunchShape;
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

#[test]
fn unlowered_statements_are_refused_by_the_emitter_and_named() {
    let cases = [
        (Stmt::WgmmaCommit, "WgmmaCommit"),
        (Stmt::WgmmaWait { pending: 0 }, "WgmmaWait"),
        (Stmt::GridDepWait, "GridDepWait"),
    ];
    let mut reached = 0;
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
        reached += 1;
    }
    assert!(
        reached > 0,
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
