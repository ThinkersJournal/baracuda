// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exclusive warp scans (unpopped 0.58.4: `Max`, `Min` and `Prod` joined `Sum`/`And`/`Or`/`Xor` in the oracle, lane 0 = the fold's identity:
//! `-inf`/`+inf`/1 on a float, the type's minimum/maximum/1 on an integer).
//!
//! The CUDA lowering is generic over the scan kind only where the FOLD is: an integer `Max` / `Min` lowers (this change adds the identity,
//! the type's least / greatest value); a float `Max` / `Min` and every `Prod` are refused BY NAME by the fold, before the identity is reached
//! (a float's NaN rule is not pinned down; there is no `Prod` fold). Nothing is a silent default. Each fixture passes `check_kernel`, so
//! every refusal asserted is the emitter's.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped_vocab::ElementKind;

/// One warp: every lane contributes the constant 3 of type `t` and stores its exclusive scan under `op`.
fn scan_kernel(t: ElementKind, op: ReduceOp, exclusive: bool) -> KernelFn {
    let init = if matches!(t, ElementKind::F32) {
        ScalarExpr::Const(3.0)
    } else {
        ScalarExpr::ConstInt(3, t)
    };
    KernelFn::new(
        "scan",
        t,
        vec![Param::buffer("out", t, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(t), LocalDecl::scalar(t)],
        Block(vec![
            Stmt::Let { local: 0, init },
            Stmt::Collective {
                kind: CollectiveKind::Scan { op, exclusive },
                value: ScalarExpr::Local(0),
                result: 1,
                scope: Scope::Subgroup,
            },
            Stmt::Store {
                place: Place::Logical {
                    buffer: 0,
                    coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
                },
                value: ScalarExpr::Local(1),
            },
        ]),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]))
}

fn lowered(t: ElementKind, op: ReduceOp, exclusive: bool) -> String {
    let k = scan_kernel(t, op, exclusive);
    check_kernel(&k)
        .unwrap_or_else(|e| panic!("{t:?} {op:?}: fixture refused by the checker: {e:?}"));
    Cuda.lower_kernel(&k)
        .unwrap_or_else(|e| panic!("{t:?} {op:?} exclusive={exclusive} must lower: {e:?}"))
        .source
}

fn refusal(t: ElementKind, op: ReduceOp) -> String {
    let k = scan_kernel(t, op, true);
    check_kernel(&k)
        .unwrap_or_else(|e| panic!("{t:?} {op:?}: fixture refused by the checker: {e:?}"));
    match Cuda.lower_kernel(&k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(
                !detail.contains("ill-formed"),
                "checker text, not the emitter's: {detail}"
            );
            detail
        }
        Err(other) => panic!("untyped refusal: {other:?}"),
        Ok(g) => panic!(
            "{t:?} {op:?} exclusive must be refused, got source:\n{}",
            g.source
        ),
    }
}

#[test]
fn an_exclusive_integer_max_min_scan_hands_lane_zero_the_types_extreme() {
    use ElementKind::*;
    for (t, min_lit, max_lit) in [
        (I32, "(-2147483647 - 1)", "2147483647"),
        (I64, "(-9223372036854775807ll - 1)", "9223372036854775807ll"),
        (U32, "0", "4294967295u"),
    ] {
        let s = lowered(t, ReduceOp::Max, true);
        assert!(
            s.contains(min_lit),
            "{t:?} exclusive Max: lane 0 must get the type's minimum {min_lit}:
{s}"
        );
        let s = lowered(t, ReduceOp::Min, true);
        assert!(
            s.contains(max_lit),
            "{t:?} exclusive Min: lane 0 must get the type's maximum {max_lit}:\n{s}"
        );
    }
}

/// The shift that makes a scan exclusive is emitted only when asked for: the inclusive scan of the same kernel has no identity in its source.
#[test]
fn an_inclusive_scan_carries_no_identity() {
    let s = lowered(ElementKind::I32, ReduceOp::Max, false);
    assert!(
        !s.contains("(-2147483647 - 1)"),
        "control: an inclusive Max has no lane-0 identity:\n{s}"
    );
    let s = lowered(ElementKind::I32, ReduceOp::Max, true);
    assert!(
        s.contains("(-2147483647 - 1)") && s.contains("__shfl_up_sync(0xffffffffu, _v, 1u, 32)"),
        "an exclusive Max shifts by a lane and hands lane 0 the minimum:\n{s}"
    );
}

#[test]
fn float_max_min_and_every_prod_exclusive_scan_are_refused_by_name() {
    use ElementKind::*;
    let d = refusal(F32, ReduceOp::Max);
    assert!(d.contains("float `Max` collective"), "{d}");
    let d = refusal(F32, ReduceOp::Min);
    assert!(d.contains("float `Min` collective"), "{d}");
    for t in [F32, I32, U32] {
        let d = refusal(t, ReduceOp::Prod);
        assert!(
            d.contains("`Prod`"),
            "{t:?}: the Prod refusal must name the operation: {d}"
        );
    }
}

/// The emitted source must be real CUDA: compile it with NVRTC for sm_89 (PTX only, nothing runs).
#[cfg(all(feature = "nvrtc", feature = "device-tests"))]
mod nvrtc {
    use super::*;
    use baracuda_cuda_emit::NvrtcCompiler;
    use unpopped::jit::Compiler;
    use unpopped_vocab::TargetId;

    #[test]
    fn exclusive_integer_max_min_scans_compile_for_sm89() {
        let c = NvrtcCompiler::new(TargetId::parse("cuda:sm89").expect("token")).expect("nvrtc");
        for t in [ElementKind::I32, ElementKind::I64, ElementKind::U32] {
            for op in [ReduceOp::Max, ReduceOp::Min] {
                let g = Cuda
                    .lower_kernel(&scan_kernel(t, op, true))
                    .expect("lowers");
                c.compile(&g.source, &g.name, 10_000)
                    .unwrap_or_else(|e| panic!("{t:?} {op:?}: {e}\n{}", g.source));
            }
        }
    }
}
