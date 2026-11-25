// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP S1/S2/S7 slice 1: the Hopper statements of Unpopped 0.56.x in `Cuda::lower_kernel`. Each is either **spelled**
//! (the PTX it must print is asserted) or **refused by name** (the decline must contain the statement's name), never
//! approximated. Nothing here needs a GPU: the source-shape tests are plain text, and `ptxas_accepts_sm90` (feature
//! `nvrtc`) assembles for `sm_90a` without running anything. **Hopper/Blackwell EXECUTION of these statements is
//! hardware-gated (the dev box is sm_89) and is not tested here**; semantics (mbarrier phase parity) are the oracle's
//! and are NOT yet cross-checked against hardware (ROADMAP S6).

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
// A barrier is a `Shared` allocation of 64-bit elements. `U64` has no CUDA spelling in the emitter's typed table (declined by name), so the fixture uses `I64`.
const B64: ElementKind = ElementKind::I64;

fn bar() -> Place {
    Place::Alloc {
        alloc: 0,
        index: ScalarExpr::ConstInt(0, I32),
    }
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "k",
        I32,
        vec![Param::buffer(
            "out",
            I32,
            unpopped::kernel::BufferRole::ReadWrite,
            Some(1),
        )],
        vec![LocalDecl::scalar(I32)],
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "bar",
        MemorySpace::Shared,
        B64,
        Extent::Static(1),
    )])
    .with_launch(LaunchShape::new([1, 1, 1], [32, 1, 1]))
}

fn lowered(body: Vec<Stmt>) -> unpopped::backend::GeneratedKernel {
    let k = kernel(body);
    check_kernel(&k).expect("the fixture is a well-formed kernel");
    Cuda.lower_kernel(&k)
        .unwrap_or_else(|e| panic!("expected a lowering, got {e:?}"))
}

fn c(n: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(n, I32)
}

#[test]
fn griddepcontrol_is_spelled() {
    let g = lowered(vec![Stmt::GridDepWait, Stmt::GridDepLaunchDependents]);
    for want in [
        "griddepcontrol.wait;",
        "griddepcontrol.launch_dependents;",
        "__CUDA_ARCH__ < 900",
    ] {
        assert!(g.source.contains(want), "missing `{want}`:\n{}", g.source);
    }
    let i = g.interface().expect("interface");
    assert!(
        format!("{i:?}").contains("sm_90"),
        "the interface must say it needs sm_90: {i:?}"
    );
}

#[test]
fn mbarrier_family_is_spelled() {
    let g = lowered(vec![
        Stmt::MbarInit {
            bar: bar(),
            count: c(32),
        },
        Stmt::MbarArrive {
            bar: bar(),
            tx: None,
        },
        Stmt::MbarArrive {
            bar: bar(),
            tx: Some(c(1024)),
        },
        Stmt::MbarCompleteTx {
            bar: bar(),
            bytes: c(1024),
        },
        Stmt::MbarWait {
            bar: bar(),
            parity: c(0),
        },
    ]);
    for want in [
        "mbarrier.init.shared::cta.b64 [%0], %1;",
        "mbarrier.arrive.shared::cta.b64 _, [%0];",
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;",
        "mbarrier.complete_tx.shared::cta.b64 [%0], %1;",
        "mbarrier.try_wait.parity.shared::cta.b64 q, [%1], %2;",
        "__CUDA_ARCH__ < 900",
    ] {
        assert!(g.source.contains(want), "missing `{want}`:\n{}", g.source);
    }
    // The parity is checked, the wait loops, and the count is range-checked: a wrong value traps, never wraps.
    assert!(g.source.contains("if (_p != 0 && _p != 1) __trap();"));
    assert!(g.source.contains("while (_w == 0u)"));
    assert!(g.source.contains("if (_c < 1 || _c > 1048575) __trap();"));
}

/// Control: a kernel with none of these statements prints none of their PTX and no sm_90 guard (the query can say "no").
#[test]
fn a_plain_kernel_carries_no_sm90_text() {
    let g = lowered(vec![Stmt::Return]);
    for not in ["mbarrier", "griddepcontrol", "__CUDA_ARCH__ < 900"] {
        assert!(!g.source.contains(not), "unexpected `{not}`:\n{}", g.source);
    }
}

fn declined_naming(body: Vec<Stmt>, name: &str) {
    let k = kernel(body);
    check_kernel(&k).unwrap_or_else(|e| panic!("`{name}` fixture is not well-formed: {e:?}"));
    match Cuda.lower_kernel(&k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains(name), "decline must name `{name}`: {detail}")
        }
        other => panic!("`{name}` must be refused by name, got {other:?}"),
    }
}

/// Not spelled yet in this slice: each is refused and the decline names the statement (S2).
#[test]
fn the_unspelled_hopper_statements_are_refused_by_name() {
    declined_naming(vec![Stmt::WgmmaCommit], "WgmmaCommit");
    declined_naming(vec![Stmt::WgmmaWait { pending: 0 }], "WgmmaWait");
}

/// `ptxas` (via NVRTC) accepts the spelled statements for `sm_90a`: PTX generation only, nothing runs.
#[cfg(feature = "nvrtc")]
#[test]
fn ptxas_accepts_sm90() {
    use baracuda_cuda_emit::NvrtcCompiler;
    use unpopped::jit::Compiler;
    use unpopped_vocab::TargetId;
    let g = lowered(vec![
        Stmt::MbarInit {
            bar: bar(),
            count: c(32),
        },
        Stmt::MbarArrive {
            bar: bar(),
            tx: Some(c(1024)),
        },
        Stmt::MbarCompleteTx {
            bar: bar(),
            bytes: c(1024),
        },
        Stmt::MbarWait {
            bar: bar(),
            parity: c(0),
        },
        Stmt::GridDepWait,
        Stmt::GridDepLaunchDependents,
    ]);
    let nv = NvrtcCompiler::new(TargetId::parse("cuda:sm90a").expect("token")).expect("nvrtc target");
    nv.compile(&g.source, &g.name, 5000)
        .unwrap_or_else(|e| panic!("sm_90a rejected the source: {e}\n{}", g.source));
    // Control (the query can say "no"): the same source for `sm_80` is refused by the `#error` guard.
    let sm80 =
        NvrtcCompiler::new(TargetId::parse("cuda:sm80").expect("token")).expect("nvrtc target");
    assert!(
        sm80.compile(&g.source, &g.name, 5000).is_err(),
        "sm_80 must refuse a source that needs sm_90"
    );
}
