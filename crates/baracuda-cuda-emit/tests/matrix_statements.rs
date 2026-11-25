// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP S1 slice 4: `MatrixLoad` (`ldmatrix`, sm_75) and `MatrixStore` (`stmatrix`, sm_90) in `Cuda::lower_kernel`. Source-shape
//! tests need no GPU; `ptxas_accepts` (feature `nvrtc`) assembles only. **Execution on hardware is not tested here** (the lane
//! distribution is the oracle's, taken from PTX; ROADMAP S6 cross-checks it on a device).

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, GeneratedKernel};
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const H: ElementKind = ElementKind::F16;

fn c(n: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(n, I32)
}

fn tile() -> Place {
    Place::Alloc {
        alloc: 0,
        index: c(0),
    }
}

fn lowered(count: u32, body: Vec<Stmt>) -> GeneratedKernel {
    let k = KernelFn::new(
        "m",
        I32,
        vec![Param::buffer("out", H, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(I32)],
        Block(body),
    )
    .with_allocs(vec![
        AllocDecl::new("tile", MemorySpace::Shared, H, Extent::Static(64 * count)),
        AllocDecl::new("frag", MemorySpace::Private, H, Extent::Static(2 * count)),
    ])
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    check_kernel(&k).unwrap_or_else(|e| panic!("the fixture is not well-formed: {e:?}"));
    Cuda.lower_kernel(&k)
        .unwrap_or_else(|e| panic!("expected a lowering, got {e:?}"))
}

fn load(count: u32, trans: bool) -> Stmt {
    Stmt::MatrixLoad {
        count,
        trans,
        src: tile(),
        dst: 1,
    }
}

fn store(count: u32, trans: bool) -> Stmt {
    Stmt::MatrixStore {
        count,
        trans,
        dst: tile(),
        src: 1,
    }
}

#[test]
fn ldmatrix_is_spelled_for_sm75() {
    let g = lowered(4, vec![load(4, true)]);
    for want in [
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];",
        "__CUDA_ARCH__ < 750",
        "if (_a & 15u) __trap();",
        "+ 8 > 256) __trap();",
    ] {
        assert!(g.source.contains(want), "missing `{want}`:\n{}", g.source);
    }
    assert!(
        !g.source.contains("__CUDA_ARCH__ < 900"),
        "ldmatrix needs sm_75, not sm_90"
    );
    let un = lowered(1, vec![load(1, false)]);
    assert!(
        un.source.contains("m8n8.x1.shared.b16 {%0}, [%1];"),
        "{}",
        un.source
    );
}

#[test]
fn stmatrix_is_spelled_for_sm90() {
    let g = lowered(2, vec![store(2, false)]);
    for want in [
        "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};",
        "__CUDA_ARCH__ < 900",
    ] {
        assert!(g.source.contains(want), "missing `{want}`:\n{}", g.source);
    }
    let i = g.interface().expect("interface");
    assert!(format!("{i:?}").contains("sm_90"), "{i:?}");
}

/// Control: no matrix statement, no matrix PTX (the query can say "no").
#[test]
fn a_plain_kernel_carries_no_matrix_text() {
    let g = lowered(1, vec![Stmt::Return]);
    for not in ["ldmatrix", "stmatrix", "__CUDA_ARCH__ < 750"] {
        assert!(!g.source.contains(not), "unexpected `{not}`:\n{}", g.source);
    }
}

#[cfg(feature = "nvrtc")]
#[test]
fn ptxas_accepts() {
    use baracuda_cuda_emit::NvrtcCompiler;
    use unpopped::jit::Compiler;
    use unpopped_vocab::TargetId;
    let t = |tok: &str| NvrtcCompiler::new(TargetId::parse(tok).expect("token")).expect("nvrtc");
    let (sm80, sm90) = (t("cuda:sm80"), t("cuda:sm90a"));
    let ld = lowered(4, vec![load(4, false), load(4, true)]);
    let st = lowered(4, vec![store(4, false), store(4, true)]);
    for (nv, g, what) in [
        (&sm80, &ld, "ldmatrix sm80"),
        (&sm90, &ld, "ldmatrix sm90a"),
        (&sm90, &st, "stmatrix sm90a"),
    ] {
        nv.compile(&g.source, &g.name, 5000)
            .unwrap_or_else(|e| panic!("{what} rejected: {e}\n{}", g.source));
    }
    // Control: stmatrix is refused below sm_90.
    assert!(sm80.compile(&st.source, &st.name, 5000).is_err());
}
