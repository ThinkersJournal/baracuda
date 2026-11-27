// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP S12a-1: the tensor-memory management statements `Tcgen05Alloc`, `Tcgen05RelinquishPermit` and `Tcgen05Dealloc`, and the `Tensor`
//! allocation they act on, lowered to `tcgen05.alloc` / `.relinquish_alloc_permit` / `.dealloc` with `.cta_group::1` (PTX ISA 9.5 9.7.18.7.1).
//!
//! RUNTIME UNVERIFIED on hardware: the lane's box is an RTX 4070 (sm_89), which has no tensor memory. What is checked is the emitted text,
//! the arch guard, and that NVRTC and then `ptxas` for `sm_100a` accept the source; that is no claim beyond what ptxas accepted. The same
//! source for `sm_89` is refused by the arch guard (and by ptxas: "Instruction 'tcgen05.alloc' not supported on .target 'sm_89'").
//!
//! Held, not lowered here (pm-kernels 2026-11-27, ROADMAP S12 (b)): `tcgen05.commit` with a multicast and a `ctaMask`, `mbarrier.arrive` with an
//! explicit count or a `ctaMask`, `mbarrier.expect_tx` with a multicast and a `ctaMask`.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{BarrierKind, Block, BufferRole, KernelFn, Param, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

const NCOLS: u32 = 32;

/// One warp: lane 0's warp allocates `ncols` columns, the block synchronises, then the warp relinquishes the permit and deallocates.
fn kernel(ncols: u32, extent: u32) -> KernelFn {
    KernelFn::new(
        "tmem",
        ElementKind::F32,
        vec![Param::buffer(
            "out",
            ElementKind::F32,
            BufferRole::Write,
            Some(1),
        )],
        vec![],
        Block(vec![
            Stmt::Tcgen05Alloc { d: 0, ncols },
            Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            },
            Stmt::Tcgen05RelinquishPermit,
            Stmt::Tcgen05Dealloc { d: 0, ncols },
        ]),
    )
    .with_allocs(vec![AllocDecl::new(
        "tmem",
        MemorySpace::Tensor,
        ElementKind::F32,
        Extent::Static(extent),
    )])
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]))
}

fn good() -> KernelFn {
    kernel(NCOLS, 128 * NCOLS)
}

#[test]
fn the_tensor_memory_management_statements_lower_to_tcgen05_text() {
    let k = good();
    check_kernel(&k).expect("well-formed");
    let g = Cuda.lower_kernel(&k).expect("lowers");
    let s = &g.source;
    for want in [
        "__shared__ unsigned int tm0;",
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;",
        "\"r\"(32u)",
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;",
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;",
        "#error \"tcgen05 needs sm_100a or newer\"",
    ] {
        assert!(s.contains(want), "missing `{want}`:\n{s}");
    }
    // The interface says what the kernel needs, and the order of the three statements is kept.
    let (a, r, d) = (
        s.find("tcgen05.alloc").unwrap(),
        s.find("tcgen05.relinquish_alloc_permit").unwrap(),
        s.find("tcgen05.dealloc").unwrap(),
    );
    assert!(a < r && r < d, "alloc, relinquish, dealloc in order:\n{s}");
    let iface = g.interface().expect("an interface");
    assert!(
        format!("{iface:?}").contains("tcgen05 (sm_100a or newer)"),
        "{iface:?}"
    );
}

/// Controls: a kernel with no tcgen05 statement has no guard and no requirement (so the assertions above can fail), and an ordinary
/// `Shared` allocation still becomes an array.
#[test]
fn a_kernel_without_tcgen05_has_no_guard() {
    let k = KernelFn::new(
        "plain",
        ElementKind::F32,
        vec![Param::buffer(
            "out",
            ElementKind::F32,
            BufferRole::Write,
            Some(1),
        )],
        vec![],
        Block(vec![Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Full,
        }]),
    )
    .with_allocs(vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        ElementKind::F32,
        Extent::Static(8),
    )])
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]));
    let g = Cuda.lower_kernel(&k).expect("lowers");
    assert!(
        !g.source.contains("tcgen05") && g.source.contains("__shared__ float sh0[8];"),
        "{}",
        g.source
    );
}

fn refused(k: &KernelFn) -> String {
    match Cuda.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        Err(other) => panic!("untyped refusal: {other:?}"),
        Ok(g) => panic!("must be refused, got:\n{}", g.source),
    }
}

/// A column count that is not a power of two, or an allocation of the wrong extent, is refused by name (by the checker or by the emitter's
/// backstop: both are a typed refusal naming the statement or the allocation).
#[test]
fn a_bad_column_count_or_extent_is_refused_by_name() {
    let d = refused(&kernel(48, 128 * 48));
    assert!(d.contains("Tcgen05") || d.contains("ill-formed"), "{d}");
    let d = refused(&kernel(NCOLS, 128 * NCOLS + 1));
    assert!(d.contains("Tcgen05") || d.contains("ill-formed"), "{d}");
}

/// An ordinary load of the `Tensor` allocation is not lowered (only the tcgen05 statements reach tensor memory): the address slot is not an array.
#[test]
fn a_tensor_allocation_is_not_an_array() {
    let k = KernelFn::new(
        "tmem_read",
        ElementKind::F32,
        vec![Param::buffer(
            "out",
            ElementKind::F32,
            BufferRole::Write,
            Some(1),
        )],
        vec![unpopped::kernel::LocalDecl::scalar(ElementKind::F32)],
        Block(vec![Stmt::Load {
            local: 0,
            place: unpopped::kernel::Place::Alloc {
                alloc: 0,
                index: ScalarExpr::Exec(ExecId::ThreadIdx(0)),
            },
        }]),
    )
    .with_allocs(vec![AllocDecl::new(
        "tmem",
        MemorySpace::Tensor,
        ElementKind::F32,
        Extent::Static(128 * NCOLS),
    )])
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]));
    let d = refused(&k);
    assert!(
        d.contains("ill-formed") || d.contains("not lowered") || d.contains("Tensor"),
        "{d}"
    );
}

#[cfg(all(feature = "nvrtc", feature = "device-tests"))]
mod nvrtc {
    use super::*;
    use baracuda_cuda_emit::NvrtcCompiler;
    use baracuda_driver::require_optional;
    use unpopped::jit::Compiler;
    use unpopped_vocab::TargetId;

    fn ptxas() -> Option<std::path::PathBuf> {
        let exe = if cfg!(windows) { "ptxas.exe" } else { "ptxas" };
        let mut cands: Vec<std::path::PathBuf> = Vec::new();
        if let Some(p) = std::env::var_os("CUDA_PATH") {
            cands.push(std::path::PathBuf::from(p).join("bin").join(exe));
        }
        if let Some(path) = std::env::var_os("PATH") {
            cands.extend(std::env::split_paths(&path).map(|d| d.join(exe)));
        }
        cands.into_iter().find(|c| c.is_file())
    }

    /// The source compiles with NVRTC for `sm_100a`, and `ptxas -arch=sm_100a` accepts the PTX. Nothing runs (no sm_100 device here).
    #[test]
    fn the_source_is_accepted_by_nvrtc_and_ptxas_for_sm_100a() {
        let exe = require_optional!(ptxas(), "ptxas from the CUDA toolkit (CUDA_PATH or PATH)");
        let k = good();
        let g = Cuda.lower_kernel(&k).expect("lowers");
        let c = NvrtcCompiler::new(TargetId::parse("cuda:sm100a").expect("token"))
            .expect("nvrtc target sm_100a");
        let ptx = c
            .compile(&g.source, &g.name, 10_000)
            .unwrap_or_else(|e| panic!("nvrtc sm_100a: {e}\n{}", g.source));
        let dir = std::env::temp_dir().join(format!("baracuda-tc1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let (src, out) = (dir.join("k.ptx"), dir.join("k.cubin"));
        std::fs::write(&src, &ptx).expect("write ptx");
        let r = std::process::Command::new(&exe)
            .args(["-arch=sm_100a", "-o"])
            .arg(&out)
            .arg(&src)
            .output()
            .expect("ptxas runs");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            r.status.success(),
            "ptxas -arch=sm_100a refused it:\nstdout: {}\nstderr: {}\n{}",
            String::from_utf8_lossy(&r.stdout),
            String::from_utf8_lossy(&r.stderr),
            g.source
        );
    }

    /// The control: the same source for `sm_89` is refused (by the arch guard), so the acceptance above is about the target.
    #[test]
    fn the_same_source_is_refused_for_sm_89() {
        let g = Cuda.lower_kernel(&good()).expect("lowers");
        let c = NvrtcCompiler::new(TargetId::parse("cuda:sm89").expect("token")).expect("nvrtc");
        let e = c
            .compile(&g.source, &g.name, 10_000)
            .expect_err("sm_89 has no tensor memory");
        assert!(e.to_string().contains("tcgen05 needs sm_100a"), "{e}");
    }
}
