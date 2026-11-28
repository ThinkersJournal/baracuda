// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP S12a-5: `Tcgen05Mma` lowered to `tcgen05.mma.cta_group::1.kind::f16` (PTX ISA 9.5 9.7.18.10.10.1), with the shared-memory matrix
//! descriptors of 9.7.18.4.1 and the instruction descriptor of 9.7.18.4.2 (Table 53).
//!
//! Built: K-major `Canonical` operands without swizzle, `F16` / `Bf16` inputs, an `F32` accumulator, `m = 128`, `n` a multiple of 16 up to 256,
//! optional negation of `A` / `B`. Refused by name: `m = 64`, a swizzle, MN-major, an `F16` accumulator, a `B` type that differs from `A`.
//!
//! RUNTIME UNVERIFIED on hardware (no sm_100 device on the lane's box, an RTX 4070 / sm_89): what is checked is the emitted text, with the
//! descriptor words recomputed here from the ISA's bit positions, and that NVRTC and `ptxas -arch=sm_100a` accept the source.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, Major, OperandLayout, Param, Place, Stmt, Swizzle,
    check_kernel,
};
use unpopped_vocab::ElementKind;

const LBO: u32 = 128;
const SBO: u32 = 256;

fn u(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, ElementKind::U32)
}
fn alloc(a: u32) -> Place {
    Place::Alloc {
        alloc: a,
        index: u(0),
    }
}
fn k_major() -> OperandLayout {
    OperandLayout::Canonical {
        major: Major::K,
        swizzle: Swizzle::None,
        lbo: Some(LBO),
        sbo: SBO,
    }
}

#[derive(Clone, Copy)]
struct Shape {
    n: u32,
    ab: ElementKind,
    acc: ElementKind,
    m: u32,
    a_layout: OperandLayout,
    b_layout: OperandLayout,
    b_ty: Option<ElementKind>,
    negate_a: bool,
    negate_b: bool,
}

fn plain(n: u32) -> Shape {
    Shape {
        n,
        ab: ElementKind::Bf16,
        acc: ElementKind::F32,
        m: 128,
        a_layout: k_major(),
        b_layout: k_major(),
        b_ty: None,
        negate_a: false,
        negate_b: false,
    }
}

/// Allocations: 0 `A` and 1 `B` (`Shared`), 2 the accumulator (`Tensor`), 3 the barrier. One warp: thread 0 issues and commits.
fn kernel(s: Shape) -> KernelFn {
    let cols = (s.n).next_power_of_two().max(32);
    let acc_elem = if s.acc == ElementKind::F16 {
        ElementKind::U32
    } else {
        ElementKind::F32
    };
    KernelFn::new(
        "mma",
        ElementKind::F32,
        vec![Param::buffer(
            "out",
            ElementKind::F32,
            BufferRole::Write,
            Some(1),
        )],
        vec![],
        Block(vec![
            Stmt::Tcgen05Alloc { d: 2, ncols: cols },
            Stmt::MbarInit {
                bar: alloc(3),
                count: u(1),
            },
            Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            },
            Stmt::If {
                cond: ScalarExpr::Binary(
                    BinaryOp::CmpEq,
                    Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
                    Box::new(u(0)),
                ),
                then: Block(vec![
                    Stmt::Tcgen05Mma {
                        n: s.n,
                        ab: s.ab,
                        acc: s.acc,
                        a: alloc(0),
                        b: alloc(1),
                        d: 2,
                        d_col: u(0),
                        scale_d: u(0),
                        a_layout: s.a_layout,
                        b_layout: s.b_layout,
                        m: s.m,
                        d_lane: 0,
                        b_ty: s.b_ty,
                        negate_a: s.negate_a,
                        negate_b: s.negate_b,
                    },
                    Stmt::Tcgen05Commit { bar: alloc(3) },
                ]),
                els: Block(vec![]),
            },
            Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            },
            Stmt::Tcgen05RelinquishPermit,
            Stmt::Tcgen05Dealloc { d: 2, ncols: cols },
        ]),
    )
    .with_allocs(vec![
        AllocDecl::new("a", MemorySpace::Shared, s.ab, Extent::Static(128 * 16)),
        AllocDecl::new("b", MemorySpace::Shared, s.ab, Extent::Static(s.n * 16)),
        AllocDecl::new(
            "tmem",
            MemorySpace::Tensor,
            acc_elem,
            Extent::Static(128 * cols),
        ),
        AllocDecl::new(
            "bar",
            MemorySpace::Shared,
            ElementKind::I64,
            Extent::Static(1),
        ),
    ])
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]))
}

fn refused(s: Shape) -> String {
    match Cuda.lower_kernel(&kernel(s)) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        Err(other) => panic!("untyped refusal: {other:?}"),
        Ok(g) => panic!("must be refused, got:\n{}", g.source),
    }
}

/// The instruction descriptor, recomputed from Table 53's bit positions (not from the emitter's expression): dtype F32 = 1 at 4, atype at 7,
/// btype at 10, negate A / B at 13 / 14, N >> 3 at 17, M >> 4 at 24.
fn idesc(n: u32, bf16: bool, neg_a: bool, neg_b: bool) -> u32 {
    let t = u32::from(bf16);
    let mut v = 0u32;
    v |= 1 << 4;
    v |= t << 7;
    v |= t << 10;
    v |= u32::from(neg_a) << 13;
    v |= u32::from(neg_b) << 14;
    v |= (n >> 3) << 17;
    v |= (128u32 >> 4) << 24;
    v
}

/// The constant part of a descriptor (everything but the address): LBO >> 4 at 16, SBO >> 4 at 32, the fixed pattern 0b001 at 46..48.
fn desc_hi() -> u64 {
    (u64::from(LBO >> 4) << 16) | (u64::from(SBO >> 4) << 32) | (0b001u64 << 46)
}

#[test]
fn tcgen05_mma_lowers_to_the_f16_kind_with_both_descriptors() {
    for (n, bf16, na, nb) in [
        (16u32, true, false, false),
        (128, true, false, false),
        (256, false, true, false),
        (64, true, true, true),
    ] {
        let mut s = plain(n);
        s.ab = if bf16 {
            ElementKind::Bf16
        } else {
            ElementKind::F16
        };
        s.negate_a = na;
        s.negate_b = nb;
        let k = kernel(s);
        check_kernel(&k).expect("well-formed");
        let g = Cuda.lower_kernel(&k).expect("lowers");
        let src = &g.source;
        for want in [
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;".to_string(),
            "#error \"tcgen05 needs sm_100a or newer\"".to_string(),
            format!("unsigned int _id = {}u;", idesc(n, bf16, na, nb)),
            format!("| {}ull;", desc_hi()),
        ] {
            assert!(src.contains(&want), "n={n}: missing `{want}`:\n{src}");
        }
        // The text carries no sparsity, no `.ws`, no collector, no `cta_group::2`, no `[a-tmem]` form.
        for held in [
            "cta_group::2",
            ".ws",
            "collector",
            ".sp",
            "ashift",
            "block_scale",
        ] {
            assert!(!src.contains(held), "n={n}: `{held}` leaked in:\n{src}");
        }
    }
}

#[test]
fn what_is_not_built_is_refused_by_name() {
    let mut s = plain(64);
    s.m = 64;
    let d = refused(s);
    assert!(
        d.contains("Tcgen05Mma") || d.contains("ill-formed"),
        "m = 64: {d}"
    );

    let mut s = plain(64);
    s.acc = ElementKind::F16;
    let d = refused(s);
    assert!(d.contains("Tcgen05Mma") && d.contains("accumulator"), "{d}");

    let mut s = plain(64);
    s.a_layout = OperandLayout::Canonical {
        major: Major::K,
        swizzle: Swizzle::B128,
        lbo: None,
        sbo: 256,
    };
    let d = refused(s);
    assert!(
        d.contains("Tcgen05Mma") || d.contains("ill-formed"),
        "swizzle: {d}"
    );

    let mut s = plain(64);
    s.b_layout = OperandLayout::Canonical {
        major: Major::MN,
        swizzle: Swizzle::None,
        lbo: Some(LBO),
        sbo: SBO,
    };
    let d = refused(s);
    assert!(
        d.contains("Tcgen05Mma") || d.contains("ill-formed"),
        "MN-major: {d}"
    );

    // RowMajor is the oracle's layout, not one the descriptor can read.
    let mut s = plain(64);
    s.a_layout = OperandLayout::RowMajor;
    let d = refused(s);
    assert!(
        d.contains("Tcgen05Mma") || d.contains("ill-formed"),
        "RowMajor: {d}"
    );

    // A bad N is refused by name (the checker allows a multiple of 8; M = 128 needs a multiple of 16).
    let d = refused(plain(8));
    assert!(d.contains("Tcgen05Mma") && d.contains("n = 8"), "{d}");
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

    /// The MMA kernels (n = 16, 64, 256; bf16, f16; negated) are accepted by NVRTC and `ptxas -arch=sm_100a`. Nothing runs.
    #[test]
    fn the_mma_kernels_are_accepted_by_nvrtc_and_ptxas_for_sm_100a() {
        let exe = require_optional!(ptxas(), "ptxas from the CUDA toolkit (CUDA_PATH or PATH)");
        let c = NvrtcCompiler::new(TargetId::parse("cuda:sm100a").expect("token"))
            .expect("nvrtc target sm_100a");
        for (n, ab, neg) in [
            (16u32, ElementKind::Bf16, false),
            (64, ElementKind::F16, true),
            (256, ElementKind::Bf16, false),
        ] {
            let mut s = plain(n);
            s.ab = ab;
            s.negate_a = neg;
            s.negate_b = neg;
            let g = Cuda.lower_kernel(&kernel(s)).expect("lowers");
            let ptx = c
                .compile(&g.source, &g.name, 10_000)
                .unwrap_or_else(|e| panic!("nvrtc sm_100a n={n}: {e}\n{}", g.source));
            let dir = std::env::temp_dir().join(format!("baracuda-tc5-{}-{n}", std::process::id()));
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
                "ptxas -arch=sm_100a refused n={n}:\nstdout: {}\nstderr: {}\n{}",
                String::from_utf8_lossy(&r.stdout),
                String::from_utf8_lossy(&r.stderr),
                g.source
            );
        }
    }
}
