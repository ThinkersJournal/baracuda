// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP S1 slice 3 (prerequisite of `MatrixLoad`/`MatrixStore`/`Wgmma`): `Private` (per-thread) static arrays and
//! 16-bit float (`F16`, `Bf16`) allocation elements in `Cuda::lower_kernel`. A 16-bit float is **storage only** (its bits
//! are carried as `unsigned short`; copies move it, a `Load`/`Store` of it is refused by name, never approximated).
//! Nothing here needs a GPU; `ptxas_accepts` (feature `nvrtc`) assembles for `sm_80` without running anything.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::CacheHint;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;

fn c(n: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(n, I32)
}

fn alloc(i: u32) -> Place {
    Place::Alloc {
        alloc: i,
        index: c(0),
    }
}

fn kernel(allocs: Vec<AllocDecl>, body: Vec<Stmt>) -> KernelFn {
    // The buffer has the element type of the first allocation when that is a half (a copy needs equal types), else `F16`.
    let buf = match allocs.first().map(|a| a.elem) {
        Some(e @ (ElementKind::F16 | ElementKind::Bf16)) => e,
        _ => ElementKind::F16,
    };
    KernelFn::new(
        "k",
        I32,
        vec![Param::buffer("out", buf, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(I32)],
        Block(body),
    )
    .with_allocs(allocs)
    .with_launch(LaunchShape::new([1, 1, 1], [32, 1, 1]))
}

fn lowered(allocs: Vec<AllocDecl>, body: Vec<Stmt>) -> unpopped::backend::GeneratedKernel {
    let k = kernel(allocs, body);
    check_kernel(&k).unwrap_or_else(|e| panic!("the fixture is not well-formed: {e:?}"));
    Cuda.lower_kernel(&k)
        .unwrap_or_else(|e| panic!("expected a lowering, got {e:?}"))
}

fn private_i32() -> Vec<AllocDecl> {
    vec![AllocDecl::new(
        "r",
        MemorySpace::Private,
        I32,
        Extent::Static(4),
    )]
}

fn private_body() -> Vec<Stmt> {
    vec![
        Stmt::Store {
            place: alloc(0),
            value: c(7),
        },
        Stmt::Load {
            local: 0,
            place: alloc(0),
        },
    ]
}

#[test]
fn private_array_is_a_zeroed_thread_local_array() {
    let g = lowered(private_i32(), private_body());
    assert!(g.source.contains("int sh0[4] = {};"), "{}", g.source);
    assert!(
        !g.source.contains("__shared__ int sh0"),
        "a Private array must not be shared:\n{}",
        g.source
    );
}

/// Control: a `Shared` array of the same shape is still `__shared__` (the query can say "no").
#[test]
fn shared_array_stays_shared() {
    let g = lowered(
        vec![AllocDecl::new(
            "s",
            MemorySpace::Shared,
            I32,
            Extent::Static(4),
        )],
        private_body(),
    );
    assert!(g.source.contains("__shared__ int sh0[4];"), "{}", g.source);
}

fn half_copy(elem: ElementKind) -> (Vec<AllocDecl>, Vec<Stmt>) {
    (
        vec![AllocDecl::new(
            "h",
            MemorySpace::Shared,
            elem,
            Extent::Static(8),
        )],
        vec![Stmt::CopyAsync {
            dst: alloc(0),
            src: Place::Logical {
                buffer: 0,
                coords: vec![c(0)],
            },
            elems: 8,
            valid: None,
            cache: CacheHint::Auto,
            l2_prefetch: None,
        }],
    )
}

#[test]
fn half_allocations_are_stored_as_their_bits_and_copied() {
    for elem in [ElementKind::F16, ElementKind::Bf16] {
        let (a, b) = half_copy(elem);
        let g = lowered(a, b);
        assert!(
            g.source.contains("__shared__ unsigned short sh0[8];"),
            "{elem:?}:\n{}",
            g.source
        );
        // 8 x 2 bytes = one 16-byte cp.async.
        assert!(
            g.source.contains("cp.async.ca.shared.global"),
            "{elem:?}:\n{}",
            g.source
        );
        assert!(g.source.contains("\"n\"(16)"), "{elem:?}:\n{}", g.source);
    }
}

/// Arithmetic on the bits would be wrong, so a `Load`/`Store` of a 16-bit float allocation is refused, naming the statement.
#[test]
fn half_load_and_store_are_refused_by_name() {
    let a = vec![AllocDecl::new(
        "h",
        MemorySpace::Shared,
        ElementKind::F16,
        Extent::Static(8),
    )];
    for (body, name) in [
        (
            vec![Stmt::Load {
                local: 0,
                place: alloc(0),
            }],
            "Load",
        ),
        (
            vec![Stmt::Store {
                place: alloc(0),
                value: c(1),
            }],
            "Store",
        ),
    ] {
        let k = kernel(a.clone(), body);
        if check_kernel(&k).is_err() {
            continue; // the IR's own checker refuses a mistyped access first; never emitted either way
        }
        match Cuda.lower_kernel(&k) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(
                    detail.contains(name) && detail.contains("16-bit float"),
                    "{detail}"
                )
            }
            other => panic!("`{name}` of a half allocation must be refused, got {other:?}"),
        }
    }
}

#[cfg(feature = "nvrtc")]
#[test]
fn ptxas_accepts() {
    use baracuda_cuda_emit::NvrtcCompiler;
    use unpopped::jit::Compiler;
    use unpopped_vocab::TargetId;
    let nv =
        NvrtcCompiler::new(TargetId::parse("cuda:sm80").expect("token")).expect("nvrtc target");
    let g = lowered(private_i32(), private_body());
    nv.compile(&g.source, &g.name, 5000)
        .unwrap_or_else(|e| panic!("sm_80 rejected a Private array: {e}\n{}", g.source));
    for elem in [ElementKind::F16, ElementKind::Bf16] {
        let (a, b) = half_copy(elem);
        let g = lowered(a, b);
        nv.compile(&g.source, &g.name, 5000)
            .unwrap_or_else(|e| panic!("sm_80 rejected a {elem:?} copy: {e}\n{}", g.source));
    }
}
