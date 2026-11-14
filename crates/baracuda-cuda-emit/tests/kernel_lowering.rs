//! `Cuda::lower_kernel`, stage 1 of ROADMAP U20: the signature and an empty body, and a typed
//! `UnsupportedKernel` (never an approximation) for everything else, naming what it declined.
//!
//! Stage 1 lowers none of the four statements ruled by board 434 C (`AsyncWait`, `CopyAsync`, `AsyncCommit`,
//! `Collective`); these tests pin that each of the three async statements is declined by name until stage 3.
//! `Collective` needs a launch-scope kernel the checker accepts and is added with its lowering in stage 3.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;
const U32: ElementKind = ElementKind::U32;
const F32: ElementKind = ElementKind::F32;

fn detail(r: Result<unpopped::backend::GeneratedKernel, LowerError>) -> String {
    match r {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        Err(other) => panic!("expected UnsupportedKernel, got {other:?}"),
        Ok(k) => panic!("expected a decline, got kernel source:\n{}", k.source),
    }
}

/// `k(src: F32 read, out: I32 read-write, n: U32 scalar)` with no statements.
fn empty() -> KernelFn {
    KernelFn::new(
        "k",
        I32,
        vec![
            Param::buffer("src", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", I32, BufferRole::ReadWrite, Some(1)),
            Param::scalar("n", U32, 0),
        ],
        vec![LocalDecl::scalar(U32)],
        Block(Vec::new()),
    )
}

#[test]
fn an_empty_kernel_gets_its_signature_and_an_empty_body() {
    let k = empty();
    check_kernel(&k).expect("the fixture is a well-formed kernel");
    let g = Cuda
        .lower_kernel(&k)
        .expect("a kernel with no statements lowers");
    assert_eq!(g.name, "baracuda_cuda_kernel_k");
    assert!(
        g.source.contains(
            "extern \"C\" __global__ void baracuda_cuda_kernel_k(const float* p0, int* p1, unsigned int p2) {\n}\n"
        ),
        "signature, `const` on the read-only buffer only:\n{}",
        g.source
    );
    assert!(g.interface().is_some(), "the interface half is attached");
}

#[test]
fn every_statement_is_declined_by_name() {
    let mut k = KernelFn::new(
        "k",
        I32,
        vec![],
        vec![LocalDecl::scalar(U32)],
        Block(vec![Stmt::Let {
            local: 0,
            init: ScalarExpr::ConstInt(1, U32),
        }]),
    );
    check_kernel(&k).expect("a Let of a constant is well-formed");
    let d = detail(Cuda.lower_kernel(&k));
    assert!(d.contains("statement `Let`"), "{d}");
    k.body = Block(Vec::new());
    assert!(
        Cuda.lower_kernel(&k).is_ok(),
        "control: the same kernel without the statement lowers"
    );
}

fn async_kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "k",
        I32,
        vec![Param::buffer("src", I32, BufferRole::Read, Some(1))],
        vec![],
        Block(body),
    )
    .with_allocs(vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        I32,
        Extent::Static(16),
    )])
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
}

#[test]
fn the_async_statements_are_declined_by_name_not_lowered() {
    let tid = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    let copy = Stmt::CopyAsync {
        dst: Place::Alloc {
            alloc: 0,
            index: tid.clone(),
        },
        src: Place::Logical {
            buffer: 0,
            coords: vec![tid],
        },
        elems: 1,
    };
    let cases = [
        (
            "CopyAsync",
            vec![copy, Stmt::AsyncCommit, Stmt::AsyncWait { pending: 0 }],
        ),
        (
            "AsyncCommit",
            vec![Stmt::AsyncCommit, Stmt::AsyncWait { pending: 0 }],
        ),
        ("AsyncWait", vec![Stmt::AsyncWait { pending: 0 }]),
    ];
    for (name, body) in cases {
        let k = async_kernel(body);
        let checked = check_kernel(&k);
        assert!(
            checked.is_ok(),
            "{name}: the fixture must pass the checker, got {checked:?}"
        );
        let d = detail(Cuda.lower_kernel(&k));
        assert!(
            d.contains(&format!("statement `{name}`")),
            "{name} is named in the decline: {d}"
        );
    }
}

#[test]
fn a_launch_shape_an_allocation_and_an_ill_formed_kernel_are_declined() {
    let launch = empty().with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]));
    assert!(detail(Cuda.lower_kernel(&launch)).contains("launch shape"));

    // A scalar parameter bound to a local that was never declared is refused by the checker.
    let bad = KernelFn::new(
        "bad",
        I32,
        vec![Param::scalar("n", U32, 7)],
        vec![],
        Block(Vec::new()),
    );
    assert!(
        check_kernel(&bad).is_err(),
        "control: the checker refuses it"
    );
    assert!(detail(Cuda.lower_kernel(&bad)).contains("ill-formed"));
}

#[test]
fn an_element_type_outside_the_cuda_table_is_declined_by_name() {
    let k = KernelFn::new(
        "h",
        I32,
        vec![Param::buffer(
            "x",
            ElementKind::F16,
            BufferRole::Read,
            Some(1),
        )],
        vec![],
        Block(Vec::new()),
    );
    check_kernel(&k).expect("the checker accepts an F16 buffer");
    let d = detail(Cuda.lower_kernel(&k));
    assert!(
        d.contains("buffer `x`") && d.contains("no CUDA spelling"),
        "{d}"
    );
}
