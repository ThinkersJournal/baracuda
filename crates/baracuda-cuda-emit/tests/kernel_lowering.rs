//! `Cuda::lower_kernel`, stages 1 and 2 of ROADMAP U20: the signature and argument block, the memory and control subset,
//! and a typed `UnsupportedKernel` (never an approximation) for everything else, naming what it declined.
//!
//! Stage 2 lowers none of the four statements ruled by board 434 C (`AsyncWait`, `CopyAsync`, `AsyncCommit`,
//! `Collective`); these tests pin that each of the three async statements is declined by name until stage 3.
//! `Collective` needs a launch-scope kernel the checker accepts and is added with its lowering in stage 3.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::{BinaryOp, ScalarExpr};
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
fn an_empty_kernel_gets_its_signature_and_argument_block() {
    let k = empty();
    check_kernel(&k).expect("the fixture is a well-formed kernel");
    let g = Cuda
        .lower_kernel(&k)
        .expect("a kernel with no statements lowers");
    assert_eq!(g.name, "baracuda_cuda_kernel_k");
    // Pointers first (`const` on the read-only buffer only), then one by-value argument block.
    assert!(
        g.source
            .contains("baracuda_cuda_kernel_k(const float* p0, int* p1, Args_k a) {"),
        "signature:\n{}",
        g.source
    );
    // Block: per buffer shape[R], stride[R], base (8-byte), then the scalar `n` (param 2, a u32).
    for want in [
        "long long b0_shape[1];",
        "long long b0_stride[1];",
        "long long b0_base;",
        "long long b1_shape[1];",
        "unsigned int s2;",
        "static_assert(sizeof(Args_k) == 56 && alignof(Args_k) == 8,",
    ] {
        assert!(g.source.contains(want), "missing `{want}`:\n{}", g.source);
    }
    // The launcher builds the bytes from the interface: the same offsets, size and alignment.
    let i = g.interface().expect("the interface half is attached");
    let pb = i
        .params_block
        .as_ref()
        .expect("the argument block layout is declared");
    assert_eq!((pb.size, pb.align, pb.by_address), (56, 8, false));
    let at = |n: &str| pb.fields.iter().find(|f| f.name == n).map(|f| f.offset);
    assert_eq!(at("b0_shape0"), Some(0));
    assert_eq!(at("b0_stride0"), Some(8));
    assert_eq!(at("b0_base"), Some(16));
    assert_eq!(at("b1_shape0"), Some(24));
    assert_eq!(at("b1_base"), Some(40));
    assert_eq!(at("s2"), Some(48));
    assert_eq!(i.entry_symbol.as_deref(), Some("baracuda_cuda_kernel_k"));
}

/// `saxpy`-shaped: `if i < n { out[i] = x[i] * 2.0 }` with `i = blockIdx.x * blockDim.x + threadIdx.x`.
fn scale() -> KernelFn {
    let u = |e: ScalarExpr| Box::new(e);
    let i = ScalarExpr::Add(
        u(ScalarExpr::Mul(
            u(ScalarExpr::Exec(ExecId::BlockIdx(0))),
            u(ScalarExpr::Exec(ExecId::BlockDim(0))),
        )),
        u(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
    );
    let at_i = |buffer| Place::Logical {
        buffer,
        coords: vec![ScalarExpr::Local(1)],
    };
    KernelFn::new(
        "scale",
        F32,
        vec![
            Param::buffer("x", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
            Param::scalar("n", U32, 0),
        ],
        vec![
            LocalDecl::scalar(U32),
            LocalDecl::scalar(U32),
            LocalDecl::scalar(F32),
        ],
        Block(vec![
            Stmt::Let { local: 1, init: i },
            Stmt::If {
                cond: ScalarExpr::Binary(
                    BinaryOp::CmpLt,
                    u(ScalarExpr::Local(1)),
                    u(ScalarExpr::Local(0)),
                ),
                then: Block(vec![
                    Stmt::Load {
                        local: 2,
                        place: at_i(0),
                    },
                    Stmt::Store {
                        place: at_i(1),
                        value: ScalarExpr::Mul(u(ScalarExpr::Local(2)), u(ScalarExpr::Const(2.0))),
                    },
                ]),
                els: Block(Vec::new()),
            },
        ]),
    )
    .with_launch(LaunchShape::new([256, 1, 1], [4, 1, 1]))
}

#[test]
fn a_guarded_elementwise_kernel_lowers_with_checked_addressing() {
    let k = scale();
    check_kernel(&k).expect("the fixture is well-formed");
    let g = Cuda
        .lower_kernel(&k)
        .expect("the memory and control subset lowers");
    for want in [
        "__launch_bounds__(256) baracuda_cuda_kernel_scale(const float* p0, float* p1, Args_scale a)",
        "unsigned int l1 = (unsigned int)(((blockIdx.x * blockDim.x) + threadIdx.x));",
        "if ((((float)((l1) < (a.s2)))) != 0) {",
        "if (c < 0 || c >= a.b0_shape[0]) __trap();",
        "l2 = (float)(p0[",
        "= (float)((l2 * 2.0f));",
    ] {
        assert!(g.source.contains(want), "missing `{want}`:\n{}", g.source);
    }
    let lb = g
        .interface()
        .and_then(|i| i.launch_bounds)
        .expect("launch bounds");
    assert_eq!(lb.max_threads, 256);
}

#[test]
fn shared_memory_a_block_barrier_and_a_loop_lower() {
    let k = red();
    check_kernel(&k).expect("the fixture is well-formed");
    let g = Cuda.lower_kernel(&k).expect("lowers");
    for want in [
        "__shared__ int sh0[4];",
        "if (_i1 < 0 || _i1 >= 4) __trap();",
        "__syncthreads();",
        "if (_t",
        "== 0) __trap();",
        // signed `+` wraps: computed in the unsigned type
        "(int)((unsigned int)(l0) + (unsigned int)(l1))",
    ] {
        assert!(
            g.source.contains(want),
            "missing `{want}`:
{}",
            g.source
        );
    }
}

/// Shared memory, a block barrier and a counted loop.
fn red() -> KernelFn {
    let u = |e: ScalarExpr| Box::new(e);
    KernelFn::new(
        "red",
        I32,
        vec![Param::buffer("out", I32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(I32), LocalDecl::scalar(I32)],
        Block(vec![
            Stmt::Store {
                place: Place::Alloc {
                    alloc: 0,
                    index: ScalarExpr::Exec(ExecId::ThreadIdx(0)),
                },
                value: ScalarExpr::ConstInt(1, I32),
            },
            Stmt::Barrier {
                scope: unpopped::exec::Scope::Block,
                kind: unpopped::kernel::BarrierKind::Full,
            },
            Stmt::Let {
                local: 0,
                init: ScalarExpr::ConstInt(0, I32),
            },
            Stmt::For {
                local: 1,
                start: ScalarExpr::ConstInt(0, I32),
                end: ScalarExpr::ConstInt(4, I32),
                step: ScalarExpr::ConstInt(1, I32),
                body: Block(vec![Stmt::Assign {
                    local: 0,
                    value: ScalarExpr::Add(u(ScalarExpr::Local(0)), u(ScalarExpr::Local(1))),
                }]),
            },
        ]),
    )
    .with_allocs(vec![AllocDecl::new(
        "s",
        MemorySpace::Shared,
        I32,
        Extent::Static(4),
    )])
    .with_launch(LaunchShape::new([4, 1, 1], [1, 1, 1]))
}

#[test]
fn rank_limits_and_unlowered_nodes_are_declined_by_name() {
    // No declared rank, and a rank above the bound.
    let mut k = empty();
    k.params[0] = Param::buffer("src", F32, BufferRole::Read, None);
    assert!(detail(Cuda.lower_kernel(&k)).contains("no declared rank"));
    let mut k = empty();
    k.params[0] = Param::buffer("src", F32, BufferRole::Read, Some(5));
    assert!(detail(Cuda.lower_kernel(&k)).contains("rank 5"));
    // An expression node stage 2 does not lower (a unary op), named.
    let mut k = scale();
    k.body = Block(vec![Stmt::Let {
        local: 2,
        init: ScalarExpr::Unary(unpopped::ir::UnaryOp::Neg, Box::new(ScalarExpr::Const(1.0))),
    }]);
    check_kernel(&k).expect("well-formed");
    assert!(detail(Cuda.lower_kernel(&k)).contains("expression node `Unary`"));
    // A subgroup barrier is declined, a block barrier lowers (above).
    let mut k = scale();
    k.body = Block(vec![Stmt::Barrier {
        scope: unpopped::exec::Scope::Subgroup,
        kind: unpopped::kernel::BarrierKind::Full,
    }]);
    if check_kernel(&k).is_ok() {
        assert!(detail(Cuda.lower_kernel(&k)).contains("Subgroup"));
    }
}

#[test]
fn a_statement_stage_2_does_not_lower_is_declined_by_name() {
    // `Let` lowers in stage 2 (a `MaskedStore` and the async statements do not).
    let mut k = scale();
    k.body = Block(vec![Stmt::MaskedStore {
        place: Place::Logical {
            buffer: 1,
            coords: vec![ScalarExpr::ConstInt(0, U32)],
        },
        value: ScalarExpr::Const(1.0),
        mask: ScalarExpr::ConstInt(1, U32),
    }]);
    check_kernel(&k).expect("well-formed");
    let d = detail(Cuda.lower_kernel(&k));
    assert!(d.contains("statement `MaskedStore`"), "{d}");
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
fn a_cluster_launch_a_dynamic_allocation_and_an_ill_formed_kernel_are_declined() {
    let cluster =
        empty().with_launch(LaunchShape::new([4, 1, 1], [2, 1, 1]).with_cluster([2, 1, 1]));
    assert!(detail(Cuda.lower_kernel(&cluster)).contains("cluster"));

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

/// The emitted source must be real CUDA: compile it with NVRTC (PTX only, nothing runs). `sm_89` always; `sm_61` when the
/// installed NVRTC accepts it (CUDA 13.x does not: set `BARACUDA_NVRTC_PATH` to a 12.x one, see `docs/sm61-p40-design.md`).
#[cfg(all(feature = "nvrtc", feature = "device-tests"))]
mod nvrtc {
    use super::*;
    use baracuda_cuda_emit::NvrtcCompiler;
    use baracuda_driver::require_optional;
    use unpopped::jit::Compiler;
    use unpopped_vocab::TargetId;

    fn compile(token: &str, k: &KernelFn) -> Result<(), String> {
        let g = Cuda.lower_kernel(k).map_err(|e| format!("{e:?}"))?;
        let c = NvrtcCompiler::new(TargetId::parse(token).expect("token")).expect("nvrtc target");
        c.compile(&g.source, &g.name, 10_000).map(|_| ())
    }

    #[test]
    fn lowered_kernels_compile_for_sm89() {
        for k in [scale(), red()] {
            if let Err(e) = compile("cuda:sm89", &k) {
                panic!("{}: {e}", k.name);
            }
        }
    }

    #[test]
    fn lowered_kernels_compile_for_sm61_when_nvrtc_can() {
        let probe = r#"extern "C" __global__ void k(int *x) { x[0] = __dp4a(x[0], x[1], 0); }"#;
        require_optional!(
            NvrtcCompiler::new(TargetId::parse("cuda:sm61").expect("token"))
                .expect("nvrtc target")
                .compile(probe, "k", 5000)
                .map(|_| ()),
            "an NVRTC that targets sm_61 (CUDA 12.x; set BARACUDA_NVRTC_PATH)"
        );
        for k in [scale(), red()] {
            if let Err(e) = compile("cuda:sm61", &k) {
                panic!("{}: {e}", k.name);
            }
        }
    }
}
