// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Cuda::lower_kernel`, stages 1 and 2 of ROADMAP U20: the signature and argument block, the memory and control subset,
//! and a typed `UnsupportedKernel` (never an approximation) for everything else, naming what it declined.
//!
//! Stage 3 lowers the four statements ruled by board 434 C (`AsyncWait`, `CopyAsync`, `AsyncCommit`, `Collective`) for NVIDIA
//! with a patent risk note; `cp.async` says it needs sm_80 in the interface, and the collectives are subgroup-scope only.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, DtypeSupport, LowerError};
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CacheHint, CollectiveKind, KernelError, KernelFn, LocalDecl, Param, Place,
    ShuffleMode, Stmt, VoteKind, check_kernel,
};
use unpopped_vocab::{ElementKind, TargetId};

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
                unroll: unpopped::kernel::Unroll::None,
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

fn async_body() -> Vec<Stmt> {
    let tid = ScalarExpr::Exec(ExecId::ThreadIdx(0));
    vec![
        Stmt::CopyAsync {
            dst: Place::Alloc {
                alloc: 0,
                index: tid.clone(),
            },
            src: Place::Logical {
                buffer: 0,
                coords: vec![tid],
            },
            elems: 1,
            valid: None,
            cache: CacheHint::Auto,
            l2_prefetch: None,
        },
        Stmt::AsyncCommit,
        Stmt::AsyncWait { pending: 0 },
    ]
}

#[test]
fn the_async_statements_lower_to_cp_async_and_say_they_need_sm80() {
    let k = async_kernel(async_body());
    check_kernel(&k).expect("the fixture must pass the checker");
    let g = Cuda.lower_kernel(&k).expect("cp.async lowers");
    for want in [
        "#error \"cp.async needs sm_80 or newer\"",
        "cp.async.ca.shared.global [%0], [%1], %2;",
        "cp.async.commit_group;",
        "cp.async.wait_group %0;",
        "\"n\"(0)",
        "if (((unsigned long long)_g | (unsigned long long)_s) & 3) __trap();",
        "patent risk note (board 434 C",
    ] {
        // `_g` is parenthesised by the emitter; match the pieces rather than the exact spacing
        let alt = want.replace(
            "((unsigned long long)_g | (unsigned long long)_s)",
            "(((unsigned long long)_g) | (unsigned long long)_s)",
        );
        assert!(
            g.source.contains(want) || g.source.contains(&alt),
            "missing `{want}`:\n{}",
            g.source
        );
    }
    let i = g.interface().expect("interface");
    assert!(
        i.requires.iter().any(|r| r.contains("sm_80")),
        "the consumer can decline before compiling: {:?}",
        i.requires
    );
    // 4, 8 or 16 bytes only: 3 x f32 is 12.
    let mut bad = async_body();
    if let Stmt::CopyAsync { elems, .. } = &mut bad[0] {
        *elems = 3;
    }
    let k = async_kernel(bad);
    if check_kernel(&k).is_ok() {
        assert!(detail(Cuda.lower_kernel(&k)).contains("cp.async moves 4, 8 or 16"));
    }
}

/// `async_body` with its `CopyAsync` fields replaced.
fn async_with(valid: Option<ScalarExpr>, cache: CacheHint, l2_prefetch: Option<u32>) -> Vec<Stmt> {
    let mut b = async_body();
    if let Stmt::CopyAsync {
        valid: v,
        cache: c,
        l2_prefetch: l,
        ..
    } = &mut b[0]
    {
        *v = valid;
        *c = cache;
        *l = l2_prefetch;
    }
    b
}

fn async_source(valid: Option<ScalarExpr>, cache: CacheHint, l2: Option<u32>) -> String {
    let k = async_kernel(async_with(valid, cache, l2));
    check_kernel(&k).expect("the fixture must pass the checker");
    Cuda.lower_kernel(&k).expect("cp.async lowers").source
}

// U13. TEXT-LEVEL pins of the emitted instruction: no GPU ran. A device test (a lane with `valid` 0 reads zeros) needs the forge.
#[test]
fn cp_async_valid_emits_the_src_size_zero_fill_form() {
    let src = async_source(Some(ScalarExpr::ConstInt(0, I32)), CacheHint::Auto, None);
    // the 4th operand is src-size in bytes; without it the hardware reads all `cp-size` bytes and zero-fills nothing
    assert!(
        src.contains("[%0], [%1], %2, %3;"),
        "no src-size operand:
{src}"
    );
    assert!(
        src.contains("\"r\"(_n)"),
        "src-size not passed:
{src}"
    );
    assert!(src.contains("unsigned _n = (unsigned)_v * 4u;"), "{src}");
    // `valid` 0 reads nothing: the source place (which traps when out of range) is only evaluated when _v > 0
    let guard = src
        .find("if (_v > 0) {")
        .expect("source place not guarded by _v > 0");
    let read = src.rfind("_g = &p0[").expect("source address assigned");
    assert!(
        guard < read,
        "source place evaluated outside the _v > 0 guard:
{src}"
    );
    assert!(
        src.contains("_v > 1) __trap();"),
        "valid above elems must trap:
{src}"
    );
}

#[test]
fn cp_async_without_valid_has_no_src_size_operand() {
    // the seeded-miss counterpart: the default form must stay the 3 operand copy
    let src = async_source(None, CacheHint::Auto, None);
    assert!(src.contains("[%0], [%1], %2;"), "{src}");
    assert!(
        !src.contains("%3"),
        "src-size leaked into a copy that has no `valid`:
{src}"
    );
}

#[test]
fn cp_async_cache_operator_and_l2_prefetch_are_emitted_or_refused() {
    // `.cg` is 16 byte copies only and an L2 prefetch is 64/128/256: unpopped's checker refuses both first; the emitter
    // refuses them too (a backstop), so a kernel that skips the checker is not silently lowered to `.ca` / no prefetch.
    for (cache, l2, want) in [
        (
            CacheHint::Global,
            None,
            "`.cg` cache operator copies 16 bytes only",
        ),
        (CacheHint::Auto, Some(100), "L2 prefetch of 100 bytes"),
    ] {
        let k = async_kernel(async_with(None, cache, l2));
        assert!(check_kernel(&k).is_err(), "the checker accepted {want}");
        assert!(
            detail(Cuda.lower_kernel(&k)).contains("ill-formed"),
            "{want}"
        );
    }
    let src = async_source(None, CacheHint::All, Some(128));
    assert!(
        src.contains("cp.async.ca.shared.global.L2::128B [%0]"),
        "{src}"
    );
}

/// One warp: every lane contributes 1.0 and receives the warp's sum.
fn warp_sum(kind: CollectiveKind, scope: unpopped::exec::Scope, block: u32) -> KernelFn {
    // A vote's result is a u32, every other collective's is the value's type.
    let out = if matches!(kind, CollectiveKind::Vote(_)) {
        U32
    } else {
        F32
    };
    KernelFn::new(
        "warp",
        F32,
        vec![Param::buffer("out", out, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(out)],
        Block(vec![
            Stmt::Let {
                local: 0,
                init: ScalarExpr::Const(1.0),
            },
            Stmt::Collective {
                kind,
                value: ScalarExpr::Local(0),
                result: 1,
                scope,
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
    .with_launch(LaunchShape::new([block, 1, 1], [1, 1, 1]))
}

#[test]
fn subgroup_collectives_lower_to_warp_intrinsics_and_the_rest_declines_by_name() {
    use unpopped::exec::Scope;
    use unpopped::ir::ReduceOp;
    let sum = warp_sum(CollectiveKind::Reduce(ReduceOp::Sum), Scope::Subgroup, 32);
    check_kernel(&sum).expect("well-formed");
    let g = Cuda.lower_kernel(&sum).expect("a warp reduce lowers");
    assert!(
        g.source
            .contains("__shfl_xor_sync(0xffffffffu, _v, _o, 32)")
            && g.source.contains("_v = ((_v) + (_n));"),
        "{}",
        g.source
    );
    for (kind, want) in [
        (
            CollectiveKind::Shuffle {
                mode: ShuffleMode::Xor,
                operand: ScalarExpr::ConstInt(1, U32),
            },
            "__shfl_xor_sync(0xffffffffu, _v, (int)(1u), 32)",
        ),
        (
            CollectiveKind::Vote(VoteKind::Ballot),
            "__ballot_sync(0xffffffffu, _v != 0)",
        ),
        (
            CollectiveKind::Scan {
                op: ReduceOp::Sum,
                exclusive: false,
            },
            "__shfl_up_sync(0xffffffffu, _v, _o, 32)",
        ),
        (
            CollectiveKind::Broadcast {
                lane: ScalarExpr::ConstInt(0, U32),
            },
            "__shfl_sync(0xffffffffu, _v, (int)(0u), 32)",
        ),
    ] {
        let k = warp_sum(kind, Scope::Subgroup, 32);
        if let Err(e) = check_kernel(&k) {
            panic!("{want}: fixture refused by the checker: {e:?}");
        }
        let g = Cuda.lower_kernel(&k).expect("lowers");
        assert!(g.source.contains(want), "missing `{want}`:\n{}", g.source);
    }
    // Declined by name: a block-scope collective, a block that is not whole warps, a signed shuffle distance. (A float Max lowered
    // as a refusal until ROADMAP S13: it now lowers with a NaN-propagating fold, see `exclusive_scans.rs`.)
    let k = warp_sum(CollectiveKind::Reduce(ReduceOp::Sum), Scope::Block, 32);
    check_kernel(&k).expect("well-formed");
    assert!(detail(Cuda.lower_kernel(&k)).contains("Block-scope `Collective`"));
    let k = warp_sum(CollectiveKind::Reduce(ReduceOp::Sum), Scope::Subgroup, 48);
    check_kernel(&k).expect("well-formed");
    assert!(detail(Cuda.lower_kernel(&k)).contains("multiple of 32"));
    let k = warp_sum(CollectiveKind::Reduce(ReduceOp::Max), Scope::Subgroup, 32);
    check_kernel(&k).expect("well-formed");
    let g = Cuda.lower_kernel(&k).expect("a float Max reduce lowers");
    assert!(g.source.contains("(_v) != (_v)"), "{}", g.source);
    let k = warp_sum(
        CollectiveKind::Shuffle {
            mode: ShuffleMode::Up,
            operand: ScalarExpr::ConstInt(1, I32),
        },
        Scope::Subgroup,
        32,
    );
    check_kernel(&k).expect("well-formed");
    assert!(detail(Cuda.lower_kernel(&k)).contains("must be a u32"));
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

/// KISS 0.24.0 element rows (unpopped 0.60). TWO gates, asserted separately:
/// 1. the CHECKER refusal (Unpopped): `check_kernel` gives `KernelError::DtypeNotComputed { dtype }` for a kernel declaring one of
///    these as a buffer, and `lower_kernel` reports it as an ill-formed kernel naming the dtype. The emitter's own by-name decline
///    ("no CUDA spelling") is unreachable through `lower_kernel` for these kinds until a checker admits them;
/// 2. the EMITTER's own answer, asserted directly: `Cuda.supports_dtype` says `Unsupported` and `scalar_ctype` is `None` for each,
///    so the day a checker admits a row the emitter still declines. `F32` is the positive control for both.
///
/// Rows to lower later are recorded in ROADMAP S14.
#[test]
fn the_seven_new_element_rows_are_each_declined_by_name() {
    for (kind, name) in [
        (ElementKind::F128, "F128"),
        (ElementKind::I128, "I128"),
        (ElementKind::U128, "U128"),
        (ElementKind::I256, "I256"),
        (ElementKind::U256, "U256"),
        (ElementKind::F8E3M4, "F8E3M4"),
        (ElementKind::F8UE4M3, "F8UE4M3"),
    ] {
        let k = KernelFn::new(
            "w",
            I32,
            vec![Param::buffer("x", kind, BufferRole::Read, Some(1))],
            vec![],
            Block(Vec::new()),
        );
        // Gate 1. Unpopped 0.62 admits some of these rows as buffers (I128 at least): then the EMITTER declines the buffer by name
        // ("no CUDA spelling"); a row the checker still refuses is refused as an ill-formed kernel naming the dtype.
        let d = detail(Cuda.lower_kernel(&k));
        match check_kernel(&k) {
            Err(KernelError::DtypeNotComputed { dtype }) => {
                assert_eq!(dtype, kind, "{name}: the checker names another dtype");
                assert!(
                    d.contains("ill-formed") && d.contains("DtypeNotComputed") && d.contains(name),
                    "{name}: {d}"
                );
            }
            Ok(()) => assert!(
                !d.contains("ill-formed")
                    && d.contains("buffer `x`")
                    && d.contains("no CUDA spelling"),
                "{name}: the checker admits it, so the emitter must decline it by name: {d}"
            ),
            other => panic!("{name}: an unexpected checker answer: {other:?}"),
        }
        // Gate 2: the emitter's own table, independent of the checker.
        let sm89 = TargetId::parse("cuda:sm89").expect("token");
        assert!(
            matches!(Cuda.supports_dtype(kind, sm89), DtypeSupport::Unsupported),
            "{name}: the emitter's own dtype gate must decline"
        );
        assert_eq!(
            baracuda_cuda_emit::cfamily_shadow::scalar_ctype(kind),
            None,
            "{name}: the emitter's own ctype table has no spelling"
        );
    }
    // Positive control: the same two calls answer yes for F32, so the assertions above can fail.
    let sm89 = TargetId::parse("cuda:sm89").expect("token");
    assert!(matches!(
        Cuda.supports_dtype(ElementKind::F32, sm89),
        DtypeSupport::Supported
    ));
    assert_eq!(
        baracuda_cuda_emit::cfamily_shadow::scalar_ctype(ElementKind::F32),
        Some("float")
    );
}

/// Unpopped 0.61.3+ (so 0.62): `check_kernel` COMPUTES `I128`/`U128`/`I256`/`U256`. The carry-flag lowering is held (ROADMAP S14), so the emitter
/// must REFUSE them BY NAME before emitting, wherever the type can appear on the kernel surface: a buffer, a scalar parameter, a local
/// declaration, the kernel's compute type and a constant. CPU-only (nothing is compiled or run). Positive control: the same shapes with `I32` lower.
#[test]
fn wide_integers_are_refused_by_name_on_every_kernel_surface() {
    use unpopped::kernel::LocalDecl;
    // The shapes, each parameterised by the element type.
    type Shape = (&'static str, fn(ElementKind) -> KernelFn);
    let shapes: [Shape; 5] = [
        ("buffer", |t| {
            KernelFn::new(
                "w",
                I32,
                vec![Param::buffer("x", t, BufferRole::ReadWrite, Some(1))],
                vec![],
                Block(Vec::new()),
            )
        }),
        ("scalar parameter", |t| {
            KernelFn::new(
                "w",
                I32,
                vec![Param::scalar("x", t, 0)],
                vec![LocalDecl::scalar(t)],
                Block(Vec::new()),
            )
        }),
        ("local", |t| {
            KernelFn::new(
                "w",
                I32,
                vec![Param::scalar("n", U32, 0)],
                vec![LocalDecl::scalar(U32), LocalDecl::scalar(t)],
                Block(Vec::new()),
            )
        }),
        ("constant", |t| {
            KernelFn::new(
                "w",
                I32,
                vec![Param::scalar("n", U32, 0)],
                vec![LocalDecl::scalar(U32), LocalDecl::scalar(t)],
                Block(vec![Stmt::Let {
                    local: 1,
                    init: ScalarExpr::ConstInt(1, t),
                }]),
            )
        }),
        ("compute type", |t| {
            KernelFn::new(
                "w",
                t,
                vec![Param::scalar("n", U32, 0)],
                vec![LocalDecl::scalar(U32)],
                Block(Vec::new()),
            )
        }),
    ];
    let mut checker_admits = 0;
    for kind in [
        ElementKind::I128,
        ElementKind::U128,
        ElementKind::I256,
        ElementKind::U256,
    ] {
        let name = format!("{kind:?}");
        for (what, make) in &shapes {
            let k = make(kind);
            // Whatever the checker says, the emitter must not emit and must not panic.
            let r = std::panic::catch_unwind(|| Cuda.lower_kernel(&k));
            let r = r.unwrap_or_else(|_| panic!("{name} {what}: the emitter PANICKED"));
            let d = match r {
                Err(LowerError::UnsupportedKernel { detail }) => detail,
                Err(other) => panic!("{name} {what}: expected UnsupportedKernel, got {other:?}"),
                Ok(g) => panic!(
                    "{name} {what}: the emitter EMITTED:
{}",
                    g.source
                ),
            };
            if check_kernel(&k).is_ok() {
                checker_admits += 1;
                assert!(
                    !d.contains("ill-formed"),
                    "{name} {what}: the checker admits it, so this must be the emitter's own refusal: {d}"
                );
            }
            assert!(
                d.contains(&name) || d.contains("no CUDA spelling"),
                "{name} {what}: the refusal must name the dtype: {d}"
            );
        }
    }
    // The census must not be vacuous: the checker admits at least the buffer shape of some wide row.
    assert!(
        checker_admits > 0,
        "the checker admitted none of the wide-integer shapes, so only its own refusal was exercised"
    );
    // Positive control: the same five shapes with `I32` lower, so the refusals above are about the type.
    for (what, make) in &shapes {
        let k = make(I32);
        check_kernel(&k)
            .unwrap_or_else(|e| panic!("control {what}: the shape must be well-formed: {e:?}"));
        Cuda.lower_kernel(&k)
            .unwrap_or_else(|e| panic!("control {what}: an I32 kernel must lower: {e:?}"));
    }
}

/// The PLAN path (`try_generate` over `Cuda`) for the wide integers: an elementwise plan at `I128` / `U128` / `I256` / `U256` must come back
/// as a typed `InadmissiblePlan` naming the dtype (Unpopped's plan admission, before the backend), never a panic and never source. Control: the
/// same plan at `F32` emits (integer elementwise plans are declined by the bespoke surface at every width today).
#[test]
fn wide_integer_plans_are_declined_not_emitted_or_panicked() {
    use unpopped::ir::BinaryOp;
    use unpopped::{OpDef, input, try_generate};
    use unpopped_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};
    let at = |dt: ElementKind| {
        let op = OpDef::elementwise("max", 2, &[dt], input(0).binary(BinaryOp::Max, input(1)));
        let a = OperandDesc::new(1, &[1 << 16], &[1], dt, 1);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        std::panic::catch_unwind(|| try_generate(&op, &key, &Cuda))
    };
    for dt in [
        ElementKind::I128,
        ElementKind::U128,
        ElementKind::I256,
        ElementKind::U256,
    ] {
        match at(dt).unwrap_or_else(|_| panic!("{dt:?}: the plan path PANICKED")) {
            Err(e) => {
                let t = format!("{e:?}");
                assert!(
                    t.contains("InadmissiblePlan") && t.contains(&format!("{dt:?}")),
                    "{dt:?}: the decline must be the plan admission naming the dtype: {t}"
                );
            }
            Ok(k) => panic!(
                "{dt:?}: the plan path EMITTED:
{}",
                k.source
            ),
        }
    }
    // Control: the bespoke elementwise surface is float-only today (an `I32` plan is declined the same way), so the control is `F32`.
    let k = at(F32)
        .expect("no panic")
        .expect("control: an F32 plan must emit");
    assert!(k.source.contains("__global__"), "{}", k.source);
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
        for k in [
            scale(),
            red(),
            async_kernel(async_body()),
            warp_sum(
                CollectiveKind::Reduce(unpopped::ir::ReduceOp::Sum),
                unpopped::exec::Scope::Subgroup,
                32,
            ),
        ] {
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

/// unpopped 0.61's integer bit operations on the KERNEL-IR surface. A float operand is refused by the checker by name
/// (`TypeError::IntegerOnlyOperand`); an `I32` operand is well typed and this emitter declines it BY NODE (`Unary` is not lowered
/// yet), never a panic. The kernel surface lowers no `Unary` node at all today (`Neg` declines the same way).
#[test]
fn the_integer_bit_ops_on_the_kernel_surface_are_refused_by_name_or_declined() {
    use unpopped::ir::UnaryOp;
    let build = |dt: ElementKind, op: UnaryOp| {
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
            "bitop",
            dt,
            vec![
                Param::buffer("x", dt, BufferRole::Read, Some(1)),
                Param::buffer("out", dt, BufferRole::Write, Some(1)),
                Param::scalar("n", U32, 0),
            ],
            vec![
                LocalDecl::scalar(U32),
                LocalDecl::scalar(U32),
                LocalDecl::scalar(dt),
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
                            value: ScalarExpr::Unary(op, u(ScalarExpr::Local(2))),
                        },
                    ]),
                    els: Block(Vec::new()),
                },
            ]),
        )
        .with_launch(LaunchShape::new([256, 1, 1], [4, 1, 1]))
    };
    // Controls: `scale()` (same shape, a `Mul` body) lowers, so a decline below is about the node. And `Neg` declines with the
    // same message: the kernel surface lowers NO `Unary` node yet, so the bit-op decline is by node, not specific to them.
    assert!(
        Cuda.lower_kernel(&scale()).is_ok(),
        "control: scale() must lower"
    );
    let neg = detail(Cuda.lower_kernel(&build(F32, UnaryOp::Neg)));
    assert!(
        neg.contains("`Unary`") && neg.contains("not lowered"),
        "{neg}"
    );
    for op in [UnaryOp::Popc, UnaryOp::Clz, UnaryOp::Brev] {
        let f = build(F32, op);
        assert!(
            matches!(
                check_kernel(&f),
                Err(KernelError::Type(
                    unpopped::types::TypeError::IntegerOnlyOperand { .. }
                ))
            ),
            "{op:?} on F32: the checker must refuse it by name: {:?}",
            check_kernel(&f)
        );
        let i = build(I32, op);
        assert!(
            check_kernel(&i).is_ok(),
            "{op:?} on I32 is well typed: {:?}",
            check_kernel(&i)
        );
        let d = detail(std::panic::catch_unwind(|| Cuda.lower_kernel(&i)).expect("no panic"));
        assert!(
            d.contains("`Unary`") && d.contains("not lowered"),
            "{op:?} on I32: {d}"
        );
    }
}

/// `Stmt::Assume` (unpopped 0.64) is a value fact an emitter may ignore: the emitted source is byte-equal with and without it,
/// at the start, in the middle and at the end of a body. Control: a statement that does lower (`Barrier`) DOES change the source,
/// so the comparison can tell two sources apart.
#[test]
fn an_assume_never_changes_the_emitted_source_of_the_rest_of_the_kernel() {
    let base = scale();
    check_kernel(&base).expect("well-formed");
    let want = Cuda.lower_kernel(&base).expect("the base lowers").source;
    let fact = || Stmt::Assume {
        cond: ScalarExpr::ConstInt(1, I32),
    };
    let n = base.body.0.len();
    assert!(n >= 1, "control: the base kernel has statements");
    for at in [0, n / 2, n] {
        let mut k = base.clone();
        k.body.0.insert(at, fact());
        check_kernel(&k).expect("a kernel with an Assume is well-formed");
        let got = Cuda
            .lower_kernel(&k)
            .expect("an Assume lowers (omitted)")
            .source;
        assert_eq!(got, want, "an Assume at {at} changed the emitted source");
    }
    let mut k = base.clone();
    k.body.0.insert(
        0,
        Stmt::Barrier {
            scope: unpopped::exec::Scope::Block,
            kind: unpopped::kernel::BarrierKind::Full,
        },
    );
    check_kernel(&k).expect("well-formed");
    let with_barrier = Cuda
        .lower_kernel(&k)
        .expect("a block barrier lowers")
        .source;
    assert_ne!(
        with_barrier, want,
        "control: a barrier must change the source"
    );
}
