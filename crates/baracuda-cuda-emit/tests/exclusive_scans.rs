// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exclusive warp scans (unpopped 0.58.4: `Max`, `Min` and `Prod` joined `Sum`/`And`/`Or`/`Xor` in the oracle, lane 0 = the fold's identity:
//! `-inf`/`+inf`/1 on a float, the type's minimum/maximum/1 on an integer).
//!
//! The CUDA lowering is generic over the scan kind only where the FOLD is: an integer `Max` / `Min` lowers with the type's least / greatest
//! value as the identity; a float `Max` / `Min` lowers with a NaN-propagating fold (a NaN anywhere in the fold gives NaN, the oracle's rule)
//! and `-inf` / `+inf`; `Prod` lowers for float and integer (wrapping on a signed integer) with identity 1. All use the existing constant
//! full-mask shuffle pattern (no vote, no mask operand in the IR). Each fixture passes `check_kernel`.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped_vocab::ElementKind;

/// One warp: every lane contributes the constant 3 of type `t` and stores its exclusive scan under `op`.
fn scan_kernel(t: ElementKind, op: ReduceOp, exclusive: bool) -> KernelFn {
    let init = if matches!(t, ElementKind::F32 | ElementKind::F64) {
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
fn float_max_min_and_prod_exclusive_scans_lower_with_their_identities() {
    use ElementKind::*;
    for (t, neg_inf, pos_inf) in [
        (
            F32,
            "__int_as_float(0xff800000)",
            "__int_as_float(0x7f800000)",
        ),
        (
            F64,
            "__longlong_as_double(0xfff0000000000000ll)",
            "__longlong_as_double(0x7ff0000000000000ll)",
        ),
    ] {
        let s = lowered(t, ReduceOp::Max, true);
        assert!(
            s.contains(neg_inf),
            "{t:?} exclusive Max: lane 0 is -inf:\n{s}"
        );
        assert!(
            s.contains("(_v) != (_v)") && s.contains("(_n) != (_n)"),
            "{t:?} Max: the fold propagates NaN:\n{s}"
        );
        let s = lowered(t, ReduceOp::Min, true);
        assert!(
            s.contains(pos_inf),
            "{t:?} exclusive Min: lane 0 is +inf:\n{s}"
        );
    }
    for (t, one) in [
        (F32, "(float)1"),
        (F64, "(double)1"),
        (I32, "(int)1"),
        (U32, "(unsigned int)1"),
    ] {
        let s = lowered(t, ReduceOp::Prod, true);
        assert!(
            s.contains(one),
            "{t:?} exclusive Prod: lane 0 is 1 ({one}):\n{s}"
        );
    }
    // Controls: an integer Max has no NaN test, and an inclusive float Max has no identity.
    let s = lowered(I32, ReduceOp::Max, true);
    assert!(
        !s.contains("(_v) != (_v)"),
        "an integer Max has no NaN test:\n{s}"
    );
    let s = lowered(F32, ReduceOp::Max, false);
    assert!(
        !s.contains("__int_as_float(0xff800000)"),
        "an inclusive scan has no identity:\n{s}"
    );
    // A signed integer Prod wraps through the unsigned type.
    let s = lowered(I32, ReduceOp::Prod, false);
    assert!(s.contains("(unsigned int)(_v) * (unsigned int)(_n)"), "{s}");
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

    #[test]
    fn float_max_min_and_prod_scans_compile_for_sm89() {
        let c = NvrtcCompiler::new(TargetId::parse("cuda:sm89").expect("token")).expect("nvrtc");
        for t in [ElementKind::F32, ElementKind::F64] {
            for op in [ReduceOp::Max, ReduceOp::Min, ReduceOp::Prod] {
                for exclusive in [false, true] {
                    let k = scan_kernel(t, op, exclusive);
                    check_kernel(&k).expect("well-formed");
                    let g = Cuda.lower_kernel(&k).expect("lowers");
                    c.compile(&g.source, &g.name, 10_000)
                        .unwrap_or_else(|e| panic!("{t:?} {op:?} {exclusive}: {e}\n{}", g.source));
                }
            }
        }
    }
}

/// Values on the device (ROADMAP 6.21: NVRTC proves syntax, not values). The lowered exclusive Max / Min scan is launched on a 64-thread block
/// (two warps: each scans its own 32 lanes, so lane 32 is a second "lane 0") and compared with unpopped's oracle (`run_launch`) AND with a
/// hand-written left fold. Data: a prefix with the type's extreme value, all lanes equal to the extreme, strictly ascending, strictly descending.
#[cfg(all(feature = "nvrtc", feature = "device-tests"))]
mod device {
    use super::*;
    use baracuda_cuda_emit::NvrtcCompiler;
    use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream, init};
    use unpopped::jit::Compiler;
    use unpopped::oracle::TypedBuffer;
    use unpopped::oracle::kernel_eval::KernelArg;
    use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
    use unpopped_vocab::TargetId;

    const THREADS: usize = 64;

    fn kernel_with(t: ElementKind, kind: CollectiveKind) -> KernelFn {
        KernelFn::new(
            "scan",
            t,
            vec![
                Param::buffer("inp", t, BufferRole::Read, Some(1)),
                Param::buffer("out", t, BufferRole::Write, Some(1)),
            ],
            vec![LocalDecl::scalar(t), LocalDecl::scalar(t)],
            Block(vec![
                Stmt::Load {
                    local: 0,
                    place: Place::Logical {
                        buffer: 0,
                        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
                    },
                },
                Stmt::Collective {
                    kind,
                    value: ScalarExpr::Local(0),
                    result: 1,
                    scope: Scope::Subgroup,
                },
                Stmt::Store {
                    place: Place::Logical {
                        buffer: 1,
                        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
                    },
                    value: ScalarExpr::Local(1),
                },
            ]),
        )
        .with_launch(LaunchShape::new([THREADS as u32, 1, 1], [1, 1, 1]))
    }

    fn width(t: ElementKind) -> usize {
        match t {
            ElementKind::I32 | ElementKind::U32 | ElementKind::F32 => 4,
            _ => 8,
        }
    }

    fn range(t: ElementKind) -> (i128, i128) {
        match t {
            ElementKind::I32 => (i32::MIN.into(), i32::MAX.into()),
            ElementKind::I64 => (i64::MIN.into(), i64::MAX.into()),
            _ => (0, u32::MAX.into()),
        }
    }

    fn enc(t: ElementKind, vals: &[i128]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| v.to_le_bytes()[..width(t)].to_vec())
            .collect()
    }

    fn dec(t: ElementKind, bytes: &[u8]) -> Vec<i128> {
        bytes
            .chunks(width(t))
            .map(|c| match t {
                ElementKind::I32 => i128::from(i32::from_le_bytes(c.try_into().unwrap())),
                ElementKind::I64 => i128::from(i64::from_le_bytes(c.try_into().unwrap())),
                ElementKind::F64 => i128::from(u64::from_le_bytes(c.try_into().unwrap())),
                _ => i128::from(u32::from_le_bytes(c.try_into().unwrap())),
            })
            .collect()
    }

    /// The spec: per warp, lane 0 = identity, lane i = fold of lanes 0..i.
    fn by_hand(t: ElementKind, op: ReduceOp, x: &[i128]) -> Vec<i128> {
        let (lo, hi) = range(t);
        let ident = if matches!(op, ReduceOp::Max) { lo } else { hi };
        let mut out = Vec::new();
        for warp in x.chunks(32) {
            let mut acc = ident;
            for v in warp {
                out.push(acc);
                acc = if matches!(op, ReduceOp::Max) {
                    acc.max(*v)
                } else {
                    acc.min(*v)
                };
            }
        }
        out
    }

    fn cases(t: ElementKind, op: ReduceOp) -> Vec<(&'static str, Vec<i128>)> {
        let (lo, hi) = range(t);
        let ext = if matches!(op, ReduceOp::Max) { hi } else { lo };
        let other = if matches!(op, ReduceOp::Max) { lo } else { hi };
        let mid = |i: i128| lo / 2 + hi / 2 + i;
        let mut prefix: Vec<i128> = (0..THREADS as i128).map(|i| mid(i * 7 % 13)).collect();
        prefix[3] = ext; // the extreme inside the prefix of warp 0
        prefix[40] = ext; // and inside warp 1
        vec![
            ("extreme in a prefix", prefix),
            ("every lane the extreme", vec![ext; THREADS]),
            ("every lane the opposite extreme", vec![other; THREADS]),
            ("ascending", (0..THREADS as i128).map(mid).collect()),
            (
                "descending",
                (0..THREADS as i128).map(|i| mid(-i)).collect(),
            ),
        ]
    }

    fn device_run(
        ctx: &Context,
        stream: &Stream,
        t: ElementKind,
        op: ReduceOp,
        x: &[i128],
    ) -> Vec<i128> {
        device_run_kind(
            ctx,
            stream,
            t,
            op,
            CollectiveKind::Scan {
                op,
                exclusive: true,
            },
            x,
        )
    }

    fn device_run_kind(
        ctx: &Context,
        stream: &Stream,
        t: ElementKind,
        op: ReduceOp,
        kind: CollectiveKind,
        x: &[i128],
    ) -> Vec<i128> {
        let k = kernel_with(t, kind);
        check_kernel(&k).expect("well-formed");
        let g = Cuda.lower_kernel(&k).expect("lowers");
        let cc = NvrtcCompiler::new(TargetId::parse("cuda:sm89").expect("token")).expect("nvrtc");
        let ptx = cc
            .compile(&g.source, &g.name, 10_000)
            .unwrap_or_else(|e| panic!("{t:?} {op:?}: {e}\n{}", g.source));
        let module = Module::load_ptx(ctx, &String::from_utf8(ptx).expect("utf8")).expect("load");
        let func = module.get_function(&g.name).expect("function");
        let bytes = enc(t, x);
        let d_in = DeviceBuffer::<u8>::from_slice(ctx, &bytes).expect("upload");
        let d_out = DeviceBuffer::<u8>::zeros(ctx, bytes.len()).expect("alloc");
        let (p_in, p_out) = (d_in.as_raw(), d_out.as_raw());
        // The argument block of two rank-1 buffers: { shape, stride, base } each (see the emitted `Args_scan`).
        let blk: [i64; 6] = [THREADS as i64, 1, 0, THREADS as i64, 1, 0];
        // SAFETY: the kernel is `(T* p0, T* p1, Args_scan a)` with `Args_scan` = 6 x i64 (the static_assert in the source pins 48 bytes
        // for two rank-1 buffers); both buffers hold `THREADS` elements and outlive the launch (synchronised below).
        unsafe {
            func.launch()
                .grid((1, 1, 1))
                .block((THREADS as u32, 1, 1))
                .stream(stream)
                .arg(&p_in)
                .arg(&p_out)
                .arg(&blk)
                .launch()
                .expect("launch");
        }
        stream.synchronize().expect("sync");
        let mut got = vec![0u8; bytes.len()];
        d_out.copy_to_host(&mut got).expect("download");
        dec(t, &got)
    }

    fn oracle_run(t: ElementKind, op: ReduceOp, x: &[i128]) -> Vec<i128> {
        oracle_run_kind(
            t,
            CollectiveKind::Scan {
                op,
                exclusive: true,
            },
            x,
        )
    }

    fn oracle_run_kind(t: ElementKind, kind: CollectiveKind, x: &[i128]) -> Vec<i128> {
        let k = kernel_with(t, kind);
        let n = THREADS as i64;
        let mut args = vec![
            KernelArg::Buffer(TypedBuffer::new(t, vec![n], vec![1], enc(t, x))),
            KernelArg::Buffer(TypedBuffer::new(
                t,
                vec![n],
                vec![1],
                vec![0; THREADS * width(t)],
            )),
        ];
        run_launch_with_subgroup(&k, &mut args, Some(32)).expect("the oracle runs it");
        match &args[1] {
            KernelArg::Buffer(b) => dec(t, &b.bytes),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn exclusive_integer_max_min_scans_match_the_oracle_on_the_device() {
        init().expect("driver init");
        let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
        let stream = Stream::new(&ctx).expect("stream");
        for t in [ElementKind::I32, ElementKind::I64, ElementKind::U32] {
            for op in [ReduceOp::Max, ReduceOp::Min] {
                for (name, x) in cases(t, op) {
                    let want = by_hand(t, op, &x);
                    assert_eq!(
                        oracle_run(t, op, &x),
                        want,
                        "{t:?} {op:?} {name}: the oracle and the hand fold disagree"
                    );
                    let got = device_run(&ctx, &stream, t, op, &x);
                    assert_eq!(got, want, "{t:?} {op:?} {name}: device != oracle");
                }
            }
        }
    }

    /// The element as an `f64` (a float's bit pattern decoded), for the hand fold.
    fn fval(t: ElementKind, bits: i128) -> f64 {
        match t {
            ElementKind::F32 => f64::from(f32::from_bits(bits as u32)),
            _ => f64::from_bits(bits as u64),
        }
    }

    fn fbits(t: ElementKind, v: f64) -> i128 {
        match t {
            ElementKind::F32 => i128::from((v as f32).to_bits()),
            _ => i128::from(v.to_bits()),
        }
    }

    /// Equal as values: a NaN equals a NaN, `-0.0` equals `+0.0` (signed zeros are not pinned), an integer compares exactly.
    fn same(t: ElementKind, a: i128, b: i128) -> bool {
        if matches!(t, ElementKind::F32 | ElementKind::F64) {
            let (x, y) = (fval(t, a), fval(t, b));
            (x.is_nan() && y.is_nan()) || x == y
        } else {
            a == b
        }
    }

    /// An integer wrapped into `t`'s width.
    fn wrap(t: ElementKind, v: i128) -> i128 {
        match t {
            ElementKind::I32 => i128::from(v as i32),
            ElementKind::I64 => i128::from(v as i64),
            _ => i128::from(v as u32),
        }
    }

    /// The fold of two elements: a float `Max` / `Min` propagates NaN, `Prod` multiplies (wrapping on an integer).
    fn fold(t: ElementKind, op: ReduceOp, a: i128, b: i128) -> i128 {
        if matches!(t, ElementKind::F32 | ElementKind::F64) {
            let (x, y) = (fval(t, a), fval(t, b));
            let r = match op {
                ReduceOp::Max if x.is_nan() || y.is_nan() => f64::NAN,
                ReduceOp::Min if x.is_nan() || y.is_nan() => f64::NAN,
                ReduceOp::Max => x.max(y),
                ReduceOp::Min => x.min(y),
                _ => x * y,
            };
            fbits(t, r)
        } else {
            match op {
                ReduceOp::Max => a.max(b),
                ReduceOp::Min => a.min(b),
                _ => wrap(t, a * b),
            }
        }
    }

    fn identity(t: ElementKind, op: ReduceOp) -> i128 {
        match (t, op) {
            (ElementKind::F32 | ElementKind::F64, ReduceOp::Max) => fbits(t, f64::NEG_INFINITY),
            (ElementKind::F32 | ElementKind::F64, ReduceOp::Min) => fbits(t, f64::INFINITY),
            (ElementKind::F32 | ElementKind::F64, _) => fbits(t, 1.0),
            _ => 1,
        }
    }

    /// Per warp: `0` = inclusive scan, `1` = exclusive scan (lane 0 the identity), `2` = reduce (every lane the whole fold).
    fn spec(t: ElementKind, op: ReduceOp, shape: u8, x: &[i128]) -> Vec<i128> {
        let mut out = Vec::new();
        for warp in x.chunks(32) {
            let total = warp
                .iter()
                .copied()
                .reduce(|a, b| fold(t, op, a, b))
                .unwrap();
            let mut acc = identity(t, op);
            for (i, v) in warp.iter().enumerate() {
                match shape {
                    0 => {
                        acc = if i == 0 { *v } else { fold(t, op, acc, *v) };
                        out.push(acc);
                    }
                    1 => {
                        out.push(acc);
                        acc = fold(t, op, acc, *v);
                    }
                    _ => out.push(total),
                }
            }
        }
        out
    }

    fn float_cases(t: ElementKind, op: ReduceOp) -> Vec<(&'static str, Vec<i128>)> {
        let f = |v: f64| fbits(t, v);
        let mut nan_prefix: Vec<i128> = (0..THREADS).map(|i| f((i % 9) as f64 - 4.0)).collect();
        nan_prefix[3] = f(f64::NAN); // inside warp 0
        nan_prefix[40] = f(f64::NAN); // inside warp 1
        if matches!(op, ReduceOp::Prod) {
            // Powers of two and signs only: every product and every partial product is exact in f32 and f64, so the
            // order of the folds (a scan folds in a different order from the left fold) cannot change the value.
            let cyc = [2.0, 0.5, -1.0, 1.0, -2.0, 0.5, 1.0];
            let mut zero = vec![0.0; THREADS];
            for (i, z) in zero.iter_mut().enumerate() {
                *z = cyc[i % cyc.len()];
            }
            zero[5] = 0.0;
            zero[37] = 0.0;
            return vec![
                (
                    "powers of two and signs",
                    (0..THREADS).map(|i| f(cyc[i % cyc.len()])).collect(),
                ),
                ("a zero in each warp", zero.iter().map(|v| f(*v)).collect()),
                ("a NaN in each warp", nan_prefix),
                ("every lane 1", vec![f(1.0); THREADS]),
            ];
        }
        vec![
            (
                "ascending",
                (0..THREADS).map(|i| f(i as f64 - 20.0)).collect(),
            ),
            (
                "descending",
                (0..THREADS).map(|i| f(20.0 - i as f64)).collect(),
            ),
            ("a NaN in each warp", nan_prefix),
            (
                "infinities",
                (0..THREADS)
                    .map(|i| match i % 5 {
                        0 => f(f64::INFINITY),
                        1 => f(f64::NEG_INFINITY),
                        _ => f(i as f64),
                    })
                    .collect(),
            ),
            ("all equal", vec![f(1.5); THREADS]),
        ]
    }

    fn int_prod_cases() -> Vec<(&'static str, Vec<i128>)> {
        vec![
            (
                "odd numbers (wraps)",
                (0..THREADS as i128).map(|i| 2 * i + 3).collect(),
            ),
            ("every lane 1", vec![1; THREADS]),
            (
                "a zero in each warp",
                (0..THREADS as i128)
                    .map(|i| if i == 5 || i == 37 { 0 } else { i % 4 + 1 })
                    .collect(),
            ),
        ]
    }

    /// ROADMAP S13: float `Max` / `Min` (NaN propagating, `-inf` / `+inf` identity) and `Prod` (identity 1), as an inclusive scan, an
    /// exclusive scan and a reduce, on the device against the oracle AND a hand fold. NVRTC proves syntax, not values (6.21).
    #[test]
    fn float_max_min_and_prod_collectives_match_the_oracle_on_the_device() {
        init().expect("driver init");
        let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
        let stream = Stream::new(&ctx).expect("stream");
        let mut checked = 0usize;
        let mut work: Vec<(ElementKind, ReduceOp)> = Vec::new();
        for t in [ElementKind::F32, ElementKind::F64] {
            for op in [ReduceOp::Max, ReduceOp::Min, ReduceOp::Prod] {
                work.push((t, op));
            }
        }
        for t in [ElementKind::I32, ElementKind::I64, ElementKind::U32] {
            work.push((t, ReduceOp::Prod));
        }
        for (t, op) in work {
            let cases = if matches!(t, ElementKind::F32 | ElementKind::F64) {
                float_cases(t, op)
            } else {
                int_prod_cases()
            };
            for (name, x) in cases {
                for (shape, kind) in [
                    (
                        0u8,
                        CollectiveKind::Scan {
                            op,
                            exclusive: false,
                        },
                    ),
                    (
                        1u8,
                        CollectiveKind::Scan {
                            op,
                            exclusive: true,
                        },
                    ),
                    (2u8, CollectiveKind::Reduce(op)),
                ] {
                    let want = spec(t, op, shape, &x);
                    let oracle = oracle_run_kind(t, kind.clone(), &x);
                    let got = device_run_kind(&ctx, &stream, t, op, kind, &x);
                    for lane in 0..THREADS {
                        assert!(
                            same(t, oracle[lane], want[lane]),
                            "{t:?} {op:?} shape {shape} {name} lane {lane}: the oracle {:?} and the hand fold {:?} disagree",
                            oracle[lane],
                            want[lane]
                        );
                        assert!(
                            same(t, got[lane], want[lane]),
                            "{t:?} {op:?} shape {shape} {name} lane {lane}: device {:?} != oracle {:?}",
                            got[lane],
                            want[lane]
                        );
                        checked += 1;
                    }
                }
            }
        }
        eprintln!("S13 collectives: {checked} lane results compared (device = oracle = hand fold)");
        assert!(checked > 0);
    }
}
