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

    fn kernel(t: ElementKind, op: ReduceOp) -> KernelFn {
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
                    kind: CollectiveKind::Scan {
                        op,
                        exclusive: true,
                    },
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
            ElementKind::I32 | ElementKind::U32 => 4,
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
        let k = kernel(t, op);
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
        let k = kernel(t, op);
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
}
