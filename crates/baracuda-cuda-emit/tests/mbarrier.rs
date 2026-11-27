// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP S12a-2: the plain `mbarrier` statements `MbarInit`, `MbarArrive` (no transaction count) and `MbarWait`, lowered to the sm_80
//! spellings `mbarrier.init`, `mbarrier.arrive` and a `mbarrier.test_wait.parity` loop on a `Shared` allocation of 64-bit elements.
//!
//! Unlike the tcgen05 statements these run on the lane's box (an RTX 4070, sm_89): the device test below launches the lowered kernel and
//! compares it with unpopped's oracle and with the spec. Not lowered, and not built (pm-kernels 2026-11-27, ROADMAP S12 (b)): an
//! `mbarrier.arrive` with an explicit count or a `ctaMask`, an `expect_tx` multicast; `MbarArrive` with a transaction count and `MbarCompleteTx`
//! are refused by name.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll, check_kernel,
};
use unpopped_vocab::ElementKind;

const THREADS: u32 = 64;
const HALF: u32 = 32;
/// Steps of warp 0's dependent chain before it writes the data, and its multiplier and increment (a 32-bit LCG).
const SPIN: i64 = 20_000;
const LCG_A: i64 = 1_664_525;
const LCG_C: i64 = 1_013_904_223;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn u32c(v: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(v, ElementKind::U32)
}

fn lt(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Binary(BinaryOp::CmpLt, Box::new(a), Box::new(b))
}

fn bar() -> Place {
    Place::Alloc {
        alloc: 1,
        index: u32c(0),
    }
}

/// Warp 0 first runs a LONG DEPENDENT CHAIN (a linear congruential generator, `SPIN` steps, whose final value it stores to `out`, so no
/// compiler can fold or drop it), then writes `data[t] = chain(t)` and arrives at `bar`; warp 1 waits for `bar` (parity 0) and reads
/// `data[t - 32]` into `out[t]`. Thread 0 initialised the barrier with an arrival count of 32 (warp 0's arrivals) before a block barrier,
/// and `data` is zeroed before it. A `MbarWait` that did not wait would let warp 1 read zeros long before warp 0 has written the data.
fn kernel(arrive: Stmt) -> KernelFn {
    let t = tid();
    let data_at = |i: ScalarExpr| Place::Alloc { alloc: 0, index: i };
    let out_at = |i: ScalarExpr| Place::Logical {
        buffer: 0,
        coords: vec![i],
    };
    KernelFn::new(
        "mbar",
        ElementKind::U32,
        vec![Param::buffer(
            "out",
            ElementKind::U32,
            BufferRole::Write,
            Some(1),
        )],
        vec![
            LocalDecl::scalar(ElementKind::U32),
            LocalDecl::scalar(ElementKind::U32),
            LocalDecl::scalar(ElementKind::U32),
        ],
        Block(vec![
            Stmt::If {
                cond: lt(t.clone(), u32c(1)),
                then: Block(vec![Stmt::MbarInit {
                    bar: bar(),
                    count: u32c(i64::from(HALF)),
                }]),
                els: Block(vec![]),
            },
            // Shared memory keeps the previous launch's bytes: zero the data before the barrier, so a read that came too early is visibly wrong.
            Stmt::If {
                cond: lt(t.clone(), u32c(i64::from(HALF))),
                then: Block(vec![Stmt::Store {
                    place: data_at(t.clone()),
                    value: u32c(0),
                }]),
                els: Block(vec![]),
            },
            Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            },
            Stmt::If {
                cond: lt(t.clone(), u32c(i64::from(HALF))),
                then: Block(vec![
                    Stmt::Let {
                        local: 2,
                        init: t.clone(),
                    },
                    Stmt::For {
                        local: 1,
                        start: u32c(0),
                        end: u32c(SPIN),
                        step: u32c(1),
                        unroll: Unroll::default(),
                        body: Block(vec![Stmt::Assign {
                            local: 2,
                            value: ScalarExpr::Add(
                                Box::new(ScalarExpr::Mul(
                                    Box::new(ScalarExpr::Local(2)),
                                    Box::new(u32c(LCG_A)),
                                )),
                                Box::new(u32c(LCG_C)),
                            ),
                        }]),
                    },
                    // The data IS the chain's final value, so the store cannot be moved before the chain (nor the arrival, which is
                    // `asm volatile` with a memory clobber, before the store).
                    Stmt::Store {
                        place: data_at(t.clone()),
                        value: ScalarExpr::Local(2),
                    },
                    arrive,
                    Stmt::Store {
                        place: out_at(t.clone()),
                        value: ScalarExpr::Local(2),
                    },
                ]),
                els: Block(vec![
                    Stmt::MbarWait {
                        bar: bar(),
                        parity: u32c(0),
                    },
                    Stmt::Load {
                        local: 0,
                        place: data_at(ScalarExpr::Sub(
                            Box::new(t.clone()),
                            Box::new(u32c(i64::from(HALF))),
                        )),
                    },
                    Stmt::Store {
                        place: out_at(t),
                        value: ScalarExpr::Local(0),
                    },
                ]),
            },
        ]),
    )
    .with_allocs(vec![
        AllocDecl::new(
            "data",
            MemorySpace::Shared,
            ElementKind::U32,
            Extent::Static(HALF),
        ),
        AllocDecl::new(
            "bar",
            MemorySpace::Shared,
            ElementKind::I64,
            Extent::Static(1),
        ),
    ])
    .with_launch(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]))
}

fn good() -> KernelFn {
    kernel(Stmt::MbarArrive {
        bar: bar(),
        tx: None,
    })
}

#[test]
fn the_plain_mbarrier_statements_lower_to_sm80_spellings() {
    let k = good();
    check_kernel(&k).expect("well-formed");
    let g = Cuda.lower_kernel(&k).expect("lowers");
    let s = &g.source;
    for want in [
        "mbarrier.init.shared::cta.b64 [%0], %1;",
        "mbarrier.arrive.shared::cta.b64 %0, [%1];",
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;",
        "__cvta_generic_to_shared(&sh1[",
        "#error \"mbarrier needs sm_80 or newer\"",
    ] {
        assert!(s.contains(want), "missing `{want}`:\n{s}");
    }
    assert!(
        format!("{:?}", g.interface().expect("an interface")).contains("mbarrier (sm_80 or newer)")
    );
}

/// Controls: an arrival that carries a transaction count, and a barrier that is not a 64-bit shared element, are refused by name, and the same
/// kernel without those lowers (the test above).
#[test]
fn an_arrive_with_a_transaction_count_and_a_bad_barrier_are_refused_by_name() {
    let k = kernel(Stmt::MbarArrive {
        bar: bar(),
        tx: Some(u32c(16)),
    });
    check_kernel(&k).expect("well-formed");
    match Cuda.lower_kernel(&k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(
                detail.contains("transaction count") && !detail.contains("ill-formed"),
                "{detail}"
            );
        }
        other => panic!("must be refused by the emitter: {other:?}"),
    }
}

#[cfg(all(feature = "nvrtc", feature = "device-tests"))]
mod device {
    use super::*;
    use baracuda_cuda_emit::NvrtcCompiler;
    use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream, init};
    use unpopped::jit::Compiler;
    use unpopped::oracle::TypedBuffer;
    use unpopped::oracle::kernel_eval::KernelArg;
    use unpopped::oracle::kernel_launch::run_launch;
    use unpopped_vocab::TargetId;

    /// The final value of warp 0's chain for thread `t`.
    fn chain(t: u32) -> u32 {
        (0..SPIN).fold(t, |x, _| {
            x.wrapping_mul(LCG_A as u32).wrapping_add(LCG_C as u32)
        })
    }

    fn spec() -> Vec<u32> {
        (0..THREADS)
            .map(|t| if t < HALF { chain(t) } else { chain(t - HALF) })
            .collect()
    }

    fn dec(b: &[u8]) -> Vec<u32> {
        b.chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn oracle() -> Vec<u32> {
        let k = good();
        let mut args = vec![KernelArg::Buffer(TypedBuffer::new(
            ElementKind::U32,
            vec![i64::from(THREADS)],
            vec![1],
            vec![0; THREADS as usize * 4],
        ))];
        run_launch(&k, &mut args).expect("the oracle runs it");
        match &args[0] {
            KernelArg::Buffer(b) => dec(&b.bytes),
            other => panic!("{other:?}"),
        }
    }

    fn device() -> Vec<u32> {
        init().expect("driver init");
        let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
        let stream = Stream::new(&ctx).expect("stream");
        let g = Cuda.lower_kernel(&good()).expect("lowers");
        let cc = NvrtcCompiler::new(TargetId::parse("cuda:sm89").expect("token")).expect("nvrtc");
        let ptx = cc
            .compile(&g.source, &g.name, 10_000)
            .unwrap_or_else(|e| panic!("nvrtc: {e}\n{}", g.source));
        let module = Module::load_ptx(&ctx, &String::from_utf8(ptx).expect("utf8")).expect("load");
        let func = module.get_function(&g.name).expect("function");
        let n = THREADS as usize * 4;
        let d_out = DeviceBuffer::<u8>::zeros(&ctx, n).expect("alloc");
        let p_out = d_out.as_raw();
        // The argument block of one rank-1 buffer: { shape, stride, base } (see the emitted `Args_mbar`).
        let blk: [i64; 3] = [i64::from(THREADS), 1, 0];
        // SAFETY: the kernel is `(int* p0, Args_mbar a)` with `Args_mbar` = 3 x i64; the buffer holds `THREADS` ints and outlives the launch.
        unsafe {
            func.launch()
                .grid((1, 1, 1))
                .block((THREADS, 1, 1))
                .stream(&stream)
                .arg(&p_out)
                .arg(&blk)
                .launch()
                .expect("launch");
        }
        stream.synchronize().expect("sync");
        let mut got = vec![0u8; n];
        d_out.copy_to_host(&mut got).expect("download");
        dec(&got)
    }

    /// The lowered kernel on the device (sm_89, through `gpu-run`), the oracle and the spec all agree: warp 1 saw warp 0's data after the
    /// phase completed (32 arrivals), so the wait released only after every arrival.
    #[test]
    fn the_lowered_barrier_kernel_matches_the_oracle_and_the_spec_on_the_device() {
        let want = spec();
        assert_eq!(oracle(), want, "the oracle and the spec disagree");
        for run in 0..20 {
            assert_eq!(device(), want, "device != spec on run {run}");
        }
    }

    /// The same source for `sm_75` is refused by the arch guard (a control: the acceptance above is about the target).
    #[test]
    fn the_same_source_is_refused_below_sm_80() {
        let g = Cuda.lower_kernel(&good()).expect("lowers");
        let c = NvrtcCompiler::new(TargetId::parse("cuda:sm75").expect("token")).expect("nvrtc");
        match c.compile(&g.source, &g.name, 10_000) {
            Err(e) => assert!(e.to_string().contains("mbarrier needs sm_80"), "{e}"),
            Ok(_) => panic!("sm_70 must be refused"),
        }
    }
}
