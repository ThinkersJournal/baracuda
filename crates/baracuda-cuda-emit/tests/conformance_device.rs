// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP S7 slice 1 (and S6 on the same fixtures): ONE test that runs kernel-layer kernels through the CUDA emitter **on the device**
//! and compares every output byte with Unpopped's oracle (`unpopped::oracle::run_kernel`), which is an independent implementation of the
//! same semantics. A disagreement is a defect in the emitter or in the oracle, and the test names the case and the first differing byte.
//!
//! Before this test the kernel-layer lowering was only ever compiled (`kernel_lowering.rs`'s `nvrtc` module), never run.
//!
//! Cases: a guarded global elementwise kernel (control flow, scalar parameter, bounds-checked addressing), `Shared` + a block `Barrier`,
//! a `Private` per-thread array (S1 slice 3) and a **named barrier** over two warps (the 0.56.13 fix: on the previous emitter the barrier
//! was a `__syncthreads()`).
//!
//! **Hardware limit:** this box is sm_89. The Hopper statements (mbarrier, `griddepcontrol`, `cp.async.bulk`, `stmatrix`, cluster barriers)
//! cannot run here and are not covered; their evidence is the `sm_90a` NVRTC assemble tests.
//!
//! Run with `pwsh scripts/gpu-run.ps1 -Project baracuda -- cargo gpu-test -p baracuda-cuda-emit --features nvrtc,device-tests --test conformance_device`
//! (the machine-wide GPU lock; see `docs/DEVICE-TESTS.md`). Not compiled without both features, so the hosted suite never builds it.
#![cfg(all(feature = "nvrtc", feature = "device-tests"))]

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::{Context, Device, DeviceBuffer, Module, Stream, init};
use unpopped::backend::{Backend, GeneratedKernel};
use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::jit::Compiler;
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
};
use unpopped::oracle::TypedBuffer;
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch;
use unpopped_vocab::{ArchSku, ElementKind};

const F32: ElementKind = ElementKind::F32;
const U32: ElementKind = ElementKind::U32;

fn b(e: ScalarExpr) -> Box<ScalarExpr> {
    Box::new(e)
}

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn cu(n: i64) -> ScalarExpr {
    ScalarExpr::ConstInt(n, U32)
}

fn at(buffer: u32, i: ScalarExpr) -> Place {
    Place::Logical {
        buffer,
        coords: vec![i],
    }
}

fn alloc_at(alloc: u32, i: ScalarExpr) -> Place {
    Place::Alloc { alloc, index: i }
}

/// What one parameter holds, for both sides.
enum Arg {
    /// A buffer of `dtype` elements (little-endian bytes); an output starts as zeros.
    Buf(ElementKind, Vec<u8>),
    /// An integer scalar parameter.
    Int(i128),
}

struct Case {
    tag: &'static str,
    k: KernelFn,
    args: Vec<Arg>,
    /// Text the emitted source must contain (a spelling the device result cannot prove by itself).
    must_spell: Option<&'static str>,
}

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `if i < n { out[i] = x[i] * 2.0 }`, `i = blockIdx.x * blockDim.x + threadIdx.x`; 1000 elements over 4 blocks of 256.
fn scale() -> Case {
    let i = ScalarExpr::Add(
        b(ScalarExpr::Mul(
            b(ScalarExpr::Exec(ExecId::BlockIdx(0))),
            b(ScalarExpr::Exec(ExecId::BlockDim(0))),
        )),
        b(tid()),
    );
    let k = KernelFn::new(
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
                    b(ScalarExpr::Local(1)),
                    b(ScalarExpr::Local(0)),
                ),
                then: Block(vec![
                    Stmt::Load {
                        local: 2,
                        place: at(0, ScalarExpr::Local(1)),
                    },
                    Stmt::Store {
                        place: at(1, ScalarExpr::Local(1)),
                        value: ScalarExpr::Mul(b(ScalarExpr::Local(2)), b(ScalarExpr::Const(2.0))),
                    },
                ]),
                els: Block(Vec::new()),
            },
        ]),
    )
    .with_launch(LaunchShape::new([256, 1, 1], [4, 1, 1]));
    let xs: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5 - 100.0).collect();
    Case {
        tag: "scale",
        k,
        args: vec![
            Arg::Buf(F32, f32_bytes(&xs)),
            Arg::Buf(F32, vec![0; 1000 * 4]),
            Arg::Int(1000),
        ],
        must_spell: None,
    }
}

/// `sh[tid] = tid + 1; barrier; out[tid] = sh[last - tid]` over `threads` threads, the barrier being `kind`.
fn reverse_through_shared(tag: &'static str, threads: u32, kind: BarrierKind) -> Case {
    let last = i64::from(threads) - 1;
    let named = matches!(kind, BarrierKind::Named { .. });
    let k = KernelFn::new(
        tag,
        U32,
        vec![Param::buffer("out", U32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(U32)],
        Block(vec![
            Stmt::Store {
                place: alloc_at(0, tid()),
                value: ScalarExpr::Add(b(tid()), b(cu(1))),
            },
            Stmt::Barrier {
                scope: Scope::Block,
                kind,
            },
            Stmt::Load {
                local: 0,
                place: alloc_at(0, ScalarExpr::Sub(b(cu(last)), b(tid()))),
            },
            Stmt::Store {
                place: at(0, tid()),
                value: ScalarExpr::Local(0),
            },
        ]),
    )
    .with_allocs(vec![AllocDecl::new(
        "sh",
        MemorySpace::Shared,
        U32,
        Extent::Static(threads),
    )])
    .with_launch(LaunchShape::new([threads, 1, 1], [1, 1, 1]));
    Case {
        tag,
        k,
        args: vec![Arg::Buf(U32, vec![0; threads as usize * 4])],
        must_spell: named.then_some("bar.sync"),
    }
}

/// `r[0] = 7; r[1] = tid * 2 + 1` in a per-thread array, then `out[tid] = r[1] + r[0]` (the oracle traps on a read of an unwritten element, so both are written).
fn private_array() -> Case {
    let k = KernelFn::new(
        "private",
        U32,
        vec![Param::buffer("out", U32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
        Block(vec![
            Stmt::Store {
                place: alloc_at(0, cu(0)),
                value: cu(7),
            },
            Stmt::Store {
                place: alloc_at(0, cu(1)),
                value: ScalarExpr::Add(b(ScalarExpr::Mul(b(tid()), b(cu(2)))), b(cu(1))),
            },
            Stmt::Load {
                local: 0,
                place: alloc_at(0, cu(1)),
            },
            Stmt::Load {
                local: 1,
                place: alloc_at(0, cu(0)),
            },
            Stmt::Store {
                place: at(0, tid()),
                value: ScalarExpr::Add(b(ScalarExpr::Local(0)), b(ScalarExpr::Local(1))),
            },
        ]),
    )
    .with_allocs(vec![AllocDecl::new(
        "r",
        MemorySpace::Private,
        U32,
        Extent::Static(4),
    )])
    .with_launch(LaunchShape::new([8, 1, 1], [1, 1, 1]));
    Case {
        tag: "private",
        k,
        args: vec![Arg::Buf(U32, vec![0; 8 * 4])],
        must_spell: None,
    }
}

/// 128 threads in two groups of 64: threads below 64 use named barrier 1, the others barrier 2, each group reversing its own half of
/// `Shared`. The barriers sit under thread-dependent control, which a `Full` barrier may not, so a `__syncthreads()` lowering is not a
/// valid reading of this kernel (the oracle accepts it only because the barriers are named).
fn named_barrier_groups() -> Case {
    let half = |lo: i64, id: u32| {
        // index of the partner within the group: lo + 63 - (tid - lo) = 2*lo + 63 - tid
        Block(vec![
            Stmt::Store {
                place: alloc_at(0, tid()),
                value: ScalarExpr::Add(b(tid()), b(cu(1))),
            },
            Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Named { id, count: 64 },
            },
            Stmt::Load {
                local: 0,
                place: alloc_at(0, ScalarExpr::Sub(b(cu(2 * lo + 63)), b(tid()))),
            },
            Stmt::Store {
                place: at(0, tid()),
                value: ScalarExpr::Local(0),
            },
        ])
    };
    let k = KernelFn::new(
        "named_groups",
        U32,
        vec![Param::buffer("out", U32, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(U32)],
        Block(vec![Stmt::If {
            cond: ScalarExpr::Binary(BinaryOp::CmpLt, b(tid()), b(cu(64))),
            then: half(0, 1),
            els: half(64, 2),
        }]),
    )
    .with_allocs(vec![AllocDecl::new(
        "sh",
        MemorySpace::Shared,
        U32,
        Extent::Static(128),
    )])
    .with_launch(LaunchShape::new([128, 1, 1], [1, 1, 1]));
    Case {
        tag: "named_groups",
        k,
        args: vec![Arg::Buf(U32, vec![0; 128 * 4])],
        must_spell: Some("bar.sync"),
    }
}

fn cases() -> Vec<Case> {
    vec![
        scale(),
        reverse_through_shared("shared_block_barrier", 128, BarrierKind::Full),
        // The named barrier synchronises exactly 64 threads (two warps); the block is 64 threads, so both sides agree only if it really waits.
        reverse_through_shared("named_barrier", 64, BarrierKind::Named { id: 1, count: 64 }),
        named_barrier_groups(),
        private_array(),
    ]
}

/// The oracle's output bytes for each buffer parameter, in parameter order.
fn oracle_outputs(c: &Case) -> Result<Vec<Vec<u8>>, String> {
    let mut args: Vec<KernelArg> = c
        .args
        .iter()
        .map(|a| match a {
            Arg::Buf(dt, bytes) => {
                let n = (bytes.len() / elem_bytes(*dt)) as i64;
                KernelArg::Buffer(TypedBuffer::new(*dt, vec![n], vec![1], bytes.clone()))
            }
            Arg::Int(v) => KernelArg::Int(*v),
        })
        .collect();
    run_launch(&c.k, &mut args).map_err(|e| format!("{}: the oracle trapped: {e:?}", c.tag))?;
    Ok(args
        .into_iter()
        .filter_map(|a| match a {
            KernelArg::Buffer(t) => Some(t.bytes),
            _ => None,
        })
        .collect())
}

fn elem_bytes(t: ElementKind) -> usize {
    match t {
        ElementKind::F32 | ElementKind::U32 | ElementKind::I32 => 4,
        other => panic!("conformance_device: unsupported {other:?}"),
    }
}

/// The by-value argument block bytes, built from `KernelInterface::params_block` (so a layout bug in the emitter shows up here).
fn params_block(g: &GeneratedKernel, c: &Case) -> Result<Vec<u8>, String> {
    let pb = g
        .interface()
        .and_then(|i| i.params_block.clone())
        .ok_or_else(|| format!("{}: no params block in the interface", c.tag))?;
    let mut out = vec![0u8; pb.size as usize];
    for f in &pb.fields {
        let name = f.name.as_str();
        let idx: usize = name[1..]
            .split(|ch: char| !ch.is_ascii_digit())
            .next()
            .and_then(|d| d.parse().ok())
            .ok_or_else(|| format!("{}: unparsable params field `{name}`", c.tag))?;
        let value: i128 = if name.contains("_shape") {
            match &c.args[idx] {
                Arg::Buf(dt, bytes) => (bytes.len() / elem_bytes(*dt)) as i128,
                Arg::Int(_) => return Err(format!("{}: shape of a scalar", c.tag)),
            }
        } else if name.contains("_stride") {
            1
        } else if name.ends_with("_base") {
            0
        } else {
            match &c.args[idx] {
                Arg::Int(v) => *v,
                Arg::Buf(..) => return Err(format!("{}: scalar field for a buffer", c.tag)),
            }
        };
        let w = match f.ty {
            ElementKind::I64 | ElementKind::U64 => 8,
            ElementKind::I32 | ElementKind::U32 => 4,
            other => return Err(format!("{}: params field type {other:?}", c.tag)),
        };
        let off = f.offset as usize;
        out[off..off + w].copy_from_slice(&value.to_le_bytes()[..w]);
    }
    Ok(out)
}

fn run_on_device(
    ctx: &Context,
    stream: &Stream,
    cc: &NvrtcCompiler,
    c: &Case,
) -> Result<Vec<Vec<u8>>, String> {
    check_kernel(&c.k).map_err(|e| format!("{}: ill-formed: {e:?}", c.tag))?;
    let g = Cuda
        .lower_kernel(&c.k)
        .map_err(|e| format!("{}: the emitter declined: {e:?}", c.tag))?;
    if let Some(t) = c.must_spell {
        if !g.source.contains(t) {
            return Err(format!(
                "{}: the emitted source does not contain `{t}`:
{}",
                c.tag, g.source
            ));
        }
        if g.source.contains("__syncthreads") {
            return Err(format!(
                "{}: a named-barrier kernel printed __syncthreads():
{}",
                c.tag, g.source
            ));
        }
    }
    let ptx = cc
        .compile(&g.source, &g.name, 10_000)
        .map_err(|e| format!("{}: nvrtc: {e}\n{}", c.tag, g.source))?;
    let ptx = String::from_utf8(ptx).map_err(|e| format!("{}: ptx utf8: {e}", c.tag))?;
    let module = Module::load_ptx(ctx, &ptx).map_err(|e| format!("{}: load: {e:?}", c.tag))?;
    let func = module
        .get_function(&g.name)
        .map_err(|e| format!("{}: function: {e:?}", c.tag))?;
    let block = params_block(&g, c)?;
    let mut blk = [0u8; 512];
    if block.len() > blk.len() {
        return Err(format!("{}: params block of {} bytes", c.tag, block.len()));
    }
    blk[..block.len()].copy_from_slice(&block);
    let bufs: Vec<DeviceBuffer<u8>> = c
        .args
        .iter()
        .filter_map(|a| match a {
            Arg::Buf(_, bytes) => {
                Some(DeviceBuffer::from_slice(ctx, bytes).expect("upload a buffer"))
            }
            Arg::Int(_) => None,
        })
        .collect();
    let ptrs: Vec<_> = bufs.iter().map(DeviceBuffer::as_raw).collect();
    let shape = c.k.launch.expect("every case has a launch shape");
    // SAFETY: the emitted signature is (buffer pointers in parameter order, one by-value argument block); `blk` holds at least the
    // block's `size` bytes (CUDA copies the parameter size the kernel declares), every device buffer outlives the launch
    // (the stream is synchronized before they drop), and the grid and block come from the kernel's own launch shape.
    unsafe {
        let mut l = func
            .launch()
            .grid((shape.grid[0], shape.grid[1], shape.grid[2]))
            .block((shape.block[0], shape.block[1], shape.block[2]))
            .stream(stream);
        for p in &ptrs {
            l = l.arg(p);
        }
        l.arg(&blk)
            .launch()
            .map_err(|e| format!("{}: launch: {e:?}", c.tag))?;
    }
    stream
        .synchronize()
        .map_err(|e| format!("{}: sync: {e:?}", c.tag))?;
    bufs.iter()
        .map(|d| {
            let mut got = vec![0u8; d.len()];
            d.copy_to_host(&mut got)
                .map_err(|e| format!("{}: download: {e:?}", c.tag))?;
            Ok(got)
        })
        .collect()
}

#[test]
fn emitted_kernels_agree_with_the_oracle_on_the_device() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let (cc_major, cc_minor) = device.compute_capability().expect("compute capability");
    eprintln!(
        "conformance_device: device {:?}, compute capability {cc_major}.{cc_minor}",
        device.name().expect("device name")
    );
    let cc = NvrtcCompiler::new(ArchSku::Sm89.into()).expect("Sm89 is a well-formed cuda: target");
    let all = cases();
    let n = all.len();
    let mut report = Vec::new();
    for c in &all {
        let want = match oracle_outputs(c) {
            Ok(w) => w,
            Err(e) => {
                report.push(e);
                continue;
            }
        };
        let got = match run_on_device(&ctx, &stream, &cc, c) {
            Ok(g) => g,
            Err(e) => {
                report.push(e);
                continue;
            }
        };
        eprintln!(
            "conformance_device: {} ... compared {} buffer(s)",
            c.tag,
            want.len()
        );
        for (bi, (w, g)) in want.iter().zip(&got).enumerate() {
            if w != g {
                let at = w.iter().zip(g).position(|(a, b)| a != b).unwrap_or(0);
                report.push(format!(
                    "{}: buffer {bi} differs from the oracle at byte {at} (oracle {:#04x}, device {:#04x}; {} bytes)",
                    c.tag,
                    w.get(at).copied().unwrap_or(0),
                    g.get(at).copied().unwrap_or(0),
                    w.len()
                ));
            }
        }
        // The control: the comparison can say "no". A kernel that wrote nothing would equal an all-zero oracle only if the oracle were
        // all zero too, so require at least one output byte to be non-zero on the oracle side.
        assert!(
            want.iter().any(|w| w.iter().any(|x| *x != 0)),
            "{}: the oracle produced only zeros, the comparison proves nothing",
            c.tag
        );
    }
    assert!(
        report.is_empty(),
        "the CUDA emitter disagrees with the oracle in {} place(s) over {n} kernels:\n{}",
        report.len(),
        report.join("\n")
    );
}
