// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP S1 slice 5: named barriers (`bar.sync`/`bar.arrive`), the `Cluster` barrier (`barrier.cluster`) and the cluster launch
//! (`__cluster_dims__`) in `Cuda::lower_kernel`. Before this slice a `Barrier` of any kind in `Block` scope printed `__syncthreads()`,
//! which waits for the wrong threads; the tests pin that it no longer does. Source shape needs no GPU; `ptxas_accepts` (feature
//! `nvrtc`) assembles only. **Hopper execution is not tested here** (sm_89 box; ROADMAP S6).

use baracuda_cuda_emit::Cuda;
use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::exec::{LaunchShape, Scope};
use unpopped::kernel::{BarrierKind, Block, KernelFn, LocalDecl, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

const I32: ElementKind = ElementKind::I32;

fn kernel(launch: LaunchShape, body: Vec<Stmt>) -> KernelFn {
    KernelFn::new("b", I32, vec![], vec![LocalDecl::scalar(I32)], Block(body)).with_launch(launch)
}

fn plain() -> LaunchShape {
    LaunchShape::new([128, 1, 1], [1, 1, 1])
}

fn clustered() -> LaunchShape {
    LaunchShape::new([128, 1, 1], [2, 1, 1]).with_cluster([2, 1, 1])
}

fn barrier(scope: Scope, kind: BarrierKind) -> Stmt {
    Stmt::Barrier { scope, kind }
}

fn lowered(launch: LaunchShape, body: Vec<Stmt>) -> GeneratedKernel {
    let k = kernel(launch, body);
    check_kernel(&k).unwrap_or_else(|e| panic!("the fixture is not well-formed: {e:?}"));
    Cuda.lower_kernel(&k)
        .unwrap_or_else(|e| panic!("expected a lowering, got {e:?}"))
}

#[test]
fn named_barriers_are_spelled_and_are_not_syncthreads() {
    let g = lowered(
        plain(),
        vec![
            barrier(Scope::Block, BarrierKind::NamedArrive { id: 2, count: 64 }),
            barrier(Scope::Block, BarrierKind::Named { id: 2, count: 64 }),
        ],
    );
    for want in [
        "asm volatile(\"bar.arrive %0, %1;\" :: \"n\"(2), \"n\"(64) : \"memory\");",
        "asm volatile(\"bar.sync %0, %1;\" :: \"n\"(2), \"n\"(64) : \"memory\");",
    ] {
        assert!(g.source.contains(want), "missing `{want}`:\n{}", g.source);
    }
    assert!(!g.source.contains("__syncthreads"), "{}", g.source);
    // Control: the full block barrier is still `__syncthreads()`.
    let f = lowered(plain(), vec![barrier(Scope::Block, BarrierKind::Full)]);
    assert!(f.source.contains("__syncthreads();"), "{}", f.source);
}

#[test]
fn a_bad_named_barrier_is_never_emitted() {
    for kind in [
        BarrierKind::Named { id: 16, count: 64 },
        BarrierKind::Named { id: 1, count: 48 },
        BarrierKind::Named { id: 1, count: 256 },
    ] {
        let k = kernel(plain(), vec![barrier(Scope::Block, kind)]);
        if check_kernel(&k).is_err() {
            continue; // the IR's own checker refuses it first
        }
        match Cuda.lower_kernel(&k) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(detail.contains("Barrier"), "{detail}")
            }
            other => panic!("{kind:?} must be refused, got {other:?}"),
        }
    }
}

#[test]
fn cluster_barrier_and_launch_are_spelled_for_sm90() {
    let g = lowered(
        clustered(),
        vec![barrier(Scope::Cluster, BarrierKind::Full)],
    );
    for want in [
        "__cluster_dims__(2, 1, 1)",
        "barrier.cluster.arrive.release.aligned;",
        "barrier.cluster.wait.acquire.aligned;",
        "__CUDA_ARCH__ < 900",
    ] {
        assert!(g.source.contains(want), "missing `{want}`:\n{}", g.source);
    }
    // Control: a plain launch has no cluster text and no sm_90 guard.
    let p = lowered(plain(), vec![barrier(Scope::Block, BarrierKind::Full)]);
    for not in ["__cluster_dims__", "barrier.cluster", "__CUDA_ARCH__ < 900"] {
        assert!(!p.source.contains(not), "unexpected `{not}`:\n{}", p.source);
    }
}

#[cfg(feature = "nvrtc")]
#[test]
fn ptxas_accepts() {
    use baracuda_cuda_emit::NvrtcCompiler;
    use unpopped::jit::Compiler;
    use unpopped_vocab::TargetId;
    let t = |tok: &str| NvrtcCompiler::new(TargetId::parse(tok).expect("token")).expect("nvrtc");
    let (sm80, sm90) = (t("cuda:sm80"), t("cuda:sm90a"));
    let named = lowered(
        plain(),
        vec![
            barrier(Scope::Block, BarrierKind::NamedArrive { id: 2, count: 64 }),
            barrier(Scope::Block, BarrierKind::Named { id: 2, count: 64 }),
        ],
    );
    let cl = lowered(
        clustered(),
        vec![barrier(Scope::Cluster, BarrierKind::Full)],
    );
    for (nv, g, what) in [
        (&sm80, &named, "named sm80"),
        (&sm90, &named, "named sm90a"),
        (&sm90, &cl, "cluster sm90a"),
    ] {
        nv.compile(&g.source, &g.name, 5000)
            .unwrap_or_else(|e| panic!("{what} rejected: {e}\n{}", g.source));
    }
    assert!(
        sm80.compile(&cl.source, &cl.name, 5000).is_err(),
        "cluster must be refused below sm_90"
    );
}
