// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Stmt::ProxyFence` (unpopped 0.57.0, ROADMAP U8c) lowers to `fence.proxy.async.shared::cta` and nothing else; a space other
//! than `Shared` is refused with a reason. Source-shape tests need no GPU; nothing runs here.

use baracuda_cuda_emit::Cuda;
use unpopped::backend::Backend;
use unpopped::exec::{LaunchShape, MemorySpace};
use unpopped::kernel::{Block, KernelFn, LocalDecl, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

fn kernel(body: Vec<Stmt>) -> KernelFn {
    KernelFn::new(
        "f",
        ElementKind::I32,
        vec![],
        vec![LocalDecl::scalar(ElementKind::I32)],
        Block(body),
    )
    .with_launch(LaunchShape::new([128, 1, 1], [1, 1, 1]))
}

#[test]
fn shared_proxy_fence_is_spelled_exactly() {
    let k = kernel(vec![Stmt::ProxyFence {
        space: MemorySpace::Shared,
    }]);
    check_kernel(&k).expect("the fixture is well-formed");
    let g = Cuda.lower_kernel(&k).expect("a shared proxy fence lowers");
    assert!(
        g.source
            .contains("asm volatile(\"fence.proxy.async.shared::cta;\" ::: \"memory\");"),
        "{}",
        g.source
    );
}

/// Control (the query can say "no"): a kernel without the statement has no fence.
#[test]
fn a_kernel_without_the_statement_has_no_fence() {
    let k = kernel(vec![Stmt::Return]);
    check_kernel(&k).expect("well-formed");
    let g = Cuda.lower_kernel(&k).expect("lowers");
    assert!(!g.source.contains("fence.proxy"), "{}", g.source);
}

/// The checker admits only `Shared`; the emitter repeats that as a second line, so the other spaces never print a fence.
#[test]
fn other_spaces_are_refused() {
    for space in [MemorySpace::Global, MemorySpace::Private] {
        let k = kernel(vec![Stmt::ProxyFence { space }]);
        let refused = check_kernel(&k).is_err() || Cuda.lower_kernel(&k).is_err();
        assert!(refused, "{space:?} must not lower");
    }
}
