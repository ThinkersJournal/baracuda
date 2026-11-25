// SPDX-License-Identifier: MIT OR Apache-2.0
//! `UniquePlan::run` must not return `Ok` with an empty result.
//!
//! Up to 0.1.0-alpha.189 it sorted into the scratch buffer and returned `Ok(())` without writing `values`, `counts` or `counter`
//! (ROADMAP E0b). It now declines with `Error::Unsupported` until the dedup stage is wired.
#![cfg(feature = "device-tests")]

use baracuda_cutlass::Error;
use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, TensorMut, TensorRef, UniqueArgs, UniqueDescriptor, UniquePlan,
    Workspace, contiguous_stride,
};

#[test]
fn unique_plan_declines_instead_of_returning_nothing() {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let x = [3f32, 1.0, 3.0, 2.0, 1.0];
    let n = x.len() as i32;
    let sh = [1, n];
    let dx = DeviceBuffer::from_slice(&ctx, &x).unwrap();
    let mut ss: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, x.len()).unwrap();
    let mut si: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, x.len()).unwrap();
    let mut dv: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, x.len()).unwrap();
    let mut dc: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, x.len()).unwrap();
    let mut dn: DeviceBuffer<i32> = DeviceBuffer::from_slice(&ctx, &[-77i32]).unwrap();
    let desc = UniqueDescriptor {
        batch: 1,
        row_len: n,
        max_unique: n,
        element: ElementKind::F32,
    };
    let plan = UniquePlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let r = plan.run(
        &stream,
        Workspace::None,
        UniqueArgs::<f32> {
            input: TensorRef::new(dx.as_slice(), sh, contiguous_stride(sh)).unwrap(),
            sorted_scratch: TensorMut::new(ss.as_slice_mut(), sh, contiguous_stride(sh)).unwrap(),
            sorted_idx_scratch: TensorMut::new(si.as_slice_mut(), sh, contiguous_stride(sh))
                .unwrap(),
            values: TensorMut::new(dv.as_slice_mut(), sh, contiguous_stride(sh)).unwrap(),
            counts: TensorMut::new(dc.as_slice_mut(), sh, contiguous_stride(sh)).unwrap(),
            counter: TensorMut::new(dn.as_slice_mut(), [1], contiguous_stride([1])).unwrap(),
        },
    );
    assert!(
        matches!(r, Err(Error::Unsupported(_))),
        "UniquePlan::run must decline (Error::Unsupported), got {r:?}"
    );
    stream.synchronize().unwrap();
    let mut k = [0i32; 1];
    dn.copy_to_host(&mut k).unwrap();
    assert_eq!(k[0], -77, "a declined run must not touch the counter");
}
