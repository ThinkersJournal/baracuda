//! The step functions reject inputs that would make the kernel read or write
//! out of bounds, or write NaN, instead of launching.
//!
//! Each test names the behaviour it guards: the first three returned `Ok(())`
//! (or launched) before this file's change, because four of the six step
//! functions compared tensor counts only, `step` on a half-precision plan
//! accepted half-precision moments the shim reads as `f32`, and a step index
//! of 0 with bias correction divided by zero in the kernel.

// Same gating as the other smoke tests: without an arch feature the shim is not
// built and the FFI does not link.
#![cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_optim::{
    AdamConfig, AdamStepPlan, Error, SgdConfig, SgdStepPlan, TensorList, TensorListMut,
};
use half::f16;

fn setup() -> (Context, Stream) {
    baracuda_driver::init().unwrap();
    let ctx = Context::new(&Device::get(0).unwrap()).unwrap();
    let stream = Stream::new(&ctx).unwrap();
    (ctx, stream)
}

fn zeros<T: baracuda_types::DeviceRepr>(ctx: &Context, n: usize) -> DeviceBuffer<T> {
    DeviceBuffer::zeros(ctx, n).unwrap()
}

#[test]
fn adam_f32_state_rejects_a_shorter_moment_tensor() {
    let (ctx, stream) = setup();
    let (mut p0, mut p1) = (zeros::<f32>(&ctx, 8), zeros::<f32>(&ctx, 8));
    let (g0, g1) = (zeros::<f32>(&ctx, 8), zeros::<f32>(&ctx, 8));
    let (mut m0, mut m1) = (zeros::<f32>(&ctx, 8), zeros::<f32>(&ctx, 8));
    // Same count as the others, one tensor shorter: the old count-only check accepted this.
    let (mut v0, mut v1) = (zeros::<f32>(&ctx, 8), zeros::<f32>(&ctx, 4));
    let plan = AdamStepPlan::<f32>::new(AdamConfig::default());
    let r = plan.step_with_f32_state(
        &mut TensorListMut::from_buffers([&mut p0, &mut p1]).unwrap(),
        &TensorList::new(&[&g0, &g1]).unwrap(),
        &mut TensorListMut::from_buffers([&mut m0, &mut m1]).unwrap(),
        &mut TensorListMut::from_buffers([&mut v0, &mut v1]).unwrap(),
        1,
        &stream,
    );
    assert!(matches!(r, Err(Error::LengthMismatch(_))), "{r:?}");
}

#[test]
fn sgd_rejects_a_shorter_momentum_tensor() {
    let (ctx, stream) = setup();
    let (mut p0, mut p1) = (zeros::<f32>(&ctx, 8), zeros::<f32>(&ctx, 8));
    let (g0, g1) = (zeros::<f32>(&ctx, 8), zeros::<f32>(&ctx, 8));
    let (mut b0, mut b1) = (zeros::<f32>(&ctx, 8), zeros::<f32>(&ctx, 4));
    let plan = SgdStepPlan::<f32>::new(SgdConfig::default());
    let r = plan.step(
        &mut TensorListMut::from_buffers([&mut p0, &mut p1]).unwrap(),
        &TensorList::new(&[&g0, &g1]).unwrap(),
        &mut TensorListMut::from_buffers([&mut b0, &mut b1]).unwrap(),
        true,
        &stream,
    );
    assert!(matches!(r, Err(Error::LengthMismatch(_))), "{r:?}");
}

#[test]
fn half_precision_step_rejects_half_moments() {
    let (ctx, stream) = setup();
    let (mut p, g) = (zeros::<f16>(&ctx, 8), zeros::<f16>(&ctx, 8));
    let (mut m, mut v) = (zeros::<f16>(&ctx, 8), zeros::<f16>(&ctx, 8));
    let plan = AdamStepPlan::<f16>::new(AdamConfig::default());
    let r = plan.step(
        &mut TensorListMut::from_buffers([&mut p]).unwrap(),
        &TensorList::new(&[&g]).unwrap(),
        &mut TensorListMut::from_buffers([&mut m]).unwrap(),
        &mut TensorListMut::from_buffers([&mut v]).unwrap(),
        1,
        &stream,
    );
    assert!(matches!(r, Err(Error::InvalidArgument(_))), "{r:?}");
}

#[test]
fn adam_step_zero_with_bias_correction_is_an_error() {
    let (ctx, stream) = setup();
    let (mut p, g, mut m, mut v) = (
        zeros::<f32>(&ctx, 8),
        zeros::<f32>(&ctx, 8),
        zeros::<f32>(&ctx, 8),
        zeros::<f32>(&ctx, 8),
    );
    let plan = AdamStepPlan::<f32>::new(AdamConfig::default()); // bias_correction: true
    let r = plan.step(
        &mut TensorListMut::from_buffers([&mut p]).unwrap(),
        &TensorList::new(&[&g]).unwrap(),
        &mut TensorListMut::from_buffers([&mut m]).unwrap(),
        &mut TensorListMut::from_buffers([&mut v]).unwrap(),
        0,
        &stream,
    );
    assert!(matches!(r, Err(Error::InvalidArgument(_))), "{r:?}");
}
