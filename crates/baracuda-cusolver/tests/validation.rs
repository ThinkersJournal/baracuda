//! The dense functions return an error for buffers that are too small for the
//! sizes they are given (the matrix arguments, by LAPACK's rules), instead of
//! handing cuSOLVER a pointer it would write past. Before this check `getrf::<f32>(.., 1000, 1000, &mut a, 1000, ..)`
//! over a 4-element `a` was accepted by the safe API.
#![cfg(feature = "device-tests")]

use baracuda_cusolver::{DnHandle, Fill, getrf, potrf};
use baracuda_driver::{Context, Device, DeviceBuffer};

fn setup() -> (Context, DnHandle) {
    baracuda_driver::init().unwrap();
    let ctx = Context::new(&Device::get(0).unwrap()).unwrap();
    (ctx, DnHandle::new().unwrap())
}

#[test]
fn getrf_rejects_a_matrix_larger_than_its_buffer() {
    let (ctx, h) = setup();
    let mut a: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).unwrap();
    let mut ipiv: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1000).unwrap();
    let mut info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let r = getrf::<f32>(&h, 1000, 1000, &mut a, 1000, &mut ipiv, &mut info);
    assert!(r.is_err(), "{r:?}");
}

#[test]
fn potrf_rejects_a_leading_dimension_below_n() {
    let (ctx, h) = setup();
    let mut a: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 25).unwrap();
    let mut info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    assert!(potrf::<f32>(&h, Fill::Lower, 5, &mut a, 3, &mut info).is_err());
}
