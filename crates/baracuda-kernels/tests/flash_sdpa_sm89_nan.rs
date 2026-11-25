// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: the sm89 flash kernels (`flash_sdpa_sm89`, regular and strided) propagate NaN instead of turning it into zeros.
//!
//! Same contract and vectors as `attention_nan_rows.rs`, through the FFI symbols `baracuda_kernels_flash_sdpa_sm89_{f16,bf16}_run` and `..._strided_run` (contiguous strides).
//! These kernels have no additive mask, so the mask vectors do not apply; a fully masked row cannot be expressed (causal never masks a whole row).
//! A NaN value at a masked-out position is UNSPECIFIED and not asserted.
#![cfg(feature = "device-tests")]

use core::ffi::c_void;

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use half::{bf16, f16};

const B: i32 = 1;
const H: i32 = 1;
const Q: usize = 3;
const K: usize = 4;
const D: usize = 8;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[derive(Clone)]
struct Inputs {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    mask: Vec<f64>, // [Q, K] additive
}

fn base() -> Inputs {
    Inputs {
        q: (0..Q * D).map(|i| 0.1 * ((i % 5) as f64 - 2.0)).collect(),
        k: (0..K * D).map(|i| 0.1 * ((i % 7) as f64 - 3.0)).collect(),
        v: (0..K * D).map(|i| 1.0 + (i % 9) as f64).collect(),
        mask: vec![0.0; Q * K],
    }
}

fn row_is_nan(y: &[f64], r: usize) -> bool {
    y[r * D..(r + 1) * D].iter().all(|v| v.is_nan())
}
fn row_finite(y: &[f64], r: usize) -> bool {
    y[r * D..(r + 1) * D].iter().all(|v| v.is_finite())
}
fn row_zero(y: &[f64], r: usize) -> bool {
    y[r * D..(r + 1) * D].iter().all(|v| *v == 0.0)
}

/// Runs the kernel under test; returns (y [Q*D], lse [Q]) as f64.
type Runner = dyn Fn(&Context, &Stream, &Inputs, bool) -> (Vec<f64>, Vec<f64>);

fn check(run: &Runner, name: &str) {
    let (ctx, stream) = setup();
    let ok = base();
    let (y0, _) = run(&ctx, &stream, &ok, false);
    assert!(
        (0..Q).all(|r| row_finite(&y0, r)),
        "{name}: the clean baseline must be finite"
    );

    // NaN in query row 1: only row 1 is NaN.
    let mut c = ok.clone();
    c.q[D] = f64::NAN;
    let (y, lse) = run(&ctx, &stream, &c, false);
    assert!(
        row_is_nan(&y, 1),
        "{name}: NaN in a query row must give a NaN output row, got {:?}",
        &y[D..2 * D]
    );
    assert!(
        lse[1].is_nan(),
        "{name}: lse of the NaN query row must be NaN, got {}",
        lse[1]
    );
    assert!(
        row_finite(&y, 0) && row_finite(&y, 2),
        "{name}: other rows must stay finite"
    );

    // NaN key 2: every row attends to it (weights > 0).
    let mut c = ok.clone();
    c.k[2 * D] = f64::NAN;
    let (y, _) = run(&ctx, &stream, &c, false);
    assert!(
        (0..Q).all(|r| row_is_nan(&y, r)),
        "{name}: a NaN key every row attends to must poison every row"
    );

    // NaN value at key 2.
    let mut c = ok.clone();
    c.v[2 * D] = f64::NAN;
    let (y, _) = run(&ctx, &stream, &c, false);
    assert!(
        (0..Q).all(|r| y[r * D].is_nan()),
        "{name}: a NaN value every row attends to must reach every row"
    );

    // Causal: key 2 is attended only by row 2. A NaN value there must reach row 2. Whether rows 0 and 1 (key 2 masked out, weight 0) stay
    // NaN-free is UNSPECIFIED (tile-skipping kernels ignore it, dense decompositions propagate it): nothing is asserted about them.
    let mut c = ok.clone();
    c.v[2 * D] = f64::NAN;
    let (y, _) = run(&ctx, &stream, &c, true);
    assert!(
        y[2 * D].is_nan(),
        "{name}: row 2 attends to key 2, so it must be NaN"
    );
}

#[allow(dead_code)]
fn check_mask(run: &Runner, name: &str) {
    let (ctx, stream) = setup();
    let ok = base();
    // NaN mask entry in row 0.
    let mut c = ok.clone();
    c.mask[1] = f64::NAN;
    let (y, _) = run(&ctx, &stream, &c, false);
    assert!(
        row_is_nan(&y, 0) && row_finite(&y, 1) && row_finite(&y, 2),
        "{name}: a NaN mask entry poisons its row only"
    );
    // +inf mask entry in row 0.
    let mut c = ok.clone();
    c.mask[2] = f64::INFINITY;
    let (y, _) = run(&ctx, &stream, &c, false);
    assert!(
        row_is_nan(&y, 0) && row_finite(&y, 1) && row_finite(&y, 2),
        "{name}: a +inf mask entry poisons its row only"
    );
    // Fully masked row 0 (all -inf): zeros and lse -inf, unchanged.
    let mut c = ok.clone();
    for j in 0..K {
        c.mask[j] = f64::NEG_INFINITY;
    }
    let (y, lse) = run(&ctx, &stream, &c, false);
    assert!(
        row_zero(&y, 0) && row_finite(&y, 1),
        "{name}: a fully masked row stays zeros, got {:?}",
        &y[0..D]
    );
    assert!(
        lse[0] == f64::NEG_INFINITY,
        "{name}: lse of a fully masked row stays -inf, got {}",
        lse[0]
    );
}

use baracuda_kernels_sys as sys;

macro_rules! sm89_runner {
    ($fname:ident, $T:ty, $to:expr, $from:expr, $sym:path, $strided:tt) => {
        fn $fname(
            ctx: &Context,
            stream: &Stream,
            x: &Inputs,
            causal: bool,
        ) -> (Vec<f64>, Vec<f64>) {
            let q: Vec<$T> = x.q.iter().map(|v| $to(*v)).collect();
            let k: Vec<$T> = x.k.iter().map(|v| $to(*v)).collect();
            let v: Vec<$T> = x.v.iter().map(|v| $to(*v)).collect();
            let dq = DeviceBuffer::from_slice(ctx, &q).unwrap();
            let dk = DeviceBuffer::from_slice(ctx, &k).unwrap();
            let dv = DeviceBuffer::from_slice(ctx, &v).unwrap();
            let mut dy: DeviceBuffer<$T> = DeviceBuffer::zeros(ctx, Q * D).unwrap();
            let mut dl: DeviceBuffer<$T> = DeviceBuffer::zeros(ctx, Q).unwrap();
            let p = |b: &DeviceBuffer<$T>| b.as_slice().as_raw().0 as *const c_void;
            let (sq, sk, sy) = (
                [(Q * D) as i64, (Q * D) as i64, D as i64],
                [(K * D) as i64, (K * D) as i64, D as i64],
                [(Q * D) as i64, (Q * D) as i64, D as i64],
            );
            let status = unsafe {
                sm89_call!(
                    $sym,
                    $strided,
                    causal,
                    stream,
                    p(&dq),
                    p(&dk),
                    p(&dv),
                    dy,
                    dl,
                    sq,
                    sk,
                    sy
                )
            };
            assert_eq!(status, 0, "kernel status {status}");
            stream.synchronize().unwrap();
            let mut y = vec![$to(0.0); Q * D];
            let mut l = vec![$to(0.0); Q];
            dy.copy_to_host(&mut y).unwrap();
            dl.copy_to_host(&mut l).unwrap();
            (
                y.iter().map(|v| $from(*v)).collect(),
                l.iter().map(|v| $from(*v)).collect(),
            )
        }
    };
}

macro_rules! sm89_call {
    ($sym:path, false, $causal:expr, $stream:expr, $q:expr, $k:expr, $v:expr, $dy:ident, $dl:ident, $sq:ident, $sk:ident, $sy:ident) => {
        $sym(
            B,
            H,
            Q as i32,
            K as i32,
            D as i32,
            D as i32,
            0.35f32,
            $causal as i32,
            $q,
            $k,
            $v,
            $dy.as_slice_mut().as_raw().0 as *mut c_void,
            $dl.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            $stream.as_raw(),
        )
    };
    ($sym:path, true, $causal:expr, $stream:expr, $q:expr, $k:expr, $v:expr, $dy:ident, $dl:ident, $sq:ident, $sk:ident, $sy:ident) => {
        $sym(
            B,
            H,
            Q as i32,
            K as i32,
            D as i32,
            D as i32,
            $sq.as_ptr(),
            $sk.as_ptr(),
            $sk.as_ptr(),
            $sy.as_ptr(),
            0.35f32,
            $causal as i32,
            $q,
            $k,
            $v,
            $dy.as_slice_mut().as_raw().0 as *mut c_void,
            $dl.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            $stream.as_raw(),
        )
    };
}

sm89_runner!(
    sm89_f16,
    f16,
    f16::from_f64,
    |v: f16| v.to_f64(),
    sys::baracuda_kernels_flash_sdpa_sm89_f16_run,
    false
);
sm89_runner!(
    sm89_bf16,
    bf16,
    bf16::from_f64,
    |v: bf16| v.to_f64(),
    sys::baracuda_kernels_flash_sdpa_sm89_bf16_run,
    false
);
sm89_runner!(
    sm89s_f16,
    f16,
    f16::from_f64,
    |v: f16| v.to_f64(),
    sys::baracuda_kernels_flash_sdpa_sm89_f16_strided_run,
    true
);
sm89_runner!(
    sm89s_bf16,
    bf16,
    bf16::from_f64,
    |v: bf16| v.to_f64(),
    sys::baracuda_kernels_flash_sdpa_sm89_bf16_strided_run,
    true
);

#[test]
fn sm89_f16_nan_rows() {
    check(&sm89_f16, "flash_sdpa_sm89 f16");
}
#[test]
fn sm89_bf16_nan_rows() {
    check(&sm89_bf16, "flash_sdpa_sm89 bf16");
}
#[test]
fn sm89_strided_f16_nan_rows() {
    check(&sm89s_f16, "flash_sdpa_sm89 strided f16");
}
#[test]
fn sm89_strided_bf16_nan_rows() {
    check(&sm89s_bf16, "flash_sdpa_sm89 strided bf16");
}
