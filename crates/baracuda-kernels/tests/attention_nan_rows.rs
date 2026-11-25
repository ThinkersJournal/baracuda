// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: attention kernels propagate NaN instead of turning it into zeros.
//!
//! Entry points under test are the ones Fuel binds: `baracuda_kernels_flash_sdpa_<dtype>_run`, `baracuda_kernels_sdpa_<dtype>_arbmask_run`
//! (both via the FFI) and the naive `SdpaPlan` (f32, f64). Rules (torch.softmax inside attention):
//! - a NaN in a query row makes that row's output NaN and no other row's;
//! - a NaN key or value that a row attends to (weight > 0) makes that row NaN;
//! - a NaN or `+inf` entry in a row's additive mask makes that row NaN;
//! - a NaN value at a position with weight 0 (masked out for that row) is UNSPECIFIED and not asserted here;
//! - a row that is fully masked (all `-inf`, no NaN) stays all zeros with `lse = -inf` (the inference convention; unchanged).
//!
//! Before ROADMAP E0k a query row of NaN scored all NaN, the row max stayed `-inf`, and the kernel took its "fully masked" branch: zeros.
#![cfg(feature = "device-tests")]

use core::ffi::c_void;

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, SdpaArgs, SdpaDescriptor, SdpaPlan, TensorMut, TensorRef,
    Workspace, contiguous_stride,
};
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

macro_rules! ffi_runner {
    ($fname:ident, $T:ty, $to:expr, $from:expr, $sym:path, $has_mask:tt) => {
        fn $fname(
            ctx: &Context,
            stream: &Stream,
            x: &Inputs,
            causal: bool,
        ) -> (Vec<f64>, Vec<f64>) {
            let q: Vec<$T> = x.q.iter().map(|v| $to(*v)).collect();
            let k: Vec<$T> = x.k.iter().map(|v| $to(*v)).collect();
            let v: Vec<$T> = x.v.iter().map(|v| $to(*v)).collect();
            let m: Vec<f32> = x.mask.iter().map(|v| *v as f32).collect();
            let dq = DeviceBuffer::from_slice(ctx, &q).unwrap();
            let dk = DeviceBuffer::from_slice(ctx, &k).unwrap();
            let dv = DeviceBuffer::from_slice(ctx, &v).unwrap();
            let dm = DeviceBuffer::from_slice(ctx, &m).unwrap();
            let mut dy: DeviceBuffer<$T> = DeviceBuffer::zeros(ctx, Q * D).unwrap();
            let mut dl: DeviceBuffer<$T> = DeviceBuffer::zeros(ctx, Q).unwrap();
            let p = |b: &DeviceBuffer<$T>| b.as_slice().as_raw().0 as *const c_void;
            let mp = dm.as_slice().as_raw().0 as *const c_void;
            let _ = mp; // unused by the no-mask entry (flash_sdpa)
            let status = unsafe {
                ffi_call!(
                    $sym,
                    $has_mask,
                    causal,
                    stream,
                    p(&dq),
                    p(&dk),
                    p(&dv),
                    mp,
                    dy,
                    dl
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

macro_rules! ffi_call {
    ($sym:path, false, $causal:expr, $stream:expr, $q:expr, $k:expr, $v:expr, $m:expr, $dy:ident, $dl:ident) => {
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
    ($sym:path, true, $causal:expr, $stream:expr, $q:expr, $k:expr, $v:expr, $m:expr, $dy:ident, $dl:ident) => {
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
            $m,
            $dy.as_slice_mut().as_raw().0 as *mut c_void,
            $dl.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            $stream.as_raw(),
        )
    };
}

use baracuda_kernels_sys as sys;
ffi_runner!(
    flash_f32,
    f32,
    |v: f64| v as f32,
    |v: f32| v as f64,
    sys::baracuda_kernels_flash_sdpa_f32_run,
    false
);
ffi_runner!(
    flash_f64,
    f64,
    |v: f64| v,
    |v: f64| v,
    sys::baracuda_kernels_flash_sdpa_f64_run,
    false
);
ffi_runner!(
    flash_f16,
    f16,
    f16::from_f64,
    |v: f16| v.to_f64(),
    sys::baracuda_kernels_flash_sdpa_f16_run,
    false
);
ffi_runner!(
    flash_bf16,
    bf16,
    bf16::from_f64,
    |v: bf16| v.to_f64(),
    sys::baracuda_kernels_flash_sdpa_bf16_run,
    false
);
ffi_runner!(
    arb_f32,
    f32,
    |v: f64| v as f32,
    |v: f32| v as f64,
    sys::baracuda_kernels_sdpa_f32_arbmask_run,
    true
);
ffi_runner!(
    arb_f64,
    f64,
    |v: f64| v,
    |v: f64| v,
    sys::baracuda_kernels_sdpa_f64_arbmask_run,
    true
);
ffi_runner!(
    arb_f16,
    f16,
    f16::from_f64,
    |v: f16| v.to_f64(),
    sys::baracuda_kernels_sdpa_f16_arbmask_run,
    true
);
ffi_runner!(
    arb_bf16,
    bf16,
    bf16::from_f64,
    |v: bf16| v.to_f64(),
    sys::baracuda_kernels_sdpa_bf16_arbmask_run,
    true
);

#[test]
fn flash_sdpa_f32_nan_rows() {
    check(&flash_f32, "flash_sdpa f32");
}
#[test]
fn flash_sdpa_f64_nan_rows() {
    check(&flash_f64, "flash_sdpa f64");
}
#[test]
fn flash_sdpa_f16_nan_rows() {
    check(&flash_f16, "flash_sdpa f16");
}
#[test]
fn flash_sdpa_bf16_nan_rows() {
    check(&flash_bf16, "flash_sdpa bf16");
}
#[test]
fn arbmask_f32_nan_rows() {
    check(&arb_f32, "arbmask f32");
    check_mask(&|c, s, x, causal| arb_f32(c, s, x, causal), "arbmask f32");
}
#[test]
fn arbmask_f64_nan_rows() {
    check(&arb_f64, "arbmask f64");
    check_mask(&|c, s, x, causal| arb_f64(c, s, x, causal), "arbmask f64");
}
#[test]
fn arbmask_f16_nan_rows() {
    check(&arb_f16, "arbmask f16");
    check_mask(&|c, s, x, causal| arb_f16(c, s, x, causal), "arbmask f16");
}
#[test]
fn arbmask_bf16_nan_rows() {
    check(&arb_bf16, "arbmask bf16");
    check_mask(&|c, s, x, causal| arb_bf16(c, s, x, causal), "arbmask bf16");
}

macro_rules! naive {
    ($name:ident, $T:ty, $elem:expr, $to:expr, $from:expr) => {
        fn $name(ctx: &Context, stream: &Stream, x: &Inputs, causal: bool) -> (Vec<f64>, Vec<f64>) {
            let q: Vec<$T> = x.q.iter().map(|v| $to(*v)).collect();
            let k: Vec<$T> = x.k.iter().map(|v| $to(*v)).collect();
            let v: Vec<$T> = x.v.iter().map(|v| $to(*v)).collect();
            let m: Vec<$T> = x.mask.iter().map(|v| $to(*v)).collect();
            let dq = DeviceBuffer::from_slice(ctx, &q).unwrap();
            let dk = DeviceBuffer::from_slice(ctx, &k).unwrap();
            let dv = DeviceBuffer::from_slice(ctx, &v).unwrap();
            let dm = DeviceBuffer::from_slice(ctx, &m).unwrap();
            let mut dy: DeviceBuffer<$T> = DeviceBuffer::zeros(ctx, Q * D).unwrap();
            let mut da: DeviceBuffer<$T> = DeviceBuffer::zeros(ctx, Q * K).unwrap();
            let desc = SdpaDescriptor {
                batch_size: B,
                num_heads: H,
                query_len: Q as i32,
                key_len: K as i32,
                d_k: D as i32,
                d_v: D as i32,
                scale: 0.35,
                is_causal: causal,
                has_mask: true,
                element: $elem,
            };
            let plan = SdpaPlan::<$T>::select(stream, &desc, PlanPreference::default()).unwrap();
            let (sq, sk, sv, sa, sy) = (
                [B, H, Q as i32, D as i32],
                [B, H, K as i32, D as i32],
                [B, H, K as i32, D as i32],
                [B, H, Q as i32, K as i32],
                [B, H, Q as i32, D as i32],
            );
            plan.run(
                stream,
                Workspace::None,
                SdpaArgs {
                    q: TensorRef::new(dq.as_slice(), sq, contiguous_stride(sq)).unwrap(),
                    k: TensorRef::new(dk.as_slice(), sk, contiguous_stride(sk)).unwrap(),
                    v: TensorRef::new(dv.as_slice(), sv, contiguous_stride(sv)).unwrap(),
                    mask: Some(TensorRef::new(dm.as_slice(), sa, contiguous_stride(sa)).unwrap()),
                    y: TensorMut::new(dy.as_slice_mut(), sy, contiguous_stride(sy)).unwrap(),
                    attn: TensorMut::new(da.as_slice_mut(), sa, contiguous_stride(sa)).unwrap(),
                },
            )
            .unwrap();
            stream.synchronize().unwrap();
            let mut y = vec![$to(0.0); Q * D];
            let mut a = vec![$to(0.0); Q * K];
            dy.copy_to_host(&mut y).unwrap();
            da.copy_to_host(&mut a).unwrap();
            // The naive plan returns attn rather than lse: a fully masked row is a zero attn row; report lse as -inf then, NaN if NaN.
            let lse = (0..Q)
                .map(|r| {
                    let row = &a[r * K..(r + 1) * K];
                    if row.iter().any(|v| $from(*v).is_nan()) {
                        f64::NAN
                    } else if row.iter().all(|v| $from(*v) == 0.0) {
                        f64::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
                .collect();
            (y.iter().map(|v| $from(*v)).collect(), lse)
        }
    };
}
naive!(
    naive_f32,
    f32,
    ElementKind::F32,
    |v: f64| v as f32,
    |v: f32| v as f64
);
naive!(naive_f64, f64, ElementKind::F64, |v: f64| v, |v: f64| v);

#[test]
fn naive_sdpa_f32_nan_rows() {
    check(&naive_f32, "naive sdpa f32");
    check_mask(&naive_f32, "naive sdpa f32");
}
#[test]
fn naive_sdpa_f64_nan_rows() {
    check(&naive_f64, "naive sdpa f64");
    check_mask(&naive_f64, "naive sdpa f64");
}
