// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU probe: does the vendored FlashAttention-2 BACKWARD (forced via `prefer_backend`) follow the attention NaN table
//! (ROADMAP E0k)? Our bespoke backward gave a finite `dV` for a NaN row until alpha.210; this measures FA2 for the same conflation.
//!
//! Table: a NaN or `+inf` query row gives a NaN row and NaN `dV` (torch: `dV = P^T dO`, and `P` of that row is NaN). A finite control
//! must give all-finite gradients. The measured NaN counts are printed; the assertions are the table, EXCEPT a NaN KEY, where FA2 is
//! measured to differ from torch (`dV` NaN only in the poisoned key's row: the forward stores `lse = +inf` for a NaN row) and the test pins
//! that divergence by name.
#![cfg(feature = "fa2_backward")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    BackendKind, ElementKind, FlashSdpaArgs, FlashSdpaBackwardArgs, FlashSdpaBackwardDescriptor,
    FlashSdpaBackwardPlan, FlashSdpaDescriptor, FlashSdpaPlan, PlanPreference, TensorMut,
    TensorRef, Workspace, contiguous_stride,
};
use half::f16;

const H: usize = 1;
const Q: usize = 128;
const K: usize = 128;
const D: usize = 64;

struct Grads {
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
}

fn nans(v: &[f32]) -> usize {
    v.iter().filter(|x| x.is_nan()).count()
}

fn run(q: &[f32], k: &[f32], v: &[f32]) -> Grads {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let cv = |x: &[f32]| x.iter().map(|a| f16::from_f32(*a)).collect::<Vec<f16>>();
    let dq_in = DeviceBuffer::from_slice(&ctx, &cv(q)).unwrap();
    let dk_in = DeviceBuffer::from_slice(&ctx, &cv(k)).unwrap();
    let dv_in = DeviceBuffer::from_slice(&ctx, &cv(v)).unwrap();
    let dy_in = DeviceBuffer::from_slice(&ctx, &cv(&vec![0.5_f32; H * Q * D])).unwrap();
    let mut y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, H * Q * D).unwrap();
    let mut lse_t: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, H * Q).unwrap();
    let (sq, sk, sl) = (
        [1, H as i32, Q as i32, D as i32],
        [1, H as i32, K as i32, D as i32],
        [1, H as i32, Q as i32],
    );
    let scale = 1.0 / (D as f32).sqrt();
    let pref = PlanPreference {
        prefer_backend: Some(BackendKind::FlashAttentionV2),
        ..Default::default()
    };

    let fdesc = FlashSdpaDescriptor::new(
        1,
        H as i32,
        Q as i32,
        K as i32,
        D as i32,
        D as i32,
        scale,
        false,
        ElementKind::F16,
    );
    let fplan = FlashSdpaPlan::<f16>::select(&stream, &fdesc, pref).expect("fw select");
    assert_eq!(fplan.backend(), BackendKind::FlashAttentionV2);
    let fws_bytes = fplan.workspace_size();
    let mut fws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, fws_bytes).unwrap();
    fplan
        .run(
            &stream,
            Workspace::Borrowed(fws.as_slice_mut()),
            FlashSdpaArgs {
                q: TensorRef::new(dq_in.as_slice(), sq, contiguous_stride(sq)).unwrap(),
                k: TensorRef::new(dk_in.as_slice(), sk, contiguous_stride(sk)).unwrap(),
                v: TensorRef::new(dv_in.as_slice(), sk, contiguous_stride(sk)).unwrap(),
                y: TensorMut::new(y.as_slice_mut(), sq, contiguous_stride(sq)).unwrap(),
                lse: TensorMut::new(lse_t.as_slice_mut(), sl, contiguous_stride(sl)).unwrap(),
                mask: None,
                alibi_slopes: None,
            },
        )
        .expect("fw run");
    stream.synchronize().expect("sync");
    // The f32 LSE the FA2 launcher wrote lives in the forward workspace; feed it back unchanged.
    let mut ws_host = vec![0u8; fws_bytes];
    fws.copy_to_host(&mut ws_host).unwrap();
    let lse_f32: Vec<f32> = (0..H * Q)
        .map(|i| f32::from_le_bytes(ws_host[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();
    let dlse = DeviceBuffer::from_slice(&ctx, &lse_f32).unwrap();

    let bdesc = FlashSdpaBackwardDescriptor::new(
        1,
        H as i32,
        Q as i32,
        K as i32,
        D as i32,
        D as i32,
        scale,
        false,
        ElementKind::F16,
    );
    let bplan = FlashSdpaBackwardPlan::<f16>::select(&stream, &bdesc, pref).expect("bw select");
    assert_eq!(bplan.backend(), BackendKind::FlashAttentionV2);
    let mut bws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, bplan.workspace_size()).unwrap();
    let mut gq: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, H * Q * D).unwrap();
    let mut gk: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, H * K * D).unwrap();
    let mut gv: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, H * K * D).unwrap();
    let mut dws: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, H * Q).unwrap();
    bplan
        .run(
            &stream,
            Workspace::Borrowed(bws.as_slice_mut()),
            FlashSdpaBackwardArgs {
                q: TensorRef::new(dq_in.as_slice(), sq, contiguous_stride(sq)).unwrap(),
                k: TensorRef::new(dk_in.as_slice(), sk, contiguous_stride(sk)).unwrap(),
                v: TensorRef::new(dv_in.as_slice(), sk, contiguous_stride(sk)).unwrap(),
                y: TensorRef::new(y.as_slice(), sq, contiguous_stride(sq)).unwrap(),
                lse: TensorRef::new(lse_t.as_slice(), sl, contiguous_stride(sl)).unwrap(),
                dy: TensorRef::new(dy_in.as_slice(), sq, contiguous_stride(sq)).unwrap(),
                d_ws: TensorMut::new(dws.as_slice_mut(), sl, contiguous_stride(sl)).unwrap(),
                dq: TensorMut::new(gq.as_slice_mut(), sq, contiguous_stride(sq)).unwrap(),
                dk: TensorMut::new(gk.as_slice_mut(), sk, contiguous_stride(sk)).unwrap(),
                dv: TensorMut::new(gv.as_slice_mut(), sk, contiguous_stride(sk)).unwrap(),
                lse_f32: Some(TensorRef::new(dlse.as_slice(), sl, contiguous_stride(sl)).unwrap()),
                alibi_slopes: None,
            },
        )
        .expect("bw run");
    stream.synchronize().expect("sync");
    let rd = |b: &DeviceBuffer<f16>, n: usize| {
        let mut h = vec![f16::ZERO; n];
        b.copy_to_host(&mut h).unwrap();
        h.iter().map(|x| x.to_f32()).collect::<Vec<f32>>()
    };
    Grads {
        dq: rd(&gq, H * Q * D),
        dk: rd(&gk, H * K * D),
        dv: rd(&gv, H * K * D),
    }
}

fn base() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        (0..H * Q * D)
            .map(|i| 0.05 * ((i % 5) as f32 - 2.0))
            .collect(),
        // Positive keys so an `inf` query component scores +inf on every key.
        (0..H * K * D)
            .map(|i| 0.05 * ((i % 7) as f32 + 1.0))
            .collect(),
        (0..H * K * D).map(|i| 1.0 + (i % 9) as f32).collect(),
    )
}

#[test]
fn finite_control_gives_finite_gradients() {
    let (q, k, v) = base();
    let g = run(&q, &k, &v);
    eprintln!(
        "control nan counts dq={} dk={} dv={}",
        nans(&g.dq),
        nans(&g.dk),
        nans(&g.dv)
    );
    assert!(g.dq.iter().chain(&g.dk).chain(&g.dv).all(|x| x.is_finite()));
}

fn poisoned_query(val: f32, name: &str) {
    let (mut q, k, v) = base();
    q[3 * D + 7] = val;
    let g = run(&q, &k, &v);
    eprintln!(
        "{name}: nan counts dq={}/{} dk={}/{} dv={}/{}",
        nans(&g.dq),
        g.dq.len(),
        nans(&g.dk),
        g.dk.len(),
        nans(&g.dv),
        g.dv.len()
    );
    // torch: the poisoned row's P is NaN, so dV = P^T dO is NaN everywhere; dQ of that row is NaN.
    assert!(
        g.dv.iter().all(|x| x.is_nan()),
        "{name}: dV must be all NaN, {} of {} were",
        nans(&g.dv),
        g.dv.len()
    );
    assert!(
        g.dq[3 * D..4 * D].iter().all(|x| x.is_nan()),
        "{name}: dQ row 3 must be NaN"
    );
}

#[test]
fn nan_query_gives_nan_dv() {
    poisoned_query(f32::NAN, "nan_q");
}

#[test]
fn inf_query_gives_nan_dv() {
    poisoned_query(f32::INFINITY, "inf_q");
}

/// KNOWN DIVERGENCE from the torch convention (NaN key): `dV` is NaN only in the poisoned key's row (64 of 8192 elements), where torch
/// gives NaN for every `dV` (row i's softmax over a NaN score is all NaN, so `P` and `dV = P^T dO` are NaN); `dQ` and `dK` are all NaN,
/// as torch. Measured on the device (RTX 4070 Laptop, sm_89, through `gpu-run`) with f16, `d = 64`, `h = 1`, 128 x 128, non-causal,
/// against the vendored FlashAttention-2 pinned in `kernels-sys/vendor/flash-attention/VENDOR.md` (v2.8.3, commit `060c9188`).
/// CAUSE: read from vendor `softmax.h`, NOT mutant-isolated, so a HYPOTHESIS until a mutant on that vendored line shows it: the forward
/// stores `lse = +inf` for a row whose softmax sum is NaN (`sum == 0 || sum != sum`), and the backward's `exp(s * scale - lse)` is then
/// `0` for every finite score and NaN only for the NaN score itself. The NaN still reaches the caller through `dQ` and `dK`.
/// This pins what FA2 does today, not what torch does; if the vendored kernel changes, the pin fails and says so. The bespoke
/// backward gives all-NaN `dV` (alpha.210). Fuel binds no flash backward; a consumer that does must check it against the union vector
/// table.
#[test]
fn nan_key_fa2_backward_dv_is_nan_only_for_the_poisoned_key() {
    let (q, mut k, v) = base();
    k[5 * D + 2] = f32::NAN;
    let g = run(&q, &k, &v);
    eprintln!(
        "nan_k: nan counts dq={} dk={} dv={}",
        nans(&g.dq),
        nans(&g.dk),
        nans(&g.dv)
    );
    assert!(g.dq.iter().all(|x| x.is_nan()), "dQ must be all NaN");
    assert!(g.dk.iter().all(|x| x.is_nan()), "dK must be all NaN");
    assert_eq!(
        nans(&g.dv),
        D,
        "FA2 today: dV is NaN only in the poisoned key's row (torch: all of dV)"
    );
    assert!(g.dv[5 * D..6 * D].iter().all(|x| x.is_nan()));
}
