// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: flash attention BACKWARD with NaN / inf inputs (the NaN sweep row "attention backward").
//!
//! Entry point: `baracuda_kernels_flash_sdpa_{f32,f16,bf16}_run` then `..._backward_..._run` (the FFI Fuel binds). The oracle is a CPU f64
//! attention forward + backward written with plain IEEE arithmetic (NaN propagates through `+ * exp` as in torch autograd; the row
//! maximum is NaN-propagating). For every probe the test compares the NaN PATTERN of dQ, dK, dV element by element (NaN where the oracle is
//! NaN, finite where it is finite) and the finite values within a dtype tolerance. Probes: finite control, NaN in Q / K / V / dO, a `+inf`
//! query component (scores `+-inf`). The positive control is the finite case, which must match the oracle on values.
//! A NaN V at a position with weight 0 (masked out) is UNSPECIFIED and is not probed (non-causal shapes attend everywhere).
#![cfg(feature = "device-tests")]

use core::ffi::c_void;

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels_sys as sys;
use half::{bf16, f16};

const Q: usize = 64;
const K: usize = 64;
const D: usize = 32;
const SCALE: f32 = 0.17;

#[derive(Clone)]
struct Inp {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    dy: Vec<f64>,
}

fn base() -> Inp {
    let g = |n: usize, a: f64, b: f64| -> Vec<f64> {
        (0..n).map(|i| ((i as f64) * a + b).sin() * 0.8).collect()
    };
    Inp {
        q: g(Q * D, 0.013, -0.5),
        k: g(K * D, 0.017, 0.2),
        v: g(K * D, 0.011, -0.1),
        dy: g(Q * D, 0.019, 0.3),
    }
}

#[allow(clippy::needless_range_loop)]
/// CPU f64 attention backward with IEEE NaN propagation. Returns (dq, dk, dv).
fn oracle(x: &Inp) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let s = SCALE as f64;
    let mut p = vec![0.0; Q * K];
    let mut y = vec![0.0; Q * D];
    for i in 0..Q {
        let mut sc = vec![0.0; K];
        for j in 0..K {
            sc[j] = s * (0..D).map(|d| x.q[i * D + d] * x.k[j * D + d]).sum::<f64>();
        }
        let m = sc.iter().fold(f64::NEG_INFINITY, |a, &b| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        });
        let e: Vec<f64> = sc.iter().map(|v| (v - m).exp()).collect();
        let z: f64 = e.iter().sum();
        for j in 0..K {
            p[i * K + j] = e[j] / z;
        }
        for d in 0..D {
            y[i * D + d] = (0..K).map(|j| p[i * K + j] * x.v[j * D + d]).sum();
        }
    }
    let mut dv = vec![0.0; K * D];
    for j in 0..K {
        for d in 0..D {
            dv[j * D + d] = (0..Q).map(|i| p[i * K + j] * x.dy[i * D + d]).sum();
        }
    }
    let mut dq = vec![0.0; Q * D];
    let mut dk = vec![0.0; K * D];
    let mut ds = vec![0.0; Q * K];
    for i in 0..Q {
        let dp: Vec<f64> = (0..K)
            .map(|j| (0..D).map(|d| x.dy[i * D + d] * x.v[j * D + d]).sum())
            .collect();
        let dd: f64 = (0..K).map(|j| p[i * K + j] * dp[j]).sum();
        for j in 0..K {
            ds[i * K + j] = p[i * K + j] * (dp[j] - dd);
        }
    }
    for i in 0..Q {
        for d in 0..D {
            dq[i * D + d] = s * (0..K).map(|j| ds[i * K + j] * x.k[j * D + d]).sum::<f64>();
        }
    }
    for j in 0..K {
        for d in 0..D {
            dk[j * D + d] = s * (0..Q).map(|i| ds[i * K + j] * x.q[i * D + d]).sum::<f64>();
        }
    }
    (dq, dk, dv)
}

macro_rules! suite {
    ($m:ident, $T:ty, $to:expr, $from:expr, $fw:path, $bw:path, $tol:expr) => {
        mod $m {
            use super::*;

            fn run(x: &Inp) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
                init().expect("driver init");
                let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
                let stream = Stream::new(&ctx).expect("stream");
                let up = |v: &Vec<f64>| -> DeviceBuffer<$T> {
                    let h: Vec<$T> = v.iter().map(|a| $to(*a)).collect();
                    DeviceBuffer::from_slice(&ctx, &h).unwrap()
                };
                let (dq_in, dk_in, dv_in, ddy) = (up(&x.q), up(&x.k), up(&x.v), up(&x.dy));
                let z = |n: usize| -> DeviceBuffer<$T> { DeviceBuffer::zeros(&ctx, n).unwrap() };
                let mut y = z(Q * D);
                let mut lse = z(Q);
                let mut d_ws = z(Q);
                let (mut gq, mut gk, mut gv) = (z(Q * D), z(K * D), z(K * D));
                let p = |b: &DeviceBuffer<$T>| b.as_slice().as_raw().0 as *const c_void;
                let st = unsafe {
                    $fw(
                        1, 1, Q as i32, K as i32, D as i32, D as i32, SCALE, 0,
                        p(&dq_in), p(&dk_in), p(&dv_in),
                        y.as_slice_mut().as_raw().0 as *mut c_void,
                        lse.as_slice_mut().as_raw().0 as *mut c_void,
                        core::ptr::null_mut(), 0, stream.as_raw(),
                    )
                };
                assert_eq!(st, 0, "forward status {st}");
                let st = unsafe {
                    $bw(
                        1, 1, Q as i32, K as i32, D as i32, D as i32, SCALE, 0,
                        p(&dq_in), p(&dk_in), p(&dv_in), p(&y),
                        lse.as_slice().as_raw().0 as *const c_void,
                        p(&ddy),
                        d_ws.as_slice_mut().as_raw().0 as *mut c_void,
                        gq.as_slice_mut().as_raw().0 as *mut c_void,
                        gk.as_slice_mut().as_raw().0 as *mut c_void,
                        gv.as_slice_mut().as_raw().0 as *mut c_void,
                        core::ptr::null_mut(), 0, stream.as_raw(),
                    )
                };
                assert_eq!(st, 0, "backward status {st}");
                stream.synchronize().unwrap();
                let dl = |b: &DeviceBuffer<$T>, n: usize| -> Vec<f64> {
                    let mut h = vec![$to(0.0); n];
                    b.copy_to_host(&mut h).unwrap();
                    h.iter().map(|a| $from(*a)).collect()
                };
                (dl(&gq, Q * D), dl(&gk, K * D), dl(&gv, K * D))
            }

            /// Compare the kernel to the oracle on the dtype-rounded inputs; returns a list of mismatch descriptions (empty = agree).
            #[allow(clippy::needless_range_loop)]
            fn probe(name: &str, mut x: Inp) -> Vec<String> {
                // Round inputs through the dtype so the oracle sees exactly what the kernel sees.
                for v in [&mut x.q, &mut x.k, &mut x.v, &mut x.dy] {
                    for a in v.iter_mut() {
                        *a = $from($to(*a));
                    }
                }
                let (oq, ok, ov) = oracle(&x);
                let (gq, gk, gv) = run(&x);
                let mut bad = Vec::new();
                for (label, g, o, cols) in [("dQ", &gq, &oq, Q), ("dK", &gk, &ok, K), ("dV", &gv, &ov, K)] {
                    let mut nan_extra = 0; // kernel NaN where oracle finite
                    let mut nan_missing = 0; // kernel finite where oracle NaN
                    let mut off = 0;
                    let mut first = String::new();
                    for i in 0..g.len() {
                        let (a, b) = (g[i], o[i]);
                        if a.is_nan() && !b.is_nan() {
                            nan_extra += 1;
                            if first.is_empty() { first = format!("row {} col {} kernel NaN oracle {b}", i / D, i % D); }
                        } else if !a.is_nan() && b.is_nan() {
                            nan_missing += 1;
                            if first.is_empty() { first = format!("row {} col {} kernel {a} oracle NaN", i / D, i % D); }
                        } else if !a.is_nan() && (a - b).abs() > $tol * (1.0 + b.abs()) {
                            off += 1;
                            if first.is_empty() { first = format!("row {} col {} kernel {a} oracle {b}", i / D, i % D); }
                        }
                    }
                    let _ = cols;
                    if nan_extra + nan_missing + off > 0 {
                        bad.push(format!("{name} {label}: kernel-NaN-oracle-finite={nan_extra} kernel-finite-oracle-NaN={nan_missing} value-off={off}; first: {first}"));
                    }
                }
                bad
            }

            #[test]
            fn finite_control_matches_the_cpu_oracle() {
                let b = probe("control", base());
                assert!(b.is_empty(), "{b:#?}");
            }

            #[test]
            fn nan_and_inf_inputs_match_the_ieee_oracle() {
                let mut bad = Vec::new();
                let mut c = base(); c.q[5 * D + 3] = f64::NAN;
                bad.extend(probe("NaN in Q row 5", c));
                let mut c = base(); c.k[9 * D] = f64::NAN;
                bad.extend(probe("NaN in K row 9", c));
                let mut c = base(); c.v[9 * D] = f64::NAN;
                bad.extend(probe("NaN in V row 9", c));
                let mut c = base(); c.dy[5 * D + 2] = f64::NAN;
                bad.extend(probe("NaN in dO row 5", c));
                let mut c = base(); c.q[5 * D + 3] = f64::INFINITY;
                bad.extend(probe("+inf in Q row 5", c));
                assert!(bad.is_empty(), "{bad:#?}");
            }
        }
    };
}

suite!(
    f32s,
    f32,
    |v: f64| v as f32,
    |v: f32| v as f64,
    sys::baracuda_kernels_flash_sdpa_f32_run,
    sys::baracuda_kernels_flash_sdpa_backward_f32_run,
    2e-4
);
suite!(
    f16s,
    f16,
    f16::from_f64,
    |v: f16| v.to_f64(),
    sys::baracuda_kernels_flash_sdpa_f16_run,
    sys::baracuda_kernels_flash_sdpa_backward_f16_run,
    2e-2
);
suite!(
    bf16s,
    bf16,
    bf16::from_f64,
    |v: bf16| v.to_f64(),
    sys::baracuda_kernels_flash_sdpa_bf16_run,
    sys::baracuda_kernels_flash_sdpa_backward_bf16_run,
    8e-2
);
