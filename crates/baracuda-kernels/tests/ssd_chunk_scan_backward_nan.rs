// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `SsdChunkScanBackwardPlan` (Mamba-2 SSD backward) against an IEEE f64 reverse-mode oracle, finite and with NaN / inf inputs
//! (NaN sweep row "ssd_chunk_scan backward").
//!
//! Recurrence per (b, h): `decay_t = exp(dt_t * A_h)`, `state_t[d][n] = decay_t * state_{t-1}[d][n] + dt_t * B[t][n] * x[t][d]`,
//! `y[t][d] = sum_n C[t][n] * state_t[d][n]`. The oracle runs that recurrence and its adjoint in f64 with plain IEEE arithmetic, so a NaN
//! propagates exactly as torch autograd would. For every probe the test compares the NaN PATTERN of dx, dB, dC, ddt, dA element by element and
//! the finite values within a dtype tolerance. The finite control is the positive control (it must match on values).
#![cfg(feature = "mamba")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, SsdChunkScanBackwardArgs, SsdChunkScanBackwardDescriptor,
    SsdChunkScanBackwardPlan, TensorMut, TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

const B: usize = 2;
const L: usize = 16;
const H: usize = 2;
const D: usize = 8;
const N: usize = 4;
const CHUNK: i32 = 4;

#[derive(Clone)]
struct Inp {
    x: Vec<f64>,  // [B, L, H, D]
    dt: Vec<f64>, // [B, L, H]
    a: Vec<f64>,  // [H]
    b: Vec<f64>,  // [B, L, H, N]
    c: Vec<f64>,  // [B, L, H, N]
    dy: Vec<f64>, // [B, L, H, D]
}

fn base() -> Inp {
    let g = |n: usize, f: f64, o: f64, s: f64| -> Vec<f64> {
        (0..n).map(|i| ((i as f64) * f + o).sin() * s).collect()
    };
    Inp {
        x: g(B * L * H * D, 0.071, 0.3, 0.8),
        dt: (0..B * L * H)
            .map(|i| 0.1 + 0.05 * (((i as f64) * 0.37).sin() + 1.0))
            .collect(),
        a: vec![-0.5, -0.8],
        b: g(B * L * H * N, 0.093, -0.2, 0.5),
        c: g(B * L * H * N, 0.057, 0.9, 0.5),
        dy: g(B * L * H * D, 0.043, -0.6, 0.7),
    }
}

struct Grads {
    dx: Vec<f64>,
    db: Vec<f64>,
    dc: Vec<f64>,
    ddt: Vec<f64>,
    da: Vec<f64>,
}

#[allow(clippy::needless_range_loop)]
fn oracle(x: &Inp) -> Grads {
    let mut g = Grads {
        dx: vec![0.0; B * L * H * D],
        db: vec![0.0; B * L * H * N],
        dc: vec![0.0; B * L * H * N],
        ddt: vec![0.0; B * L * H],
        da: vec![0.0; H],
    };
    for bi in 0..B {
        for h in 0..H {
            let ix = |t: usize, d: usize| ((bi * L + t) * H + h) * D + d;
            let in_ = |t: usize, n: usize| ((bi * L + t) * H + h) * N + n;
            let it = |t: usize| (bi * L + t) * H + h;
            // forward states: st[t] is the state AFTER step t
            let mut st = vec![vec![0.0f64; D * N]; L];
            let mut prev = vec![0.0f64; D * N];
            for t in 0..L {
                let decay = (x.dt[it(t)] * x.a[h]).exp();
                for d in 0..D {
                    for n in 0..N {
                        st[t][d * N + n] =
                            decay * prev[d * N + n] + x.dt[it(t)] * x.b[in_(t, n)] * x.x[ix(t, d)];
                    }
                }
                prev = st[t].clone();
            }
            let mut gs = vec![0.0f64; D * N];
            for t in (0..L).rev() {
                let dt = x.dt[it(t)];
                let decay = (dt * x.a[h]).exp();
                let sp = if t == 0 {
                    vec![0.0; D * N]
                } else {
                    st[t - 1].clone()
                };
                for d in 0..D {
                    for n in 0..N {
                        gs[d * N + n] += x.dy[ix(t, d)] * x.c[in_(t, n)];
                        g.dc[in_(t, n)] += x.dy[ix(t, d)] * st[t][d * N + n];
                    }
                }
                for d in 0..D {
                    for n in 0..N {
                        let a = gs[d * N + n];
                        g.dx[ix(t, d)] += a * dt * x.b[in_(t, n)];
                        g.db[in_(t, n)] += a * dt * x.x[ix(t, d)];
                        g.ddt[it(t)] +=
                            a * (sp[d * N + n] * decay * x.a[h] + x.b[in_(t, n)] * x.x[ix(t, d)]);
                        g.da[h] += a * sp[d * N + n] * decay * dt;
                    }
                }
                for d in 0..D {
                    for n in 0..N {
                        gs[d * N + n] *= decay;
                    }
                }
            }
        }
    }
    g
}

trait Conv: Copy + baracuda_kernels::Element {
    const EK: ElementKind;
    fn from64(v: f64) -> Self;
    fn to64(self) -> f64;
}
impl Conv for f32 {
    const EK: ElementKind = ElementKind::F32;
    fn from64(v: f64) -> Self {
        v as f32
    }
    fn to64(self) -> f64 {
        self as f64
    }
}
impl Conv for f16 {
    const EK: ElementKind = ElementKind::F16;
    fn from64(v: f64) -> Self {
        f16::from_f64(v)
    }
    fn to64(self) -> f64 {
        f16::to_f64(self)
    }
}
impl Conv for bf16 {
    const EK: ElementKind = ElementKind::Bf16;
    fn from64(v: f64) -> Self {
        bf16::from_f64(v)
    }
    fn to64(self) -> f64 {
        bf16::to_f64(self)
    }
}

fn run<T: Conv>(x: &Inp) -> Grads {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let up = |v: &Vec<f64>| -> DeviceBuffer<T> {
        let h: Vec<T> = v.iter().map(|a| T::from64(*a)).collect();
        DeviceBuffer::from_slice(&ctx, &h).unwrap()
    };
    let z = |n: usize| -> DeviceBuffer<T> { DeviceBuffer::zeros(&ctx, n).unwrap() };
    let (dx_in, ddt_in, da_in, db_in, dc_in, ddy) =
        (up(&x.x), up(&x.dt), up(&x.a), up(&x.b), up(&x.c), up(&x.dy));
    let (mut gx, mut gb, mut gc, mut gdt, mut ga) = (
        z(B * L * H * D),
        z(B * L * H * N),
        z(B * L * H * N),
        z(B * L * H),
        z(H),
    );
    let desc = SsdChunkScanBackwardDescriptor {
        batch_size: B as i32,
        seq_len: L as i32,
        num_heads: H as i32,
        head_dim: D as i32,
        state_dim: N as i32,
        chunk_size: CHUNK,
        element: T::EK,
    };
    let plan = SsdChunkScanBackwardPlan::<T>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let mut ws = DeviceBuffer::<u8>::zeros(&ctx, plan.workspace_size().max(1)).unwrap();
    let sx: [i32; 4] = [B as i32, L as i32, H as i32, D as i32];
    let sdt: [i32; 3] = [B as i32, L as i32, H as i32];
    let sa: [i32; 1] = [H as i32];
    let sbn: [i32; 4] = [B as i32, L as i32, H as i32, N as i32];
    plan.run(
        &stream,
        Workspace::Borrowed(ws.as_slice_mut()),
        SsdChunkScanBackwardArgs {
            x: TensorRef::new(dx_in.as_slice(), sx, contiguous_stride(sx)).unwrap(),
            dt: TensorRef::new(ddt_in.as_slice(), sdt, contiguous_stride(sdt)).unwrap(),
            a: TensorRef::new(da_in.as_slice(), sa, contiguous_stride(sa)).unwrap(),
            b: TensorRef::new(db_in.as_slice(), sbn, contiguous_stride(sbn)).unwrap(),
            c: TensorRef::new(dc_in.as_slice(), sbn, contiguous_stride(sbn)).unwrap(),
            dy: TensorRef::new(ddy.as_slice(), sx, contiguous_stride(sx)).unwrap(),
            dx: TensorMut::new(gx.as_slice_mut(), sx, contiguous_stride(sx)).unwrap(),
            d_b: TensorMut::new(gb.as_slice_mut(), sbn, contiguous_stride(sbn)).unwrap(),
            d_c: TensorMut::new(gc.as_slice_mut(), sbn, contiguous_stride(sbn)).unwrap(),
            d_dt: TensorMut::new(gdt.as_slice_mut(), sdt, contiguous_stride(sdt)).unwrap(),
            d_a: TensorMut::new(ga.as_slice_mut(), sa, contiguous_stride(sa)).unwrap(),
        },
    )
    .expect("run");
    stream.synchronize().unwrap();
    let dl = |b: &DeviceBuffer<T>, n: usize| -> Vec<f64> {
        let mut h = vec![T::from64(0.0); n];
        b.copy_to_host(&mut h).unwrap();
        h.iter().map(|a| a.to64()).collect()
    };
    Grads {
        dx: dl(&gx, B * L * H * D),
        db: dl(&gb, B * L * H * N),
        dc: dl(&gc, B * L * H * N),
        ddt: dl(&gdt, B * L * H),
        da: dl(&ga, H),
    }
}

/// Compare kernel to oracle on dtype-rounded inputs; returns mismatch descriptions (empty = agree).
fn probe<T: Conv>(name: &str, mut x: Inp, tol: f64) -> Vec<String> {
    for v in [&mut x.x, &mut x.dt, &mut x.a, &mut x.b, &mut x.c, &mut x.dy] {
        for a in v.iter_mut() {
            *a = T::from64(*a).to64();
        }
    }
    let o = oracle(&x);
    let k = run::<T>(&x);
    // Non-vacuity: a NaN probe must make the oracle NaN somewhere (and the control must be fully finite).
    let oracle_nans: usize = [&o.dx, &o.db, &o.dc, &o.ddt, &o.da]
        .iter()
        .map(|v| v.iter().filter(|a| a.is_nan()).count())
        .sum();
    let kernel_nans: usize = [&k.dx, &k.db, &k.dc, &k.ddt, &k.da]
        .iter()
        .map(|v| v.iter().filter(|a| a.is_nan()).count())
        .sum();
    println!("{name}: oracle NaN cells {oracle_nans}, kernel NaN cells {kernel_nans}");
    assert_eq!(
        oracle_nans == 0,
        name == "control",
        "{name}: the oracle must be NaN-free exactly for the control"
    );
    let mut bad = Vec::new();
    for (label, g, r) in [
        ("dx", &k.dx, &o.dx),
        ("dB", &k.db, &o.db),
        ("dC", &k.dc, &o.dc),
        ("ddt", &k.ddt, &o.ddt),
        ("dA", &k.da, &o.da),
    ] {
        let (mut extra, mut missing, mut off) = (0, 0, 0);
        let mut first = String::new();
        for i in 0..g.len() {
            let (a, b) = (g[i], r[i]);
            let what = if a.is_nan() && !b.is_nan() {
                extra += 1;
                true
            } else if !a.is_nan() && b.is_nan() {
                missing += 1;
                true
            } else if !a.is_nan() && (a - b).abs() > tol * (1.0 + b.abs()) {
                off += 1;
                true
            } else {
                false
            };
            if what && first.is_empty() {
                first = format!("[{i}] kernel {a} oracle {b}");
            }
        }
        if extra + missing + off > 0 {
            bad.push(format!(
                "{name} {label}: kernel-NaN-oracle-finite={extra} kernel-finite-oracle-NaN={missing} value-off={off}; first {first}"
            ));
        }
    }
    bad
}

fn xd(b: usize, t: usize, h: usize, d: usize) -> usize {
    ((b * L + t) * H + h) * D + d
}
fn xn(b: usize, t: usize, h: usize, n: usize) -> usize {
    ((b * L + t) * H + h) * N + n
}
fn xt(b: usize, t: usize, h: usize) -> usize {
    (b * L + t) * H + h
}

fn all_probes<T: Conv>(tol: f64) -> Vec<String> {
    let mut bad = Vec::new();
    bad.extend(probe::<T>("control", base(), tol));
    let mut c = base();
    c.dy[xd(0, 9, 1, 3)] = f64::NAN;
    bad.extend(probe::<T>("NaN in dy (b0 t9 h1 d3)", c, tol));
    let mut c = base();
    c.x[xd(1, 5, 0, 2)] = f64::NAN;
    bad.extend(probe::<T>("NaN in x (b1 t5 h0 d2)", c, tol));
    let mut c = base();
    c.b[xn(0, 6, 1, 1)] = f64::NAN;
    bad.extend(probe::<T>("NaN in B (b0 t6 h1 n1)", c, tol));
    let mut c = base();
    c.c[xn(1, 11, 0, 2)] = f64::NAN;
    bad.extend(probe::<T>("NaN in C (b1 t11 h0 n2)", c, tol));
    let mut c = base();
    c.dt[xt(0, 3, 0)] = f64::NAN;
    bad.extend(probe::<T>("NaN in dt (b0 t3 h0)", c, tol));
    let mut c = base();
    c.a[1] = f64::NAN;
    bad.extend(probe::<T>("NaN in A (h1)", c, tol));
    bad
}

#[test]
fn f32_matches_the_ieee_oracle() {
    let bad = all_probes::<f32>(2e-3);
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn f16_matches_the_ieee_oracle() {
    let bad = all_probes::<f16>(5e-2);
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn bf16_matches_the_ieee_oracle() {
    let bad = all_probes::<bf16>(2e-1);
    assert!(bad.is_empty(), "{bad:#?}");
}
