// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `SelectiveScanBackwardPlan` (Mamba-1) against an IEEE f64 reverse-mode oracle, finite and with NaN inputs (NaN sweep row
//! "selective_scan backward"). Configuration: no `z` gate, no `D` skip, no delta bias, no softplus (the plain recurrence).
//!
//! Recurrence per (b, d, n): `decay = exp(delta * A[d][n])`, `h_t = decay * h_{t-1} + delta * B[t][n] * u[t][d]`,
//! `y[t][d] = sum_n C[t][n] * h_t[d][n]`. The oracle runs it and its adjoint in f64 with plain IEEE arithmetic, so a NaN propagates exactly as
//! torch autograd would. Each probe compares the NaN PATTERN of du, ddelta, dA, dB, dC cell by cell and the finite values within a dtype
//! tolerance; the finite control is the positive control, and each NaN probe asserts the oracle really is NaN somewhere (non-vacuous).
#![cfg(feature = "mamba")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, SelectiveScanBackwardArgs, SelectiveScanBackwardDescriptor,
    SelectiveScanBackwardPlan, TensorMut, TensorRef, Workspace, contiguous_stride,
};
use half::{bf16, f16};

const BSZ: usize = 2;
const L: usize = 16;
const DIM: usize = 4;
const N: usize = 4;

#[derive(Clone)]
struct Inp {
    u: Vec<f64>,
    delta: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    dy: Vec<f64>,
}

fn base() -> Inp {
    let g = |n: usize, f: f64, o: f64, s: f64| -> Vec<f64> {
        (0..n).map(|i| ((i as f64) * f + o).sin() * s).collect()
    };
    Inp {
        u: g(BSZ * L * DIM, 0.071, 0.3, 0.8),
        delta: (0..BSZ * L * DIM)
            .map(|i| 0.1 + 0.05 * (((i as f64) * 0.37).sin() + 1.0))
            .collect(),
        a: (0..DIM * N).map(|i| -0.3 - 0.1 * (i as f64)).collect(),
        b: g(BSZ * L * N, 0.093, -0.2, 0.5),
        c: g(BSZ * L * N, 0.057, 0.9, 0.5),
        dy: g(BSZ * L * DIM, 0.043, -0.6, 0.7),
    }
}

struct Grads {
    du: Vec<f64>,
    ddelta: Vec<f64>,
    da: Vec<f64>,
    db: Vec<f64>,
    dc: Vec<f64>,
}

fn iud(b: usize, t: usize, d: usize) -> usize {
    (b * L + t) * DIM + d
}
fn ibn(b: usize, t: usize, n: usize) -> usize {
    (b * L + t) * N + n
}

#[allow(clippy::needless_range_loop)]
fn oracle(x: &Inp) -> Grads {
    let mut g = Grads {
        du: vec![0.0; BSZ * L * DIM],
        ddelta: vec![0.0; BSZ * L * DIM],
        da: vec![0.0; DIM * N],
        db: vec![0.0; BSZ * L * N],
        dc: vec![0.0; BSZ * L * N],
    };
    for bi in 0..BSZ {
        for d in 0..DIM {
            let mut hs = vec![vec![0.0f64; N]; L];
            let mut prev = vec![0.0f64; N];
            for t in 0..L {
                let dl = x.delta[iud(bi, t, d)];
                for n in 0..N {
                    let decay = (dl * x.a[d * N + n]).exp();
                    hs[t][n] = decay * prev[n] + dl * x.b[ibn(bi, t, n)] * x.u[iud(bi, t, d)];
                }
                prev = hs[t].clone();
            }
            let mut gs = [0.0f64; N];
            for t in (0..L).rev() {
                let dl = x.delta[iud(bi, t, d)];
                for n in 0..N {
                    let decay = (dl * x.a[d * N + n]).exp();
                    let hp = if t == 0 { 0.0 } else { hs[t - 1][n] };
                    gs[n] += x.dy[iud(bi, t, d)] * x.c[ibn(bi, t, n)];
                    g.dc[ibn(bi, t, n)] += x.dy[iud(bi, t, d)] * hs[t][n];
                    g.du[iud(bi, t, d)] += gs[n] * dl * x.b[ibn(bi, t, n)];
                    g.db[ibn(bi, t, n)] += gs[n] * dl * x.u[iud(bi, t, d)];
                    g.ddelta[iud(bi, t, d)] += gs[n]
                        * (hp * decay * x.a[d * N + n] + x.b[ibn(bi, t, n)] * x.u[iud(bi, t, d)]);
                    g.da[d * N + n] += gs[n] * hp * decay * dl;
                    gs[n] *= decay;
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
    let (du_in, dd_in, da_in, db_in, dc_in, ddy) = (
        up(&x.u),
        up(&x.delta),
        up(&x.a),
        up(&x.b),
        up(&x.c),
        up(&x.dy),
    );
    let (mut gu, mut gdd, mut gda, mut gb, mut gc) = (
        z(BSZ * L * DIM),
        z(BSZ * L * DIM),
        z(DIM * N),
        z(BSZ * L * N),
        z(BSZ * L * N),
    );
    let desc = SelectiveScanBackwardDescriptor {
        batch_size: BSZ as i32,
        seq_len: L as i32,
        dim: DIM as i32,
        dstate: N as i32,
        delta_softplus: false,
        element: T::EK,
    };
    let plan = SelectiveScanBackwardPlan::<T>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let mut ws = DeviceBuffer::<u8>::zeros(&ctx, plan.workspace_size().max(1)).unwrap();
    let sud: [i32; 3] = [BSZ as i32, L as i32, DIM as i32];
    let sa: [i32; 2] = [DIM as i32, N as i32];
    let sbc: [i32; 3] = [BSZ as i32, L as i32, N as i32];
    plan.run(
        &stream,
        Workspace::Borrowed(ws.as_slice_mut()),
        SelectiveScanBackwardArgs {
            u: TensorRef::new(du_in.as_slice(), sud, contiguous_stride(sud)).unwrap(),
            delta: TensorRef::new(dd_in.as_slice(), sud, contiguous_stride(sud)).unwrap(),
            a: TensorRef::new(da_in.as_slice(), sa, contiguous_stride(sa)).unwrap(),
            b: TensorRef::new(db_in.as_slice(), sbc, contiguous_stride(sbc)).unwrap(),
            c: TensorRef::new(dc_in.as_slice(), sbc, contiguous_stride(sbc)).unwrap(),
            d_skip: None,
            z: None,
            delta_bias: None,
            dy: TensorRef::new(ddy.as_slice(), sud, contiguous_stride(sud)).unwrap(),
            du: TensorMut::new(gu.as_slice_mut(), sud, contiguous_stride(sud)).unwrap(),
            d_b: TensorMut::new(gb.as_slice_mut(), sbc, contiguous_stride(sbc)).unwrap(),
            d_c: TensorMut::new(gc.as_slice_mut(), sbc, contiguous_stride(sbc)).unwrap(),
            d_delta: TensorMut::new(gdd.as_slice_mut(), sud, contiguous_stride(sud)).unwrap(),
            d_a: TensorMut::new(gda.as_slice_mut(), sa, contiguous_stride(sa)).unwrap(),
            d_d: None,
            dz: None,
            d_delta_bias: None,
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
        du: dl(&gu, BSZ * L * DIM),
        ddelta: dl(&gdd, BSZ * L * DIM),
        da: dl(&gda, DIM * N),
        db: dl(&gb, BSZ * L * N),
        dc: dl(&gc, BSZ * L * N),
    }
}

fn probe<T: Conv>(name: &str, mut x: Inp, tol: f64) -> Vec<String> {
    for v in [
        &mut x.u,
        &mut x.delta,
        &mut x.a,
        &mut x.b,
        &mut x.c,
        &mut x.dy,
    ] {
        for a in v.iter_mut() {
            *a = T::from64(*a).to64();
        }
    }
    let o = oracle(&x);
    let k = run::<T>(&x);
    let count = |g: &Grads| -> usize {
        [&g.du, &g.ddelta, &g.da, &g.db, &g.dc]
            .iter()
            .map(|v| v.iter().filter(|a| a.is_nan()).count())
            .sum()
    };
    let (oracle_nans, kernel_nans) = (count(&o), count(&k));
    println!("{name}: oracle NaN cells {oracle_nans}, kernel NaN cells {kernel_nans}");
    assert_eq!(
        oracle_nans == 0,
        name == "control",
        "{name}: the oracle must be NaN-free exactly for the control"
    );
    let mut bad = Vec::new();
    for (label, g, r) in [
        ("du", &k.du, &o.du),
        ("ddelta", &k.ddelta, &o.ddelta),
        ("dA", &k.da, &o.da),
        ("dB", &k.db, &o.db),
        ("dC", &k.dc, &o.dc),
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

fn all_probes<T: Conv>(tol: f64) -> Vec<String> {
    let mut bad = Vec::new();
    bad.extend(probe::<T>("control", base(), tol));
    let mut c = base();
    c.dy[iud(0, 9, 3)] = f64::NAN;
    bad.extend(probe::<T>("NaN in dy (b0 t9 d3)", c, tol));
    let mut c = base();
    c.u[iud(1, 5, 2)] = f64::NAN;
    bad.extend(probe::<T>("NaN in u (b1 t5 d2)", c, tol));
    let mut c = base();
    c.b[ibn(0, 6, 1)] = f64::NAN;
    bad.extend(probe::<T>("NaN in B (b0 t6 n1)", c, tol));
    let mut c = base();
    c.c[ibn(1, 11, 2)] = f64::NAN;
    bad.extend(probe::<T>("NaN in C (b1 t11 n2)", c, tol));
    let mut c = base();
    c.delta[iud(0, 3, 0)] = f64::NAN;
    bad.extend(probe::<T>("NaN in delta (b0 t3 d0)", c, tol));
    let mut c = base();
    c.a[DIM + 2] = f64::NAN;
    bad.extend(probe::<T>("NaN in A (d1 n2)", c, tol));
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
