//! Real-GPU numerics check of `BatchPagedDecodePlan` against a contiguous
//! SDPA reference.
//!
//! The reference is computed on the host in `f64` from the *contiguous* K/V
//! history of each request (the test gathers it out of the paged store by
//! hand, using the same page table it hands the kernel): for every query head
//! `h`, `softmax(sm_scale * q_h . K_g^T) V_g` with `g = h / group_size`. The
//! kernel inputs are rounded to the element type first and the reference reads
//! those rounded values, so the only error measured is the kernel's.
//!
//! Coverage: `f16` and `bf16`; `head_dim` 64 / 128 / 256; GQA group sizes 1
//! and 4; `page_size` 16; several pages per request; shuffled physical page
//! ids; pages the page table does not name are filled with large garbage so a
//! stray read shows up; `last_page_len` of 1, mid-page, `page_size`, and 0
//! (with at least two pages, so the request is non-empty).
//!
//! The maximum absolute error per case is printed (`--nocapture`) and the
//! per-dtype worst case is asserted below a bound with headroom.
//!
//! Gated behind `device-tests` (a missing adapter FAILS, it does not skip).

#![cfg(feature = "flashinfer")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{CaptureMode, Context, Device, DeviceBuffer, Event, Stream, init};
use baracuda_flashinfer::attention::{
    BatchPagedDecodeArgs, BatchPagedDecodeDescriptor, BatchPagedDecodePlan, PagedKvCacheDescriptor,
};
use baracuda_flashinfer::{PlanPreference, TensorMut, TensorRef, Workspace, contiguous_stride};
use baracuda_kernels::Element;
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Host-side conversion for the two element types under test.
trait Host: Element + Copy {
    fn from_f32(x: f32) -> Self;
    fn to_f32(self) -> f32;
    fn zero() -> Self;
}
impl Host for f16 {
    fn from_f32(x: f32) -> Self {
        f16::from_f32(x)
    }
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }
    fn zero() -> Self {
        f16::ZERO
    }
}
impl Host for bf16 {
    fn from_f32(x: f32) -> Self {
        bf16::from_f32(x)
    }
    fn to_f32(self) -> f32 {
        bf16::to_f32(self)
    }
    fn zero() -> Self {
        bf16::ZERO
    }
}

/// Deterministic xorshift in `[-1, 1)`.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

struct Case {
    head_dim: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    /// Per request: (number of pages, last_page_len).
    reqs: &'static [(usize, usize)],
}

const PS: usize = 16;
/// Edge cases for `last_page_len`: full page, 1 row, mid-page, 0 (>= 2 pages).
const REQS: &[(usize, usize)] = &[(3, PS), (2, 1), (4, 9), (2, 0), (1, PS), (1, 1)];

/// Extra measurements for one case: reference magnitude, launch time, CUDA-graph capture.
#[derive(Default)]
struct Bench {
    /// Largest |reference output| (so `max_o_err / max_ref` is a normalised error).
    max_ref: f64,
    /// Largest |err| / max(|ref|, 0.05): a relative error that ignores near-zero outputs.
    max_rel: f64,
    /// Milliseconds per eager launch (200 launches between two events, after 20 warm-ups).
    eager_ms: f64,
    /// Milliseconds per layer-launch when 8 launches are captured into one graph; `None` if capture failed.
    graph_ms_per_launch: Option<f64>,
    /// `"ok"`, or the error that stopped capture / instantiation / launch.
    graph_note: String,
    /// Max |graph output - eager output| (0 expected: the kernel is deterministic).
    graph_vs_eager: f64,
}

/// Returns the max absolute error of `o` and of `lse` against the reference.
fn run_case<T: Host>(
    ctx: &Context,
    stream: &Stream,
    c: &Case,
    seed: u64,
    bench: Option<&mut Bench>,
) -> (f64, f64) {
    let hd = c.head_dim;
    let group = c.num_qo_heads / c.num_kv_heads;
    let batch = c.reqs.len();
    let used: usize = c.reqs.iter().map(|r| r.0).sum();
    // Spare pages the page table never names.
    let total_pages = used + 5;
    let mut rng = Rng(seed);

    // Shuffled physical page ids (Fisher-Yates), then carved per request.
    let mut ids: Vec<i32> = (0..total_pages as i32).collect();
    for i in (1..ids.len()).rev() {
        let j = (rng.next().abs() * 1e6) as usize % (i + 1);
        ids.swap(i, j);
    }
    let mut indices = Vec::new();
    let mut indptr = vec![0i32];
    let mut last_page_len = Vec::new();
    let mut next = 0usize;
    for &(np, lpl) in c.reqs {
        indices.extend_from_slice(&ids[next..next + np]);
        next += np;
        indptr.push(indices.len() as i32);
        last_page_len.push(lpl as i32);
    }
    let unused: Vec<i32> = ids[next..].to_vec();

    // Paged store [total_pages, kv_heads, page_size, hd]; unused pages get
    // large garbage so a read of them cannot hide.
    let page_elems = c.num_kv_heads * c.page_size * hd;
    let mut k_store = vec![T::zero(); total_pages * page_elems];
    let mut v_store = vec![T::zero(); total_pages * page_elems];
    for p in 0..total_pages {
        let garbage = unused.contains(&(p as i32));
        for e in 0..page_elems {
            let (kv, vv) = if garbage {
                (50.0, -50.0)
            } else {
                (rng.next(), rng.next())
            };
            k_store[p * page_elems + e] = T::from_f32(kv);
            v_store[p * page_elems + e] = T::from_f32(vv);
        }
    }
    let q_h: Vec<T> = (0..batch * c.num_qo_heads * hd)
        .map(|_| T::from_f32(rng.next()))
        .collect();

    // Device.
    let q_dev = DeviceBuffer::from_slice(ctx, &q_h).expect("upload q");
    let k_dev = DeviceBuffer::from_slice(ctx, &k_store).expect("upload k");
    let v_dev = DeviceBuffer::from_slice(ctx, &v_store).expect("upload v");
    let idx_dev = DeviceBuffer::from_slice(ctx, &indices).expect("upload indices");
    let ptr_dev = DeviceBuffer::from_slice(ctx, &indptr).expect("upload indptr");
    let lpl_dev = DeviceBuffer::from_slice(ctx, &last_page_len).expect("upload last_page_len");
    let mut o_dev: DeviceBuffer<T> =
        DeviceBuffer::zeros(ctx, batch * c.num_qo_heads * hd).expect("alloc o");
    let mut lse_dev: DeviceBuffer<f32> =
        DeviceBuffer::zeros(ctx, batch * c.num_qo_heads).expect("alloc lse");

    let sm_scale = 1.0 / (hd as f32).sqrt();
    let desc = BatchPagedDecodeDescriptor {
        batch_size: batch as i32,
        num_qo_heads: c.num_qo_heads as i32,
        sm_scale,
        paged_kv: PagedKvCacheDescriptor {
            page_size: c.page_size as i32,
            num_total_pages: total_pages as i32,
            num_kv_heads: c.num_kv_heads as i32,
            head_dim: hd as i32,
            element: T::KIND,
        },
    };
    let plan = BatchPagedDecodePlan::<T>::select(stream, &desc, PlanPreference::default())
        .expect("select");
    let mut ws_dev: DeviceBuffer<u8> =
        DeviceBuffer::zeros(ctx, plan.workspace_size().max(1)).expect("alloc ws");

    let q_shape = [batch as i32, c.num_qo_heads as i32, hd as i32];
    let cache_shape = [
        total_pages as i32,
        c.num_kv_heads as i32,
        c.page_size as i32,
        hd as i32,
    ];
    let lse_shape = [batch as i32, c.num_qo_heads as i32];
    let launch = |o_dev: &mut DeviceBuffer<T>,
                  lse_dev: &mut DeviceBuffer<f32>,
                  ws_dev: &mut DeviceBuffer<u8>| {
        plan.run(
            stream,
            Workspace::Borrowed(ws_dev.as_slice_mut()),
            BatchPagedDecodeArgs {
                q: TensorRef::new(q_dev.as_slice(), q_shape, contiguous_stride(q_shape)).unwrap(),
                k_data: TensorRef::new(
                    k_dev.as_slice(),
                    cache_shape,
                    contiguous_stride(cache_shape),
                )
                .unwrap(),
                v_data: TensorRef::new(
                    v_dev.as_slice(),
                    cache_shape,
                    contiguous_stride(cache_shape),
                )
                .unwrap(),
                indices: TensorRef::new(idx_dev.as_slice(), [indices.len() as i32], [1]).unwrap(),
                indptr: TensorRef::new(ptr_dev.as_slice(), [batch as i32 + 1], [1]).unwrap(),
                last_page_len: TensorRef::new(lpl_dev.as_slice(), [batch as i32], [1]).unwrap(),
                o: TensorMut::new(o_dev.as_slice_mut(), q_shape, contiguous_stride(q_shape))
                    .unwrap(),
                lse: TensorMut::new(
                    lse_dev.as_slice_mut(),
                    lse_shape,
                    contiguous_stride(lse_shape),
                )
                .unwrap(),
            },
        )
    };
    launch(&mut o_dev, &mut lse_dev, &mut ws_dev).expect("paged decode run");
    stream.synchronize().expect("sync");

    let mut o_host = vec![T::zero(); batch * c.num_qo_heads * hd];
    o_dev.copy_to_host(&mut o_host).expect("download o");
    let mut lse_host = vec![0f32; batch * c.num_qo_heads];
    lse_dev.copy_to_host(&mut lse_host).expect("download lse");

    // Reference: gather each request's contiguous K/V from the paged store.
    let (mut max_o, mut max_lse) = (0f64, 0f64);
    let (mut max_ref, mut max_rel) = (0f64, 0f64);
    for (b, &(np, lpl)) in c.reqs.iter().enumerate() {
        let kv_len = (np - 1) * c.page_size + lpl;
        assert!(kv_len > 0, "case must be non-empty");
        let pages = &indices[indptr[b] as usize..indptr[b + 1] as usize];
        let mut req_max = 0f64;
        for h in 0..c.num_qo_heads {
            let g = h / group;
            let qv = &q_h[(b * c.num_qo_heads + h) * hd..][..hd];
            let at = |store: &[T], t: usize, d: usize| -> f64 {
                let page = pages[t / c.page_size] as usize;
                let row = t % c.page_size;
                store[page * page_elems + (g * c.page_size + row) * hd + d].to_f32() as f64
            };
            let scores: Vec<f64> = (0..kv_len)
                .map(|t| {
                    (0..hd)
                        .map(|d| qv[d].to_f32() as f64 * at(&k_store, t, d))
                        .sum::<f64>()
                        * sm_scale as f64
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let den: f64 = scores.iter().map(|s| (s - m).exp()).sum();
            // FlashInfer reports lse in base 2.
            let lse_ref = (m + den.ln()) * std::f64::consts::LOG2_E;
            let got_lse = lse_host[b * c.num_qo_heads + h] as f64;
            max_lse = max_lse.max((got_lse - lse_ref).abs());
            for d in 0..hd {
                let want: f64 = (0..kv_len)
                    .map(|t| (scores[t] - m).exp() / den * at(&v_store, t, d))
                    .sum();
                let got = o_host[(b * c.num_qo_heads + h) * hd + d].to_f32() as f64;
                max_o = max_o.max((got - want).abs());
                max_ref = max_ref.max(want.abs());
                max_rel = max_rel.max((got - want).abs() / want.abs().max(0.05));
                req_max = req_max.max((got - want).abs());
            }
        }
        eprintln!(
            "  req {b}: pages={np} last_page_len={lpl} kv_len={kv_len} max|o err|={req_max:.3e}"
        );
    }
    if let Some(b) = bench {
        b.max_ref = max_ref;
        b.max_rel = max_rel;
        for _ in 0..20 {
            launch(&mut o_dev, &mut lse_dev, &mut ws_dev).expect("warm-up");
        }
        stream.synchronize().expect("sync");
        let (e0, e1) = (Event::new(ctx).unwrap(), Event::new(ctx).unwrap());
        let iters = 200;
        e0.record(stream).unwrap();
        for _ in 0..iters {
            launch(&mut o_dev, &mut lse_dev, &mut ws_dev).expect("timed launch");
        }
        e1.record(stream).unwrap();
        e1.synchronize().unwrap();
        b.eager_ms = Event::elapsed_time_ms(&e0, &e1).unwrap() as f64 / iters as f64;

        // CUDA-graph capture: 8 launches (Fuel's 8 layers) into one graph.
        let layers = 8usize;
        let mut launch_err: Option<String> = None;
        let captured = stream.capture(CaptureMode::ThreadLocal, |_| {
            for _ in 0..layers {
                if let Err(e) = launch(&mut o_dev, &mut lse_dev, &mut ws_dev) {
                    launch_err = Some(format!("launch under capture: {e:?}"));
                    break;
                }
            }
            Ok(())
        });
        b.graph_note = match (captured, launch_err) {
            (_, Some(e)) => e,
            (Err(e), None) => format!("capture: {e:?}"),
            (Ok(graph), None) => match graph.instantiate() {
                Err(e) => format!("instantiate: {e:?}"),
                Ok(exec) => {
                    // Poison o, run the graph, and compare with the eager result.
                    o_dev
                        .copy_from_host(&vec![T::from_f32(7.0); o_host.len()])
                        .expect("poison o");
                    exec.launch(stream).expect("graph launch");
                    stream.synchronize().expect("sync");
                    let mut o_graph = vec![T::zero(); o_host.len()];
                    o_dev.copy_to_host(&mut o_graph).expect("download o");
                    b.graph_vs_eager = o_graph
                        .iter()
                        .zip(&o_host)
                        .map(|(a, e)| (a.to_f32() as f64 - e.to_f32() as f64).abs())
                        .fold(0.0, f64::max);
                    e0.record(stream).unwrap();
                    for _ in 0..50 {
                        exec.launch(stream).expect("graph launch");
                    }
                    e1.record(stream).unwrap();
                    e1.synchronize().unwrap();
                    b.graph_ms_per_launch = Some(
                        Event::elapsed_time_ms(&e0, &e1).unwrap() as f64 / (50 * layers) as f64,
                    );
                    "ok".to_owned()
                }
            },
        };
    }
    (max_o, max_lse)
}

fn sweep<T: Host>(label: &str, o_bound: f64, lse_bound: f64) {
    let (ctx, stream) = setup();
    let (mut worst_o, mut worst_lse) = (0f64, 0f64);
    for &hd in &[64usize, 128, 256] {
        // (qo heads, kv heads): MHA and GQA group 4.
        for &(qo, kv) in &[(4usize, 4usize), (8, 2)] {
            let c = Case {
                head_dim: hd,
                num_qo_heads: qo,
                num_kv_heads: kv,
                page_size: PS,
                reqs: REQS,
            };
            let (eo, el) = run_case::<T>(
                &ctx,
                &stream,
                &c,
                0x9E37_79B9 ^ (hd as u64 * 131 + qo as u64),
                None,
            );
            eprintln!(
                "paged_decode {label} hd={hd} qo={qo} kv={kv}: max|o err|={eo:.3e} max|lse err|={el:.3e}"
            );
            worst_o = worst_o.max(eo);
            worst_lse = worst_lse.max(el);
        }
    }
    eprintln!("paged_decode {label} WORST: max|o err|={worst_o:.3e} max|lse err|={worst_lse:.3e}");
    assert!(
        worst_o < o_bound,
        "{label}: o error {worst_o:e} >= {o_bound:e}"
    );
    assert!(
        worst_lse < lse_bound,
        "{label}: lse error {worst_lse:e} >= {lse_bound:e}"
    );
}

#[test]
fn paged_decode_matches_contiguous_sdpa_f16() {
    sweep::<f16>("f16", 1e-3, 1e-4);
}

#[test]
fn paged_decode_matches_contiguous_sdpa_bf16() {
    sweep::<bf16>("bf16", 6e-3, 1e-4);
}

/// Fuel's GAP-380 fixtures (8 layers, 16 heads x d64, cap 256), `page_size` 16: one sequence at the cap (16 full
/// pages), the same with 4 KV heads (GQA group 4), a batch of 8 at the cap, and a ragged batch. Numerics against the
/// reference, milliseconds per launch (events, after warm-up) and whether 8 launches capture into a CUDA graph and
/// replay with the eager result. The device is named in the output.
fn fuel_fixtures<T: Host>(label: &str, o_bound: f64) {
    let (ctx, stream) = setup();
    let device = Device::get(0)
        .expect("device 0")
        .name()
        .expect("device name");
    static AT_CAP: &[(usize, usize)] = &[(16, 16)];
    static BATCH8: &[(usize, usize)] = &[(16, 16); 8];
    static RAGGED8: &[(usize, usize)] = &[
        (16, 16),
        (12, 7),
        (9, 16),
        (16, 1),
        (3, 16),
        (14, 5),
        (8, 16),
        (16, 9),
    ];
    let cases: [(&str, usize, usize, &'static [(usize, usize)]); 4] = [
        ("batch1 mha16 kv256", 16, 16, AT_CAP),
        ("batch1 gqa16/4 kv256", 16, 4, AT_CAP),
        ("batch8 mha16 kv256", 16, 16, BATCH8),
        ("batch8 mha16 ragged", 16, 16, RAGGED8),
    ];
    for (name, qo, kv, reqs) in cases {
        let c = Case {
            head_dim: 64,
            num_qo_heads: qo,
            num_kv_heads: kv,
            page_size: PS,
            reqs,
        };
        let mut b = Bench::default();
        let (eo, el) = run_case::<T>(&ctx, &stream, &c, 0xF0E1_D2C3, Some(&mut b));
        let graph = match b.graph_ms_per_launch {
            Some(ms) => format!(
                "{ms:.4} ms/launch (8 captured), vs eager {:.2e}",
                b.graph_vs_eager
            ),
            None => format!("FAILED: {}", b.graph_note),
        };
        eprintln!(
            "FUEL {device} {label} {name}: max|o err|={eo:.3e} rel={:.3e} (max|ref|={:.3}) lse={el:.3e} eager={:.4} ms/launch graph={graph}",
            b.max_rel, b.max_ref, b.eager_ms
        );
        assert!(
            eo < o_bound,
            "{label} {name}: o error {eo:e} >= {o_bound:e}"
        );
        assert_eq!(
            b.graph_note, "ok",
            "{label} {name}: CUDA graph capture/replay"
        );
        assert_eq!(
            b.graph_vs_eager, 0.0,
            "{label} {name}: graph replay differs from eager"
        );
    }
}

#[test]
fn paged_decode_fuel_fixtures_f16() {
    fuel_fixtures::<f16>("f16", 1e-3);
}

#[test]
fn paged_decode_fuel_fixtures_bf16() {
    fuel_fixtures::<bf16>("bf16", 6e-3);
}
