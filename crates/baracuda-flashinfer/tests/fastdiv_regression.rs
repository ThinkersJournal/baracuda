//! Regression tests for the vendored FlashInfer `fastdiv.cuh`, through the plans that use it.
//!
//! The 2026-05 vendor patch (a pure-C++ replacement for `cuda::fast_mod_div`) gave quotient 0 for every power-of-two
//! divisor and a wrong quotient for almost every other divisor (4999 of the divisors 1..5000). `uint_fastdiv` is used by
//! `page.cuh` (`paged_kv_t::page_size`, so paged decode and paged prefill) and by `default_prefill_params.cuh`
//! (`group_size`, so paged AND ragged prefill whenever the GQA group size is not 1). The cascade merge kernels
//! (`cascade.cuh`) do not include it: they are checked for that in the PR, not here.
//!
//! Each test compares a plan with an f64 reference computed on the host from the same inputs, with a power-of-two and a
//! non-power-of-two `page_size` (16 and 12) and a power-of-two and a non-power-of-two GQA group size (4 and 3), over
//! requests that span several pages. With the old header these fail with errors of order 0.1-1 (reading page 0 for every
//! token, wrong head mapping); with the fixed header they agree to the element type's rounding.
//!
//! Gated behind `device-tests` (a missing adapter FAILS).

#![cfg(feature = "flashinfer")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_flashinfer::attention::{
    BatchPagedPrefillArgs, BatchPagedPrefillDescriptor, BatchPagedPrefillPlan,
    BatchRaggedPrefillArgs, BatchRaggedPrefillDescriptor, BatchRaggedPrefillPlan,
    PagedKvCacheDescriptor,
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

const HD: usize = 64;

struct Case {
    page_size: usize,
    qo_heads: usize,
    kv_heads: usize,
    causal: bool,
    /// Per request: (query rows, KV tokens); `kv >= qo` so a causal row always sees a token.
    reqs: &'static [(usize, usize)],
}

static REQS: &[(usize, usize)] = &[(7, 40), (4, 29), (5, 5)];

/// f64 attention over the host copies: `k(b, t, g, d)` / `v(b, t, g, d)` fetch token `t` of request `b`, KV head `g`.
/// Query row `i` of a request with `kv` tokens and `qo` rows sees tokens `0..kv` (non-causal) or `0..=i + kv - qo`.
fn reference<T: Host>(
    c: &Case,
    q: &[T],
    k: &dyn Fn(usize, usize, usize, usize) -> f64,
    v: &dyn Fn(usize, usize, usize, usize) -> f64,
) -> Vec<f64> {
    let group = c.qo_heads / c.kv_heads;
    let scale = 1.0 / (HD as f64).sqrt();
    let rows: usize = c.reqs.iter().map(|r| r.0).sum();
    let mut out = vec![0f64; rows * c.qo_heads * HD];
    let mut row0 = 0;
    for (b, &(qo, kv)) in c.reqs.iter().enumerate() {
        for i in 0..qo {
            let visible = if c.causal { i + kv - qo + 1 } else { kv };
            for h in 0..c.qo_heads {
                let g = h / group;
                let qv = &q[((row0 + i) * c.qo_heads + h) * HD..][..HD];
                let scores: Vec<f64> = (0..visible)
                    .map(|t| {
                        (0..HD)
                            .map(|d| qv[d].to_f32() as f64 * k(b, t, g, d))
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let den: f64 = scores.iter().map(|s| (s - m).exp()).sum();
                for d in 0..HD {
                    out[((row0 + i) * c.qo_heads + h) * HD + d] = (0..visible)
                        .map(|t| (scores[t] - m).exp() / den * v(b, t, g, d))
                        .sum();
                }
            }
        }
        row0 += qo;
    }
    out
}

fn max_err<T: Host>(o: &[T], want: &[f64]) -> f64 {
    o.iter()
        .zip(want)
        .map(|(g, w)| (g.to_f32() as f64 - w).abs())
        .fold(0.0, f64::max)
}

fn rand_vec<T: Host>(rng: &mut Rng, n: usize) -> Vec<T> {
    (0..n).map(|_| T::from_f32(rng.next())).collect()
}

/// Paged prefill over several pages per request, shuffled physical pages. Max abs error of `o`.
fn paged_prefill_err<T: Host>(ctx: &Context, stream: &Stream, c: &Case, seed: u64) -> f64 {
    let ps = c.page_size;
    let mut rng = Rng(seed);
    let pages_of = |kv: usize| kv.div_ceil(ps);
    let used: usize = c.reqs.iter().map(|r| pages_of(r.1)).sum();
    let total_pages = used + 3;
    let mut ids: Vec<i32> = (0..total_pages as i32).collect();
    for i in (1..ids.len()).rev() {
        let j = (rng.next().abs() * 1e6) as usize % (i + 1);
        ids.swap(i, j);
    }
    let (mut indices, mut indptr, mut last_page_len, mut q_indptr) =
        (Vec::new(), vec![0i32], Vec::new(), vec![0i32]);
    let mut next = 0;
    for &(qo, kv) in c.reqs {
        let np = pages_of(kv);
        indices.extend_from_slice(&ids[next..next + np]);
        next += np;
        indptr.push(indices.len() as i32);
        last_page_len.push((kv - (np - 1) * ps) as i32);
        q_indptr.push(q_indptr.last().unwrap() + qo as i32);
    }
    let page_elems = c.kv_heads * ps * HD;
    let k_store: Vec<T> = rand_vec(&mut rng, total_pages * page_elems);
    let v_store: Vec<T> = rand_vec(&mut rng, total_pages * page_elems);
    let rows = *q_indptr.last().unwrap() as usize;
    let q_h: Vec<T> = rand_vec(&mut rng, rows * c.qo_heads * HD);

    let paged = PagedKvCacheDescriptor {
        page_size: ps as i32,
        num_total_pages: total_pages as i32,
        num_kv_heads: c.kv_heads as i32,
        head_dim: HD as i32,
        element: T::KIND,
    };
    let desc = BatchPagedPrefillDescriptor::new(
        c.reqs.len() as i32,
        rows as i32,
        c.qo_heads as i32,
        1.0 / (HD as f32).sqrt(),
        paged,
    )
    .with_causal(c.causal);
    let plan = BatchPagedPrefillPlan::<T>::select(stream, &desc, PlanPreference::default())
        .expect("select");

    let q_dev = DeviceBuffer::from_slice(ctx, &q_h).unwrap();
    let qp_dev = DeviceBuffer::from_slice(ctx, &q_indptr).unwrap();
    let k_dev = DeviceBuffer::from_slice(ctx, &k_store).unwrap();
    let v_dev = DeviceBuffer::from_slice(ctx, &v_store).unwrap();
    let idx_dev = DeviceBuffer::from_slice(ctx, &indices).unwrap();
    let ptr_dev = DeviceBuffer::from_slice(ctx, &indptr).unwrap();
    let lpl_dev = DeviceBuffer::from_slice(ctx, &last_page_len).unwrap();
    let mut o_dev: DeviceBuffer<T> = DeviceBuffer::zeros(ctx, rows * c.qo_heads * HD).unwrap();
    let mut lse_dev: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, rows * c.qo_heads).unwrap();
    let qo_shape = [rows as i32, c.qo_heads as i32, HD as i32];
    let cache_shape = [total_pages as i32, c.kv_heads as i32, ps as i32, HD as i32];
    let lse_shape = [rows as i32, c.qo_heads as i32];
    let b = c.reqs.len() as i32;
    plan.run(
        stream,
        Workspace::None,
        BatchPagedPrefillArgs {
            q: TensorRef::new(q_dev.as_slice(), qo_shape, contiguous_stride(qo_shape)).unwrap(),
            q_indptr: TensorRef::new(qp_dev.as_slice(), [b + 1], [1]).unwrap(),
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
            kv_indices: TensorRef::new(idx_dev.as_slice(), [indices.len() as i32], [1]).unwrap(),
            kv_indptr: TensorRef::new(ptr_dev.as_slice(), [b + 1], [1]).unwrap(),
            last_page_len: TensorRef::new(lpl_dev.as_slice(), [b], [1]).unwrap(),
            o: TensorMut::new(o_dev.as_slice_mut(), qo_shape, contiguous_stride(qo_shape)).unwrap(),
            lse: TensorMut::new(
                lse_dev.as_slice_mut(),
                lse_shape,
                contiguous_stride(lse_shape),
            )
            .unwrap(),
        },
    )
    .expect("paged prefill run");
    stream.synchronize().expect("sync");
    let mut o_host = vec![T::zero(); rows * c.qo_heads * HD];
    o_dev.copy_to_host(&mut o_host).unwrap();

    let at = |store: &[T], b: usize, t: usize, g: usize, d: usize| -> f64 {
        let page = indices[indptr[b] as usize + t / ps] as usize;
        store[page * page_elems + (g * ps + t % ps) * HD + d].to_f32() as f64
    };
    let want = reference(
        c,
        &q_h,
        &|b, t, g, d| at(&k_store, b, t, g, d),
        &|b, t, g, d| at(&v_store, b, t, g, d),
    );
    max_err(&o_host, &want)
}

/// Ragged (non-paged) prefill: only `group_size` is a `uint_fastdiv` here. Max abs error of `o`.
fn ragged_prefill_err<T: Host>(ctx: &Context, stream: &Stream, c: &Case, seed: u64) -> f64 {
    let mut rng = Rng(seed);
    let (mut q_indptr, mut kv_indptr) = (vec![0i32], vec![0i32]);
    for &(qo, kv) in c.reqs {
        q_indptr.push(q_indptr.last().unwrap() + qo as i32);
        kv_indptr.push(kv_indptr.last().unwrap() + kv as i32);
    }
    let rows = *q_indptr.last().unwrap() as usize;
    let kv_rows = *kv_indptr.last().unwrap() as usize;
    let q_h: Vec<T> = rand_vec(&mut rng, rows * c.qo_heads * HD);
    let k_h: Vec<T> = rand_vec(&mut rng, kv_rows * c.kv_heads * HD);
    let v_h: Vec<T> = rand_vec(&mut rng, kv_rows * c.kv_heads * HD);
    let desc = BatchRaggedPrefillDescriptor {
        batch_size: c.reqs.len() as i32,
        total_num_rows: rows as i32,
        total_kv_rows: kv_rows as i32,
        num_qo_heads: c.qo_heads as i32,
        num_kv_heads: c.kv_heads as i32,
        head_dim: HD as i32,
        sm_scale: 1.0 / (HD as f32).sqrt(),
        causal: c.causal,
        enable_kv_split: false,
        element: T::KIND,
    };
    let plan = BatchRaggedPrefillPlan::<T>::select(stream, &desc, PlanPreference::default())
        .expect("select");
    let q_dev = DeviceBuffer::from_slice(ctx, &q_h).unwrap();
    let qp_dev = DeviceBuffer::from_slice(ctx, &q_indptr).unwrap();
    let k_dev = DeviceBuffer::from_slice(ctx, &k_h).unwrap();
    let v_dev = DeviceBuffer::from_slice(ctx, &v_h).unwrap();
    let kp_dev = DeviceBuffer::from_slice(ctx, &kv_indptr).unwrap();
    let mut o_dev: DeviceBuffer<T> = DeviceBuffer::zeros(ctx, rows * c.qo_heads * HD).unwrap();
    let mut lse_dev: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, rows * c.qo_heads).unwrap();
    let qo_shape = [rows as i32, c.qo_heads as i32, HD as i32];
    let kv_shape = [kv_rows as i32, c.kv_heads as i32, HD as i32];
    let lse_shape = [rows as i32, c.qo_heads as i32];
    let b = c.reqs.len() as i32;
    plan.run(
        stream,
        Workspace::None,
        BatchRaggedPrefillArgs {
            q: TensorRef::new(q_dev.as_slice(), qo_shape, contiguous_stride(qo_shape)).unwrap(),
            q_indptr: TensorRef::new(qp_dev.as_slice(), [b + 1], [1]).unwrap(),
            k_data: TensorRef::new(k_dev.as_slice(), kv_shape, contiguous_stride(kv_shape))
                .unwrap(),
            v_data: TensorRef::new(v_dev.as_slice(), kv_shape, contiguous_stride(kv_shape))
                .unwrap(),
            kv_indptr: TensorRef::new(kp_dev.as_slice(), [b + 1], [1]).unwrap(),
            o: TensorMut::new(o_dev.as_slice_mut(), qo_shape, contiguous_stride(qo_shape)).unwrap(),
            lse: TensorMut::new(
                lse_dev.as_slice_mut(),
                lse_shape,
                contiguous_stride(lse_shape),
            )
            .unwrap(),
        },
    )
    .expect("ragged prefill run");
    stream.synchronize().expect("sync");
    let mut o_host = vec![T::zero(); rows * c.qo_heads * HD];
    o_dev.copy_to_host(&mut o_host).unwrap();
    let at = |store: &[T], b: usize, t: usize, g: usize, d: usize| -> f64 {
        store[((kv_indptr[b] as usize + t) * c.kv_heads + g) * HD + d].to_f32() as f64
    };
    let want = reference(
        c,
        &q_h,
        &|b, t, g, d| at(&k_h, b, t, g, d),
        &|b, t, g, d| at(&v_h, b, t, g, d),
    );
    max_err(&o_host, &want)
}

/// (label, page_size, qo_heads, kv_heads): page size 16 (power of two) and 12 (not); GQA group 4 (power of two) and 3.
const SHAPES: [(&str, usize, usize, usize); 4] = [
    ("ps16 group4", 16, 8, 2),
    ("ps12 group4", 12, 8, 2),
    ("ps16 group3", 16, 6, 2),
    ("ps12 group3", 12, 6, 2),
];

fn sweep<T: Host>(
    kind: &str,
    label: &str,
    bound: f64,
    run: fn(&Context, &Stream, &Case, u64) -> f64,
) {
    let (ctx, stream) = setup();
    let mut worst = 0f64;
    for (name, ps, qo, kv) in SHAPES {
        for causal in [false, true] {
            let c = Case {
                page_size: ps,
                qo_heads: qo,
                kv_heads: kv,
                causal,
                reqs: REQS,
            };
            let e = run(&ctx, &stream, &c, 0xABCD_1234 ^ (ps * 31 + qo) as u64);
            eprintln!("fastdiv {kind} {label} {name} causal={causal}: max|o err|={e:.3e}");
            worst = worst.max(e);
        }
    }
    eprintln!("fastdiv {kind} {label} WORST max|o err|={worst:.3e} (bound {bound:.0e})");
    assert!(worst < bound, "{kind} {label}: {worst:e} >= {bound:e}");
}

#[test]
fn paged_prefill_multi_page_matches_reference_f16() {
    sweep::<f16>("paged-prefill", "f16", 2e-3, paged_prefill_err::<f16>);
}

#[test]
fn paged_prefill_multi_page_matches_reference_bf16() {
    sweep::<bf16>("paged-prefill", "bf16", 1e-2, paged_prefill_err::<bf16>);
}

#[test]
fn ragged_prefill_gqa_matches_reference_f16() {
    sweep::<f16>("ragged-prefill", "f16", 2e-3, ragged_prefill_err::<f16>);
}

#[test]
fn ragged_prefill_gqa_matches_reference_bf16() {
    sweep::<bf16>("ragged-prefill", "bf16", 1e-2, ragged_prefill_err::<bf16>);
}

/// Paged KV append (decode, one token per request). The kernel (`AppendPagedKVCacheDecodeKernel`, `page.cuh`) computes
/// `seq_len = (pages - 1) * page_size + last_page_len`, then the page `indptr[b] + (seq_len - 1) / page_size` and slot
/// `(seq_len - 1) % page_size`, both through `uint_fastdiv`. So `last_page_len` here COUNTS the new token, which is what
/// this test uses (the plan's doc says to increment it after the call, i.e. a slot one higher; see the PR). The whole
/// cache is compared with the expected contents: the new rows at the expected slots, everything else untouched.
fn append_mismatches(
    ctx: &Context,
    stream: &Stream,
    ps: usize,
    heads: usize,
    reqs: &[(usize, usize)],
) -> usize {
    use baracuda_flashinfer::attention::{
        PagedKvAppendArgs, PagedKvAppendDescriptor, PagedKvAppendPlan,
    };
    let hd = 64;
    let used: usize = reqs.iter().map(|r| r.0).sum();
    let total_pages = used + 2;
    let mut rng = Rng(0x5151_7A7A ^ ps as u64);
    let mut ids: Vec<i32> = (0..total_pages as i32).collect();
    for i in (1..ids.len()).rev() {
        let j = (rng.next().abs() * 1e6) as usize % (i + 1);
        ids.swap(i, j);
    }
    let (mut indices, mut indptr, mut lpl) = (Vec::new(), vec![0i32], Vec::new());
    let mut next = 0;
    for &(np, l) in reqs {
        indices.extend_from_slice(&ids[next..next + np]);
        next += np;
        indptr.push(indices.len() as i32);
        lpl.push(l as i32);
    }
    let cache_len = total_pages * heads * ps * hd;
    let batch = reqs.len();
    let key: Vec<f32> = (0..batch * heads * hd)
        .map(|i| 1.0 + i as f32 * 0.01)
        .collect();
    let value: Vec<f32> = key.iter().map(|x| -x).collect();
    let mut k_want = vec![5.0f32; cache_len];
    let mut v_want = vec![5.0f32; cache_len];
    for (b, &(np, l)) in reqs.iter().enumerate() {
        let seq_len = (np - 1) * ps + l;
        let page = indices[indptr[b] as usize + (seq_len - 1) / ps] as usize;
        let slot = (seq_len - 1) % ps;
        for h in 0..heads {
            for d in 0..hd {
                let at = ((page * heads + h) * ps + slot) * hd + d;
                k_want[at] = key[(b * heads + h) * hd + d];
                v_want[at] = value[(b * heads + h) * hd + d];
            }
        }
    }
    let paged = PagedKvCacheDescriptor {
        page_size: ps as i32,
        num_total_pages: total_pages as i32,
        num_kv_heads: heads as i32,
        head_dim: hd as i32,
        element: baracuda_kernels::ElementKind::F32,
    };
    let plan = PagedKvAppendPlan::<f32>::select(
        stream,
        &PagedKvAppendDescriptor {
            batch_size: batch as i32,
            paged_kv: paged,
        },
        PlanPreference::default(),
    )
    .expect("select");
    let key_dev = DeviceBuffer::from_slice(ctx, &key).unwrap();
    let val_dev = DeviceBuffer::from_slice(ctx, &value).unwrap();
    let mut k_dev = DeviceBuffer::from_slice(ctx, &vec![5.0f32; cache_len]).unwrap();
    let mut v_dev = DeviceBuffer::from_slice(ctx, &vec![5.0f32; cache_len]).unwrap();
    let idx_dev = DeviceBuffer::from_slice(ctx, &indices).unwrap();
    let ptr_dev = DeviceBuffer::from_slice(ctx, &indptr).unwrap();
    let lpl_dev = DeviceBuffer::from_slice(ctx, &lpl).unwrap();
    let row_shape = [batch as i32, heads as i32, hd as i32];
    let cache_shape = [total_pages as i32, heads as i32, ps as i32, hd as i32];
    plan.run(
        stream,
        Workspace::None,
        PagedKvAppendArgs {
            key: TensorRef::new(key_dev.as_slice(), row_shape, contiguous_stride(row_shape))
                .unwrap(),
            value: TensorRef::new(val_dev.as_slice(), row_shape, contiguous_stride(row_shape))
                .unwrap(),
            k_data: TensorMut::new(
                k_dev.as_slice_mut(),
                cache_shape,
                contiguous_stride(cache_shape),
            )
            .unwrap(),
            v_data: TensorMut::new(
                v_dev.as_slice_mut(),
                cache_shape,
                contiguous_stride(cache_shape),
            )
            .unwrap(),
            indices: TensorRef::new(idx_dev.as_slice(), [indices.len() as i32], [1]).unwrap(),
            indptr: TensorRef::new(ptr_dev.as_slice(), [batch as i32 + 1], [1]).unwrap(),
            last_page_len: TensorRef::new(lpl_dev.as_slice(), [batch as i32], [1]).unwrap(),
        },
    )
    .expect("append run");
    stream.synchronize().expect("sync");
    let (mut k_got, mut v_got) = (vec![0f32; cache_len], vec![0f32; cache_len]);
    k_dev.copy_to_host(&mut k_got).unwrap();
    v_dev.copy_to_host(&mut v_got).unwrap();
    k_got.iter().zip(&k_want).filter(|(a, b)| a != b).count()
        + v_got.iter().zip(&v_want).filter(|(a, b)| a != b).count()
}

#[test]
fn paged_kv_append_multi_page_lands_where_the_kernel_formula_says() {
    let (ctx, stream) = setup();
    let mut total = 0;
    for (ps, heads) in [(16usize, 4usize), (12, 4), (16, 3), (12, 3)] {
        // (pages, last_page_len counting the new token): first slot, mid, last slot, single page.
        let reqs = [(3, 1), (2, 5), (4, ps), (1, ps), (2, ps - 1)];
        let bad = append_mismatches(&ctx, &stream, ps, heads, &reqs);
        eprintln!("fastdiv append ps={ps} heads={heads}: mismatching cache elements = {bad}");
        total += bad;
    }
    assert_eq!(total, 0, "paged KV append wrote to unexpected cache slots");
}
