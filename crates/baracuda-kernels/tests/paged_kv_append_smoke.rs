//! Direct-FFI smoke test for FlashInfer `paged_kv_append_decode`
//! (Phase 46/66, vLLM-style serving).
//!
//! Validates the lifecycle:
//!  - `can_implement` accepts a normal config; rejects head_dim=0.
//!  - `run` appends a single decode-step token to a pre-built paged KV-
//!    cache structure for f16 / bf16 / f32 dtypes.
//!  - Output K/V pages contain the expected key/value vectors at the
//!    correct paged position after the call, and no other cache element changes.
//!
//! Layout (kHND, as the launcher builds it):
//!  - `k_data` / `v_data` — `[num_pages, num_heads, page_size, head_dim]`
//!  - `indices` — `[total_pages_across_batch]`
//!  - `indptr` — `[batch_size + 1]` (page-offset into `indices`)
//!  - `last_page_len` — `[batch_size]`, in `[1, page_size]`, COUNTING the token being appended
//!  - `key` / `value` — `[batch_size, num_heads, head_dim]` (one new
//!    decode-step token per request)
//!
//! Contract (upstream FlashInfer's, see `paged_kv_append.rs`): the page table already includes the new token, so the
//! token lands in slot `last_page_len[b] - 1` of request `b`'s last page. These tests used to expect slot
//! `last_page_len[b]` (the old doc's "increment after the call"): the f32 test failed on a device with `got=0
//! want=0.0699` (the kernel wrote slot 2, not 3), and the f16/bf16 tests used `head_dim` 32, which the launcher
//! rejects with status 3 (supported: 64, 128, 256), so they had never run to completion.
//!
//! Gated behind the `device-tests` feature; run with
//! `cargo gpu-test -p baracuda-kernels --features flashinfer,device-tests --test paged_kv_append_smoke`.

#![cfg(all(
    any(feature = "sm80", feature = "sm89", feature = "sm90a"),
    feature = "flashinfer"
))]
#![cfg(feature = "device-tests")]

use core::ffi::c_void;

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
fn paged_kv_append_can_implement_rejects_bad_head_dim() {
    let ok = unsafe {
        baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_can_implement(
            2, 16, 8, 128,
        )
    };
    assert_eq!(
        ok, 0,
        "normal config (B=2 page=16 H=8 D=128) should succeed"
    );

    let bad = unsafe {
        baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_can_implement(
            2, 16, 8, 0,
        )
    };
    assert_ne!(bad, 0, "head_dim=0 must be rejected");
}

/// Requests as `(pages, last_page_len counting the new token)`. For page size `ps`: "3 tokens already in the page, one
/// new" (slot 3), the first slot of a fresh last page (`last_page_len` 1), the last slot (`ps`), and single-page requests.
fn requests(ps: usize) -> Vec<(usize, usize)> {
    vec![(2, 4), (3, 1), (2, ps), (1, 1), (1, ps), (4, ps - 1)]
}

/// One generated test per element type. Builds a shuffled page table for [`requests`], fills K and V with a sentinel,
/// runs the launcher, and compares the WHOLE cache with the expected one: the new key (value) at head-major slot
/// `last_page_len - 1` of each request's last page, the sentinel everywhere else.
macro_rules! append_test {
    ($name:ident, $run:path, $t:ty, $from:expr, $to:expr) => {
        #[test]
        fn $name() {
            let (ctx, stream) = setup();
            let from = $from;
            let to = $to;
            let (num_heads, head_dim) = (4usize, 64usize);
            for ps in [16usize, 12] {
                let reqs = requests(ps);
                let batch = reqs.len();
                let used: usize = reqs.iter().map(|r| r.0).sum();
                let num_pages = used + 2;
                // Physical page ids in a fixed shuffle (a plain 0..n would hide a page-table bug).
                let mut ids: Vec<i32> = (0..num_pages as i32).collect();
                for i in (1..ids.len()).rev() {
                    ids.swap(i, (i * 7 + 3) % (i + 1));
                }
                let (mut indices, mut indptr, mut last_page_len) =
                    (Vec::new(), vec![0i32], Vec::new());
                let mut next = 0;
                for &(np, l) in &reqs {
                    indices.extend_from_slice(&ids[next..next + np]);
                    next += np;
                    indptr.push(indices.len() as i32);
                    last_page_len.push(l as i32);
                }
                let cache_len = num_pages * num_heads * ps * head_dim;
                let key_f32: Vec<f32> = (0..batch * num_heads * head_dim)
                    .map(|i| 1.0 + (i % 997) as f32 * 0.01)
                    .collect();
                let key: Vec<$t> = key_f32.iter().map(|&x| from(x)).collect();
                let value: Vec<$t> = key_f32.iter().map(|&x| from(-x)).collect();
                let sentinel = from(5.0);
                let mut k_want = vec![sentinel; cache_len];
                let mut v_want = vec![sentinel; cache_len];
                for (b, &(np, l)) in reqs.iter().enumerate() {
                    // Contract: the last page of request b, slot last_page_len - 1.
                    let page = indices[indptr[b] as usize + np - 1] as usize;
                    let slot = l - 1;
                    for h in 0..num_heads {
                        for d in 0..head_dim {
                            let at = ((page * num_heads + h) * ps + slot) * head_dim + d;
                            k_want[at] = key[(b * num_heads + h) * head_dim + d];
                            v_want[at] = value[(b * num_heads + h) * head_dim + d];
                        }
                    }
                }

                let mut dev_k = DeviceBuffer::from_slice(&ctx, &vec![sentinel; cache_len]).unwrap();
                let mut dev_v = DeviceBuffer::from_slice(&ctx, &vec![sentinel; cache_len]).unwrap();
                let mut dev_indices = DeviceBuffer::from_slice(&ctx, &indices).unwrap();
                let mut dev_indptr = DeviceBuffer::from_slice(&ctx, &indptr).unwrap();
                let mut dev_lpl = DeviceBuffer::from_slice(&ctx, &last_page_len).unwrap();
                let dev_key = DeviceBuffer::from_slice(&ctx, &key).unwrap();
                let dev_value = DeviceBuffer::from_slice(&ctx, &value).unwrap();
                let status = unsafe {
                    $run(
                        batch as i32,
                        ps as i32,
                        num_heads as i32,
                        head_dim as i32,
                        dev_k.as_slice_mut().as_raw().0 as *mut c_void,
                        dev_v.as_slice_mut().as_raw().0 as *mut c_void,
                        dev_indices.as_slice_mut().as_raw().0 as *mut c_void,
                        dev_indptr.as_slice_mut().as_raw().0 as *mut c_void,
                        dev_lpl.as_slice_mut().as_raw().0 as *mut c_void,
                        dev_key.as_slice().as_raw().0 as *const c_void,
                        dev_value.as_slice().as_raw().0 as *const c_void,
                        stream.as_raw(),
                    )
                };
                assert_eq!(status, 0, "paged_kv_append status (page_size {ps})");
                stream.synchronize().expect("sync");

                let mut k_got = vec![from(0.0); cache_len];
                let mut v_got = vec![from(0.0); cache_len];
                dev_k.copy_to_host(&mut k_got).unwrap();
                dev_v.copy_to_host(&mut v_got).unwrap();
                let bad = |got: &[$t], want: &[$t]| {
                    got.iter()
                        .zip(want)
                        .filter(|(a, b)| to(**a) != to(**b))
                        .count()
                };
                assert_eq!(
                    (bad(&k_got, &k_want), bad(&v_got, &v_want)),
                    (0, 0),
                    "page_size {ps}: K/V cache differs from the expected contents \
                     (new token at slot last_page_len - 1 of the last page, nothing else touched)"
                );
            }
        }
    };
}

append_test!(
    paged_kv_append_f32_places_the_token_at_last_page_len_minus_1,
    baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_f32_run,
    f32,
    |x: f32| x,
    |x: f32| x
);
append_test!(
    paged_kv_append_f16_places_the_token_at_last_page_len_minus_1,
    baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_f16_run,
    f16,
    f16::from_f32,
    f16::to_f32
);
append_test!(
    paged_kv_append_bf16_places_the_token_at_last_page_len_minus_1,
    baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_bf16_run,
    bf16,
    bf16::from_f32,
    bf16::to_f32
);
