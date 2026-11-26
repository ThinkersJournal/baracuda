// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `SearchsortedPlan` uses the NaN-largest total order of the sort plans (f32, f64; the plan has no f16 / bf16).
//!
//! A NaN query: `left` = the index of the first NaN of the sequence (the count of non-NaN elements; the length when there is none), `right` =
//! the length. Identical to torch.searchsorted whenever the sequence has no NaN; for a NaN query against a sequence that CONTAINS a NaN torch
//! returns the length for `left` too (torch's behaviour with a NaN in the sequence is itself undefined), this plan returns the first NaN index so
//! that `searchsorted(sort(x), NaN)` lands on the NaN block. Before ROADMAP E0m a NaN query returned 0 for both sides.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, SearchsortedArgs, SearchsortedDescriptor, SearchsortedPlan,
    TensorMut, TensorRef, Workspace, contiguous_stride,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// NaN-largest order.
fn less(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

fn reference(seq: &[f64], q: f64, right: bool) -> i32 {
    seq.iter()
        .filter(|&&x| if right { !less(q, x) } else { less(x, q) })
        .count() as i32
}

macro_rules! tests {
    ($vec:ident, $prop:ident, $T:ty, $elem:expr) => {
        fn run(ctx: &Context, s: &Stream, seq: &[f64], q: &[f64], right: bool) -> Vec<i32> {
            let seq: Vec<$T> = seq.iter().map(|v| *v as $T).collect();
            let q: Vec<$T> = q.iter().map(|v| *v as $T).collect();
            let dseq = DeviceBuffer::from_slice(ctx, &seq).unwrap();
            let dq = DeviceBuffer::from_slice(ctx, &q).unwrap();
            let mut dout: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, q.len()).unwrap();
            let (ns, nq) = (seq.len() as i32, q.len() as i32);
            let desc = SearchsortedDescriptor {
                num_queries: q.len() as i64,
                len_sorted: ns,
                right,
                element: $elem,
            };
            let plan = SearchsortedPlan::<$T>::select(s, &desc, PlanPreference::default()).unwrap();
            plan.run(
                s,
                Workspace::None,
                SearchsortedArgs::<$T> {
                    sorted_seq: TensorRef::new(dseq.as_slice(), [ns], contiguous_stride([ns]))
                        .unwrap(),
                    values: TensorRef::new(dq.as_slice(), [nq], contiguous_stride([nq])).unwrap(),
                    output: TensorMut::new(dout.as_slice_mut(), [nq], contiguous_stride([nq]))
                        .unwrap(),
                },
            )
            .unwrap();
            s.synchronize().unwrap();
            let mut g = vec![0i32; q.len()];
            dout.copy_to_host(&mut g).unwrap();
            g
        }

        #[test]
        fn $vec() {
            let (ctx, s) = setup();
            let n = f64::NAN;
            // The four probe vectors: [seq, queries, left, right].
            let cases: [(&[f64], &[f64], &[i32], &[i32]); 3] = [
                (
                    &[-f64::INFINITY, 1.0, f64::INFINITY],
                    &[-f64::INFINITY, f64::INFINITY, n, 1.0],
                    &[0, 2, 3, 1],
                    &[1, 3, 3, 2],
                ),
                (
                    &[1.0, 2.0, 2.0, 3.0, n, n],
                    &[n, 0.0, 2.0, 3.0, 5.0, f64::INFINITY, -f64::INFINITY],
                    &[4, 0, 1, 3, 4, 4, 0],
                    &[6, 0, 3, 4, 4, 4, 0],
                ),
                (&[1.0, 2.0, 3.0], &[n, n], &[3, 3], &[3, 3]),
            ];
            for (seq, q, left, right) in cases {
                assert_eq!(
                    run(&ctx, &s, seq, q, false),
                    left,
                    "left over {seq:?} for {q:?}"
                );
                assert_eq!(
                    run(&ctx, &s, seq, q, true),
                    right,
                    "right over {seq:?} for {q:?}"
                );
            }
        }

        #[test]
        fn $prop() {
            let (ctx, s) = setup();
            let mut g = Lcg(0x5ea);
            for round in 0..40 {
                let len = (g.next() % 40) as usize + 1;
                let mut seq: Vec<f64> = (0..len)
                    .map(|_| match g.next() % 12 {
                        0 if round % 2 == 1 => f64::NAN,
                        1 => f64::INFINITY,
                        2 => f64::NEG_INFINITY,
                        k => (k as f64) - 6.0,
                    })
                    .collect();
                // The sequence the plan expects: sorted in the NaN-largest order.
                seq.sort_by(|a, b| {
                    if less(*a, *b) {
                        std::cmp::Ordering::Less
                    } else if less(*b, *a) {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Equal
                    }
                });
                let q: Vec<f64> = (0..50)
                    .map(|_| match g.next() % 9 {
                        0 => f64::NAN,
                        1 => f64::INFINITY,
                        2 => f64::NEG_INFINITY,
                        k => (k as f64) - 5.0 + 0.5 * ((g.next() % 2) as f64),
                    })
                    .collect();
                for right in [false, true] {
                    let got = run(&ctx, &s, &seq, &q, right);
                    for (i, qv) in q.iter().enumerate() {
                        let want = reference(&seq, *qv as $T as f64, right);
                        assert_eq!(got[i], want, "right={right} q={qv} seq={seq:?}");
                    }
                }
            }
        }
    };
}

mod f32_ {
    use super::*;
    tests!(vectors_f32, property_f32, f32, ElementKind::F32);
}
mod f64_ {
    use super::*;
    tests!(vectors_f64, property_f64, f64, ElementKind::F64);
}
