// SPDX-License-Identifier: MIT OR Apache-2.0
//! Real-GPU test: `HistogramPlan` ignores NaN (and out-of-range values, including +-inf).
//!
//! torch.histc: "elements lower than min and higher than max and NaN elements are ignored". Before ROADMAP E0j a NaN passed the range test
//! (`v < lo || v > hi` is false for NaN) and was counted in bin 0.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, HistogramArgs, HistogramDescriptor, HistogramPlan, PlanPreference, TensorMut,
    TensorRef, Workspace, contiguous_stride,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

macro_rules! test {
    ($name:ident, $T:ty, $elem:expr) => {
        #[test]
        fn $name() {
            let (ctx, stream) = setup();
            let nan = <$T>::NAN;
            // [0, 1] in 4 bins: 0.1 -> bin 0, 0.5 -> bin 2, 1.0 (== hi) -> last bin; NaN, +-inf and out-of-range are ignored.
            let input: Vec<$T> = vec![
                nan,
                0.1,
                0.5,
                <$T>::INFINITY,
                <$T>::NEG_INFINITY,
                nan,
                1.0,
                -0.5,
                2.0,
                nan,
            ];
            let n = input.len() as i64;
            let dev_in = DeviceBuffer::from_slice(&ctx, &input).unwrap();
            let mut dev_out: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 4).unwrap();
            let desc = HistogramDescriptor {
                numel: n,
                num_bins: 4,
                lo: 0.0,
                hi: 1.0,
                element: $elem,
            };
            let plan =
                HistogramPlan::<$T>::select(&stream, &desc, PlanPreference::default()).unwrap();
            plan.run(
                &stream,
                Workspace::None,
                HistogramArgs::<$T> {
                    input: TensorRef::new(
                        dev_in.as_slice(),
                        [n as i32],
                        contiguous_stride([n as i32]),
                    )
                    .unwrap(),
                    output: TensorMut::new(dev_out.as_slice_mut(), [4], contiguous_stride([4]))
                        .unwrap(),
                },
            )
            .unwrap();
            stream.synchronize().unwrap();
            let mut got = vec![0i32; 4];
            dev_out.copy_to_host(&mut got).unwrap();
            assert_eq!(
                got,
                vec![1, 0, 1, 1],
                "NaN must not be counted (it landed in bin 0 before)"
            );
        }
    };
}

test!(histogram_ignores_nan_f32, f32, ElementKind::F32);
test!(histogram_ignores_nan_f64, f64, ElementKind::F64);
