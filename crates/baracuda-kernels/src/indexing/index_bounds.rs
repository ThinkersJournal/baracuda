// SPDX-License-Identifier: MIT OR Apache-2.0
//! `IndexBoundsPlan`: the device-side bounds check that lets a caller raise on an out-of-range
//! index, as torch does (ROADMAP E4).
//!
//! The indexing plans (`IndexSelectPlan`, `GatherPlan`, `ScatterPlan`, `ScatterAddPlan`,
//! `IndexAddPlan` and their backwards) skip an out-of-range or negative index and leave the output
//! cell untouched: a kernel cannot raise. This plan counts such values in an index buffer into a
//! device `u32` word without touching `src` or `out`:
//!
//! 1. `plan.run(...)` on the same stream, before or after the indexing launch (it reads only the
//!    index buffer). No host sync is added, so a captured graph replays it.
//! 2. `plan.check(...)` at the next synchronisation point: it synchronises the stream, reads the
//!    word and returns [`Error::IndexOutOfRange`] when it is non-zero.
//!
//! The index buffer is read as a flat, contiguous array of `numel` values (the shape the indexing
//! plans take; a strided index view must be made contiguous first). `bound` is the extent of the
//! indexed axis (`src_dim_size`). With `clear = false` the count accumulates onto the word, so one
//! word can cover every index buffer of a step.

use baracuda_cutlass::{Error, Result};
use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_kernels_types::{IndexElement, IndexElementKind, TensorRef};
use core::ffi::c_void;

use super::gather::map_status;

/// Args for [`IndexBoundsPlan::run`].
#[derive(Debug)]
pub struct IndexBoundsArgs<'a, I: IndexElement> {
    /// The index buffer to check (read as `numel()` contiguous values).
    pub index: TensorRef<'a, I, 1>,
    /// Zero the counter word before counting (`false` accumulates).
    pub clear: bool,
}

/// Counts index values outside `[0, bound)` on the device.
#[derive(Copy, Clone, Debug)]
pub struct IndexBoundsPlan {
    bound: i64,
}

impl IndexBoundsPlan {
    /// A check against the axis extent `bound` (`>= 0`).
    pub fn new(bound: i64) -> Result<Self> {
        if bound < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IndexBoundsPlan: bound must be non-negative",
            ));
        }
        Ok(Self { bound })
    }

    /// Enqueue the check; the count lands in `flag[0]`.
    pub fn run<I: IndexElement>(
        &self,
        stream: &Stream,
        flag: &mut DeviceBuffer<u32>,
        args: IndexBoundsArgs<'_, I>,
    ) -> Result<()> {
        if flag.is_empty() {
            return Err(Error::BufferTooSmall { needed: 1, got: 0 });
        }
        let n = args.index.numel();
        if (args.index.data().len() as i64) < n {
            return Err(Error::BufferTooSmall {
                needed: n as usize,
                got: args.index.data().len(),
            });
        }
        let idx = args.index.data().as_raw().0 as *const c_void;
        let flag_ptr = flag.as_raw().0 as *mut c_void;
        let clear = i32::from(args.clear);
        let status = match I::KIND {
            IndexElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_bounds_i32_run(
                    n,
                    self.bound,
                    idx,
                    flag_ptr,
                    clear,
                    stream.as_raw(),
                )
            },
            IndexElementKind::I64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_index_bounds_i64_run(
                    n,
                    self.bound,
                    idx,
                    flag_ptr,
                    clear,
                    stream.as_raw(),
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::IndexBoundsPlan: index dtype must be i32 or i64",
                ));
            }
        };
        map_status(status)
    }

    /// Synchronise `stream`, read the counter word and raise if any index was out of range.
    pub fn check(stream: &Stream, flag: &DeviceBuffer<u32>) -> Result<()> {
        stream.synchronize()?;
        let mut host = vec![0u32; flag.len()];
        flag.copy_to_host(&mut host)?;
        match host.first().copied().unwrap_or(0) {
            0 => Ok(()),
            count => Err(Error::IndexOutOfRange { count }),
        }
    }
}
