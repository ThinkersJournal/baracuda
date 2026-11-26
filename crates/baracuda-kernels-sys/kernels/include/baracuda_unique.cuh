// baracuda_unique.cuh
//
// `unique` / `unique_consecutive` FW (no BW — set-valued op).
//
// `unique_consecutive` is the trailblazer primitive — it walks the
// already-sorted (or assumed-sorted by the caller) input row and emits
// one cell per "run start" (position 0 OR a position where the value
// differs from its predecessor), IN ORDER, with the run length as the
// count (a block scan numbers the runs; see the kernel).
//
// `unique` is the same kernel with the contract that the caller
// pre-sorted the row (calls SortPlan first). Since the per-row sort
// then a per-row consecutive-dedup are two separate launches at the
// Rust plan layer, the kernel here only implements the consecutive
// variant. The Plan's `unique` op chains sort + this kernel.
//
// Status codes mirror baracuda_sort.cuh.
//
// Trailblazer dtype coverage: f32, f64, i32.

#ifndef BARACUDA_UNIQUE_CUH
#define BARACUDA_UNIQUE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda { namespace unique_ns {

template <typename T>
__device__ inline bool eq_val(T a, T b) { return a == b; }

// One block of 1024 threads per row, a loop over 1024-wide chunks. A thread's element starts a run when it differs from its predecessor
// (`!(a == b)`: a NaN never equals a NaN, so every NaN is its own run, and -0.0 == +0.0, one run). A block scan numbers the runs of the chunk
// and a carry numbers them across chunks, so output slot k holds the k-th run IN ORDER (no atomic slot race), `y_counts[k]` is the run LENGTH,
// `y_vals[k]` its first element, and `counter[row]` the actual number of runs (it can exceed `max_unique`; the overflow runs are dropped).
// A run that continues into the next chunk is finalised when that chunk finds the next run start (or at the end of the row).
template <typename T>
__global__ void unique_consecutive_kernel(
    const T*  __restrict__ x,           // [batch, row_len]
    T*        __restrict__ y_vals,      // [batch, max_unique] flat
    int32_t*  __restrict__ y_counts,    // [batch, max_unique] flat, run lengths (nullable)
    int32_t*  __restrict__ counter,     // [batch], the run count of each row
    int32_t   row_len,
    int32_t   max_unique)
{
    __shared__ int32_t scan[1024];
    __shared__ int32_t starts[1024];
    __shared__ int32_t last_slot;   // slot of the run still open at the end of the previous chunk (-1: none)
    __shared__ int32_t last_start;  // its first position
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const T* xr = x + (int64_t)row * (int64_t)row_len;
    const int64_t out_off = (int64_t)row * (int64_t)max_unique;
    if (t == 0) { last_slot = -1; last_start = 0; }
    int32_t carry = 0;  // runs started before this chunk (same value in every thread)
    for (int32_t base = 0; base < row_len; base += 1024) {
        const int32_t idx = base + t;
        int32_t flag = 0;
        if (idx < row_len) flag = (idx == 0 || !(xr[idx] == xr[idx - 1])) ? 1 : 0;
        scan[t] = flag;
        __syncthreads();
        for (int off = 1; off < 1024; off <<= 1) {
            int32_t v = (t >= off) ? scan[t - off] : 0;
            __syncthreads();
            scan[t] += v;
            __syncthreads();
        }
        const int32_t runs = scan[1023];
        if (flag) starts[scan[t] - 1] = idx;
        __syncthreads();
        if (runs > 0) {
            // Close the run left open by the previous chunk: it ends where this chunk's first run starts.
            if (t == 0 && last_slot >= 0 && last_slot < max_unique && y_counts)
                y_counts[out_off + last_slot] = starts[0] - last_start;
            for (int32_t k = t; k < runs; k += 1024) {
                const int32_t slot = carry + k;
                if (slot < max_unique) {
                    y_vals[out_off + slot] = xr[starts[k]];
                    if (y_counts && k + 1 < runs) y_counts[out_off + slot] = starts[k + 1] - starts[k];
                }
            }
            if (t == 0) { last_slot = carry + runs - 1; last_start = starts[runs - 1]; }
        }
        carry += runs;
        __syncthreads();  // `starts` and `scan` are reused by the next chunk
    }
    if (t == 0) {
        if (last_slot >= 0 && last_slot < max_unique && y_counts)
            y_counts[out_off + last_slot] = row_len - last_start;
        counter[row] = carry;
    }
}

template <typename T>
__host__ inline int32_t launch_unique_consecutive(
    const T* x, T* y_vals, int32_t* y_counts, int32_t* counter,
    int32_t batch, int32_t row_len, int32_t max_unique,
    cudaStream_t stream)
{
    if (batch < 0 || row_len < 0 || max_unique < 0) return 2;
    if (batch == 0) return 0;
    if (counter == nullptr) return 2;
    if (row_len == 0) {
        return cudaMemsetAsync(counter, 0, (size_t)batch * sizeof(int32_t), stream) == cudaSuccess ? 0 : 5;
    }
    if (x == nullptr) return 2;
    if (max_unique > 0 && y_vals == nullptr) return 2;
    unique_consecutive_kernel<T><<<batch, 1024, 0, stream>>>(
        x, y_vals, y_counts, counter, row_len, max_unique);
    return cudaGetLastError() == cudaSuccess ? 0 : 5;
}

}} // namespace baracuda::unique_ns

#define BARACUDA_KERNELS_UNIQUE_CONSECUTIVE_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t batch, int32_t row_len, int32_t max_unique,                                       \
        const void* x, void* y_vals, void* y_counts, void* counter,                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::unique_ns::launch_unique_consecutive<T>(                                 \
            static_cast<const T*>(x),                                                              \
            static_cast<T*>(y_vals),                                                               \
            static_cast<int32_t*>(y_counts),                                                       \
            static_cast<int32_t*>(counter),                                                        \
            batch, row_len, max_unique, stream);                                                  \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t batch, int32_t row_len, int32_t max_unique,                                       \
        const void* /*x*/, const void* /*y_vals*/,                                                \
        const void* /*y_counts*/, const void* /*counter*/)                                        \
    {                                                                                              \
        if (batch < 0 || row_len < 0 || max_unique < 0) return 2;                                 \
        return 0;                                                                                 \
    }

#endif // BARACUDA_UNIQUE_CUH
