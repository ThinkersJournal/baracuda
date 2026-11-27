// baracuda-kernels ROADMAP E4 — device-side index bounds check.
//
// The indexing kernels (gather / index_select / scatter / scatter_add / index_add and their
// backwards) SKIP an out-of-range or negative index; torch raises. This kernel is the check that
// lets a caller raise: it counts the values of a contiguous index buffer outside `[0, bound)` into
// a device `uint32` word. It changes no existing kernel or ABI, applies to every indexing op and
// both index dtypes, adds no host sync (a captured graph replays it), and never reads `src`/`out`.
//
// `clear != 0` zeroes the word on the stream first (cudaMemsetAsync, capturable); `clear == 0`
// accumulates onto it, so one word can cover several index buffers of one step.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

template <typename IndexT>
__global__ void index_bounds_kernel(const IndexT* __restrict__ index, int64_t n, int64_t bound,
                                    uint32_t* __restrict__ flag)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x * (int64_t)blockDim.x;
    uint32_t bad = 0;
    for (int64_t i = tid; i < n; i += step) {
        int64_t v = (int64_t)index[i];
        if (v < 0 || v >= bound) ++bad;
    }
    if (bad != 0) atomicAdd(flag, bad);
}

template <typename IndexT>
int32_t launch_index_bounds(const IndexT* index, int64_t n, int64_t bound, uint32_t* flag,
                            int32_t clear, cudaStream_t stream)
{
    if (n < 0 || bound < 0) return 2;
    if (flag == nullptr) return 2;
    if (n > 0 && index == nullptr) return 2;
    if (clear != 0) {
        if (cudaMemsetAsync(flag, 0, sizeof(uint32_t), stream) != cudaSuccess) return 5;
    }
    if (n == 0) return 0;
    const int block = 256;
    int64_t blocks64 = (n + block - 1) / block;
    int grid = (int)(blocks64 > 1024 ? 1024 : blocks64);
    index_bounds_kernel<IndexT><<<grid, block, 0, stream>>>(index, n, bound, flag);
    return cudaGetLastError() == cudaSuccess ? 0 : 5;
}

}  // namespace

#define BARACUDA_INDEX_BOUNDS_FFI(NAME, INDEX_T)                                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n, int64_t bound, const void* index, void* flag, int32_t clear, void* stream_ptr)  \
    {                                                                                              \
        return launch_index_bounds<INDEX_T>(static_cast<const INDEX_T*>(index), n, bound,         \
                                            static_cast<uint32_t*>(flag), clear,                   \
                                            static_cast<cudaStream_t>(stream_ptr));                \
    }

BARACUDA_INDEX_BOUNDS_FFI(index_bounds_i32, int32_t)
BARACUDA_INDEX_BOUNDS_FFI(index_bounds_i64, int64_t)
