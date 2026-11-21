// Check harness for flashinfer_paged_decode_f16_d128_g1_flat.cu (baracuda-written, MIT OR Apache-2.0).
// Runs the ORIGINAL vendored FlashInfer BatchDecodeWithPagedKVCacheDispatched<128,kNone,...> and the
// flattened kernel on identical random paged-KV inputs and compares the half outputs and the lse
// BITWISE. Result 2026-11-21 (RTX 4070 sm_89, CUDA 13.3, MSVC 14.52, vendored ref eee0d75f):
// 30 cases (5 shapes x 3 seeds x lse on/off, incl. empty request, one-full-page request, kv_len up to
// ~1500 so the kv_offset_smem refresh at (iter+2)%16==0 runs): diff_o=0 diff_lse=0.
// Control: with `s[j] *= sm_scale_log2 * 1.01f` in a scratch copy of the flat kernel the same harness
// reports diff_o=39350 diff_lse=159, so it can see a difference. This replaced running through the Rust
// paged_decode_vs_reference test (that needs a kernels-sys rebuild); GROUP_SIZE 1 only (H==num_kv_heads).
// Build (scratch): copy vendor/flashinfer/include/flashinfer to ./fi and delete `__restrict__` from
// fi/flashinfer/pos_enc.cuh (MSVC rejects it at pos_enc.cuh:1529); the "fi/flashinfer/..." and the
// flat-file include paths below are the scratch layout. nvcc -arch=sm_89 -std=c++17
//   -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING -diag-suppress 177 -O2 harness.cu
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>  // Phase 66 Tier 2: fp8 KV-cache decode

// --- MSVC nvcc per-thread-default-stream workaround --------------------
// FlashInfer's `decode.cuh` calls `cudaLaunchKernel((void*)kernel, ...)`.
// Under `-default-stream per-thread` (Windows nvcc), CUDA's headers
// `#define cudaLaunchKernel __CUDART_API_PTSZ(cudaLaunchKernel)`, which
// under MSVC interacts badly with the template overload at the FlashInfer
// call site (the template's name itself gets rewritten, then the
// non-template static-inline wrapper at `cuda_runtime_api.h:13929` and
// the templated one at `cuda_runtime.h:208` collide on overload
// resolution). We side-step the rewrite by introducing a TU-local
// explicit-signature wrapper and `#define`-aliasing `cudaLaunchKernel`
// to it BEFORE the FlashInfer header is parsed. This keeps the per-
// thread-default-stream behavior intact for our other calls while
// making the FlashInfer call site unambiguous.
namespace baracuda_paged_decode_msvc_shim {
    static inline cudaError_t launch_kernel_explicit(
        const void*  func,
        ::dim3       grid,
        ::dim3       block,
        void**       args,
        std::size_t  smem,
        cudaStream_t stream)
    {
        // Bypass the macro by calling the qualified `_ptsz` variant
        // directly when present (per-thread default stream); fall back
        // to the legacy `cudaLaunchKernel` otherwise. Both have the
        // same C signature, so this is ABI-safe.
        #ifdef __CUDART_API_PER_THREAD_DEFAULT_STREAM
        return ::cudaLaunchKernel_ptsz(func, grid, block, args, smem, stream);
        #else
        // Note: the macro is locally redefined just below, so we have
        // to undef-then-re-call through the underlying entry point.
        return ::cudaLaunchKernel(func, grid, block, args, smem, stream);
        #endif
    }
}

// Force FlashInfer's `cudaLaunchKernel((void*)kernel, ...)` to bind to
// our explicit-signature wrapper. The `void*` -> `const void*` is
// implicit; the rest of the args match exactly. This MUST come before
// the FlashInfer headers are included.
#undef cudaLaunchKernel
#define cudaLaunchKernel(func, grid, block, args, smem, stream) \
    ::baracuda_paged_decode_msvc_shim::launch_kernel_explicit( \
        (const void*)(func), (grid), (block), (args), (smem), (stream))

#include "fi/flashinfer/attention/decode.cuh"
#include "fi/flashinfer/attention/default_decode_params.cuh"
#include "fi/flashinfer/attention/variants.cuh"
#include "fi/flashinfer/layout.cuh"
#include "fi/flashinfer/page.cuh"
#undef cudaLaunchKernel
#include "C:/Projects/baracuda-wt-r6b/docs/planning/u19/flashinfer_paged_decode_f16_d128_g1_flat.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#define CK(...) do{cudaError_t e=(__VA_ARGS__); if(e!=cudaSuccess){printf("CUDA error at %d: %s\n",__LINE__,cudaGetErrorString(e)); return 2;}}while(0)
using DType = half;
using Params = flashinfer::BatchDecodeParams<DType, DType, DType, int32_t>;
using Variant = flashinfer::DefaultAttention<false,false,false,false>;
static uint32_t rng=12345; static uint32_t rnd(){ rng=rng*1664525u+1013904223u; return rng>>8; }
int run_case(int B, int H, int page, int maxlen, int seed, bool want_lse, int* ndiff_o, int* ndiff_lse){
  rng=seed; const int D=128;
  std::vector<int> lens(B); for(int b=0;b<B;b++) lens[b]= (rnd()%maxlen)+ (b==0?1:0);
  if(B>2) lens[2]=0;                // empty request
  if(B>3) lens[3]=page;             // exactly one full page
  std::vector<int> indptr(B+1,0), last(B,0);
  for(int b=0;b<B;b++){ int np=(lens[b]+page-1)/page; indptr[b+1]=indptr[b]+np; last[b]= np? lens[b]-(np-1)*page : 0; }
  int total=indptr[B]; int pool=total+7;
  std::vector<int> perm(pool); for(int i=0;i<pool;i++) perm[i]=i; for(int i=pool-1;i>0;i--) std::swap(perm[i],perm[rnd()%(i+1)]);
  std::vector<int> indices(total); for(int i=0;i<total;i++) indices[i]=perm[i];
  size_t kvn=(size_t)pool*H*page*D, qn=(size_t)B*H*D;
  std::vector<half> k(kvn), v(kvn), q(qn);
  for(auto&x:k) x=__float2half(((int)(rnd()%2001)-1000)/1000.f);
  for(auto&x:v) x=__float2half(((int)(rnd()%2001)-1000)/1000.f);
  for(auto&x:q) x=__float2half(((int)(rnd()%2001)-1000)/500.f);
  half *dk,*dv,*dq,*do1,*do2; float *dl1,*dl2; int *di,*dip,*dlp,*dri,*dkt,*doi,*dkc;
  CK(cudaMalloc(&dk,kvn*2));CK(cudaMalloc(&dv,kvn*2));CK(cudaMalloc(&dq,qn*2));CK(cudaMalloc(&do1,qn*2));CK(cudaMalloc(&do2,qn*2));
  CK(cudaMalloc(&dl1,B*H*4));CK(cudaMalloc(&dl2,B*H*4));
  CK(cudaMalloc(&di,(total+1)*4));CK(cudaMalloc(&dip,(B+1)*4));CK(cudaMalloc(&dlp,B*4));CK(cudaMalloc(&dri,B*4));CK(cudaMalloc(&dkt,B*4));CK(cudaMalloc(&doi,(B+1)*4));CK(cudaMalloc(&dkc,4));
  CK(cudaMemcpy(dk,k.data(),kvn*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dv,v.data(),kvn*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dq,q.data(),qn*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(di,indices.data(),total*4+0,cudaMemcpyHostToDevice));CK(cudaMemcpy(dip,indptr.data(),(B+1)*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dlp,last.data(),B*4,cudaMemcpyHostToDevice));
  std::vector<int> ri(B),kt(B,0),oi(B+1); for(int b=0;b<B;b++){ri[b]=b;oi[b]=b;} oi[B]=B; int kc=INT32_MAX;
  CK(cudaMemcpy(dri,ri.data(),B*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dkt,kt.data(),B*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(doi,oi.data(),(B+1)*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dkc,&kc,4,cudaMemcpyHostToDevice));
  CK(cudaMemset(do1,0xAB,qn*2));CK(cudaMemset(do2,0xAB,qn*2));CK(cudaMemset(dl1,0xAB,B*H*4));CK(cudaMemset(dl2,0xAB,B*H*4));
  flashinfer::paged_kv_t<DType,int32_t> pk(H,page,D,B,flashinfer::QKVLayout::kHND,dk,dv,di,dip,dlp,nullptr);
  Params p; p.q=dq; p.q_rope_offset=nullptr; p.paged_kv=pk; p.o=do1; p.lse=want_lse?dl1:nullptr; p.maybe_alibi_slopes=nullptr;
  p.padded_batch_size=B; p.num_qo_heads=H; p.q_stride_n=H*D; p.q_stride_h=D; p.window_left=-1; p.logits_soft_cap=0.f; p.sm_scale=1.0f/sqrtf(128.f);
  p.rope_rcp_scale=1.f; p.rope_rcp_theta=1e-4f; p.request_indices=dri; p.kv_tile_indices=dkt; p.o_indptr=doi; p.kv_chunk_size_ptr=dkc; p.block_valid_mask=nullptr; p.partition_kv=false;
  CK(flashinfer::BatchDecodeWithPagedKVCacheDispatched<128, flashinfer::PosEncodingMode::kNone, Variant, Params>(p,nullptr,nullptr,false,0));
  CK(cudaDeviceSynchronize());
  // flat: same bytes reinterpreted into the plain struct (checks layout equality too)
  static_assert(sizeof(Params)==sizeof(flat_decode::DecodeParamsHalf),"layout size");
  flat_decode::DecodeParamsHalf fp; memcpy(&fp,&p,sizeof(fp)); fp.o=do2; fp.lse=want_lse?dl2:nullptr;
  flat_decode::BatchDecodeWithPagedKVCacheKernel_f16_d128_g1_none<<<dim3(B,H),dim3(16,1,8),flat_decode::SMEM_BYTES>>>(fp);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<half> o1(qn),o2(qn); std::vector<float> l1(B*H),l2(B*H);
  CK(cudaMemcpy(o1.data(),do1,qn*2,cudaMemcpyDeviceToHost));CK(cudaMemcpy(o2.data(),do2,qn*2,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(l1.data(),dl1,B*H*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(l2.data(),dl2,B*H*4,cudaMemcpyDeviceToHost));
  *ndiff_o=0;*ndiff_lse=0; for(size_t i=0;i<qn;i++) if(memcmp(&o1[i],&o2[i],2)) (*ndiff_o)++;
  for(int i=0;i<B*H;i++) if(memcmp(&l1[i],&l2[i],4)) (*ndiff_lse)++;
  cudaFree(dk);cudaFree(dv);cudaFree(dq);cudaFree(do1);cudaFree(do2);cudaFree(dl1);cudaFree(dl2);cudaFree(di);cudaFree(dip);cudaFree(dlp);cudaFree(dri);cudaFree(dkt);cudaFree(doi);cudaFree(dkc);
  return 0;
}
int main(){
  int tot_o=0,tot_l=0,cases=0; struct C{int B,H,page,maxlen;} cs[]={{6,2,16,1500},{4,1,16,40},{8,4,16,900},{5,2,32,700},{3,2,1,300}};
  for(auto c:cs) for(int seed=1;seed<=3;seed++){ for(int lse=0;lse<2;lse++){ int a,b; if(run_case(c.B,c.H,c.page,c.maxlen,seed*7+c.B,lse,&a,&b)) return 2; tot_o+=a;tot_l+=b;cases++; printf("case B=%d H=%d page=%d maxlen=%d seed=%d lse=%d diff_o=%d diff_lse=%d\n",c.B,c.H,c.page,c.maxlen,seed,lse,a,b);} }
  printf("TOTAL cases=%d diff_o=%d diff_lse=%d\n",cases,tot_o,tot_l); return (tot_o||tot_l)?1:0;
}
