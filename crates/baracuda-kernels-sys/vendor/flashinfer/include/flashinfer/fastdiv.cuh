/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_FASTDIV_CUH_
#define FLASHINFER_FASTDIV_CUH_
#include <cstdint>

namespace flashinfer {

// baracuda: pure-C++ fast unsigned 32-bit divmod.
// Upstream FlashInfer v0.6.12 wraps `cuda::fast_mod_div<uint32_t>` from
// CCCL 2.4+'s `<cuda/cmath>`; not all toolkits ship that header (in
// particular older CCCL bundled with CUDA 12.0-12.2 doesn't). This is the
// Granlund-Montgomery round-up method ("Division by invariant integers
// using multiplication", PLDI 1994, fig. 4.1) and avoids the CCCL
// dependency. See vendor/flashinfer/VENDOR.md "Patch list" #5.
//
// Algorithm, for d >= 2 and l = ceil(log2(d)):
//   m' = floor(2^32 * (2^l - d) / d) + 1            (fits in 32 bits)
//   hi = floor(m' * n / 2^32)
//   q  = (hi + ((n - hi) >> 1)) >> (l - 1)          (== n / d for every n)
// d == 1 and the default-constructed d == 0 are identity (q = n, r = 0).
//
// The first version of this patch used the single-multiply form
// `m = 2^(32+l)/d + 1`, a 33-bit constant truncated to 32 bits: every
// power-of-two divisor (so every page_size of 2, 4, 8, 16, ...) got
// q == 0, and most other divisors were wrong too. Paged decode then read
// page 0 for every token. Fixed with the tests in
// crates/baracuda-flashinfer/tests/paged_decode_vs_reference.rs.
struct uint_fastdiv {
  __host__ __device__ uint_fastdiv() : d_(0), m_(0), l_(0) {}

  __host__ uint_fastdiv(uint32_t d) : d_(d), m_(0), l_(0) {
    if (d <= 1) {
      // d == 0: placeholder, divmod is the identity. d == 1: identity.
      return;
    }
    // l = ceil(log2(d)), in [1, 32].
    uint32_t l = 0;
    while (l < 32 && (uint64_t(1) << l) < d) ++l;
    uint64_t m = ((uint64_t(1) << 32) * ((uint64_t(1) << l) - d)) / d + 1;
    m_ = uint32_t(m);
    l_ = l;
  }

  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }

  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    if (d_ <= 1) {
      q = n;
      r = 0;
      return;
    }
    uint32_t hi = uint32_t((uint64_t(m_) * uint64_t(n)) >> 32);
    q = (hi + ((n - hi) >> 1)) >> (l_ - 1);
    r = n - q * d_;
  }

 private:
  uint32_t d_;
  uint32_t m_;
  uint32_t l_;
};

__host__ __device__ __forceinline__ uint32_t operator/(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return q;
}

__host__ __device__ __forceinline__ uint32_t operator%(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return r;
}

}  // namespace flashinfer

#endif  // FLASHINFER_FASTDIV_CUH_
