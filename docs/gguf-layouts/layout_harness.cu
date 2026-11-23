// SPDX-FileCopyrightText: 2026 Eric Evans (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Compiled harness: prints sizeof and offsetof of every field of the GGUF block structs in
// crates/baracuda-kernels-sys/kernels/include/baracuda_gguf.cuh, as nvcc lays them out.
// Build: nvcc -std=c++17 -I crates/baracuda-kernels-sys/kernels/include docs/gguf-layouts/layout_harness.cu -o layout_harness
// Output lines: STRUCT <name> sizeof=<n> alignof=<n> / FIELD <name> <field> offset=<n> elem=<c type> count=<n> bytes=<n>
// Control: block_ctl { half d; uint8_t qs[16]; } must print sizeof=18, d@0, qs@2.
#include <cstddef>
#include <cstdio>
#include "baracuda_gguf.cuh"
using namespace baracuda::gguf;

struct block_ctl { half d; uint8_t qs[16]; };

#define S(T) std::printf("STRUCT %s sizeof=%zu alignof=%zu\n", #T, sizeof(T), alignof(T))
#define F(T, f, ELEM, N) std::printf("FIELD %s %s offset=%zu elem=%s count=%d bytes=%zu\n", #T, #f, offsetof(T, f), ELEM, (int)(N), sizeof(((T*)0)->f))

int main() {
    S(block_ctl); F(block_ctl, d, "half", 1); F(block_ctl, qs, "uint8_t", 16);
    S(block_q4_0); F(block_q4_0, d, "half", 1); F(block_q4_0, qs, "uint8_t", QK4_0/2);
    S(block_q4_1); F(block_q4_1, dm, "half2", 1); F(block_q4_1, qs, "uint8_t", QK4_1/2);
    S(block_q5_0); F(block_q5_0, d, "half", 1); F(block_q5_0, qh, "uint8_t", 4); F(block_q5_0, qs, "uint8_t", QK5_0/2);
    S(block_q5_1); F(block_q5_1, dm, "half2", 1); F(block_q5_1, qh, "uint8_t", 4); F(block_q5_1, qs, "uint8_t", QK5_1/2);
    S(block_q8_0); F(block_q8_0, d, "half", 1); F(block_q8_0, qs, "int8_t", QK8_0);
    S(block_q2_K); F(block_q2_K, scales, "uint8_t", QK_K/16); F(block_q2_K, qs, "uint8_t", QK_K/4); F(block_q2_K, dm, "half2", 1);
    S(block_q3_K); F(block_q3_K, hmask, "uint8_t", QK_K/8); F(block_q3_K, qs, "uint8_t", QK_K/4); F(block_q3_K, scales, "uint8_t", K_SCALE_SIZE); F(block_q3_K, d, "half", 1);
    S(block_q4_K); F(block_q4_K, dm, "half2", 1); F(block_q4_K, scales, "uint8_t", 3*QK_K/64); F(block_q4_K, qs, "uint8_t", QK_K/2);
    S(block_q5_K); F(block_q5_K, dm, "half2", 1); F(block_q5_K, scales, "uint8_t", K_SCALE_SIZE); F(block_q5_K, qh, "uint8_t", QK_K/8); F(block_q5_K, qs, "uint8_t", QK_K/2);
    S(block_q6_K); F(block_q6_K, ql, "uint8_t", QK_K/2); F(block_q6_K, qh, "uint8_t", QK_K/4); F(block_q6_K, scales, "int8_t", QK_K/16); F(block_q6_K, d, "half", 1);
    S(block_q8_K); F(block_q8_K, d, "float", 1); F(block_q8_K, qs, "int8_t", QK_K); F(block_q8_K, bsums, "int16_t", QK_K/16);
    return 0;
}
