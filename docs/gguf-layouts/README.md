# GGUF block layouts as nvcc lays them out

Derived by a COMPILED harness, not by reading the source: [`layout_harness.cu`](layout_harness.cu) includes
`crates/baracuda-kernels-sys/kernels/include/baracuda_gguf.cuh` and prints `sizeof`/`offsetof` per field;
raw output in [`layout_harness.out`](layout_harness.out).

- Compiler: `nvcc` release 13.3, V13.3.33 (build cuda_13.3.r13.3/compiler.37862127_0), host cl.exe MSVC 14.52.37025, `-std=c++17`, x86-64 Windows. Run 2026-11-23, header at origin/main ce2911956.
- Control: `block_ctl { half d; uint8_t qs[16]; }` printed sizeof=18, `d`@0, `qs`@2, as predicted before the run.
- `half2` fields (`dm`) are listed with elem `half2`, count 1, 4 bytes (two `half`, `.x` at +0 and `.y` at +2 within the field).
- The header's `static_assert`s hold (it compiled); q3_K has none, and its 110 bytes (2-byte alignment, `d` last) is the measured value.

Table: name, sizeof, alignof, then `(field, offsetof, elem, count)`.

| struct | sizeof | alignof | fields (field @offset elem x count) |
|---|---|---|---|
| `block_ctl` | 18 | 2 | `d` @0 half x1; `qs` @2 uint8_t x16 |
| `block_q4_0` | 18 | 2 | `d` @0 half x1; `qs` @2 uint8_t x16 |
| `block_q4_1` | 20 | 4 | `dm` @0 half2 x1; `qs` @4 uint8_t x16 |
| `block_q5_0` | 22 | 2 | `d` @0 half x1; `qh` @2 uint8_t x4; `qs` @6 uint8_t x16 |
| `block_q5_1` | 24 | 4 | `dm` @0 half2 x1; `qh` @4 uint8_t x4; `qs` @8 uint8_t x16 |
| `block_q8_0` | 34 | 2 | `d` @0 half x1; `qs` @2 int8_t x32 |
| `block_q2_K` | 84 | 4 | `scales` @0 uint8_t x16; `qs` @16 uint8_t x64; `dm` @80 half2 x1 |
| `block_q3_K` | 110 | 2 | `hmask` @0 uint8_t x32; `qs` @32 uint8_t x64; `scales` @96 uint8_t x12; `d` @108 half x1 |
| `block_q4_K` | 144 | 4 | `dm` @0 half2 x1; `scales` @4 uint8_t x12; `qs` @16 uint8_t x128 |
| `block_q5_K` | 176 | 4 | `dm` @0 half2 x1; `scales` @4 uint8_t x12; `qh` @16 uint8_t x32; `qs` @48 uint8_t x128 |
| `block_q6_K` | 210 | 2 | `ql` @0 uint8_t x128; `qh` @128 uint8_t x64; `scales` @192 int8_t x16; `d` @208 half x1 |
| `block_q8_K` | 292 | 4 | `d` @0 float x1; `qs` @4 int8_t x256; `bsums` @260 int16_t x16 |

As `StructLayout::new(sizeof, [FieldLayout::new(field, offsetof, elem_c_type, count)])` the rows are exactly the
numbers above; nothing is inferred from the field names.
