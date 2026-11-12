# Comprehensive CUDA parser: what is missing, row by row

Ref: `origin/main` `059d3d88`; Unpopped `0.43.2` (the registry copy `baracuda-cuda-emit` builds against). Author: baracuda lane,
PM [TASK] 2026-11-11. Docs only: no code, and the 2026-10-28 hold (*no new CUDA parser/emitter features until the seam
settles*, CireSnave) stays in force, so every row below is **scheduled, not started**.

CireSnave 2026-11-11, verbatim: *"If the CUDA parser is not comprehensive, Baracuda still has work to do to make it
comprehensively cover everything CUDA can do."*

## What exists today (measured)

- `crates/baracuda-cuda-parse/src/lib.rs`: a `tree-sitter-cuda` grammar binding, the `CUDA` frontend descriptor, and a
  residue list of ten words that **refuse** a body: `__shared__`, `atomicAdd`, `atomicCAS`, `__syncthreads`, `cublas`,
  `cudnn`, `printf`, `asm`, `cp.async`, `__shfl`. The shared walkers (`lift_elementwise`/`lift_reduction`/`lift_scan`) live
  in `unpopped::convert` and lift **idioms** (an elementwise expression, a reduction, a scan), not statements.
- `unpopped::kernel_lift` lifts an intent-layer `OpDef` **to** a `KernelFn` (elementwise, reduce, scan, scatter, window,
  ...). That is the other direction. There is **no** CST -> `KernelFn` lift of a hand-written body, and before this
  document no ROADMAP row for one.
- The Unpopped kernel IR (`unpopped-0.43.2/src/kernel.rs`, `exec.rs`) already has: `Stmt::Barrier` (`Subgroup`, `Block`,
  `Device`, `Cluster`), `Stmt::AtomicRmw`/`AtomicCas` (global buffer or `Shared` element, **integer element types in v1**,
  relaxed), `Stmt::Fence`, `Stmt::Collective` (`Shuffle`, `Vote`, `Reduce`, `Scan`, `Broadcast`), `Stmt::Mma`,
  `MemorySpace::{Global, Shared, Private, Constant}`, `Place::Alloc`, extern calls with a `cuda-ptx` asm body, and (Unpopped#335,
  unmerged at this ref) `KernelInterface`.
- The IR does **not** have: pointer values (`Place::Ptr` is named as deferred in `kernel.rs`; extern asm has "no pointer
  operands" in v1), any async-copy or pipeline statement, any texture/surface space, an expression-valued launch grid, or
  struct-typed scalar parameters (`Param` is `Buffer` or `Scalar`). Found by grep of those two files; absence of a match
  is the evidence (control: the same grep finds `Barrier`, `Collective`, `Shared`).
- `baracuda-cuda-emit` already **spells** shared memory, barriers, a Kogge-Stone warp scan, a block scan, a bitonic
  smem sort, FP atomic scatter and (behind `extern_asm_cuda`) inline PTX, for IR it generates itself. Emitting is ahead of
  lifting.
- Existing parse-side tests: `crates/baracuda-cuda-parse/tests/convert.rs`, 14 tests; the only residue assertion is
  `cuda_refuses_residue`, which checks `__shared__` (and neighbour indexing `in0[i+1]`). **No test asserts refusal of the
  other nine residue words.**

## First target: the vendored FlashInfer paged decode

`crates/baracuda-kernels-sys/vendor/flashinfer/include/flashinfer/attention/decode.cuh` (1207 lines) plus the launcher
`kernels/attention/flashinfer_paged_decode_launcher.cu`. Regex counts at this ref (`git show origin/main:<path>`, patterns
in the table), per file:

| file | lines | `__shared__` | `cp.async` | `asm` | `cooperative_groups` | `template<` | `if constexpr` | `constexpr` | `__grid_constant__` | `#pragma unroll` |
|---|---|---|---|---|---|---|---|---|---|---|
| `attention/decode.cuh` | 1207 | 4 | 33 | 5 | 7 | 11 | 10 | 44 | 2 | 42 |
| `cp_async.cuh` | 265 | 0 | 21 | 12 | 0 | 7 | 11 | 11 | 0 | 0 |
| `permuted_smem.cuh` | 191 | 0 | 9 | 0 | 0 | 11 | 8 | 9 | 0 | 0 |
| `page.cuh` | 597 | 0 | 0 | 0 | 0 | 8 | 0 | 5 | 0 | 2 |

`decode.cuh` itself has **zero** matches for `__syncthreads`, `__shfl`, votes, atomics, `mma.sync`/`ldmatrix` and
`__launch_bounds__`. Each pattern has a positive control in the same pass over the FlashInfer headers: `__syncthreads` matches
`cascade.cuh` (15), `sampling.cuh` (39) and `prefill.cuh` (2); `__shfl` matches `frag_layout_swizzle.cuh` (9), `mma.cuh` (4),
`sampling.cuh` (5) and `math.cuh` (1); atomics match `sampling.cuh` (1); `mma.sync`/`ldmatrix` match `mma.cuh` (66),
`prefill.cuh` (15) and `permuted_smem.cuh` (12); `__launch_bounds__` matches `prefill.cuh` (3). Votes (`__ballot*`,
`__any_sync`, `__all_sync`) match **nowhere** in the vendored headers (no control, so this one is only "no match in these
files"). Its barriers come through cooperative groups: 18 `.sync()` calls and 7 `cg::`/`cooperative_groups` matches in
`decode.cuh`. So the paged decode needs rows P1, P4-P8 and P11 below, and **does not need P3, P9, P10 or the vote
half of P2**. Caveat: `decode.cuh` includes `math.cuh`, which has one `__shfl`; I did not check whether the decode path
instantiates it, so P2 (shuffle) is "probably not needed", not "not needed". The prefill kernels (`prefill.cuh`, 3038
lines) add `mma`/`ldmatrix` (P10). The device check that a lifted-and-re-emitted decode must keep passing is
`crates/baracuda-flashinfer/tests/paged_decode_vs_reference.rs`.

## Rows

Dependency order as asked by the PM. **Owner** is who writes it: *Unpopped* owns the neutral IR and walker, *Baracuda*
owns `baracuda-cuda-parse` (residue list, CUDA spellings, frontend), `baracuda-cuda-emit` spelling and the device tests.
Sizes are S (days), M (a week or two), L (weeks), XL (a quarter); two sizes where two owners both do work. Sizes are
estimates, not measurements.

| id | construct family | IR status (Unpopped 0.43.2) | what has to be lifted | owner | size | existing test that will show it |
|---|---|---|---|---|---|---|
| P1 | **Shared memory and barriers** | `Place::Alloc` + `MemorySpace::Shared`, `Stmt::Barrier{Subgroup,Block,Device,Cluster}`: present | static `__shared__ T s[N]`; `extern __shared__` (dynamic; sized by `KernelInterface.dynamic_shared`); `__syncthreads()`, `__syncwarp()`; the checker's `BarrierDivergent` rule applied to hand-written control flow | Baracuda S (drop 2 residue words, spell, test), Unpopped M (declarations, dynamic size, divergence check on lifted control flow) | S + M | `baracuda-cuda-parse/tests/convert.rs::cuda_refuses_residue` (asserts `__shared__` is refused today; flips to a lift). No test covers `__syncthreads`: first new test `cuda_lifts_block_sum_with_syncthreads` in the same file |
| P2 | **Warp shuffles, votes, reductions** | `Stmt::Collective{Shuffle, Vote, Reduce, Scan, Broadcast}`: present | `__shfl_sync/_up/_down/_xor_sync`, `__ballot_sync`, `__any_sync`/`__all_sync`, `__reduce_*_sync`; a shuffle with a data-dependent source lane has no `Collective` form and must be a typed decline, not a guess | Baracuda S, Unpopped M (decide which shuffle patterns map to `Collective`) | S + M | none asserts `__shfl` refusal. Emit side is covered by `baracuda-cuda-emit/tests/contract.rs::scan_advertises_a_recipe_carrying_contract` (generated warp scan). New: `cuda_lifts_warp_sum_via_shfl_xor` |
| P3 | **Atomics and fences** | `Stmt::AtomicRmw`/`AtomicCas`/`Fence`: present, but **integer elements only in v1, relaxed** | `atomicAdd/Min/Max/CAS/Exch` on ints; `__threadfence()`/`_block`/`_system`; `cuda::atomic_ref`. Floating-point `atomicAdd` (`float`, `half2`, `bf16x2`) needs an IR extension: **IR gap G4** | Baracuda S, Unpopped M (G4: FP element atomics + their determinism class) | S + M | none asserts `atomicAdd`/`atomicCAS` refusal. Emit side spells an FP atomic scatter (`cuda.rs`, "Increment 5"); a lift round-trip test is new: `cuda_lifts_int_histogram_atomic` |
| P4 | **`cp.async` and pipelines** | **absent**: no async-copy or pipeline statement; extern asm has no pointer operands | `cp.async.{ca,cg}.shared.global`, `cp.async.commit_group`, `cp.async.wait_group`, `cuda::pipeline`/`cuda::memcpy_async`, `__pipeline_*`. Needs **G3 (pointer values)** and an async-copy `Stmt` with commit/wait groups: **IR gap G5** | Unpopped L (G3 + G5), Baracuda M | L + M | none (`cp.async` is in the residue list; no test). The paged decode is the first consumer: `decode.cuh` has 33 matches, `cp_async.cuh` 21 |
| P5 | **Inline PTX / `asm`** | extern calls with a `cuda-ptx` asm body: present (no pointer operands) | parse `asm [volatile]("..." : "=r"(x) : "r"(y))` into an `ExternDecl` (template, constraints, volatility); asm with `"memory"` clobbers or pointer operands needs G3 | Baracuda M (the asm grammar), Unpopped S | M + S | `baracuda-cuda-emit/tests/extern_ptx.rs::popc_b32_matches_its_reference` and `::fma_rn_f32_matches_its_reference` run the **emit** direction on a device; the lift direction is new (parse the asm those tests emit and round-trip it) |
| P6 | **Cooperative groups** | scopes `Subgroup`/`Block`/`Device`/`Cluster` exist on `Barrier`/`Collective` | `cg::this_thread_block()`, `.sync()`, `cg::tiled_partition<N>`, `cg::reduce`, `cg::this_grid().sync()` (a `Device` barrier), `cluster_group`. Not in the residue list today, so a body using it is **silently not refused**: first check that it is refused (typed decline), then lift | Baracuda M, Unpopped S | M + S | none exists. `decode.cuh` has 18 `.sync()` calls and 7 `cg::` matches and takes its barriers this way |
| P7 | **Templates, `constexpr`, `if constexpr`, struct parameters, `__grid_constant__`** | struct-typed scalar params **absent** (`Param` is `Buffer`/`Scalar`): **IR gap G2**. `KernelInterface.params_block` (Unpopped#335) describes the flattened block | lift **one instantiation**: template arguments taken from the launcher's explicit instantiation; `constexpr` integer const-eval; `if constexpr` folded at lift time; struct params flattened into scalars/buffers; `#pragma unroll` kept as a hint (the `Stmt::For` unroll hint is breaking, Unpopped 0.46.0; emitters treat it as optional until then) | Unpopped L (const-eval, instantiation, G2), Baracuda M (tree-sitter template nodes, finding the launcher's instantiation) | L + M | `baracuda-cuda-emit/tests/new_node_declines.rs::integer_and_packed_nodes_are_typed_declines_and_a_plain_body_still_emits` fixes the *decline contract* a not-yet-lifted node must keep. `decode.cuh`: 11 `template<`, 10 `if constexpr`, 44 `constexpr`, 2 `__grid_constant__` |
| P8 | **Launch bounds and attributes** | `KernelInterface{launch_bounds, max_regs, cooperative, params_block, buffer_guarantees, entry_symbol}` (Unpopped#335, unmerged) | `__launch_bounds__(max, min)`, `__maxnreg__`, `__restrict__`, `__grid_constant__`, `__cluster_dims__`, `__forceinline__`/`__noinline__`, `extern "C"` vs a mangled entry. **IR gap G1**: an expression-valued launch grid (the IR grid is fixed numbers) | Baracuda S (attribute parse -> interface), Unpopped S (#335) + S (G1) | S + S | none. `__launch_bounds__` has 0 matches in `decode.cuh` (3 in `prefill.cuh`, none elsewhere), so it is not on the decode critical path |
| P9 | **Texture and surface** | **absent**: `MemorySpace` has no texture/surface | `cudaTextureObject_t`, `tex1Dfetch`/`tex2D`, `cudaSurfaceObject_t`, `surf2Dread/write`. Needs a read-only-cache memory space with filtering and addressing semantics: **IR gap G6** | Unpopped L (G6), Baracuda M | L + M | none; the decline test to add lives in `new_node_declines.rs`. 0 matches in FlashInfer |
| P10 | **Tensor-core `mma` / `wmma`** | `Stmt::Mma{atom: MmaAtom, ..}`: present (which atoms are covered: not checked here) | `mma.sync.aligned.*` (arrives as asm, so needs P5), `ldmatrix`, `nvcuda::wmma` fragments (`load_matrix_sync`, `mma_sync`, `store_matrix_sync`), `wgmma` (sm_90) | Baracuda L, Unpopped M (atom coverage) | L + M | none for lift. The FlashInfer **prefill** kernels are the target (`mma.cuh` 705 lines, `prefill.cuh` 3038); the paged decode has 0 |
| P11 | **The body lift itself** | the IR can hold the result; there is no CST -> `KernelFn` walker, and **pointer values are absent (G3)**: hand-written bodies index through pointers (`DTypeKV*` in `decode.cuh`) | statement spine: declarations -> `Let`/`Load`; assignments; `for`/`while`/`if`/`break`/`continue`/`return`; indexing -> `Place::Logical`/`Place::Alloc`; `threadIdx`/`blockIdx`/`blockDim`/`gridDim` -> `ExecId`/`LaunchShape`; integer promotion and casts; every unsupported node a **typed decline** | Unpopped XL (the neutral walker; PM: XL after #332 and S4), Baracuda L (the CUDA frontend: builtins, residue -> typed decline, NVRTC re-compile and on-device check) | XL + L | `baracuda-cuda-emit/tests/lift.rs` (2 hand-written-CUDA round-trip tests) and `baracuda-cuda-parse/tests/convert.rs` (14). Goal test: lift the paged-decode instantiation, re-emit, run it through `crates/baracuda-flashinfer/tests/paged_decode_vs_reference.rs` and match its measured error |

### IR gaps these rows depend on (all Unpopped; none is a Baracuda decision)

| gap | what | blocks |
|---|---|---|
| G1 | expression-valued launch grid (grid derived from params) | P8, P11 |
| G2 | struct-typed scalar parameters (Unpopped flattens them in the emitter; the IR does not carry them) | P7, P11 |
| G3 | pointer values (`Place::Ptr`; extern asm pointer operands) | P4, P5 (memory operands), P11 |
| G4 | floating-point (and half/bf16) atomic element types | P3 |
| G5 | async-copy statement with commit/wait groups | P4 |
| G6 | texture/surface memory space | P9 |

G1 and G2 are the two Unpopped named on 2026-11-11 when it added `KernelInterface`; G3-G6 are what this review found by
reading `kernel.rs`/`exec.rs`/`kernel_extern.rs`. Each should be confirmed with Unpopped before it is scheduled.

## Suggested order, with the dependency reason

1. **P8 + P1** first: no IR gap, smallest, and they unblock the interface and the shared-memory idioms.
2. **P6, P2, P3 (integer)**: each is an existing IR statement with a CUDA spelling to recognise; P6 first because the
   paged decode needs it and a silent non-refusal is worse than a refusal.
3. **P5**, then **G3**, then **P4**: the asm lift is independent, but `cp.async` needs pointer values and an async-copy
   statement. This is the longest chain on the decode path.
4. **P7** (templates and const-eval) alongside P11's statement spine; **P11 last**, because it is the row that proves the
   others on a real kernel.
5. **P10** (prefill) and **P9** (texture) after the decode target passes.

## Not checked

Which `MmaAtom`s and `CollectiveKind` variants Unpopped covers beyond their names; whether `tree-sitter-cuda` parses
`cg::` templates, `asm` operands and `<<<>>>` without `ERROR` nodes (the crate's own tests do not say); regex counts are
text matches, not syntax (comments and string literals match); the sizes are estimates.

## Update 2026-11-12: Unpopped's answer to G1-G6 (their notice after Unpopped#336, main `0ca89050`)

The gaps above were read from Unpopped 0.43.2. Unpopped's reply (relayed, not re-read in their code by this lane) is that the
IR has moved on, so treat the IR-status column of the table as of 0.43.2 and use this mapping:

| gap | Unpopped row | state they report |
|---|---|---|
| G3 pointer values | U9c, size L | trigger met by the paged decode |
| G4 FP atomics | U12b widened | `F32`/`F64` add already built; `F16`/`Bf16`/packed is M |
| G5 async copy | covered by `CopyAsync` / `AsyncCommit` / `AsyncWait`; the remainder is U13 (predicated, zero-fill, cache hint, pipeline), M, BREAKING 0.46.0 | **this document said "absent": that was true of 0.43.2 only** |
| G6 texture/surface | U21, size L, low rank | |
| G1 expression-valued grid | U22, M | |
| G2 struct-typed scalar params | U23, M | |

Also theirs or shared: U15 packed 8/16-lane (M), U17 PDL (M), U18 `Stmt::For` unroll hint (S-M, BREAKING 0.46.0), U19 C-family
frontend (XL, with Baracuda), U20 `baracuda-cuda-emit` `lower_kernel` (L, **Baracuda's**). `KernelInterface` shipped in Unpopped
0.45.1, so the "unmerged Unpopped#335" wording above is stale. P4 therefore no longer waits on a new async-copy statement, only
on G3 (U9c) and the U13 remainder.
