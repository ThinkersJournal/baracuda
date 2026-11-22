// SPDX-License-Identifier: MIT OR Apache-2.0
//! The CUDA [`unpopped_convert::Frontend`] — the parse-side (`.cu` -> IR)
//! half of Baracuda's donation to Unpopped's neutral core.
//!
//! Moved out of `unpopped::convert` (Unpopped `deferred.md`'s "unpopped-cuda"
//! row, restated after Baracuda's ruling): the neutral crate's own module doc
//! measured that its 620 non-test lines are ~11 CST walkers shared with Slang
//! plus a narrow per-language surface -- two grammar references, two residue
//! lists, and the language descriptors. CUDA's share of that narrow surface is
//! exactly what lives here: the `tree-sitter-cuda` grammar binding
//! ([`parse_cuda`]), CUDA's residue list, and the [`CUDA`] descriptor. The
//! shared walkers (`lift_elementwise`/`lift_reduction`/`lift_scan`/`lift`)
//! and Slang's half stay in `unpopped::convert` because Slang needs them too.
//!
//! Mirrors [`baracuda-cuda-emit`](https://docs.rs/baracuda-cuda-emit)'s
//! placement on the emit side (IR -> `.cu`): both are Baracuda-owned crates
//! outside the neutral core, supplying the CUDA-specific stage from the
//! swappable-backend/swappable-frontend model. Deliberately a separate crate
//! rather than folded into `baracuda-cuda-emit` -- one crate promising both
//! directions either mis-scopes its name or hides one direction behind a
//! feature flag a consumer of the other direction doesn't want.
//!
//! **Not a correctness proof.** This crate's own test suite is CUDA-idiom
//! golden vectors -- the same source strings and equality assertions
//! `unpopped::convert`'s private CUDA tests carried, reproduced here at full
//! fidelity against the public API. It says nothing about kernels outside
//! those idioms: on Unpopped's own measurement, the shared walker accepted
//! 0/147 of fuel's real Slang kernels (4/147 after a naming fix), with one
//! confirmed silent misclassification (a `cumsum` flipping from
//! correctly-refused-as-scan to wrongly-accepted-as-elementwise) surfaced
//! while prototyping that fix. A green suite here is evidence about these
//! idioms, not a claim about the CUDA parser in general.

use tree_sitter::{Parser, Tree};
use unpopped_convert::Frontend;

/// Parse CUDA source into a tree-sitter CST (error-tolerant; unrecognized
/// constructs become `ERROR`/unhandled subtrees rather than failing).
///
/// Identical in behaviour to `unpopped::convert`'s former `parse_cuda` --
/// moved, not rewritten. `unpopped::convert`'s own module doc named this move
/// as the exact seam a per-target crate would use: "An `unpopped-cuda` crate
/// supplies its own `Frontend` and this crate stops naming `tree_sitter_cuda`
/// at all" (that crate name was never built; this one is its replacement,
/// alongside `baracuda-cuda-emit` rather than bundled with it).
///
/// `tree-sitter-cuda` 0.21 does not accept an `__align__(..)` specifier between `__shared__` and
/// the type (`extern __shared__ __align__(alignof(T)) uint8_t smem[];`, FlashInfer's dynamic
/// shared-memory idiom; issue #420). The parser is therefore handed a copy of `src` in which each
/// such `__align__(..)` span is blanked to spaces, **byte for byte** (newlines kept), so every node
/// offset and line number still refers to the caller's original `src`. Only that one span shape is
/// touched; any other source reaches the parser unchanged.
pub fn parse_cuda(src: &str) -> Option<Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_cuda::LANGUAGE.into())
        .ok()?;
    match blank_shared_align(src) {
        Some(blanked) => parser.parse(blanked, None),
        None => parser.parse(src, None),
    }
}

/// `Some(copy)` with every `__shared__ __align__( .. )` alignment span blanked, or `None` when
/// the source has none (the common case: no allocation).
fn blank_shared_align(src: &str) -> Option<Vec<u8>> {
    const SHARED: &[u8] = b"__shared__";
    const ALIGN: &[u8] = b"__align__";
    let b = src.as_bytes();
    let mut out: Option<Vec<u8>> = None;
    let mut i = 0;
    while let Some(off) = find(&b[i..], SHARED) {
        let mut j = i + off + SHARED.len();
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if b[j..].starts_with(ALIGN) {
            let start = j;
            j += ALIGN.len();
            while j < b.len() && b[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < b.len() && b[j] == b'(' {
                let mut depth = 0usize;
                let mut k = j;
                while k < b.len() {
                    match b[k] {
                        b'(' => depth += 1,
                        b')' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                if k < b.len() {
                    let buf = out.get_or_insert_with(|| b.to_vec());
                    for byte in &mut buf[start..=k] {
                        if *byte != 10 {
                            *byte = b' ';
                        }
                    }
                    j = k + 1;
                }
            }
        }
        i = j;
    }
    out
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// CUDA constructs that aren't IR-expressible -- their presence means the
/// kernel is hand-optimized (shared mem, atomics, barriers, library calls) and
/// belongs in the source language. Moved verbatim from `unpopped::convert`.
const CUDA_RESIDUE: &[&str] = &[
    "__shared__",
    "atomicAdd",
    "atomicCAS",
    "__syncthreads",
    "cublas",
    "cudnn",
    "printf",
    "asm",
    "cp.async",
    "__shfl",
];

/// CUDA: `__global__` kernels, `out`/`in{K}` buffers. Moved verbatim from
/// `unpopped::convert::CUDA` -- byte-identical field values, so every existing
/// CUDA lift is unaffected by the move.
pub const CUDA: Frontend =
    Frontend::new("cuda", "__global__", CUDA_RESIDUE, "out", "in", parse_cuda);
