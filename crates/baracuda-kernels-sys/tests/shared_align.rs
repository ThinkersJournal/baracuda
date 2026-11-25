// SPDX-License-Identifier: MIT OR Apache-2.0
//! Issue #420: `tree-sitter-cuda` 0.21 rejects `extern __shared__ __align__(alignof(T)) uint8_t smem[];`
//! (FlashInfer's dynamic-shared-memory idiom, `sampling.cuh`, 10 kernels). `parse_cuda` must accept it
//! with no ERROR node and without moving any byte offset of the source.
//! Lives here, next to the vendored `sampling.cuh` it reads (it was in `baracuda-cuda-parse/tests`, whose
//! packaged `.crate` cannot carry a file outside the crate), so it ships with the file it reads.

use baracuda_cuda_parse::parse_cuda;

fn has_error(src: &str) -> bool {
    parse_cuda(src)
        .expect("parser loads")
        .root_node()
        .has_error()
}

const ONE_LINE: &str = "template <typename TempStorage>\n__global__ void k(float* a) {\n  extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];\n  a[0] = 1.0f;\n}\n";
// sampling.cuh:787-789 shape: the declaration split over three lines, with nested template angle brackets.
const SPLIT: &str = "template <int B, int S, int R>\n__global__ void k(float* a) {\n  extern __shared__ __align__(\n      alignof(SamplingTempStorage<B, S, R>))\n      uint8_t smem_sampling[];\n  a[0] = 1.0f;\n}\n";

#[test]
fn extern_shared_align_one_line_parses_without_error() {
    assert!(!has_error(ONE_LINE));
}

#[test]
fn extern_shared_align_split_over_three_lines_parses_without_error() {
    assert!(!has_error(SPLIT));
}

#[test]
fn controls_plain_shared_parses_and_broken_source_still_errors() {
    // control 1: the same kernel without __align__ parses clean (the query can say "no error").
    assert!(!has_error(
        "__global__ void k(float* a) {\n  extern __shared__ uint8_t smem[];\n  a[0] = 1.0f;\n}\n"
    ));
    // control 2: has_error is live (genuinely broken source is still reported, the workaround hides nothing else).
    assert!(has_error("__global__ void k(float* a) { a[0] = ; }"));
}

#[test]
fn workaround_does_not_move_byte_offsets() {
    // The declaration's following statement must sit at the same byte offset the ORIGINAL source has.
    let tree = parse_cuda(ONE_LINE).expect("parser loads");
    let at = ONE_LINE.find("a[0] = 1.0f;").unwrap();
    let node = tree
        .root_node()
        .descendant_for_byte_range(at, at + 1)
        .unwrap();
    assert_eq!(&ONE_LINE[node.start_byte()..node.end_byte()].trim(), &"a");
}

/// The real file: no ERROR node may touch any line holding the `__align__` dynamic-smem declaration
/// (sampling.cuh:303, 430, 509, 739, 787-789). Other lines of the file are out of scope here.
#[test]
fn vendored_flashinfer_sampling_has_no_error_on_its_smem_declarations() {
    let src = include_str!("../vendor/flashinfer/include/flashinfer/sampling.cuh");
    let align_lines: Vec<usize> = src
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("__align__") && !l.trim_start().starts_with("//"))
        .map(|(i, _)| i)
        .collect();
    assert!(
        align_lines.len() >= 6,
        "control: expected the declaration lines, found {}",
        align_lines.len()
    );
    let tree = parse_cuda(src).expect("parser loads");
    let mut bad = Vec::new();
    let mut stack = vec![tree.root_node()];
    while let Some(n) = stack.pop() {
        if n.is_error() || n.is_missing() {
            let (s, e) = (n.start_position().row, n.end_position().row);
            // an ERROR node only counts when it is small enough to be the declaration itself (<= 4 lines)
            if e - s <= 4 && align_lines.iter().any(|l| (s..=e).contains(l)) {
                bad.push((s + 1, e + 1));
            }
        }
        let mut c = n.walk();
        stack.extend(n.children(&mut c));
    }
    assert!(
        bad.is_empty(),
        "ERROR/MISSING nodes on __align__ lines (1-based start,end): {bad:?}"
    );
}
