# R6 review: `baracuda-cutensor`

Reviewer: baracuda lane (R6 owner). Ref: `origin/main` `fb3c54018`. Items: the 103 rows for `baracuda-cutensor` in the §7.4 census, all `stable`, all `reach = public`,
all `non_exhaustive = 0`, all in `src/lib.rs` (1,564 lines). By `kind`: 52 methods, 24 variants, 14 structs, 7 fns, 4 enums, 2 types (52 + 24 + 14 + 7 + 4 + 2 = 103, counted from
`items.tsv` at this ref). The crate has no macro-generated items and no `pub` field. Method: counts and greps over `git ls-files` of `src/`, read the crate header, the four
enums, `Handle`, `Plan::contract`/`contract_trinary`, the `set_attribute` signatures and `Cargo.toml`. Function bodies beyond those were **not** read, nothing was built (cuTENSOR is not
installed on this box; `tests/gpu_smoke.rs` skips when it is absent). A verdict on the public shape.

## Verdict per group

| group | items | verdict | why |
|---|---|---|---|
| 4 enums (`DataType`, `UnaryOp`, `BinaryOp`, `WorkspaceKind`) and 24 variants | 28 | **freeze decision** | finding 1 |
| `Error`, `Result` | 2 | stable with a note | finding 2 |
| 21 `pub unsafe fn` (descriptor/op constructors, execute methods, attribute get/set) | 21 | stable with a note | finding 3 |
| 7 `as_raw` methods | 7 | stable with a note | finding 4 |
| `Plan::contract`, `reduce`, `elementwise_*`, `permute`, `contract_*` (7 execute methods; all `pub unsafe fn`, counted in the 21 above) | (21) | **freeze decision** | finding 5 |
| `Handle` and the other 13 structs, remaining methods, 7 free fns | the rest | stable with a note | findings 4, 6, 7 |

## Findings

1. **Four closed enums, 0 `#[non_exhaustive]`.** `DataType` (10 variants), `UnaryOp` (7), `BinaryOp` (4), `WorkspaceKind` (3) are plain enums mapping to `cutensor*` integer codes through a private `raw()`.
   cuTENSOR can add data types, operators and workspace tiers, and each addition is a breaking change to any caller's exhaustive `match`. `WorkspaceKind` does not derive `Eq`/`PartialEq`
   (the other three do), so it cannot be compared with `==`. Decision: `#[non_exhaustive]` on all four now, and add the missing derives.
2. **`Error` is not defined here.** `pub type Error = baracuda_core::Error<cutensorStatus_t>` (`lib.rs:115`) puts the `-sys` crate's status type into the public error type, so the freeze of
   this crate fixes the identity of `baracuda-core::Error` and of `cutensorStatus_t` from `baracuda-cutensor-sys` (a vendor-mirror crate). The `Result` alias has a defaulted second
   parameter. Same coupling as `baracuda-types` finding 5 and cutlass finding 3.
3. **21 `pub unsafe fn`, each with a `# Safety` section (21 sections, grep).** That includes all five operation constructors (`Contraction::new`, `Reduction::new`, `ElementwiseBinary::new`,
   `ElementwiseTrinary::new`, `Permutation::new`), `BlockSparseContraction::new`, `TrinaryContraction::new`, the attribute get/set methods and the execute methods. The contracts are one or two sentences
   ("all device pointers must be live, tensor-descriptor-conforming and aligned") and the execute methods defer to `Plan::contract` by "Same as". Constructors that need `unsafe` take
   `compute_desc: *const c_void`, a raw pointer where the crate has a `ComputeDescriptor` type (`lib.rs:398`) that could be passed by reference. 73 `*const`/`*mut c_void` occurrences in `src/` (grep).
4. **Raw handles escape through 7 `as_raw` methods** (`Handle`, `ComputeDescriptor`, `BlockSparseTensorDescriptor`, `TensorDescriptor`, `OperationDescriptor`, `PlanPreference`, `Plan`), each returning a
   `-sys` type (`cutensorHandle_t` and friends) with the doc "Use with care". Freezing them freezes the `-sys` type names; the escape hatch is deliberate and matches the other wrapper crates.
5. **Execute methods panic instead of returning `Err`.** `Plan::contract`, `reduce`, `elementwise_binary`, `elementwise_trinary`, `permute`, `contract_block_sparse` and `contract_trinary` each
   `assert_eq!(self.kind, OpKind::...)` (10 `assert_eq!`, 0 `assert!`, 0 `unwrap`/`expect`/`panic!` in `src/`, grep), so calling the wrong execute method on a plan aborts, although every one returns
   `Result`. The doc says "Aborts if `self` wasn't built from a `Contraction` descriptor". `OpKind` is private. Decision: keep the abort (document it as part of the frozen contract) or return an `Err` variant, which is
   observable behaviour and so a breaking change after the freeze.
6. **Thread-safety is one line.** `unsafe impl Send for Handle {}` (`lib.rs:286`), no `Sync`, no `unsafe impl` anywhere else (grep). The descriptors and `Plan<'h>` borrow `&'h Handle` and hold raw pointers, so by the
   auto-trait rules (not compiled) they are neither `Send` nor `Sync`. The cuTENSOR host API's thread-safety rules are not stated in the crate, and I did not look them up, so whether `Handle: Send`
   is correct is **not** checked here. No `assert_send` test exists (grep).
7. **Stringly numeric parameters.** Attribute ids are `i32` (`set_attribute(attr: i32, ...)` on 5 types) and 25 `modes_*: &[i32]` parameters (grep), so the Rust types do not distinguish a mode label from any other `i32`. Not
   wrong, but a typed `Attribute`/`Mode` would be a breaking change later.
8. **Census lines are current.** Loop over the 27 struct/enum/fn/type rows: 27 checked, 0 mismatched (the loop reads each row's line and looks for `struct|enum|fn|type <name>` on it). The methods and
   variants were not line-checked.
9. **No in-repo consumers.** No other crate's `Cargo.toml` depends on it (grep over `Cargo.toml` and `crates/*/Cargo.toml`: only the workspace entry); the `baracuda` facade does not list it.
   It is published (no `publish = false`), so its API is external-facing even though nothing here calls it.
10. **The header claim is not verified.** "wraps the full v2 host API surface" (crate docs) was not compared against the cuTENSOR header; `Cargo.toml` describes the crate as "Scaffolding at v0.1".
    Tests: 3 `#[test]`s, 2 in `contract_gemm.rs` and 1 in `gpu_smoke.rs`, none runnable without the library.

## Not decided here

- Whether the four enums become `#[non_exhaustive]` (finding 1) and whether `WorkspaceKind` gets `Eq`.
- Whether the execute methods keep aborting on a wrong `OpKind` (finding 5).
- Whether `Handle: Send` is sound (finding 6; needs the cuTENSOR threading rules).
- Not checked: method bodies, `OpKind`, doc-test coverage, whether the surface is "full v2".
