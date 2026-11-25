// SPDX-License-Identifier: MIT OR Apache-2.0
//! Kernel-layer lowering for the CUDA backend (ROADMAP row U20), **stages 2 and 3: the memory and control subset, then the four statements of board 434 C**.
//!
//! [`Cuda::lower_kernel`](crate::Cuda) lowers a [`KernelFn`] to one `extern "C" __global__` function. It runs
//! [`check_kernel`] itself (the input is data; the checker is the only gate) and then either spells the kernel
//! **exactly** or declines it **by name**, as `LowerError::UnsupportedKernel`. Nothing is approximated.
//!
//! # The argument block (ABI "B", PM ruling 2026-11-14)
//!
//! A buffer is addressed by logical coordinates *through its view* (shape, strides, base offset), so a launcher
//! has to pass those. The emitted function takes the buffer **pointers** first, in parameter order, then **one
//! by-value struct** `a` holding everything else:
//!
//! ```text
//! extern "C" __global__ void __launch_bounds__(256) baracuda_cuda_kernel_k(const float* p0, int* p1, Args_k a)
//! struct Args_k { long long b0_shape[R0]; long long b0_stride[R0]; long long b0_base; ...; unsigned int s2; };
//! ```
//!
//! Per buffer parameter `i` of rank `R`: `b<i>_shape` and `b<i>_stride` (`R` signed 64-bit element counts and
//! element strides) and `b<i>_base` (element offset added before address math). Per scalar parameter `i`:
//! `s<i>`. The struct is an **ordinary by-value kernel parameter**, so it works on every `sm` (Pascal included);
//! `__grid_constant__` (sm_70+) is deliberately not used. The exact layout (field order, offsets, size,
//! alignment) is attached as [`KernelInterface::params_block`], one field per array element (`b0_shape0`,
//! `b0_stride0`, ..., `b0_base`, `s2`), so a launcher builds the bytes from the interface, and the source
//! carries a `static_assert` on the size and the alignment (NVRTC has no constant `offsetof`: the offsets are C's natural sequential layout of the declared members, and the size and alignment check catches a wrong member size or count).
//!
//! The pointers plus the block must fit the classic 4 KB kernel-parameter limit and every buffer's rank must be
//! declared and at most [`MAX_RANK`]; otherwise the kernel is declined by name.
//!
//! # What stage 2 lowers
//!
//! `Let`, `Load`, `Assign`, `Store`, `If`, `For`, `While`, `Break`, `Continue`, `Return` and a `Block`-scope
//! `Barrier`; a launch shape (`blockIdx`/`threadIdx`/`blockDim`/`gridDim` for `Exec`, `__launch_bounds__`),
//! static `Shared` allocations (`__shared__`), and the expressions `Const`, `ConstInt`, `Local`, `Exec`,
//! `Add`/`Sub`/`Mul` (wrapping on integers), float `Div`, the six comparisons and `Cast`. Types: `F32`, `F64`,
//! `I32`, `U32` and `I64` for locals and every value; buffers may have any element type of the CUDA type table
//! but only the five above are loaded or stored. An out-of-bounds coordinate or `Shared` index, or a zero `For`
//! step, executes `__trap()`, as the IR's semantics trap.
//!
//! # Stage 3: the four statements ruled by board 434 C (NVIDIA only)
//!
//! `CopyAsync`, `AsyncCommit` and `AsyncWait` lower to `cp.async.ca.shared.global`, `cp.async.commit_group` and
//! `cp.async.wait_group N` (a `Shared` destination, a global source, 4, 8 or 16 bytes). They need **sm_80 or newer**:
//! the source carries an `#error` guard for an older `__CUDA_ARCH__` and the interface lists the requirement in
//! [`KernelInterface::requires`], so a consumer on Pascal declines before it compiles. Conditions the hardware needs that
//! the IR does not promise (the whole run in bounds, a unit innermost source stride, natural alignment of both addresses)
//! are checked and trap. `Collective` lowers at `Subgroup` scope only, as a warp of 32 with the full mask: `Shuffle`
//! (`Idx` any integer, `Up`/`Down`/`Xor` with a `u32` distance or mask, since a signed one means something else off the
//! ends of the warp), `Vote`, `Broadcast`, `Reduce` and `Scan` (inclusive, and exclusive for `Sum`/`And`/`Or`/`Xor`) with
//! `Sum` (wrapping), `Max`, `Min` on integers, `Sum` on floats (the order differs from the oracle's left-to-right fold, so
//! compare within a tolerance), and the integer bit ops. A float `Max`/`Min`, a `Block`-scope collective, a block that is
//! not whole warps, `CollectiveTuple`, `ReduceWith` and `ScanWith` decline by name. The emitted source says it lowers a
//! statement under a patent risk note: US11294713B2 and US12118382B2 (async copy and wait), US8539204B2 (barrier
//! aggregation), `docs/PATENT-SWEEP.md` hits 22 and 23; **not a clearance**. Targets other than NVIDIA decline these
//! statements by name in their own emitters; this crate has only the NVIDIA one.
//!
//! # Stage 4 (ROADMAP S1, slice 1): the Hopper synchronisation statements
//!
//! `MbarInit`, `MbarArrive` (with or without `expect_tx`), `MbarCompleteTx`, `MbarWait` (a `try_wait.parity` loop) and
//! `GridDepWait`/`GridDepLaunchDependents` lower to the PTX of the same names. They need **sm_90 or newer** (the source
//! carries an `#error` guard and the interface lists the requirement). The barrier is a `Shared` allocation of 64-bit
//! elements (`I64`: `U64` has no CUDA spelling in this crate's typed table). An out-of-range count, transaction byte count
//! or a parity other than 0 or 1 traps. `CopyBulk` (1-D `cp.async.bulk` global to `Shared`, completing on an `mbarrier`)
//! lowers too, with the run, alignment and 16-byte multiple checked. `Wgmma*`, `Tcgen05*`, `MatrixLoad`/`MatrixStore`
//! (they need `Private` allocations and 16-bit float element types, which this crate does not lower yet), named and
//! `Cluster` barriers and the tensor-map statements are still declined by name (ROADMAP S1 continues).
//!
//! Everything else is declined by name: any other statement (named by its variant, so a new `Stmt` variant is
//! named too), any other expression node, a `Subgroup`, `Cluster` or `Device` barrier, a cluster or multi-device
//! launch, a dynamic or non-`Shared` allocation, a result local (a callee), a packed local, `ClusterPeer`,
//! `Peer` and `Multicast` places.

use unpopped::backend::{GeneratedKernel, LowerError, Spelling};
use unpopped::exec::{ExecId, Extent, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CacheHint, CollectiveKind, KernelFn, ParamKind, Place, ShuffleMode, Stmt,
    VoteKind, check_kernel,
};
use unpopped::kernel_interface::{KernelInterface, LaunchBounds, ParamsBlock, ParamsBlockField};
use unpopped::types::{Ty, infer_ty_in};
use unpopped_vocab::ElementKind;

use crate::typed::{self, type_name};

/// The largest buffer rank the argument block carries (an over-rank kernel is declined by name).
pub(crate) const MAX_RANK: u8 = 4;

/// The classic CUDA kernel-parameter limit in bytes (pointers plus the by-value argument block).
const MAX_PARAM_BYTES: u32 = 4096;

fn refuse(detail: impl Into<String>) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: detail.into(),
    }
}

/// The leading identifier of a `Debug` form (`Let`, `AsyncWait`, `Cast`, ...), so a variant added to `Stmt` or
/// `ScalarExpr` later is still named in the decline instead of falling into a wildcard.
fn variant_name(debug: &str) -> String {
    debug
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .next()
        .unwrap_or_default()
        .to_string()
}

/// The variant name of a statement (`Let`, `AsyncWait`, ...).
pub(crate) fn stmt_name(s: &Stmt) -> String {
    variant_name(&format!("{s:?}"))
}

/// `x` as a C identifier fragment.
fn ident(x: &str) -> String {
    let s: String = x
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.is_empty() { "_".into() } else { s }
}

/// The five types stage 2 computes in.
fn computed(t: ElementKind) -> bool {
    matches!(
        t,
        ElementKind::F32
            | ElementKind::F64
            | ElementKind::I32
            | ElementKind::U32
            | ElementKind::I64
    )
}

fn is_float(t: ElementKind) -> bool {
    matches!(t, ElementKind::F32 | ElementKind::F64)
}

fn is_signed_int(t: ElementKind) -> bool {
    matches!(t, ElementKind::I32 | ElementKind::I64)
}

/// The unsigned C type of the same width, for wrapping signed arithmetic.
fn unsigned_of(t: ElementKind) -> &'static str {
    match t {
        ElementKind::I64 => "unsigned long long",
        _ => "unsigned int",
    }
}

/// Size and alignment in bytes of a field of `t` in the argument block (the C types of [`type_name`]).
fn size_align(t: ElementKind) -> Option<u32> {
    Some(match t {
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::I32 | ElementKind::U32 => 4,
        ElementKind::F64 | ElementKind::I64 => 8,
        ElementKind::I8 | ElementKind::U8 => 1,
        ElementKind::I16 | ElementKind::U16 => 2,
        _ => return None,
    })
}

/// One member of the argument block, in declaration order.
struct Field {
    /// C declaration text, e.g. `long long b0_shape[2];`.
    decl: String,
    /// Interface fields this member stands for: `(name, element type, byte offset)`.
    parts: Vec<(String, ElementKind, u32)>,
}

/// The laid-out argument block.
struct ArgBlock {
    fields: Vec<Field>,
    size: u32,
    align: u32,
}

fn round_up(x: u32, a: u32) -> u32 {
    x.div_ceil(a) * a
}

/// Lay the block out in C's natural order: sequential members, each at the next multiple of its alignment.
fn layout(k: &KernelFn) -> Result<ArgBlock, LowerError> {
    let mut fields = Vec::new();
    let mut at = 0u32;
    let mut align = 1u32;
    let place = |at: &mut u32, align: &mut u32, a: u32, bytes: u32| -> u32 {
        *at = round_up(*at, a);
        let o = *at;
        *at += bytes;
        *align = (*align).max(a);
        o
    };
    for (i, p) in k.params.iter().enumerate() {
        match &p.kind {
            ParamKind::Buffer { rank, .. } => {
                let r = match rank {
                    Some(r) if (1..=MAX_RANK).contains(r) => u32::from(*r),
                    Some(r) => {
                        return Err(refuse(format!(
                            "kernel `{}`: buffer `{}` has rank {r}, above the argument block's bound of {MAX_RANK}",
                            k.name, p.name
                        )));
                    }
                    None => {
                        return Err(refuse(format!(
                            "kernel `{}`: buffer `{}` has no declared rank, which the argument block's shape and stride arrays need",
                            k.name, p.name
                        )));
                    }
                };
                for arr in ["shape", "stride"] {
                    let o = place(&mut at, &mut align, 8, 8 * r);
                    fields.push(Field {
                        decl: format!("long long b{i}_{arr}[{r}];"),
                        parts: (0..r)
                            .map(|d| (format!("b{i}_{arr}{d}"), ElementKind::I64, o + 8 * d))
                            .collect(),
                    });
                }
                let o = place(&mut at, &mut align, 8, 8);
                fields.push(Field {
                    decl: format!("long long b{i}_base;"),
                    parts: vec![(format!("b{i}_base"), ElementKind::I64, o)],
                });
            }
            ParamKind::Scalar { ty, .. } => {
                let (t, a) = type_name(*ty).zip(size_align(*ty)).ok_or_else(|| {
                    refuse(format!(
                        "kernel `{}`: scalar `{}` of type {ty:?} has no CUDA spelling",
                        k.name, p.name
                    ))
                })?;
                let o = place(&mut at, &mut align, a, a);
                fields.push(Field {
                    decl: format!("{t} s{i};"),
                    parts: vec![(format!("s{i}"), *ty, o)],
                });
            }
            _ => {
                return Err(refuse(format!(
                    "kernel `{}`: parameter `{}` is a kind this backend has not been taught",
                    k.name, p.name
                )));
            }
        }
    }
    if fields.is_empty() {
        // An empty C struct has size 1 in CUDA C++; make that explicit so the layout is not a guess.
        fields.push(Field {
            decl: "unsigned char _unused;".into(),
            parts: vec![("_unused".into(), ElementKind::U8, 0)],
        });
        at = 1;
    }
    let size = round_up(at, align);
    Ok(ArgBlock {
        fields,
        size,
        align,
    })
}

/// The expression and statement lowerer for one kernel.
struct Cx<'a> {
    k: &'a KernelFn,
    tys: Vec<Ty>,
    /// `Some(member)` for a local bound to a scalar parameter.
    scalar_local: Vec<Option<String>>,
    out: String,
    depth: usize,
    fresh: u32,
    /// The kernel uses `cp.async` (needs sm_80 or newer).
    async_used: bool,
    /// The kernel lowers a statement under a patent risk note (board 434 C).
    flagged: bool,
    /// A statement spelled here needs `sm_90` (mbarrier `expect_tx`/`complete_tx`/`try_wait`, `griddepcontrol`).
    sm90_used: bool,
}

impl Cx<'_> {
    fn line(&mut self, text: &str) {
        for _ in 0..self.depth {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn tmp(&mut self, what: &str) -> String {
        self.fresh += 1;
        format!("_{what}{}", self.fresh)
    }

    fn scalar_ty(&self, e: &ScalarExpr) -> Result<ElementKind, LowerError> {
        let ty = infer_ty_in(e, self.k.compute, &self.tys).map_err(|er| {
            refuse(format!(
                "kernel `{}`: ill-typed expression: {er:?}",
                self.k.name
            ))
        })?;
        match ty {
            Ty::Scalar(t) if computed(t) => Ok(t),
            Ty::Scalar(t) => Err(refuse(format!(
                "kernel `{}`: a value of type {t:?} is not lowered yet (stage 2 computes in f32, f64, i32, u32 and i64)",
                self.k.name
            ))),
            other => Err(refuse(format!(
                "kernel `{}`: a packed value ({other:?}) is not lowered yet",
                self.k.name
            ))),
        }
    }

    fn local_name(&self, i: u32) -> String {
        match &self.scalar_local[i as usize] {
            Some(m) => format!("a.{m}"),
            None => format!("l{i}"),
        }
    }

    fn local_ctype(&self, i: u32) -> Result<String, LowerError> {
        let d = self.k.locals[i as usize];
        if d.lanes != 1 || !computed(d.elem) {
            return Err(refuse(format!(
                "kernel `{}`: local {i} ({:?}, {} lane(s)) is not lowered yet (stage 2 computes in f32, f64, i32, u32 and i64)",
                self.k.name, d.elem, d.lanes
            )));
        }
        type_name(d.elem).ok_or_else(|| {
            refuse(format!(
                "kernel `{}`: no CUDA spelling for local {i}",
                self.k.name
            ))
        })
    }

    /// Spell `e` exactly, or decline it by node name.
    fn ex(&self, e: &ScalarExpr) -> Result<String, LowerError> {
        let t = self.scalar_ty(e)?;
        let spelled = |s: Result<Spelling, LowerError>| -> Result<String, LowerError> {
            s?.spelled().map_err(LowerError::from)
        };
        Ok(match e {
            ScalarExpr::Const(v) => match self.k.compute {
                ElementKind::F32 => spelled(typed::const_f32(*v))?,
                ElementKind::F64 => spelled(typed::const_default(*v))?,
                c => {
                    return Err(refuse(format!(
                        "kernel `{}`: an untyped `Const` at compute type {c:?} is not lowered (only f32 and f64 compute)",
                        self.k.name
                    )));
                }
            },
            ScalarExpr::ConstInt(v, ty) => const_int(*v, *ty, &self.k.name)?,
            ScalarExpr::Local(i) => self.local_name(*i),
            ScalarExpr::Exec(id) => exec_id(*id, &self.k.name)?,
            ScalarExpr::Add(a, b) => self.arith(a, b, t, "+")?,
            ScalarExpr::Sub(a, b) => self.arith(a, b, t, "-")?,
            ScalarExpr::Mul(a, b) => self.arith(a, b, t, "*")?,
            ScalarExpr::Div(a, b) => {
                if !is_float(t) {
                    return Err(refuse(format!(
                        "kernel `{}`: integer `Div` is not lowered yet (division by zero and overflow have a defined trap in the IR)",
                        self.k.name
                    )));
                }
                format!("({} / {})", self.ex(a)?, self.ex(b)?)
            }
            ScalarExpr::Binary(op, a, b) if op.is_cmp() => {
                let sym = match op {
                    BinaryOp::CmpEq => "==",
                    BinaryOp::CmpNe => "!=",
                    BinaryOp::CmpLt => "<",
                    BinaryOp::CmpLe => "<=",
                    BinaryOp::CmpGt => ">",
                    _ => ">=",
                };
                let c = type_name(self.k.compute).ok_or_else(|| {
                    refuse(format!(
                        "kernel `{}`: no CUDA spelling for the compute type",
                        self.k.name
                    ))
                })?;
                format!("(({c})(({}) {sym} ({})))", self.ex(a)?, self.ex(b)?)
            }
            ScalarExpr::Cast(x, spec) => {
                let from = self.scalar_ty(x)?;
                spelled(typed::cast(self.ex(x)?, from, *spec))?
            }
            other => {
                return Err(refuse(format!(
                    "kernel `{}`: expression node `{}` is not lowered by the CUDA backend yet",
                    self.k.name,
                    variant_name(&format!("{other:?}"))
                )));
            }
        })
    }

    fn arith(
        &self,
        a: &ScalarExpr,
        b: &ScalarExpr,
        t: ElementKind,
        op: &str,
    ) -> Result<String, LowerError> {
        let (x, y) = (self.ex(a)?, self.ex(b)?);
        Ok(if is_signed_int(t) {
            // C signed overflow is undefined; the IR wraps. Compute in the unsigned type and convert back.
            let (c, u) = (type_name(t).unwrap_or_default(), unsigned_of(t));
            format!("(({c})(({u})({x}) {op} ({u})({y})))")
        } else {
            format!("({x} {op} {y})")
        })
    }

    /// The C lvalue of `place`, after emitting its address math and bounds checks (a `{ ... }` scope is the caller's).
    fn place(&mut self, place: &Place) -> Result<String, LowerError> {
        match place {
            Place::Logical { buffer, coords } => {
                let rank = match &self.k.params[*buffer as usize].kind {
                    ParamKind::Buffer { rank, .. } => *rank,
                    _ => None,
                };
                if rank.is_some_and(|r| usize::from(r) != coords.len()) {
                    return Err(refuse(format!(
                        "kernel `{}`: place on buffer {buffer} has {} coordinates, its declared rank is {}",
                        self.k.name,
                        coords.len(),
                        rank.unwrap_or(0)
                    )));
                }
                let o = self.tmp("o");
                self.line(&format!("long long {o} = a.b{buffer}_base;"));
                for (d, c) in coords.iter().enumerate() {
                    let v = self.ex(c)?;
                    self.line(&format!(
                        "{{ long long c = (long long)({v}); if (c < 0 || c >= a.b{buffer}_shape[{d}]) __trap(); {o} += c * a.b{buffer}_stride[{d}]; }}"
                    ));
                }
                Ok(format!("p{buffer}[{o}]"))
            }
            Place::Alloc { alloc, index } => {
                let decl = &self.k.allocs[*alloc as usize];
                let Extent::Static(n) = decl.extent else {
                    return Err(refuse(format!(
                        "kernel `{}`: allocation `{}` has a dynamic extent (not lowered yet)",
                        self.k.name, decl.name
                    )));
                };
                let v = self.ex(index)?;
                let o = self.tmp("i");
                self.line(&format!("long long {o} = (long long)({v});"));
                self.line(&format!("if ({o} < 0 || {o} >= {n}) __trap();"));
                Ok(format!("sh{alloc}[{o}]"))
            }
            other => Err(refuse(format!(
                "kernel `{}`: place `{}` is not lowered yet",
                self.k.name,
                variant_name(&format!("{other:?}"))
            ))),
        }
    }

    /// `cp.async` of `elems` consecutive elements, global to `Shared` (PTX `cp.async.ca`; sm_80 or newer). Exact, so every
    /// condition the hardware needs that the IR does not guarantee is checked and traps: bounds of the whole run, a unit
    /// innermost stride on the source, and the natural alignment of both addresses.
    /// `cp.async` of `elems` elements (ROADMAP U13). `valid` (`src-size`): only that many source elements are read and the rest
    /// of the destination is zero-filled by the hardware; `0` reads nothing, so the source place is then NOT evaluated (it
    /// may point anywhere, even out of range). `cache`: `.ca` (default) or `.cg` (16 byte copies only). `l2_prefetch`:
    /// `.L2::64B`, `128B` or `256B`.
    fn copy_async(
        &mut self,
        dst: &Place,
        src: &Place,
        elems: u32,
        valid: Option<&ScalarExpr>,
        cache: CacheHint,
        l2_prefetch: Option<u32>,
    ) -> Result<(), LowerError> {
        let (Place::Alloc { alloc, index }, Place::Logical { buffer, coords }) = (dst, src) else {
            return Err(refuse(format!(
                "kernel `{}`: `CopyAsync` between these places is not lowered (a `Shared` allocation from a global buffer is)",
                self.k.name
            )));
        };
        let elem = self.k.allocs[*alloc as usize].elem;
        let esize = size_align(elem).unwrap_or(0);
        let bytes = esize * elems;
        if !matches!(bytes, 4 | 8 | 16) {
            return Err(refuse(format!(
                "kernel `{}`: `CopyAsync` of {elems} x {elem:?} is {bytes} bytes (cp.async moves 4, 8 or 16)",
                self.k.name
            )));
        }
        let op = match cache {
            CacheHint::Auto | CacheHint::All => "ca",
            CacheHint::Global if bytes == 16 => "cg",
            CacheHint::Global => {
                return Err(refuse(format!(
                    "kernel `{}`: `CopyAsync` with the `.cg` cache operator copies 16 bytes only, this is {bytes}",
                    self.k.name
                )));
            }
            _ => {
                return Err(refuse(format!(
                    "kernel `{}`: `CopyAsync` cache hint {cache:?} is not lowered",
                    self.k.name
                )));
            }
        };
        let prefetch = match l2_prefetch {
            None => String::new(),
            Some(n @ (64 | 128 | 256)) => format!(".L2::{n}B"),
            Some(n) => {
                return Err(refuse(format!(
                    "kernel `{}`: `CopyAsync` L2 prefetch of {n} bytes (cp.async takes 64, 128 or 256)",
                    self.k.name
                )));
            }
        };
        let Extent::Static(n) = self.k.allocs[*alloc as usize].extent else {
            return Err(refuse(format!(
                "kernel `{}`: `CopyAsync` into a dynamic allocation is not lowered yet",
                self.k.name
            )));
        };
        self.async_used = true;
        self.flagged = true;
        let tn = type_name(elem).unwrap_or_default();
        self.line("{");
        self.depth += 1;
        if let Some(v) = valid {
            // Predicated copy: `_v` source elements are read, the rest of the destination is zero-filled by `src-size`.
            let s = self.place(dst)?;
            let ve = self.ex(v)?;
            self.line(&format!("long long _v = (long long)({ve});"));
            self.line(&format!("if (_v < 0 || _v > {elems}) __trap();"));
            self.line(&format!("unsigned _n = (unsigned)_v * {esize}u;"));
            self.line(&format!(
                "const {tn}* _g = &p{buffer}[0]; unsigned _s = (unsigned)__cvta_generic_to_shared(&{s});"
            ));
            self.line(&format!("if (_s & {}) __trap();", bytes - 1));
            // `_v == 0` reads nothing: the source place is not evaluated (it may be out of range).
            self.line("if (_v > 0) {");
            self.depth += 1;
            let g = self.place(src)?;
            let last = coords.len() - 1;
            let c = self.ex(&coords[last])?;
            self.line(&format!(
                "if (a.b{buffer}_stride[{last}] != 1 || (long long)({c}) + _v > a.b{buffer}_shape[{last}]) __trap();"
            ));
            self.line(&format!("_g = &{g};"));
            self.line(&format!(
                "if (((unsigned long long)_g) & {}) __trap();",
                bytes - 1
            ));
            self.depth -= 1;
            self.line("}");
            let i = self.ex(index)?;
            self.line(&format!("if ((long long)({i}) + {elems} > {n}) __trap();"));
            self.line(&format!(
                "asm volatile(\"cp.async.{op}.shared.global{prefetch} [%0], [%1], %2, %3;\" :: \"r\"(_s), \"l\"(_g), \"n\"({bytes}), \"r\"(_n) : \"memory\");"
            ));
        } else {
            let g = self.place(src)?;
            let s = self.place(dst)?;
            if elems > 1 {
                let last = coords.len() - 1;
                let c = self.ex(&coords[last])?;
                self.line(&format!(
                    "if (a.b{buffer}_stride[{last}] != 1 || (long long)({c}) + {elems} > a.b{buffer}_shape[{last}]) __trap();"
                ));
                let i = self.ex(index)?;
                self.line(&format!("if ((long long)({i}) + {elems} > {n}) __trap();"));
            }
            self.line(&format!(
                "const {tn}* _g = &{g}; unsigned _s = (unsigned)__cvta_generic_to_shared(&{s});"
            ));
            self.line(&format!(
                "if ((((unsigned long long)_g) | (unsigned long long)_s) & {}) __trap();",
                bytes - 1
            ));
            self.line(&format!(
                "asm volatile(\"cp.async.{op}.shared.global{prefetch} [%0], [%1], %2;\" :: \"r\"(_s), \"l\"(_g), \"n\"({bytes}) : \"memory\");"
            ));
        }
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// A `Subgroup`-scope collective as a warp collective with the full mask (the subgroup of an NVIDIA target is a warp of 32).
    fn collective(
        &mut self,
        kind: &CollectiveKind,
        value: &ScalarExpr,
        result: u32,
        scope: Scope,
    ) -> Result<(), LowerError> {
        let name = &self.k.name;
        if !matches!(scope, Scope::Subgroup) {
            return Err(refuse(format!(
                "kernel `{name}`: a {scope:?}-scope `Collective` is not lowered yet (stage 3 lowers the subgroup, a warp of 32)"
            )));
        }
        let threads = self.k.launch.map_or(0, |l| l.block.iter().product::<u32>());
        if threads == 0 || !threads.is_multiple_of(32) {
            return Err(refuse(format!(
                "kernel `{name}`: a subgroup `Collective` needs a block of a multiple of 32 threads, got {threads}"
            )));
        }
        let t = self.scalar_ty(value)?;
        let tn = type_name(t).unwrap_or_default();
        let v = self.ex(value)?;
        let m = "0xffffffffu";
        let lane = "((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) & 31u)";
        // `x op y` for a fold, wrapping on signed integers.
        let fold = |op: ReduceOp, x: &str, y: &str| -> Result<String, LowerError> {
            let sym = match op {
                ReduceOp::Sum => Some("+"),
                ReduceOp::And => Some("&"),
                ReduceOp::Or => Some("|"),
                ReduceOp::Xor => Some("^"),
                _ => None,
            };
            Ok(match (op, sym) {
                (ReduceOp::Max | ReduceOp::Min, _) if is_float(t) => {
                    return Err(refuse(format!(
                        "kernel `{name}`: a float `{op:?}` collective is not lowered yet (its NaN behaviour is not pinned down)"
                    )));
                }
                (ReduceOp::Max, _) => format!("(({x}) > ({y}) ? ({x}) : ({y}))"),
                (ReduceOp::Min, _) => format!("(({x}) < ({y}) ? ({x}) : ({y}))"),
                (_, Some(s)) if is_signed_int(t) && matches!(op, ReduceOp::Sum) => {
                    let u = unsigned_of(t);
                    format!("(({tn})(({u})({x}) {s} ({u})({y})))")
                }
                (_, Some(s)) => format!("(({x}) {s} ({y}))"),
                _ => {
                    return Err(refuse(format!(
                        "kernel `{name}`: a collective fold `{op:?}` is not lowered"
                    )));
                }
            })
        };
        let unsigned_operand = |e: &ScalarExpr| -> Result<String, LowerError> {
            if self.scalar_ty(e)? != ElementKind::U32 {
                return Err(refuse(format!(
                    "kernel `{name}`: a shuffle distance or mask must be a u32 (a signed one has a different meaning off the ends of the warp)"
                )));
            }
            self.ex(e)
        };
        let r = format!("l{result}");
        let mut body = Vec::new();
        let rt = match kind {
            CollectiveKind::Shuffle { mode, operand } => {
                let call = match mode {
                    ShuffleMode::Idx => {
                        format!("__shfl_sync({m}, _v, (int)({}), 32)", self.ex(operand)?)
                    }
                    ShuffleMode::Up => {
                        format!(
                            "__shfl_up_sync({m}, _v, {}, 32)",
                            unsigned_operand(operand)?
                        )
                    }
                    ShuffleMode::Down => {
                        format!(
                            "__shfl_down_sync({m}, _v, {}, 32)",
                            unsigned_operand(operand)?
                        )
                    }
                    ShuffleMode::Xor => format!(
                        "__shfl_xor_sync({m}, _v, (int)({}), 32)",
                        unsigned_operand(operand)?
                    ),
                    other => {
                        return Err(refuse(format!(
                            "kernel `{name}`: shuffle mode `{other:?}` is not lowered"
                        )));
                    }
                };
                body.push(format!("{r} = {call};"));
                tn.clone()
            }
            CollectiveKind::Broadcast { lane: l } => {
                body.push(format!(
                    "{r} = __shfl_sync({m}, _v, (int)({}), 32);",
                    self.ex(l)?
                ));
                tn.clone()
            }
            CollectiveKind::Vote(vk) => {
                let f = match vk {
                    VoteKind::All => format!("(unsigned int)__all_sync({m}, _v != 0)"),
                    VoteKind::Any => format!("(unsigned int)__any_sync({m}, _v != 0)"),
                    VoteKind::Ballot => format!("__ballot_sync({m}, _v != 0)"),
                    other => {
                        return Err(refuse(format!(
                            "kernel `{name}`: vote `{other:?}` is not lowered"
                        )));
                    }
                };
                body.push(format!("{r} = {f};"));
                "unsigned int".to_string()
            }
            CollectiveKind::Reduce(op) => {
                // Butterfly: every lane ends with the fold of the whole warp.
                body.push(format!(
                    "for (int _o = 16; _o > 0; _o >>= 1) {{ {tn} _n = __shfl_xor_sync({m}, _v, _o, 32); _v = {}; }}",
                    fold(*op, "_v", "_n")?
                ));
                body.push(format!("{r} = _v;"));
                tn.clone()
            }
            CollectiveKind::Scan { op, exclusive } => {
                // Hillis-Steele inclusive scan by `shfl_up`; an exclusive one shifts it by a lane and gives lane 0 the identity.
                body.push(format!(
                    "for (unsigned _o = 1u; _o < 32u; _o <<= 1) {{ {tn} _n = __shfl_up_sync({m}, _v, _o, 32); if ({lane} >= _o) _v = {}; }}",
                    fold(*op, "_v", "_n")?
                ));
                if *exclusive {
                    let ident = match op {
                        ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor => format!("({tn})0"),
                        ReduceOp::And => format!("({tn})(~({tn})0)"),
                        _ => {
                            return Err(refuse(format!(
                                "kernel `{name}`: an exclusive `{op:?}` scan has no identity"
                            )));
                        }
                    };
                    body.push(format!(
                        "{{ {tn} _x = __shfl_up_sync({m}, _v, 1u, 32); _v = ({lane} == 0u) ? {ident} : _x; }}"
                    ));
                }
                body.push(format!("{r} = _v;"));
                tn.clone()
            }
            other => {
                return Err(refuse(format!(
                    "kernel `{name}`: collective `{}` is not lowered yet",
                    variant_name(&format!("{other:?}"))
                )));
            }
        };
        self.flagged = true;
        self.line(&format!("{rt} {r};"));
        self.line("{");
        self.depth += 1;
        self.line(&format!("{tn} _v = ({tn})({v});"));
        for l in body {
            self.line(&l);
        }
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// The shared-window address of the `mbarrier` object at `bar` (a place of a `Shared` 64-bit allocation), as `_b`.
    fn mbar_addr(&mut self, bar: &Place) -> Result<(), LowerError> {
        let s = self.place(bar)?;
        self.line(&format!(
            "unsigned _b = (unsigned)__cvta_generic_to_shared(&{s});"
        ));
        Ok(())
    }

    /// `mbarrier.init` (PTX `mbarrier`; NVIDIA only, `sm_90` for the rest of the family). `count` below 1 traps, as the IR's does.
    fn mbar_init(&mut self, bar: &Place, count: &ScalarExpr) -> Result<(), LowerError> {
        self.sm90_used = true;
        self.flagged = true;
        let c = self.ex(count)?;
        self.line("{");
        self.depth += 1;
        self.mbar_addr(bar)?;
        self.line(&format!("long long _c = (long long)({c});"));
        self.line("if (_c < 1 || _c > 1048575) __trap();");
        self.line("asm volatile(\"mbarrier.init.shared::cta.b64 [%0], %1;\" :: \"r\"(_b), \"r\"((unsigned)_c) : \"memory\");");
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// `mbarrier.arrive` (with `expect_tx` when `tx` is given); the returned state is not used, so the sink `_`.
    fn mbar_arrive(&mut self, bar: &Place, tx: Option<&ScalarExpr>) -> Result<(), LowerError> {
        self.sm90_used = true;
        self.flagged = true;
        let t = tx.map(|e| self.ex(e)).transpose()?;
        self.line("{");
        self.depth += 1;
        self.mbar_addr(bar)?;
        if let Some(t) = t {
            self.line(&format!("long long _t = (long long)({t});"));
            self.line("if (_t < 0 || _t > 1048575) __trap();");
            self.line("asm volatile(\"mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\" :: \"r\"(_b), \"r\"((unsigned)_t) : \"memory\");");
        } else {
            self.line("asm volatile(\"mbarrier.arrive.shared::cta.b64 _, [%0];\" :: \"r\"(_b) : \"memory\");");
        }
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// `mbarrier.complete_tx`: the bytes an asynchronous copy delivered.
    fn mbar_complete_tx(&mut self, bar: &Place, bytes: &ScalarExpr) -> Result<(), LowerError> {
        self.sm90_used = true;
        self.flagged = true;
        let n = self.ex(bytes)?;
        self.line("{");
        self.depth += 1;
        self.mbar_addr(bar)?;
        self.line(&format!("long long _n = (long long)({n});"));
        self.line("if (_n < 0 || _n > 1048575) __trap();");
        self.line("asm volatile(\"mbarrier.complete_tx.shared::cta.b64 [%0], %1;\" :: \"r\"(_b), \"r\"((unsigned)_n) : \"memory\");");
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// `cp.async.bulk` (1-D, global to `Shared`, completing on an `mbarrier`; `sm_90`). `elems` x element size is a multiple of 16 bytes.
    /// Every condition the hardware needs that the IR does not guarantee is checked and traps: the whole source run in bounds on a
    /// unit innermost stride, the destination run inside the allocation, and 16-byte alignment of both addresses and the barrier's 8.
    fn copy_bulk(
        &mut self,
        dst: &Place,
        src: &Place,
        elems: u32,
        bar: &Place,
    ) -> Result<(), LowerError> {
        let (Place::Alloc { alloc, index }, Place::Logical { buffer, coords }) = (dst, src) else {
            return Err(refuse(format!(
                "kernel `{}`: `CopyBulk` between these places is not lowered (a `Shared` allocation from a global buffer is)",
                self.k.name
            )));
        };
        let elem = self.k.allocs[*alloc as usize].elem;
        let esize = size_align(elem).unwrap_or(0);
        let bytes = esize * elems;
        if bytes == 0 || bytes % 16 != 0 {
            return Err(refuse(format!(
                "kernel `{}`: `CopyBulk` of {elems} x {elem:?} is {bytes} bytes (cp.async.bulk moves a multiple of 16)",
                self.k.name
            )));
        }
        let Extent::Static(n) = self.k.allocs[*alloc as usize].extent else {
            return Err(refuse(format!(
                "kernel `{}`: `CopyBulk` into a dynamic allocation is not lowered yet",
                self.k.name
            )));
        };
        self.sm90_used = true;
        self.flagged = true;
        let tn = type_name(elem).unwrap_or_default();
        self.line("{");
        self.depth += 1;
        let g = self.place(src)?;
        let s = self.place(dst)?;
        let last = coords.len() - 1;
        let c = self.ex(&coords[last])?;
        self.line(&format!(
            "if (a.b{buffer}_stride[{last}] != 1 || (long long)({c}) + {elems} > a.b{buffer}_shape[{last}]) __trap();"
        ));
        let i = self.ex(index)?;
        self.line(&format!("if ((long long)({i}) + {elems} > {n}) __trap();"));
        self.line(&format!(
            "const {tn}* _g = &{g}; unsigned _s = (unsigned)__cvta_generic_to_shared(&{s});"
        ));
        self.line(&format!(
            "if ((((unsigned long long)_g) | (unsigned long long)_s) & 15) __trap();"
        ));
        self.mbar_addr(bar)?;
        self.line(&format!(
            "asm volatile(\"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\" :: \"r\"(_s), \"l\"(_g), \"n\"({bytes}), \"r\"(_b) : \"memory\");"
        ));
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    /// `mbarrier.try_wait.parity` in a loop until the phase of that parity has completed. The parity is 0 or 1 (anything else traps).
    fn mbar_wait(&mut self, bar: &Place, parity: &ScalarExpr) -> Result<(), LowerError> {
        self.sm90_used = true;
        self.flagged = true;
        let p = self.ex(parity)?;
        self.line("{");
        self.depth += 1;
        self.mbar_addr(bar)?;
        self.line(&format!("long long _p = (long long)({p});"));
        self.line("if (_p != 0 && _p != 1) __trap();");
        self.line("unsigned _w = 0u;");
        self.line("while (_w == 0u) {");
        self.depth += 1;
        self.line("asm volatile(\"{ .reg .pred q; mbarrier.try_wait.parity.shared::cta.b64 q, [%1], %2; selp.u32 %0, 1, 0, q; }\" : \"=r\"(_w) : \"r\"(_b), \"r\"((unsigned)_p) : \"memory\");");
        self.depth -= 1;
        self.line("}");
        self.depth -= 1;
        self.line("}");
        Ok(())
    }

    fn block(&mut self, b: &Block) -> Result<(), LowerError> {
        for s in &b.0 {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn scope(&mut self, b: &Block) -> Result<(), LowerError> {
        self.depth += 1;
        let r = self.block(b);
        self.depth -= 1;
        r
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::Let { local, init } => {
                let t = self.local_ctype(*local)?;
                let v = self.ex(init)?;
                self.line(&format!("{t} l{local} = ({t})({v});"));
            }
            Stmt::Assign { local, value } => {
                if self.scalar_local[*local as usize].is_some() {
                    return Err(refuse(format!(
                        "kernel `{}`: assignment to a parameter local",
                        self.k.name
                    )));
                }
                let t = self.local_ctype(*local)?;
                let v = self.ex(value)?;
                self.line(&format!("l{local} = ({t})({v});"));
            }
            Stmt::Load { local, place } => {
                let t = self.local_ctype(*local)?;
                self.line(&format!("{t} l{local};"));
                self.line("{");
                self.depth += 1;
                let lv = self.place(place)?;
                self.line(&format!("l{local} = ({t})({lv});"));
                self.depth -= 1;
                self.line("}");
            }
            Stmt::Store { place, value } => {
                let ty = self.scalar_ty(value)?;
                let t = type_name(ty).unwrap_or_default();
                let v = self.ex(value)?;
                self.line("{");
                self.depth += 1;
                let lv = self.place(place)?;
                self.line(&format!("{lv} = ({t})({v});"));
                self.depth -= 1;
                self.line("}");
            }
            Stmt::If { cond, then, els } => {
                let c = self.ex(cond)?;
                self.line(&format!("if (({c}) != 0) {{"));
                self.scope(then)?;
                if els.0.is_empty() {
                    self.line("}");
                } else {
                    self.line("} else {");
                    self.scope(els)?;
                    self.line("}");
                }
            }
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
                // `unroll` is a non-semantic hint (ROADMAP U18) that this emitter does not emit yet; named, not `..`,
                // so the next field unpopped adds to `Stmt::For` is a compile error here again.
                unroll: _,
            } => {
                let t = self.local_ctype(*local)?;
                let ety = self.k.locals[*local as usize].elem;
                let (a, b, c) = (self.tmp("s"), self.tmp("e"), self.tmp("t"));
                let (vs, ve, vt) = (self.ex(start)?, self.ex(end)?, self.ex(step)?);
                self.line("{");
                self.depth += 1;
                self.line(&format!(
                    "{t} {a} = ({t})({vs}); {t} {b} = ({t})({ve}); {t} {c} = ({t})({vt});"
                ));
                self.line(&format!("if ({c} == 0) __trap();"));
                let next = if is_signed_int(ety) {
                    let u = unsigned_of(ety);
                    format!("l{local} = ({t})(({u})l{local} + ({u}){c})")
                } else {
                    format!("l{local} += {c}")
                };
                self.line(&format!(
                    "for ({t} l{local} = {a}; ({c} > 0) ? (l{local} < {b}) : (l{local} > {b}); {next}) {{"
                ));
                self.scope(body)?;
                self.line("}");
                self.depth -= 1;
                self.line("}");
            }
            Stmt::While { cond, body } => {
                let c = self.ex(cond)?;
                self.line(&format!("while (({c}) != 0) {{"));
                self.scope(body)?;
                self.line("}");
            }
            Stmt::CopyAsync {
                dst,
                src,
                elems,
                valid,
                cache,
                l2_prefetch,
            } => self.copy_async(dst, src, *elems, valid.as_ref(), *cache, *l2_prefetch)?,
            Stmt::AsyncCommit => {
                self.async_used = true;
                self.flagged = true;
                self.line("asm volatile(\"cp.async.commit_group;\" ::: \"memory\");");
            }
            Stmt::AsyncWait { pending } => {
                self.async_used = true;
                self.flagged = true;
                self.line(&format!(
                    "asm volatile(\"cp.async.wait_group %0;\" :: \"n\"({pending}) : \"memory\");"
                ));
            }
            Stmt::Collective {
                kind,
                value,
                result,
                scope,
            } => self.collective(kind, value, *result, *scope)?,
            Stmt::GridDepWait => {
                self.sm90_used = true;
                self.line("asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");");
            }
            Stmt::GridDepLaunchDependents => {
                self.sm90_used = true;
                self.line("asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");");
            }
            Stmt::CopyBulk {
                dst,
                src,
                elems,
                bar,
            } => self.copy_bulk(dst, src, *elems, bar)?,
            Stmt::MbarInit { bar, count } => self.mbar_init(bar, count)?,
            Stmt::MbarArrive { bar, tx } => self.mbar_arrive(bar, tx.as_ref())?,
            Stmt::MbarCompleteTx { bar, bytes } => self.mbar_complete_tx(bar, bytes)?,
            Stmt::MbarWait { bar, parity } => self.mbar_wait(bar, parity)?,
            Stmt::Break => self.line("break;"),
            Stmt::Continue => self.line("continue;"),
            Stmt::Return => self.line("return;"),
            Stmt::Barrier { scope, .. } => {
                if matches!(scope, Scope::Block) {
                    self.line("__syncthreads();");
                } else {
                    return Err(refuse(format!(
                        "kernel `{}`: a {scope:?}-scope `Barrier` is not lowered yet (stage 2 lowers the block barrier)",
                        self.k.name
                    )));
                }
            }
            other => {
                return Err(refuse(format!(
                    "kernel `{}`: statement `{}` is not lowered by the CUDA backend yet",
                    self.k.name,
                    stmt_name(other)
                )));
            }
        }
        Ok(())
    }
}

/// An integer literal of type `t`, exactly.
fn const_int(v: i64, t: ElementKind, name: &str) -> Result<String, LowerError> {
    let bad = || {
        refuse(format!(
            "kernel `{name}`: constant {v} of type {t:?} is not lowered"
        ))
    };
    Ok(match t {
        ElementKind::I32 => {
            let x = i32::try_from(v).map_err(|_| bad())?;
            if x == i32::MIN {
                "(-2147483647 - 1)".to_string()
            } else {
                format!("{x}")
            }
        }
        ElementKind::U32 => format!("{}u", u32::try_from(v).map_err(|_| bad())?),
        ElementKind::I64 => {
            if v == i64::MIN {
                "(-9223372036854775807LL - 1LL)".to_string()
            } else {
                format!("{v}LL")
            }
        }
        _ => return Err(bad()),
    })
}

fn exec_id(id: ExecId, name: &str) -> Result<String, LowerError> {
    let ax = |a: u8| ["x", "y", "z"][usize::from(a.min(2))];
    Ok(match id {
        ExecId::ThreadIdx(a) => format!("threadIdx.{}", ax(a)),
        ExecId::BlockIdx(a) => format!("blockIdx.{}", ax(a)),
        ExecId::BlockDim(a) => format!("blockDim.{}", ax(a)),
        ExecId::GridDim(a) => format!("gridDim.{}", ax(a)),
        other => {
            return Err(refuse(format!(
                "kernel `{name}`: execution id `{}` is not lowered yet",
                variant_name(&format!("{other:?}"))
            )));
        }
    })
}

/// Lower `k` to a CUDA `__global__` function, or say why not. See the module docs for the contract.
pub(crate) fn lower_kernel(k: &KernelFn) -> Result<GeneratedKernel, LowerError> {
    check_kernel(k).map_err(|e| refuse(format!("kernel `{}` is ill-formed: {e:?}", k.name)))?;
    if k.result.is_some() || !k.results.is_empty() {
        return Err(refuse(format!(
            "kernel `{}`: a callee (a result local) is not lowered by the CUDA backend yet",
            k.name
        )));
    }
    for (i, d) in k.locals.iter().enumerate() {
        if d.lanes != 1 {
            return Err(refuse(format!(
                "kernel `{}`: local {i} is packed ({} lanes)",
                k.name, d.lanes
            )));
        }
    }
    let mut bounds = None;
    if let Some(l) = &k.launch {
        if l.subgroup.is_some() || l.cluster.is_some() || l.devices != 1 {
            return Err(refuse(format!(
                "kernel `{}`: a subgroup, cluster or multi-device launch is not lowered yet (stage 2 lowers a block and a grid)",
                k.name
            )));
        }
        let threads = l.block.iter().try_fold(1u32, |a, b| a.checked_mul(*b));
        bounds = threads.filter(|t| (1..=1024).contains(t));
        if bounds.is_none() {
            return Err(refuse(format!(
                "kernel `{}`: block {:?} is not 1..=1024 threads",
                k.name, l.block
            )));
        }
    }
    for a in &k.allocs {
        match (a.space, a.extent) {
            (MemorySpace::Shared, Extent::Static(_)) => {
                if type_name(a.elem).is_none() {
                    return Err(refuse(format!(
                        "kernel `{}`: allocation `{}` of element type {:?} has no CUDA spelling",
                        k.name, a.name, a.elem
                    )));
                }
            }
            (space, extent) => {
                return Err(refuse(format!(
                    "kernel `{}`: allocation `{}` ({space:?}, {extent:?}) is not lowered yet (stage 2 lowers static `Shared` arrays)",
                    k.name, a.name
                )));
            }
        }
    }

    let block = layout(k)?;
    let nptr = k
        .params
        .iter()
        .filter(|p| matches!(p.kind, ParamKind::Buffer { .. }))
        .count() as u32;
    if 8 * nptr + block.size > MAX_PARAM_BYTES {
        return Err(refuse(format!(
            "kernel `{}`: pointers plus the argument block are {} bytes, above the {MAX_PARAM_BYTES}-byte kernel-parameter limit",
            k.name,
            8 * nptr + block.size
        )));
    }

    // Pointers, in parameter order.
    let mut ptrs = Vec::new();
    for (i, p) in k.params.iter().enumerate() {
        if let ParamKind::Buffer { elem, role, .. } = &p.kind {
            let t = type_name(*elem).ok_or_else(|| {
                refuse(format!(
                    "kernel `{}`: buffer `{}` of element type {elem:?} has no CUDA spelling",
                    k.name, p.name
                ))
            })?;
            let q = if matches!(role, BufferRole::Read) {
                "const "
            } else {
                ""
            };
            ptrs.push(format!("{q}{t}* p{i}"));
        }
    }

    let mut cx = Cx {
        k,
        tys: k.locals.iter().map(|d| d.ty()).collect(),
        scalar_local: vec![None; k.locals.len()],
        out: String::new(),
        depth: 1,
        fresh: 0,
        async_used: false,
        flagged: false,
        sm90_used: false,
    };
    for (i, p) in k.params.iter().enumerate() {
        if let ParamKind::Scalar { local, .. } = &p.kind {
            cx.scalar_local[*local as usize] = Some(format!("s{i}"));
        }
    }
    for (i, a) in k.allocs.iter().enumerate() {
        if let Extent::Static(n) = a.extent {
            let t = type_name(a.elem).unwrap_or_default();
            cx.line(&format!("__shared__ {t} sh{i}[{n}];"));
        }
    }
    if k.launch.is_none() {
        // No launch shape: one logical thread. Whatever grid a launcher uses, only thread 0 of block 0 runs the body.
        cx.line("if ((threadIdx.x | threadIdx.y | threadIdx.z | blockIdx.x | blockIdx.y | blockIdx.z) != 0u) return;");
    }
    cx.block(&k.body)?;

    let ident_name = ident(&k.name);
    let name = format!("baracuda_cuda_kernel_{ident_name}");
    let sname = format!("Args_{ident_name}");
    let mut src = String::new();
    src.push_str("// Generated by baracuda-cuda-emit (kernel-layer lowering), do not edit.\n");
    src.push_str(&format!("// kernel: {}\n", k.name));
    src.push_str(&format!("struct {sname} {{\n"));
    for f in &block.fields {
        src.push_str(&format!("    {}\n", f.decl));
    }
    src.push_str("};\n");
    src.push_str(&format!(
        "static_assert(sizeof({sname}) == {} && alignof({sname}) == {}, \"{sname}: argument block layout differs from KernelInterface::params_block\");\n",
        block.size, block.align
    ));
    if cx.async_used {
        src.push_str("#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800\n#error \"cp.async needs sm_80 or newer\"\n#endif\n");
    }
    if cx.sm90_used {
        src.push_str(
            "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900\n#error \"mbarrier transaction counts, mbarrier.try_wait and griddepcontrol need sm_90 or newer\"\n#endif\n",
        );
    }
    if cx.flagged {
        src.push_str("// Lowers an asynchronous-copy or collective statement under a patent risk note (board 434 C, docs/PATENT-SWEEP.md hits 22 and 23). Not a clearance.\n");
    }
    let bounds_attr = bounds.map_or(String::new(), |t| format!("__launch_bounds__({t}) "));
    let mut params = ptrs;
    params.push(format!("{sname} a"));
    src.push_str(&format!(
        "extern \"C\" __global__ void {bounds_attr}{name}({}) {{\n",
        params.join(", ")
    ));
    src.push_str(&cx.out);
    src.push_str("}\n");

    let mut pb = ParamsBlock::new(block.size, block.align);
    for f in &block.fields {
        for (n, t, o) in &f.parts {
            pb = pb.with_field(ParamsBlockField::new(n.clone(), *o, *t));
        }
    }
    let mut iface = KernelInterface::of(k)
        .with_entry_symbol(name.clone())
        .with_params_block(pb);
    if let Some(t) = bounds {
        iface = iface.with_launch_bounds(LaunchBounds::new(t, None));
    }
    // One requirement string (a later `with_requires` would replace an earlier one): the newest arch wins the word `sm_90`.
    let mut needs: Vec<&str> = Vec::new();
    if cx.async_used {
        needs.push("cp.async (sm_80 or newer)");
    }
    if cx.sm90_used {
        needs.push("mbarrier / griddepcontrol (sm_90 or newer)");
    }
    if !needs.is_empty() {
        iface = iface.with_requires(needs.join("; "));
    }
    Ok(GeneratedKernel::new(name, src).with_interface(iface))
}
