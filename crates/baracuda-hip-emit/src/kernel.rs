// SPDX-License-Identifier: MIT OR Apache-2.0
//! Kernel-layer lowering for the HIP backend (ROADMAP row H1), **stage 1: the memory and control subset**.
//!
//! [`Hip::lower_kernel`](crate::Hip) lowers a [`KernelFn`] to one `extern "C" __global__` function in HIP C++. It runs
//! [`check_kernel`] itself (the input is data; the checker is the only gate) and then either spells the kernel
//! **exactly** or declines it **by name**, as `LowerError::UnsupportedKernel`. Nothing is approximated.
//!
//! **Provenance (trace and credit).** This stage is the same code shape as `baracuda-cuda-emit`'s stage 2 (this repository,
//! MIT OR Apache-2.0): the argument block (ABI "B", one by-value struct beside the buffer pointers), the place and bounds
//! checks, and the control-flow lowering are carried over unchanged because HIP C++ spells them identically
//! (`__global__`, `__shared__`, `threadIdx`/`blockIdx`/`blockDim`/`gridDim`, `__syncthreads()`, `__trap()`; the HIP
//! C++ language extensions are documented in the public ROCm HIP programming guide,
//! <https://rocm.docs.amd.com/projects/HIP/>, MIT licensed, read for these spellings only). What is deliberately **not**
//! carried over is everything that is NVIDIA-specific or whose AMD semantics have not been read:
//!
//! - `CopyAsync`/`AsyncCommit`/`AsyncWait` (`cp.async`), `Mbar*`, `CopyBulk`, `GridDep*`, `Wgmma*`, `Tcgen05*`,
//!   `MatrixLoad`/`MatrixStore`, named and `Cluster` barriers, clusters: PTX features of NVIDIA hardware. Several carry the PM's
//!   patent notes ("lowering this to a hardware operation on non-NVIDIA hardware may implicate US11803380B2 / US12204897B2");
//!   **a HIP emitter declines them, it does not translate them**.
//! - `Collective` (a warp shuffle or vote): an AMD wave is 32 or 64 lanes, and the IR's `Subgroup` scope says "the target
//!   decides"; a full-mask warp collective needs the wave size and the `__shfl*` semantics read first (ROADMAP H1 follow-up).
//! - `Cast`, `Fma`, `Bitcast`: the CUDA spellings rely on the PTX `cvt` saturation contract; the AMD conversion
//!   intrinsics have not been checked against the IR's semantics, so they are not guessed.
//! - A `Subgroup` size, a cluster or multi-device launch, a dynamic or non-`Shared` allocation, packed locals, places other than
//!   `Logical` and `Alloc`.
//!
//! What stage 1 lowers: `Let`, `Load`, `Assign`, `Store`, `If`, `For`, `While`, `Break`, `Continue`, `Return` and a
//! `Block`-scope full `Barrier` (`__syncthreads()`); a launch shape (`__launch_bounds__`); static `Shared` arrays; the expressions
//! `Const`, `ConstInt`, `Local`, `Exec`, `Add`/`Sub`/`Mul` (wrapping on integers), float `Div`, the six comparisons. Types
//! `F32`, `F64`, `I32`, `U32`, `I64`. An out-of-bounds coordinate or `Shared` index, or a zero `For` step, executes `__trap()`.
//!
//! **Not compiled by `hipcc` here** (no ROCm on the development box, and no AMD device): the tests pin the source shape, and
//! cross-check that the emitted text is valid C++ with the shared `__global__` spellings by compiling it with NVRTC when the
//! `nvrtc-syntax-check` feature is on. That is a syntax check, not evidence that a HIP compiler accepts it or that it runs on AMD.

use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::exec::{ExecId, Extent, MemorySpace, Scope};
use unpopped::ir::{BinaryOp, ScalarExpr};
use unpopped::kernel::{
    BarrierKind, Block, BufferRole, KernelFn, ParamKind, Place, Stmt, check_kernel,
};
use unpopped::kernel_interface::{KernelInterface, LaunchBounds, ParamsBlock, ParamsBlockField};
use unpopped::types::{Ty, infer_ty_in};
use unpopped_vocab::ElementKind;

/// The HIP C type of an element in the stage-1 table, or `None` outside it.
pub(crate) fn type_name(t: ElementKind) -> Option<String> {
    Some(
        match t {
            ElementKind::F32 | ElementKind::F32Strict => "float",
            ElementKind::F64 => "double",
            ElementKind::I8 => "signed char",
            ElementKind::I16 => "short",
            ElementKind::I32 => "int",
            ElementKind::I64 => "long long",
            ElementKind::U8 => "unsigned char",
            ElementKind::U16 => "unsigned short",
            ElementKind::U32 => "unsigned int",
            _ => return None,
        }
        .to_string(),
    )
}

/// An `f32` constant as a C `float` literal (`1.0f`, never the double `1.0`).
fn const_f32(v: f64) -> String {
    let f = v as f32;
    if f.is_nan() {
        "NAN".to_string()
    } else if f.is_infinite() {
        if f > 0.0 { "INFINITY" } else { "-INFINITY" }.to_string()
    } else {
        format!("{f:?}f")
    }
}

/// An `f64` constant as a C `double` literal.
fn const_default(v: f64) -> String {
    unpopped::backend::const_lit(v)
}

/// Why a statement this stage does not lower is not lowered: NVIDIA hardware, or not yet read for AMD.
fn why_not(s: &Stmt) -> &'static str {
    match s {
        Stmt::CopyAsync { .. } | Stmt::AsyncCommit | Stmt::AsyncWait { .. } => {
            "`cp.async` is a PTX feature of NVIDIA hardware, under the PM's patent notes; a HIP emitter declines it"
        }
        Stmt::MbarInit { .. }
        | Stmt::MbarArrive { .. }
        | Stmt::MbarCompleteTx { .. }
        | Stmt::MbarWait { .. }
        | Stmt::CopyBulk { .. }
        | Stmt::GridDepWait
        | Stmt::GridDepLaunchDependents => {
            "an NVIDIA `mbarrier`/bulk-copy/`griddepcontrol` feature; the IR's patent note says emitters for other manufacturers decline it"
        }
        Stmt::Wgmma { .. }
        | Stmt::WgmmaCommit
        | Stmt::WgmmaWait { .. }
        | Stmt::Tcgen05Mma { .. }
        | Stmt::Tcgen05Commit { .. }
        | Stmt::Tcgen05Ld { .. } => {
            "an NVIDIA tensor-core statement; AMD's matrix core instructions are `MmaAtom` constructors from the ISA tables, not these"
        }
        Stmt::MatrixLoad { .. } | Stmt::MatrixStore { .. } => {
            "`ldmatrix`/`stmatrix` are NVIDIA instructions; the AMD equivalent (`ds_read_tr`) is a separate named operation (ROADMAP H1)"
        }
        Stmt::Collective { .. } => {
            "a warp collective needs the AMD wave size (32 or 64) and the `__shfl*` semantics read first (ROADMAP H1)"
        }
        _ => "no HIP lowering has been written for it yet",
    }
}

/// The largest buffer rank the argument block carries (an over-rank kernel is declined by name).
pub(crate) const MAX_RANK: u8 = 4;

/// The kernel-parameter limit (the classic 4 KB; HIP's is the same order, and the smaller number is the safe one) in bytes (pointers plus the by-value argument block).
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
                        "kernel `{}`: scalar `{}` of type {ty:?} has no HIP spelling",
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
                "kernel `{}`: no HIP spelling for local {i}",
                self.k.name
            ))
        })
    }

    /// Spell `e` exactly, or decline it by node name.
    fn ex(&self, e: &ScalarExpr) -> Result<String, LowerError> {
        let t = self.scalar_ty(e)?;
        Ok(match e {
            ScalarExpr::Const(v) => match self.k.compute {
                ElementKind::F32 => const_f32(*v),
                ElementKind::F64 => const_default(*v),
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
                        "kernel `{}`: no HIP spelling for the compute type",
                        self.k.name
                    ))
                })?;
                format!("(({c})(({}) {sym} ({})))", self.ex(a)?, self.ex(b)?)
            }
            other => {
                return Err(refuse(format!(
                    "kernel `{}`: expression node `{}` is not lowered by the HIP backend yet",
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
            Stmt::Break => self.line("break;"),
            Stmt::Continue => self.line("continue;"),
            Stmt::Return => self.line("return;"),
            Stmt::Barrier { scope, kind } => {
                // `__syncthreads()` is a whole-block barrier. A kind that waits for fewer threads is not that, so it is
                // refused (never approximated); the same rule as the CUDA emitter's 0.56.13 fix.
                if !matches!(kind, BarrierKind::Full) {
                    return Err(refuse(format!(
                        "kernel `{}`: a `Barrier` of kind {kind:?} is not lowered by the HIP backend",
                        self.k.name
                    )));
                }
                if matches!(scope, Scope::Block) {
                    self.line("__syncthreads();");
                } else {
                    return Err(refuse(format!(
                        "kernel `{}`: a {scope:?}-scope `Barrier` is not lowered yet (stage 1 lowers the block barrier)",
                        self.k.name
                    )));
                }
            }
            other => {
                return Err(refuse(format!(
                    "kernel `{}`: statement `{}` is not lowered by the HIP backend ({})",
                    self.k.name,
                    stmt_name(other),
                    why_not(other)
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

/// Lower `k` to a HIP `__global__` function, or say why not. See the module docs for the contract.
pub(crate) fn lower_kernel(k: &KernelFn) -> Result<GeneratedKernel, LowerError> {
    check_kernel(k).map_err(|e| refuse(format!("kernel `{}` is ill-formed: {e:?}", k.name)))?;
    if k.result.is_some() || !k.results.is_empty() {
        return Err(refuse(format!(
            "kernel `{}`: a callee (a result local) is not lowered by the HIP backend yet",
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
                "kernel `{}`: a subgroup, cluster or multi-device launch is not lowered yet (stage 1 lowers a block and a grid; AMD waves are 32 or 64 wide, so a subgroup size is not assumed)",
                k.name
            )));
        }
        let threads = l.block.iter().try_fold(1u32, |a, b| a.checked_mul(*b));
        bounds = threads.filter(|t| (1..=1024).contains(t)); // HIP's default `__launch_bounds__` ceiling
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
                        "kernel `{}`: allocation `{}` of element type {:?} has no HIP spelling",
                        k.name, a.name, a.elem
                    )));
                }
            }
            (space, extent) => {
                return Err(refuse(format!(
                    "kernel `{}`: allocation `{}` ({space:?}, {extent:?}) is not lowered yet (stage 1 lowers static `Shared` arrays: `__shared__` is the same in HIP)",
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
                    "kernel `{}`: buffer `{}` of element type {elem:?} has no HIP spelling",
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
    let name = format!("baracuda_hip_kernel_{ident_name}");
    let sname = format!("Args_{ident_name}");
    let mut src = String::new();
    src.push_str("// Generated by baracuda-hip-emit (kernel-layer lowering), do not edit.\n");
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
    Ok(GeneratedKernel::new(name, src).with_interface(iface))
}
