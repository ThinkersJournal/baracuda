//! Kernel-layer lowering for the CUDA backend (ROADMAP row U20), **stage 1: the skeleton**.
//!
//! [`Cuda::lower_kernel`](crate::Cuda) used to be the trait default, a typed `UnsupportedKernel` for every
//! [`KernelFn`]. This is the first of three staged changes that replace it. What stage 1 does:
//!
//! - runs [`check_kernel`] itself (the input is data; the checker is the only gate), and refuses an ill-formed
//!   kernel as `UnsupportedKernel`;
//! - spells the signature: a `__global__` function with one parameter per [`Param`], a `T*` (`const T*` for a
//!   read-only buffer) or a `T` scalar, in the CUDA type table of [`crate::typed`];
//! - emits that signature with an **empty body** when, and only when, the kernel has no statements.
//!
//! Everything else is **declined by name, never approximated**: the first statement of a non-empty body
//! (named by its variant, so a new `Stmt` variant is named too), a launch shape, an allocation, a result local
//! (a callee), a buffer or scalar of an element type outside the type table, a packed local. Stage 2 lowers the
//! memory and control subset; stage 3 the four statements ruled by board 434 C (`AsyncWait`, `CopyAsync`,
//! `AsyncCommit`, `Collective`) for NVIDIA targets, with the patent risk note
//! (`docs/PATENT-SWEEP.md`, "Async scheduling instructions"). None of those four is lowered here.

use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::kernel::{BufferRole, KernelFn, ParamKind, Stmt, check_kernel};
use unpopped::kernel_interface::KernelInterface;

use crate::typed::type_name;

fn refuse(detail: impl Into<String>) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: detail.into(),
    }
}

/// The variant name of a statement (`Let`, `AsyncWait`, ...), from its `Debug` form, so a variant added to
/// `Stmt` later is still named in the decline instead of falling into a wildcard.
pub(crate) fn stmt_name(s: &Stmt) -> String {
    format!("{s:?}")
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .next()
        .unwrap_or_default()
        .to_string()
}

/// `x` as a C identifier fragment.
fn ident(x: &str) -> String {
    let s: String = x
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.is_empty() { "_".into() } else { s }
}

/// Lower `k` to a CUDA `__global__` function, or say why not. See the module docs for the stage-1 contract.
pub(crate) fn lower_kernel(k: &KernelFn) -> Result<GeneratedKernel, LowerError> {
    check_kernel(k).map_err(|e| refuse(format!("kernel `{}` is ill-formed: {e:?}", k.name)))?;
    if let Some(first) = k.body.0.first() {
        return Err(refuse(format!(
            "kernel `{}`: statement `{}` is not lowered by the CUDA backend yet (stage 1 of U20 emits a signature and an empty body only)",
            k.name,
            stmt_name(first)
        )));
    }
    if k.launch.is_some() {
        return Err(refuse(format!(
            "kernel `{}`: a launch shape is not lowered by the CUDA backend yet",
            k.name
        )));
    }
    if !k.allocs.is_empty() {
        return Err(refuse(format!(
            "kernel `{}`: an allocation is not lowered by the CUDA backend yet",
            k.name
        )));
    }
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
    let mut params = Vec::with_capacity(k.params.len());
    for (i, p) in k.params.iter().enumerate() {
        params.push(match &p.kind {
            ParamKind::Buffer { elem, role, .. } => {
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
                format!("{q}{t}* p{i}")
            }
            ParamKind::Scalar { ty, .. } => {
                let t = type_name(*ty).ok_or_else(|| {
                    refuse(format!(
                        "kernel `{}`: scalar `{}` of type {ty:?} has no CUDA spelling",
                        k.name, p.name
                    ))
                })?;
                format!("{t} p{i}")
            }
            _ => {
                return Err(refuse(format!(
                    "kernel `{}`: parameter `{}` is a kind this backend has not been taught",
                    k.name, p.name
                )));
            }
        });
    }
    let name = format!("baracuda_cuda_kernel_{}", ident(&k.name));
    let source = format!(
        "// Generated by baracuda-cuda-emit (kernel-layer lowering, stage 1: signature only), do not edit.\n// kernel: {}\nextern \"C\" __global__ void {name}({}) {{\n}}\n",
        k.name,
        params.join(", ")
    );
    Ok(GeneratedKernel::new(name, source).with_interface(KernelInterface::of(k)))
}
