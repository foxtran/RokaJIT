//! The `EeInfo` trait surface — the safe Rust contract the compiler core
//! consumes, derived from the generated bindings (never from memory of the
//! C++). See `decisions/2026-09-11-ee-info-trait-surface.md`.
//!
//! # Grouping
//!
//! The 183-method `ICorJitInfo` chain (corinfo.h:2130 `ICorStaticInfo`,
//! corinfo.h:3282 `ICorDynamicInfo`, corjit.h:171 `ICorJitInfo`) is split
//! into one sub-trait per functional group, matching the Phase 1 agent
//! split in `docs/porting-strategy.md`:
//!
//! | Sub-trait | C++ home | Phase 1 gasket group |
//! |---|---|---|
//! | [`MethodQueries`] | `ICorMethodInfo` (corinfo.h) | method queries |
//! | [`ClassQueries`] | `ICorClassInfo` (corinfo.h) | class queries |
//! | [`FieldQueries`] | `ICorFieldInfo` (corinfo.h) | field queries |
//! | [`TokensAndSignatures`] | `ICorModuleInfo` + `ICorSigInfo` | tokens/sig |
//! | [`InliningAndTailCall`] | `ICorMethodInfo` (inlining part) | inlining |
//! | [`Helpers`] | `ICorDynamicInfo` (helper part) | helpers |
//! | [`DebugInfo`] | `ICorDebugInfo` (corinfo.h) | debug |
//! | [`OutputSinks`] | `ICorJitInfo` (corjit.h) | output sinks |
//! | [`Pgo`] | `ICorJitInfo` (corjit.h) | PGO |
//! | [`Relocations`] | `ICorJitInfo` (corjit.h) | relocations |
//!
//! [`EeInfo`] is the supertrait composing them all; the compiler core holds
//! `&dyn EeInfo`. Not every one of the 183 methods has a trait entry yet —
//! each group carries a representative, compile-needed subset, and the
//! pattern is mechanical: Phase 1 agents add one trait method per gasket
//! forwarder they write.
//!
//! # Conventions (frozen)
//!
//! - **Naming** is a mechanical transform: the C++ method name in
//!   snake_case (`getMethodAttribs` → `get_method_attribs`). Never
//!   re-named, so a C++ reference lookup always maps to exactly one trait
//!   method.
//! - **Parameters and returns** use the handle newtypes
//!   ([`crate::handles`]) and wrapped enums/flags ([`crate::enums`]).
//! - **FFI-mirror structs passed by value**: the handful of large aggregate
//!   structs the EE fills for us — [`CORINFO_SIG_INFO`],
//!   [`CORINFO_RESOLVED_TOKEN`], [`CORINFO_CALL_INFO`],
//!   [`CORINFO_EH_CLAUSE`], [`CORINFO_FIELD_INFO`], [`CORJIT_FLAGS`],
//!   [`CORINFO_CONST_LOOKUP`] — cross the trait as the bindgen types
//!   themselves (they are `Copy`, layout-verified against the headers, and
//!   re-wrapping them would be pure transcription risk). Pointer fields
//!   inside them (`pSig`, …) are read-only and valid only for the duration
//!   of the current compilation, like handles.
//! - **Error style** (see the module docs on each trait for per-method
//!   notes):
//!   - Infallible value queries return plain values.
//!   - C++ failure sentinels — null handle, `CORINFO_TYPE_UNDEF`, null
//!     string — become `Option`. The wrapper maps the sentinel once, at the
//!     boundary.
//!   - The two HRESULT-returning calls (PGO) return `Result<T, i32>` with
//!     the raw `JITINTERFACE_HRESULT` as the error.
//!   - The output sinks return `()`: the EE reports OOM by longjmp, never
//!     by return value, so there is nothing to encode.
//! - **Ownership**: every string or buffer that crosses the boundary is
//!   copied into an owned Rust value (`String`, `Vec`) by the wrapper; the
//!   wrapper then discharges the C++-side obligation
//!   (`freeStringConfigValue` for host config strings, `freeArray` for
//!   `getBoundaries`/`getVars` buffers, nothing for
//!   `getMethodNameFromMetadata`, whose storage is EE-lifetime). The core
//!   never sees a borrowed `char*`. EE-allocated executable memory returned
//!   by [`OutputSinks::alloc_mem`]/[`OutputSinks::alloc_gc_info`] is
//!   EE-owned and lives as long as the compiled method.

mod class_queries;
mod debug_info;
mod field_queries;
mod helpers;
mod inlining_and_tail_call;
mod method_queries;
mod output_sinks;
mod pgo;
mod real;
mod relocations;
mod tokens_and_signatures;
pub(crate) mod wrap;

pub use class_queries::ClassQueries;
pub use debug_info::{BoundaryMap, DebugInfo, NativeVarInfo};
pub use field_queries::FieldQueries;
pub use helpers::{HelperTarget, Helpers};
pub use inlining_and_tail_call::InliningAndTailCall;
pub use method_queries::MethodQueries;
pub use output_sinks::{AllocatedChunk, ChunkRequest, OutputSinks};
pub use pgo::{Pgo, PgoResults, PgoSchemaItem, PgoSource};
pub use real::GasketEeInfo;
pub use relocations::Relocations;
pub use tokens_and_signatures::TokensAndSignatures;

// Kept in scope so the module docs above resolve their intra-doc links
// exactly as they did in the single-file layout.
#[allow(unused_imports)]
use rokajit_ffi::{
    CORINFO_CALL_INFO, CORINFO_CONST_LOOKUP, CORINFO_EH_CLAUSE, CORINFO_FIELD_INFO,
    CORINFO_RESOLVED_TOKEN, CORINFO_SIG_INFO, CORJIT_FLAGS,
};

/// The full EE surface the compiler core consumes: the composition of all
/// functional groups above. Implemented by [`GasketEeInfo`] (the
/// gasket-backed wrapper, see `real.rs`) and by `MockEe` in tests.
/// Blanket-implemented for any type implementing all groups.
pub trait EeInfo:
    MethodQueries
    + ClassQueries
    + FieldQueries
    + TokensAndSignatures
    + InliningAndTailCall
    + Helpers
    + DebugInfo
    + OutputSinks
    + Pgo
    + Relocations
{
}

impl<T> EeInfo for T where
    T: MethodQueries
        + ClassQueries
        + FieldQueries
        + TokensAndSignatures
        + InliningAndTailCall
        + Helpers
        + DebugInfo
        + OutputSinks
        + Pgo
        + Relocations
{
}
