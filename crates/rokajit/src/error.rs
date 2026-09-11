//! The error model (frozen; `decisions/2026-09-11-error-model.md`, prose in
//! `RokaJIT-internal/docs/error-model.md`).
//!
//! The compiler core reports failure as [`CompileError`]; the FFI edge maps
//! it to the EE's `CorJitResult` exactly once, in
//! [`CompileError::to_cor_jit_result`]. Core code never constructs a
//! `CorJitResult`, and FFI-edge code never constructs a `CompileError`.
//!
//! # Panic policy
//!
//! A panic is always a bug in RokaJIT — never a control-flow mechanism.
//! `catch_unwind` appears exactly once, around the body of each
//! `extern "C"` entry point in `lib.rs`; everywhere else, panics propagate
//! freely to that boundary. When the guard catches a panic and the EE info
//! object is available (it is, in `compileMethod`), the edge calls
//! `EeInfo::report_fatal_error(CorJitResult::InternalError)` so a checked
//! EE traps at the right place, then returns `CORJIT_INTERNALERROR`.

use rokajit_ee::enums::CorJitResult;

/// Why a compilation failed, in core terms.
///
/// Each variant names the *cause*; the EE-visible result code is derived by
/// the mapping below, so no variant carries a `CorJitResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// The IL is malformed (bad stack depths, type mismatches, invalid
    /// tokens the EE accepted but that don't decode). → `CORJIT_BADCODE`.
    BadIl(&'static str),

    /// A required allocation failed. → `CORJIT_OUTOFMEM`. Note: EE-side
    /// allocation failure (`allocMem` & friends) never reaches here — the
    /// EE longjmps past us; this is for Rust-side allocation failure only.
    OutOfMemory,

    /// An internal invariant broke (ICE). → `CORJIT_INTERNALERROR`. The
    /// payload is for diagnostics only.
    Internal(&'static str),

    /// The EE asked us not to compile (e.g. a flag declined the method).
    /// → `CORJIT_SKIPPED`.
    Skipped(&'static str),

    /// A feature we deliberately don't support yet (generics, EH, SIMD…)
    /// while coverage grows. → `CORJIT_IMPLLIMITATION`.
    Unsupported(&'static str),
}

impl CompileError {
    /// The frozen `CompileError` → `CorJitResult` mapping, applied at the
    /// FFI edge:
    ///
    /// | `CompileError` | `CorJitResult` |
    /// |---|---|
    /// | `BadIl` | `BadCode` |
    /// | `OutOfMemory` | `OutOfMem` |
    /// | `Internal` | `InternalError` |
    /// | `Skipped` | `Skipped` |
    /// | `Unsupported` | `ImplLimitation` |
    /// | (caught panic) | `InternalError` + `report_fatal_error` |
    ///
    /// `RecoverableError` and `R2RUnsupported` have no core variant: the
    /// former is a legacy Crossgen-ism, the latter is for R2R compilers —
    /// neither can occur in a jitting compile.
    pub fn to_cor_jit_result(&self) -> CorJitResult {
        match self {
            CompileError::BadIl(_) => CorJitResult::BadCode,
            CompileError::OutOfMemory => CorJitResult::OutOfMem,
            CompileError::Internal(_) => CorJitResult::InternalError,
            CompileError::Skipped(_) => CorJitResult::Skipped,
            CompileError::Unsupported(_) => CorJitResult::ImplLimitation,
        }
    }
}

/// The core's result type: `Ok(artifact)` on success.
pub type CompileResult<T> = Result<T, CompileError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_table_is_as_documented() {
        assert_eq!(
            CompileError::BadIl("x").to_cor_jit_result(),
            CorJitResult::BadCode
        );
        assert_eq!(
            CompileError::OutOfMemory.to_cor_jit_result(),
            CorJitResult::OutOfMem
        );
        assert_eq!(
            CompileError::Internal("x").to_cor_jit_result(),
            CorJitResult::InternalError
        );
        assert_eq!(
            CompileError::Skipped("x").to_cor_jit_result(),
            CorJitResult::Skipped
        );
        assert_eq!(
            CompileError::Unsupported("x").to_cor_jit_result(),
            CorJitResult::ImplLimitation
        );
    }
}
