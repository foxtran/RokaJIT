//! RokaJIT — a Rust reimplementation of RyuJIT, the CoreCLR JIT compiler.
//!
//! Compiled as a `cdylib` that is a drop-in replacement for CoreCLR's
//! `libclrjit.so`. The exports the EE looks up (`jitStartup`, `getJit`) are
//! owned by the C++ gasket in `rokajit-ee`; this crate provides the
//! `extern "C"` entry points the gasket forwards to.
//!
//! # Error model (frozen; `decisions/2026-09-11-error-model.md`)
//!
//! - The compiler core fails with [`error::CompileError`]; the FFI edge
//!   maps it to the EE's `CorJitResult` via
//!   [`error::CompileError::to_cor_jit_result`] — the single mapping table.
//! - Panics are bugs, never control flow. `catch_unwind` appears only at
//!   the `extern "C"` entry points below. A caught panic is reported to the
//!   EE via `EeInfo::report_fatal_error(CorJitResult::InternalError)` when
//!   the EE info object is available, then surfaces as
//!   `CORJIT_INTERNALERROR`.
//!
//! # Contracts
//!
//! - [`ir`] — the HIR/LIR IR shapes (docs: `RokaJIT-internal/docs/ir-design.md`).
//! - [`artifact`] — what a successful `compileMethod` produces and hands to
//!   the EE's output sinks.
//! - The safe EE surface (`EeInfo`, handles, enums) lives in `rokajit-ee`;
//!   this crate depends on it, never the other way.

pub mod artifact;
pub mod error;
pub mod ir;
mod spot_check;

use rokajit_ee::ee_info::{EeInfo, GasketEeInfo};
use rokajit_ffi::{
    CorJitResult, CorJitResult_CORJIT_INTERNALERROR, CORINFO_METHOD_INFO, CORINFO_OS, ICorJitInfo,
    ICorStaticInfo,
};

// Link the gasket archive whole: its `jitStartup`/`getJit` exports are
// referenced by nobody at link time (the EE resolves them with dlopen/dlsym),
// so without +whole-archive the linker would discard the gasket object
// entirely, and without +export-symbols rustc's cdylib version script
// (`local: *`) would keep them out of the dynamic symbol table.
#[link(name = "rokajit_ee_gasket", kind = "static", modifiers = "+whole-archive,+export-symbols")]
extern "C" {
    fn getJit() -> *mut std::ffi::c_void;
}

// An unreferenced `#[link]` extern block is dropped by rustc without
// emitting the -l flag at all; this #[used] anchor keeps the link (and with
// +whole-archive,+export-symbols, the gasket's exports) in the final cdylib.
#[used]
static GASKET_ANCHOR: unsafe extern "C" fn() -> *mut std::ffi::c_void = getJit;

/// The one entry point that matters for milestone M1: log what the EE asked
/// us to compile, then fail gracefully so the EE reports a JIT error instead
/// of a load failure.
#[no_mangle]
pub extern "C" fn rokajit_compile_method(
    comp: *mut ICorJitInfo,
    info: *mut CORINFO_METHOD_INFO,
    flags: std::ffi::c_uint,
    native_entry: *mut *mut u8,
    native_size_of_code: *mut u32,
) -> CorJitResult {
    // A panic crossing the FFI boundary is UB; map it to a graceful error.
    let compile = || {
        // SAFETY: the EE passes a valid CORINFO_METHOD_INFO for the duration
        // of this call.
        let info = unsafe { &*info };
        let Some(ee) = GasketEeInfo::new(comp) else {
            eprintln!("rokajit: null ICorJitInfo in compileMethod");
            return CorJitResult_CORJIT_INTERNALERROR;
        };
        let _ = (native_entry, native_size_of_code);
        spot_check::run(info, &ee);
        compile_method(info, flags, &ee)
    };
    match std::panic::catch_unwind(compile) {
        Ok(result) => result,
        Err(_) => {
            eprintln!("rokajit: panic in compileMethod");
            CorJitResult_CORJIT_INTERNALERROR
        }
    }
}

/// The compiler spine: receives the safe EE surface and the method to
/// compile. No compiler work yet (step 04) — log the request and report a
/// graceful failure to the EE.
fn compile_method(info: &CORINFO_METHOD_INFO, flags: std::ffi::c_uint, ee: &dyn EeInfo) -> CorJitResult {
    eprintln!(
        "rokajit: compileMethod ftn={:p} ILCodeSize={} maxStack={} EHcount={} flags={:#x}",
        info.ftn, info.ILCodeSize, info.maxStack, info.EHcount, flags
    );
    let _ = ee;
    CorJitResult_CORJIT_INTERNALERROR
}

#[no_mangle]
pub extern "C" fn rokajit_set_target_os(os: CORINFO_OS) {
    // A panic crossing the FFI boundary is UB; swallow it like compileMethod.
    let _ = std::panic::catch_unwind(|| eprintln!("rokajit: setTargetOS os={os}"));
}

#[no_mangle]
pub extern "C" fn rokajit_process_shutdown_work(_info: *mut ICorStaticInfo) {
    let _ = std::panic::catch_unwind(|| {});
}
