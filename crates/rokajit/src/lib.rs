//! RokaJIT — a Rust reimplementation of RyuJIT, the CoreCLR JIT compiler.
//!
//! Compiled as a `cdylib` that is a drop-in replacement for CoreCLR's
//! `libclrjit.so`. The exports the EE looks up (`jitStartup`, `getJit`) are
//! owned by the C++ gasket in `rokajit-ee`; this crate provides the
//! `extern "C"` entry points the gasket forwards to.

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
        eprintln!(
            "rokajit: compileMethod ftn={:p} ILCodeSize={} maxStack={} EHcount={} flags={:#x}",
            info.ftn, info.ILCodeSize, info.maxStack, info.EHcount, flags
        );
        let _ = (comp, native_entry, native_size_of_code);
        CorJitResult_CORJIT_INTERNALERROR
    };
    match std::panic::catch_unwind(compile) {
        Ok(result) => result,
        Err(_) => {
            eprintln!("rokajit: panic in compileMethod");
            CorJitResult_CORJIT_INTERNALERROR
        }
    }
}

#[no_mangle]
pub extern "C" fn rokajit_set_target_os(os: CORINFO_OS) {
    eprintln!("rokajit: setTargetOS os={os}");
}

#[no_mangle]
pub extern "C" fn rokajit_process_shutdown_work(_info: *mut ICorStaticInfo) {}
