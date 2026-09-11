//! `rokajit-cdy` — the cdylib edge of RokaJIT: the `librokajit.so` the EE
//! loads.
//!
//! The workspace splits into the compiler core (`rokajit`, rlib), the x64
//! backend (`rokajit-x64`, rlib), and this thin top crate (step_07.7;
//! `decisions/2026-09-11-pipeline-and-target-contracts.md` "cdylib wiring"):
//! the core cannot name `X64Target` (Cargo forbids the rokajit ↔ rokajit-x64
//! cycle), so the `extern "C"` entry points the gasket forwards to live here,
//! where both crates are in scope. This crate is part of the gasket edge:
//! `unsafe` is allowed here (and in `rokajit-ffi`), nowhere else — the
//! pointer-chasing copies out of `CORINFO_METHOD_INFO` and the writes into
//! EE-allocated chunks happen in this file and [`drain`].
//!
//! The exports the EE looks up (`jitStartup`, `getJit`) are owned by the
//! C++ gasket in `rokajit-ee`; the gasket's `compileMethod` forwards to
//! [`rokajit_compile_method`] below.
//!
//! # Error model (frozen; `decisions/2026-09-11-error-model.md`)
//!
//! The core fails with `CompileError`; this edge maps it to the EE's
//! `CorJitResult` via `CompileError::to_cor_jit_result` — the single
//! mapping table. `catch_unwind` appears only around the bodies of the
//! `extern "C"` entry points here; a caught panic surfaces as
//! `CORJIT_INTERNALERROR`.

mod drain;
mod spot_check;

use rokajit::error::CompileError;
use rokajit::pipeline::{self, MethodInfo, Tier};
use rokajit_ee::ee_info::{EeInfo, GasketEeInfo};
use rokajit_ee::handles::MethodHandle;
use rokajit_ee::host::GasketEeHost;
use rokajit_ffi::{
    CorJitResult, CorJitResult_CORJIT_INTERNALERROR, ICorJitHost, ICorJitInfo, ICorStaticInfo,
    CORINFO_METHOD_INFO, CORINFO_OS,
};
use rokajit_x64::X64Target;

// Link the gasket archive whole: its `jitStartup`/`getJit` exports are
// referenced by nobody at link time (the EE resolves them with dlopen/dlsym),
// so without +whole-archive the linker would discard the gasket object
// entirely, and without +export-symbols rustc's cdylib version script
// (`local: *`) would keep them out of the dynamic symbol table.
#[link(
    name = "rokajit_ee_gasket",
    kind = "static",
    modifiers = "+whole-archive,+export-symbols"
)]
extern "C" {
    fn getJit() -> *mut std::ffi::c_void;
}

// An unreferenced `#[link]` extern block is dropped by rustc without
// emitting the -l flag at all; this #[used] anchor keeps the link (and with
// +whole-archive,+export-symbols, the gasket's exports) in the final cdylib.
#[used]
static GASKET_ANCHOR: unsafe extern "C" fn() -> *mut std::ffi::c_void = getJit;

/// The `compileMethod` entry point: drive the full pipeline (import →
/// morph → lower → codegen → metadata), drain the artifact into the EE's
/// output sinks, and answer `CORJIT_OK` with the real entry pointer.
/// Compilation failures map to graceful `CORJIT_*` codes; panics map to
/// `CORJIT_INTERNALERROR`.
// `info` (and `comp`) are EE-issued pointers the EE guarantees valid for the
// duration of this call — the standard ICorJitInfo entry-point contract.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
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
        spot_check::run(info, &ee);
        compile_method(info, flags, &ee, native_entry, native_size_of_code)
    };
    match std::panic::catch_unwind(compile) {
        Ok(result) => result,
        Err(_) => {
            eprintln!("rokajit: panic in compileMethod");
            CorJitResult_CORJIT_INTERNALERROR
        }
    }
}

/// The compiler spine: build the owned [`MethodInfo`] snapshot, run the
/// pipeline against the concrete x64 target, drain the artifact, and fill
/// the out-params. Failures are the frozen `CompileError` → `CorJitResult`
/// mapping, logged once.
fn compile_method(
    info: &CORINFO_METHOD_INFO,
    flags: std::ffi::c_uint,
    ee: &dyn EeInfo,
    native_entry: *mut *mut u8,
    native_size_of_code: *mut u32,
) -> CorJitResult {
    eprintln!(
        "rokajit: compileMethod ftn={:p} ILCodeSize={} maxStack={} EHcount={} flags={:#x}",
        info.ftn, info.ILCodeSize, info.maxStack, info.EHcount, flags
    );
    let Some(ftn) = MethodHandle::from_raw(info.ftn) else {
        eprintln!("rokajit: null ftn in compileMethod");
        return CompileError::Internal("null ftn in CORINFO_METHOD_INFO")
            .to_cor_jit_result()
            .to_raw();
    };
    let method_info = MethodInfo {
        ftn,
        // SAFETY: the EE guarantees ILCode points at ILCodeSize readable
        // bytes for the duration of compileMethod.
        il: unsafe { std::slice::from_raw_parts(info.ILCode, info.ILCodeSize as usize) }.to_vec(),
        max_stack: info.maxStack,
        eh_count: info.EHcount,
        init_locals: info.options & rokajit_ffi::CorInfoOptions_CORINFO_OPT_INIT_LOCALS != 0,
        args: info.args,
        locals: info.locals,
    };
    match pipeline::compile(&method_info, ee, &X64Target, Tier::Tier0) {
        Ok(artifact) => {
            let (entry, size) = match drain::drain(&artifact, ftn, ee) {
                Ok(drained) => drained,
                Err(error) => {
                    let result = error.to_cor_jit_result();
                    eprintln!("rokajit: drain failed: {error:?} → {result:?}");
                    return result.to_raw();
                }
            };
            // SAFETY: the EE passes valid out-param pointers (the standard
            // compileMethod contract).
            unsafe {
                *native_entry = entry.as_ptr();
                *native_size_of_code = size;
            }
            rokajit_ffi::CorJitResult_CORJIT_OK
        }
        Err(error) => {
            let result = error.to_cor_jit_result();
            let name = rokajit_ee::ee_info::MethodQueries::print_method_name(ee, ftn);
            eprintln!("rokajit: compilation failed: {name}: {error:?} → {result:?}");
            result.to_raw()
        }
    }
}

/// Startup hook called by the gasket's `jitStartup` after the host is
/// stored (step_06 task 4). Resolves the config snapshot through the host
/// and runs the unsupported-knob warning scan — it fires at JIT load,
/// before any `compileMethod`.
#[no_mangle]
pub extern "C" fn rokajit_on_startup(host: *mut ICorJitHost) {
    // A panic crossing the FFI boundary is UB; log and swallow like the
    // other info-less entry points (error-model decision).
    let _ = std::panic::catch_unwind(|| {
        let Some(host) = GasketEeHost::new(host) else {
            eprintln!("rokajit: null ICorJitHost in jitStartup");
            return;
        };
        rokajit::config::init_from_host(&host);
    });
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
