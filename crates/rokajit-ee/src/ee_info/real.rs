//! `GasketEeInfo` — the real, gasket-backed `EeInfo` implementation.
//!
//! The struct itself is orchestrator-owned shared territory; the per-group
//! partial `impl <Group> for GasketEeInfo` blocks live in the group's own
//! trait file (`ee_info/<group>.rs`) per the step_03 split notes §1. Shape
//! decision: `decisions/2026-09-11-gasket-ee-info-shape.md`.
//!
//! Lifetime: like handles, the wrapped interface pointers are EE-owned and
//! valid only for the duration of one `compileMethod` call (`comp`) resp.
//! the process (`host`); nothing stores a `GasketEeInfo` beyond its
//! compilation.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, Ordering};

use rokajit_ffi as ffi;

/// The process-lifetime `ICorJitHost*`, captured at `jitStartup` time by
/// `gasket_core.cpp` calling `rokajit_ee_set_jit_host`. `ICorJitHost`
/// "lives at least as long as the JIT itself" (corjithost.h:12), so a
/// process-wide static is honest.
static JIT_HOST: AtomicPtr<ffi::ICorJitHost> = AtomicPtr::new(std::ptr::null_mut());

/// Called once from `gasket_core.cpp`'s `jitStartup` export. Never called
/// again; the EE hands the same host pointer for the JIT's lifetime.
#[no_mangle]
pub extern "C" fn rokajit_ee_set_jit_host(host: *mut ffi::ICorJitHost) {
    JIT_HOST.store(host, Ordering::Relaxed);
}

/// The concrete `EeInfo`: raw interface pointers plus one mechanical
/// wrapper per gasket forwarder. Carries no state of its own — every query
/// is forwarded to the EE in the same compilation.
pub struct GasketEeInfo {
    /// The `ICorJitInfo` the EE passed to `compileMethod`. Valid for this
    /// compilation only.
    comp: NonNull<ffi::ICorJitInfo>,
    /// The process-lifetime `ICorJitHost` from `jitStartup`. `None` only if
    /// the EE never called `jitStartup` (which `getJit` already refuses).
    host: Option<NonNull<ffi::ICorJitHost>>,
}

impl GasketEeInfo {
    /// Wrap the raw `ICorJitInfo*` from the gasket's `compileMethod`
    /// forward. `None` = null pointer (never happens with a conforming EE;
    /// the spine treats it as an internal error).
    pub fn new(comp: *mut ffi::ICorJitInfo) -> Option<Self> {
        let comp = NonNull::new(comp)?;
        let host = NonNull::new(JIT_HOST.load(Ordering::Relaxed));
        Some(Self { comp, host })
    }

    /// The raw `ICorJitInfo*` for this group's gasket forwarders.
    pub(crate) fn comp_raw(&self) -> *mut ffi::ICorJitInfo {
        self.comp.as_ptr()
    }

    /// The raw process-lifetime `ICorJitHost*`, for `host.rs`'s forwarders.
    pub(crate) fn host_raw(&self) -> Option<*mut ffi::ICorJitHost> {
        self.host.map(|host| host.as_ptr())
    }
}
