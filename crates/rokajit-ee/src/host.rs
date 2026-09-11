//! The `EeHost` trait — the safe Rust view of `ICorJitHost`
//! (corjithost.h:14): process-lifetime services handed to `jitStartup`,
//! independent of any single compilation.

use std::ffi::{c_char, c_int, CStr, CString};
use std::ptr::NonNull;

use rokajit_ffi as ffi;

use crate::ee_info::GasketEeInfo;

/// C++ `ICorJitHost` (corjithost.h:14). The host object outlives the JIT
/// ("lives at least as long as the JIT itself", corjithost.h:12), so this
/// trait carries no lifetime parameter.
///
/// Ownership rule (frozen): **strings cross as owned `String`s**. C++
/// `getStringConfigValue` returns a pointer the caller must hand back to
/// `freeStringConfigValue` (corjithost.h:34-40); the wrapper copies to a
/// `String` and frees the original before returning, so the core never
/// holds EE-allocated memory. Host-allocated memory (`allocateMemory`,
/// `allocateSlab`) crosses as `NonNull<u8>` and must be returned to the
/// same host via `free_memory`/`free_slab`.
pub trait EeHost {
    /// C++ `ICorJitHost::allocateMemory` (corjithost.h:18). `None` = the
    /// C++ nullptr (OOM is reported in-band by the host).
    fn allocate_memory(&self, size: usize) -> Option<NonNull<u8>>;

    /// C++ `ICorJitHost::freeMemory` (corjithost.h:21). `block` must have
    /// come from `allocate_memory` on the same host.
    fn free_memory(&self, block: NonNull<u8>);

    /// C++ `ICorJitHost::getIntConfigValue` (corjithost.h:24).
    fn get_int_config_value(&self, name: &str, default: i32) -> i32;

    /// C++ `ICorJitHost::getStringConfigValue` (corjithost.h:30), with the
    /// result copied out and the original freed via `freeStringConfigValue`
    /// inside the wrapper. `None` = the C++ nullptr (no value configured).
    fn get_string_config_value(&self, name: &str) -> Option<String>;

    /// C++ `ICorJitHost::freeStringConfigValue` (corjithost.h:38).
    ///
    /// Present for vtable coverage only: under the ownership rule above,
    /// `get_string_config_value` copies and frees internally, so the core
    /// never holds an EE-allocated string to hand back. `value` must be a
    /// pointer returned by the same host's `getStringConfigValue`.
    fn free_string_config_value(&self, value: NonNull<c_char>);

    /// C++ `ICorJitHost::allocateSlab` (corjithost.h:44). On success returns
    /// the slab together with the size the host actually granted
    /// (`*pActualSize`, >= `size`); `None` = the C++ nullptr (OOM). The
    /// header supplies a default body that falls back to `allocateMemory`,
    /// but it is a real vtable entry and is forwarded like the rest.
    fn allocate_slab(&self, size: usize) -> Option<(NonNull<u8>, usize)>;

    /// C++ `ICorJitHost::freeSlab` (corjithost.h:51). `slab` and
    /// `actual_size` must be a pair previously returned by `allocate_slab`
    /// on the same host.
    fn free_slab(&self, slab: NonNull<u8>, actual_size: usize);
}

extern "C" {
    fn rokajit_host_allocate_memory(host: *mut ffi::ICorJitHost, size: usize) -> *mut u8;
    fn rokajit_host_free_memory(host: *mut ffi::ICorJitHost, block: *mut u8);
    fn rokajit_host_get_int_config_value(
        host: *mut ffi::ICorJitHost,
        name: *const c_char,
        default_value: c_int,
    ) -> c_int;
    fn rokajit_host_get_string_config_value(
        host: *mut ffi::ICorJitHost,
        name: *const c_char,
    ) -> *const c_char;
    fn rokajit_host_free_string_config_value(host: *mut ffi::ICorJitHost, value: *const c_char);
    fn rokajit_host_allocate_slab(
        host: *mut ffi::ICorJitHost,
        size: usize,
        actual_size: *mut usize,
    ) -> *mut u8;
    fn rokajit_host_free_slab(host: *mut ffi::ICorJitHost, slab: *mut u8, actual_size: usize);
}

impl EeHost for GasketEeInfo {
    fn allocate_memory(&self, size: usize) -> Option<NonNull<u8>> {
        // `None` host = jitStartup never ran; there is nobody to ask.
        self.host()?.allocate_memory(size)
    }

    fn free_memory(&self, block: NonNull<u8>) {
        // A `None` host is unreachable here: the block could only have come
        // from this host. Leak rather than call through a null vtable.
        if let Some(host) = self.host() {
            host.free_memory(block);
        }
    }

    fn get_int_config_value(&self, name: &str, default: i32) -> i32 {
        // `None` host (jitStartup never ran) degrades to the caller's default.
        let Some(host) = self.host() else {
            return default;
        };
        host.get_int_config_value(name, default)
    }

    fn get_string_config_value(&self, name: &str) -> Option<String> {
        self.host()?.get_string_config_value(name)
    }

    fn free_string_config_value(&self, value: NonNull<c_char>) {
        // `None` host is unreachable: the pointer came from this host.
        if let Some(host) = self.host() {
            host.free_string_config_value(value);
        }
    }

    fn allocate_slab(&self, size: usize) -> Option<(NonNull<u8>, usize)> {
        self.host()?.allocate_slab(size)
    }

    fn free_slab(&self, slab: NonNull<u8>, actual_size: usize) {
        // `None` host is unreachable: the slab came from this host.
        if let Some(host) = self.host() {
            host.free_slab(slab, actual_size);
        }
    }
}

impl GasketEeInfo {
    /// The process-lifetime host as a `GasketEeHost`, for delegation.
    fn host(&self) -> Option<GasketEeHost> {
        GasketEeHost::new(self.host_raw()?)
    }
}

/// The startup-time view of `ICorJitHost`: everything `jitStartup` can use
/// before any `compileMethod` exists. `GasketEeInfo` carries the same host
/// pointer for the compilation-scoped surface; this type exists so the
/// startup path (step_06's `rokajit_on_startup`) can resolve config without
/// fabricating an `ICorJitInfo`. The wrapper bodies live here;
/// `GasketEeInfo`'s `EeHost` impl delegates.
pub struct GasketEeHost {
    host: NonNull<ffi::ICorJitHost>,
}

impl GasketEeHost {
    /// Wrap the raw `ICorJitHost*` the gasket's `jitStartup` received.
    /// `None` = null pointer (never happens with a conforming EE).
    pub fn new(host: *mut ffi::ICorJitHost) -> Option<Self> {
        Some(Self {
            host: NonNull::new(host)?,
        })
    }
}

impl EeHost for GasketEeHost {
    fn allocate_memory(&self, size: usize) -> Option<NonNull<u8>> {
        NonNull::new(unsafe { rokajit_host_allocate_memory(self.host.as_ptr(), size) })
    }

    fn free_memory(&self, block: NonNull<u8>) {
        unsafe { rokajit_host_free_memory(self.host.as_ptr(), block.as_ptr()) };
    }

    fn get_int_config_value(&self, name: &str, default: i32) -> i32 {
        // Names with interior NULs degrade to the caller's default.
        let Ok(name) = CString::new(name) else {
            return default;
        };
        unsafe { rokajit_host_get_int_config_value(self.host.as_ptr(), name.as_ptr(), default) }
    }

    fn get_string_config_value(&self, name: &str) -> Option<String> {
        let name = CString::new(name).ok()?;
        let raw =
            unsafe { rokajit_host_get_string_config_value(self.host.as_ptr(), name.as_ptr()) };
        if raw.is_null() {
            return None;
        }
        // Copy out, then discharge the C++ obligation (corjithost.h:34-38)
        // before returning. Config values are ASCII in practice; invalid
        // UTF-8 degrades to U+FFFD rather than failing the query.
        let value = unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned();
        unsafe { rokajit_host_free_string_config_value(self.host.as_ptr(), raw) };
        Some(value)
    }

    fn free_string_config_value(&self, value: NonNull<c_char>) {
        unsafe { rokajit_host_free_string_config_value(self.host.as_ptr(), value.as_ptr()) };
    }

    fn allocate_slab(&self, size: usize) -> Option<(NonNull<u8>, usize)> {
        let mut actual_size = 0usize;
        let slab =
            unsafe { rokajit_host_allocate_slab(self.host.as_ptr(), size, &mut actual_size) };
        NonNull::new(slab).map(|slab| (slab, actual_size))
    }

    fn free_slab(&self, slab: NonNull<u8>, actual_size: usize) {
        unsafe { rokajit_host_free_slab(self.host.as_ptr(), slab.as_ptr(), actual_size) };
    }
}
