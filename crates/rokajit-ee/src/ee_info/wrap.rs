//! Shared wrapper idioms for the per-group `impl <Group> for GasketEeInfo`
//! blocks (step_05 consolidation, `docs/step_05-completion.md`). One home
//! for the patterns eleven parallel authors would otherwise spell eleven
//! ways. Group files keep everything else explicit.

use rokajit_ffi as ffi;

/// Fills an FFI-mirror struct through an out-parameter: zero-initializes
/// the struct (so an EE that short-circuits still yields a defined value),
/// runs `call` with a mutable reference to it, and returns the filled
/// struct. Only for the bindgen mirror structs — plain data, no ownership,
/// where an all-zero bit pattern is a valid value. Also the compiler core's
/// safe way to *build* an out-struct it must pass to the EE (step_07.2's
/// importer constructs `CORINFO_RESOLVED_TOKEN` through it), keeping the
/// `unsafe { zeroed() }` in this crate.
pub fn zeroed_out<T>(call: impl FnOnce(&mut T)) -> T {
    // SAFETY: callers pass only FFI-mirror POD structs (the bindgen
    // CORINFO_* / CORJIT_* aggregates), for which zero-init is valid.
    let mut value: T = unsafe { std::mem::zeroed() };
    call(&mut value);
    value
}

/// The target address of an `IAT_VALUE` const lookup (a directly callable
/// entry point), `None` for any other access kind — the safe read of the
/// `CORINFO_CONST_LOOKUP` union, so the compiler core (step_07.5 codegen
/// resolving call targets) stays `unsafe`-free.
pub fn const_lookup_addr(lookup: &ffi::CORINFO_CONST_LOOKUP) -> Option<usize> {
    if lookup.accessType == ffi::InfoAccessType_IAT_VALUE {
        // SAFETY: accessType IAT_VALUE means the `addr` union member is live.
        Some(unsafe { lookup.__bindgen_anon_1.addr } as usize)
    } else {
        None
    }
}

/// The indirection-cell address of an `IAT_PVALUE` const lookup (the cell
/// holds the entry point; the EE keeps it current — e.g. a fixup precode's
/// target slot for a not-yet-compiled callee), `None` for any other access
/// kind. Same safe-union-read rationale as [`const_lookup_addr`] (step_07.7:
/// codegen resolving call targets for methods not yet compiled).
pub fn const_lookup_slot(lookup: &ffi::CORINFO_CONST_LOOKUP) -> Option<usize> {
    if lookup.accessType == ffi::InfoAccessType_IAT_PVALUE {
        // SAFETY: accessType IAT_PVALUE means the `addr` union member is live.
        Some(unsafe { lookup.__bindgen_anon_1.addr } as usize)
    } else {
        None
    }
}

/// The `printObjectDescription` string contract (corinfo.h:2444,
/// 2463-2466): a null buffer with size 0 queries the required size, which
/// INCLUDES the NUL terminator; the fetch call returns the byte count
/// EXCLUDING the terminator. `call` performs one forwarder invocation
/// (buffer, buffer size, required-size out-slot) and returns the C++
/// return value. Copies into an owned `String` (lossy on invalid UTF-8).
pub(crate) fn print_object_string(
    mut call: impl FnMut(*mut u8, usize, *mut usize) -> usize,
) -> String {
    let mut required = 0usize;
    call(std::ptr::null_mut(), 0, &mut required);
    if required == 0 {
        return String::new();
    }
    let mut buf = vec![0u8; required];
    let written = call(buf.as_mut_ptr(), buf.len(), &mut required);
    buf.truncate(written.min(buf.len()));
    if buf.last() == Some(&b'\0') {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).into_owned()
}
