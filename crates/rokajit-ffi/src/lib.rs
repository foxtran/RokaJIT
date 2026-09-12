//! JIT/EE interface definitions, generated from CoreCLR's `corinfo.h`,
//! `corjit.h`, and `corjithost.h` by bindgen (see `build.rs`).
//!
//! This crate is the exact binary contract between the CoreCLR execution
//! engine and the JIT. Everything generated here stays layout-compatible
//! with the C++ headers in `runtime/src/coreclr/inc/` by construction.
//!
//! Hand-written on purpose (bindgen gaps, see
//! `decisions/2026-09-11-bindgen-for-ffi-layouts.md`):
//! - `JIT_EE_VERSION_IDENTIFIER` below — `constexpr`, internal linkage, no
//!   symbol to link against.
//! - The vtables — bindgen opaques C++ classes; the `rokajit-ee` C++ gasket
//!   owns all vtable layout instead.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
// bindgen's enum-field glue transmutes same-typed values; the lint fires
// on generated code only (clippy 1.98).
#![allow(clippy::useless_transmute)]
// Same story for the generated raw accessors: pointer arithmetic by byte
// offset and undocumented unsafe fns are bindgen's house style.
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::ptr_offset_with_cast)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Version of the JIT/EE interface this JIT implements, returned from
/// `ICorJitCompiler::getVersionIdentifier`.
///
/// Mirrored by hand from `runtime/src/coreclr/inc/jiteeversionguid.h`
/// (`5a3e8dc8-83bf-47e5-b032-531dbd307dbe`), read at runtime commit
/// ac550b6897e4faf908a90e9ef0e7045173d0c706. Must be re-checked whenever the
/// `runtime/` checkout is updated.
pub const JIT_EE_VERSION_IDENTIFIER: GUID = GUID {
    Data1: 0x5a3e8dc8,
    Data2: 0x83bf,
    Data3: 0x47e5,
    Data4: [0xb0, 0x32, 0x53, 0x1d, 0xbd, 0x30, 0x7d, 0xbe],
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // Expected values printed from the C++ side (clang++ -std=c++20, same
    // defines/include paths as build.rs) at runtime commit
    // ac550b6897e4faf908a90e9ef0e7045173d0c706. These pin the bindgen output
    // against the authoritative C++ layout.
    #[test]
    fn corinfo_method_info_layout_matches_cpp() {
        assert_eq!(size_of::<CORINFO_METHOD_INFO>(), 272);
        assert_eq!(offset_of!(CORINFO_METHOD_INFO, ILCodeSize), 24);
        assert_eq!(offset_of!(CORINFO_METHOD_INFO, args), 48);
    }

    #[test]
    fn other_critical_layouts_match_cpp() {
        assert_eq!(size_of::<CORINFO_SIG_INFO>(), 112);
        assert_eq!(size_of::<CORJIT_FLAGS>(), 24);
        assert_eq!(size_of::<GUID>(), 16);
    }

    #[test]
    fn corjit_internalerror_value() {
        // 0x80000003 as the c_int bindgen assigns to C++ enum constants.
        assert_eq!(CorJitResult_CORJIT_INTERNALERROR, 0x80000003u32 as i32);
    }
}
