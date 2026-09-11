use std::ffi::c_void;
use std::ptr::NonNull;

use rokajit_ffi as ffi;

use super::GasketEeInfo;
use crate::enums::{CorInfoArch, RelocType};

/// Relocation recording (C++ `ICorJitInfo`, corjit.h:435/443). No-ops when
/// jitting in-process except for jump thunks; load-bearing for prejitting.
pub trait Relocations {
    /// C++ `ICorJitInfo::recordRelocation` (corjit.h:435). `location` is the
    /// executable view, `location_rw` the writable alias (`None` = same);
    /// `target` is the raw address the slot must point at.
    fn record_relocation(
        &self,
        location: NonNull<u8>,
        location_rw: Option<NonNull<u8>>,
        target: usize,
        reloc: RelocType,
        addl_delta: i32,
    );

    /// C++ `ICorJitInfo::getRelocTypeHint` (corjit.h:443). The C++
    /// `CORINFO_RELOC_NONE` "no hint" answer becomes `RelocType::NONE`.
    fn get_reloc_type_hint(&self, target: usize) -> RelocType;

    /// C++ `ICorJitInfo::getExpectedTargetArchitecture` (corjit.h:450): the
    /// architecture the VM expects the JIT to generate code for. Differs
    /// from the host architecture when the VM is cross-compiling (crossgen).
    /// Panics only on an EE contract violation (a value outside the
    /// `CorInfoArch` set, i.e. headers grew).
    fn get_expected_target_architecture(&self) -> CorInfoArch;
}

extern "C" {
    fn rokajit_ee_record_relocation(
        info: *mut ffi::ICorJitInfo,
        location: *mut c_void,
        location_rw: *mut c_void,
        target: *mut c_void,
        f_reloc_type: ffi::CorInfoReloc,
        addl_delta: i32,
    );
    fn rokajit_ee_get_reloc_type_hint(
        info: *mut ffi::ICorJitInfo,
        target: *mut c_void,
    ) -> ffi::CorInfoReloc;
    fn rokajit_ee_get_expected_target_architecture(info: *mut ffi::ICorJitInfo) -> u32;
}

impl Relocations for GasketEeInfo {
    fn record_relocation(
        &self,
        location: NonNull<u8>,
        location_rw: Option<NonNull<u8>>,
        target: usize,
        reloc: RelocType,
        addl_delta: i32,
    ) {
        // The EE writes through locationRW unconditionally; `None` means the
        // executable view is writable, so the alias is `location` itself.
        let location_rw = location_rw.unwrap_or(location);
        unsafe {
            rokajit_ee_record_relocation(
                self.comp_raw(),
                location.as_ptr().cast(),
                location_rw.as_ptr().cast(),
                target as *mut c_void,
                reloc.to_raw(),
                addl_delta,
            )
        };
    }

    fn get_reloc_type_hint(&self, target: usize) -> RelocType {
        let raw = unsafe { rokajit_ee_get_reloc_type_hint(self.comp_raw(), target as *mut c_void) };
        RelocType::from_raw(raw)
    }

    fn get_expected_target_architecture(&self) -> CorInfoArch {
        let raw = unsafe { rokajit_ee_get_expected_target_architecture(self.comp_raw()) };
        CorInfoArch::from_raw(raw)
            .expect("EE contract: getExpectedTargetArchitecture returns a CorInfoArch value")
    }
}
