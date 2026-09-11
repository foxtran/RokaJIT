use std::ptr::NonNull;

use crate::enums::RelocType;

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
}
