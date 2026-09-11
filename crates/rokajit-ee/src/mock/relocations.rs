use std::ptr::NonNull;

use super::MockEe;
use crate::ee_info::Relocations;
use crate::enums::RelocType;

impl Relocations for MockEe {
    fn record_relocation(
        &self,
        _location: NonNull<u8>,
        _location_rw: Option<NonNull<u8>>,
        _target: usize,
        _reloc: RelocType,
        _addl_delta: i32,
    ) {
    }

    fn get_reloc_type_hint(&self, _target: usize) -> RelocType {
        RelocType::NONE
    }
}
