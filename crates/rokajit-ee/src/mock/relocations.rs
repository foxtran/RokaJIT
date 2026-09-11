use std::ptr::NonNull;

use super::MockEe;
use crate::ee_info::Relocations;
use crate::enums::{CorInfoArch, RelocType};

impl Relocations for MockEe {
    fn record_relocation(
        &self,
        location: NonNull<u8>,
        _location_rw: Option<NonNull<u8>>,
        target: usize,
        reloc: RelocType,
        _addl_delta: i32,
    ) {
        self.sink_log.borrow_mut().push(format!(
            "record_relocation(loc={:#x}, target={target:#x}, {reloc:?})",
            location.as_ptr() as usize
        ));
    }

    fn get_reloc_type_hint(&self, _target: usize) -> RelocType {
        RelocType::NONE
    }

    fn get_expected_target_architecture(&self) -> CorInfoArch {
        // The one target RokaJIT generates code for.
        CorInfoArch::X64
    }
}
