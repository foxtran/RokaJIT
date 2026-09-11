use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::FieldQueries;
use crate::enums::CorInfoType;
use crate::handles::{ClassHandle, FieldHandle, MethodHandle};

impl FieldQueries for MockEe {
    fn get_field_type(&self, _field: FieldHandle) -> (CorInfoType, Option<ClassHandle>) {
        (CorInfoType::Int, None)
    }

    fn get_field_offset(&self, _field: FieldHandle) -> u32 {
        8
    }

    fn is_field_static(&self, _field: FieldHandle) -> bool {
        false
    }

    fn get_field_info(
        &self,
        _token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        _caller: MethodHandle,
        _flags: u32,
    ) -> ffi::CORINFO_FIELD_INFO {
        unsafe { std::mem::zeroed() }
    }
}
