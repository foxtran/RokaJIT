use std::ffi::c_void;
use std::ptr::NonNull;

use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::FieldQueries;
use crate::enums::CorInfoType;
use crate::handles::{ClassHandle, FieldHandle, MethodHandle, ObjectHandle};

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

    fn print_field_name(&self, _field: FieldHandle) -> String {
        "MockField".to_string()
    }

    fn get_field_class(&self, field: FieldHandle) -> ClassHandle {
        // The mock has one class; reuse the field handle's address as a
        // stable, non-null stand-in.
        ClassHandle(field.0 as ffi::CORINFO_CLASS_HANDLE)
    }

    fn get_thread_local_field_info(&self, _field: FieldHandle, _is_gc_type: bool) -> u32 {
        0
    }

    fn get_thread_local_static_blocks_info(&self) -> ffi::CORINFO_THREAD_STATIC_BLOCKS_INFO {
        unsafe { std::mem::zeroed() }
    }

    fn get_thread_local_static_info_native_aot(&self) -> ffi::CORINFO_THREAD_STATIC_INFO_NATIVEAOT {
        unsafe { std::mem::zeroed() }
    }

    fn get_array_or_string_length(&self, _obj: ObjectHandle) -> Option<u32> {
        None
    }

    fn get_thread_tls_index(&self) -> (u32, Option<NonNull<c_void>>) {
        (0, None)
    }

    fn get_addr_of_capture_thread_global(&self) -> (Option<NonNull<i32>>, Option<NonNull<c_void>>) {
        (None, None)
    }

    fn get_static_field_content(
        &self,
        _field: FieldHandle,
        _buffer: &mut [u8],
        _value_offset: i32,
        _ignore_movable_objects: bool,
    ) -> bool {
        false
    }

    fn get_static_field_current_class(&self, _field: FieldHandle) -> (Option<ClassHandle>, bool) {
        (None, false)
    }

    fn get_field_thread_local_store_id(&self, _field: FieldHandle) -> (u32, Option<NonNull<c_void>>) {
        (0, None)
    }
}
