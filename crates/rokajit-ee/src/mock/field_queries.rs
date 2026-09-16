use std::ffi::c_void;
use std::ptr::NonNull;

use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::FieldQueries;
use crate::enums::CorInfoType;
use crate::handles::{ClassHandle, FieldHandle, MethodHandle, ObjectHandle};

impl FieldQueries for MockEe {
    fn get_field_type(&self, field: FieldHandle) -> (CorInfoType, Option<ClassHandle>) {
        self.fields
            .values()
            .find(|f| f.handle == field)
            .map(|f| (f.ty, f.value_class))
            // Unknown handles keep the original constant answer.
            .unwrap_or((CorInfoType::Int, None))
    }

    fn get_field_offset(&self, field: FieldHandle) -> u32 {
        self.fields
            .values()
            .find(|f| f.handle == field)
            .map(|f| f.offset)
            .unwrap_or(8)
    }

    fn is_field_static(&self, field: FieldHandle) -> bool {
        self.fields
            .values()
            .find(|f| f.handle == field)
            .map(|f| f.is_static)
            .unwrap_or(false)
    }

    fn get_field_info(
        &self,
        token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        _caller: MethodHandle,
        _flags: u32,
    ) -> ffi::CORINFO_FIELD_INFO {
        let mut info: ffi::CORINFO_FIELD_INFO = unsafe { std::mem::zeroed() };
        let Some(field) = self
            .fields
            .values()
            .find(|f| f.handle.as_raw() == token.hField)
        else {
            return info;
        };
        if !field.is_static {
            // The instance answer: fieldAccessor CORINFO_FIELD_INSTANCE
            // (the zeroed default); nobody consults the rest.
            return info;
        }
        // The statics pack (step_10.7): a plain static answers
        // STATIC_ADDRESS with IAT_VALUE — `fieldLookup.addr` is the
        // field's final address. A canned distinct constant per field
        // stands in for it; the mock never dereferences addresses.
        info.fieldAccessor = field
            .accessor
            .unwrap_or(ffi::CORINFO_FIELD_ACCESSOR_CORINFO_FIELD_STATIC_ADDRESS);
        info.fieldFlags = ffi::CORINFO_FIELD_FLAGS_CORINFO_FLG_FIELD_STATIC
            | if field.init_class {
                ffi::CORINFO_FIELD_FLAGS_CORINFO_FLG_FIELD_INITCLASS
            } else {
                0
            }
            | if field.in_heap {
                ffi::CORINFO_FIELD_FLAGS_CORINFO_FLG_FIELD_STATIC_IN_HEAP
            } else {
                0
            };
        info.fieldType = field.ty.to_raw();
        info.structType = field
            .value_class
            .map_or(std::ptr::null_mut(), |c| c.as_raw());
        info.accessAllowed = if field.access_illegal {
            // The real EE's only non-ALLOWED answer (jitinterface.cpp:
            // FIELD_ACCESS_EXCEPTION(callerForSecurity, field)).
            info.accessCalloutHelper.helperNum =
                crate::enums::CorInfoHelpFunc::FIELD_ACCESS_EXCEPTION.to_raw();
            info.accessCalloutHelper.numArgs = 2;
            info.accessCalloutHelper.args[0].argType =
                ffi::CorInfoAccessAllowedHelperArgType_CORINFO_HELPER_ARG_TYPE_Method;
            info.accessCalloutHelper.args[0]
                .__bindgen_anon_1
                .methodHandle = 0xCA11_E700 as ffi::CORINFO_METHOD_HANDLE;
            info.accessCalloutHelper.args[1].argType =
                ffi::CorInfoAccessAllowedHelperArgType_CORINFO_HELPER_ARG_TYPE_Field;
            info.accessCalloutHelper.args[1]
                .__bindgen_anon_1
                .fieldHandle = field.handle.as_raw();
            ffi::CorInfoIsAccessAllowedResult_CORINFO_ACCESS_ILLEGAL
        } else {
            ffi::CorInfoIsAccessAllowedResult_CORINFO_ACCESS_ALLOWED
        };
        // The helper/offset pair drives the GENERICS_STATIC_HELPER
        // accessor (step_11.3D): the base helper id and the field's
        // offset into the statics block.
        info.helper = field.statics_helper.map_or(0, |h| h.to_raw());
        info.offset = field.offset;
        if field.address_via_cell {
            // IAT_PVALUE: `fieldLookup.addr` is the cell holding the
            // field's address, not the address itself.
            info.fieldLookup.accessType = ffi::InfoAccessType_IAT_PVALUE;
            info.fieldLookup.__bindgen_anon_1.addr =
                (0xCE11_0000usize + field.handle.as_raw() as usize * 8) as *mut std::ffi::c_void;
        } else {
            info.fieldLookup.accessType = ffi::InfoAccessType_IAT_VALUE;
            info.fieldLookup.__bindgen_anon_1.addr =
                (0x57A7_0000usize + field.handle.as_raw() as usize * 8) as *mut std::ffi::c_void;
        }
        info
    }

    fn print_field_name(&self, _field: FieldHandle) -> String {
        "MockField".to_string()
    }

    fn get_field_class(&self, field: FieldHandle) -> ClassHandle {
        // The mock has one class; reuse the field handle's address as a
        // stable, non-null stand-in.
        ClassHandle(field.0 as ffi::CORINFO_CLASS_HANDLE)
    }

    fn get_thread_local_field_info(&self, field: FieldHandle, _is_gc_type: bool) -> u32 {
        self.fields
            .values()
            .find(|f| f.handle == field)
            .map(|f| f.tls_index)
            .unwrap_or(0)
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

    fn get_field_thread_local_store_id(
        &self,
        _field: FieldHandle,
    ) -> (u32, Option<NonNull<c_void>>) {
        (0, None)
    }
}
