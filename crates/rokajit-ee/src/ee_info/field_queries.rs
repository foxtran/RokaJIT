use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::ptr::NonNull;

use rokajit_ffi as ffi;
use rokajit_ffi::{
    CORINFO_FIELD_INFO, CORINFO_RESOLVED_TOKEN, CORINFO_THREAD_STATIC_BLOCKS_INFO,
    CORINFO_THREAD_STATIC_INFO_NATIVEAOT,
};

use super::wrap::{print_object_string, zeroed_out};
use super::GasketEeInfo;
use crate::enums::CorInfoType;
use crate::handles::{ClassHandle, FieldHandle, MethodHandle, ObjectHandle};

/// Field metadata queries (C++ `ICorFieldInfo`).
pub trait FieldQueries {
    /// C++ `ICorFieldInfo::getFieldType` (corinfo.h:2939). The C++
    /// `structType` out-param (set for value-class fields) becomes the
    /// second tuple element. The `fieldOwnerHint` parameter is not exposed
    /// (RyuJIT passes null outside exotic diagnostics paths).
    fn get_field_type(&self, field: FieldHandle) -> (CorInfoType, Option<ClassHandle>);

    /// C++ `ICorFieldInfo::getFieldOffset` (corinfo.h:2946).
    fn get_field_offset(&self, field: FieldHandle) -> u32;

    /// C++ `ICorFieldInfo::isFieldStatic` (corinfo.h:2973).
    fn is_field_static(&self, field: FieldHandle) -> bool;

    /// C++ `ICorFieldInfo::getFieldInfo` (corinfo.h:2950): how to access a
    /// resolved field. `flags` is the raw `CORINFO_ACCESS_FLAGS` word
    /// (corinfo.h:622); it gets a flag newtype when the first consumer
    /// lands.
    fn get_field_info(
        &self,
        token: &mut CORINFO_RESOLVED_TOKEN,
        caller: MethodHandle,
        flags: u32,
    ) -> CORINFO_FIELD_INFO;

    /// C++ `ICorFieldInfo::printFieldName` (corinfo.h:2921). Debug/diagnostic
    /// names only (see `printObjectDescription`, corinfo.h:2444, for the
    /// buffer contract). The wrapper does the two-call dance — query the
    /// required size with a null buffer, fill a JIT-owned buffer, copy into
    /// an owned `String` — so truncation never happens and the C++ NUL
    /// terminator is stripped.
    fn print_field_name(&self, field: FieldHandle) -> String;

    /// C++ `ICorFieldInfo::getFieldClass` (corinfo.h:2929). Infallible: the
    /// EE always knows a field's owning class.
    fn get_field_class(&self, field: FieldHandle) -> ClassHandle;

    /// C++ `ICorFieldInfo::getThreadLocalFieldInfo` (corinfo.h:2958): the
    /// index under which the field's thread-static block is stored in TLS.
    fn get_thread_local_field_info(&self, field: FieldHandle, is_gc_type: bool) -> u32;

    /// C++ `ICorFieldInfo::getThreadLocalStaticBlocksInfo` (corinfo.h:2964).
    /// The EE-filled mirror struct is returned by value.
    fn get_thread_local_static_blocks_info(&self) -> CORINFO_THREAD_STATIC_BLOCKS_INFO;

    /// C++ `ICorFieldInfo::getThreadLocalStaticInfo_NativeAOT`
    /// (corinfo.h:2968). The EE-filled mirror struct is returned by value.
    fn get_thread_local_static_info_native_aot(&self) -> CORINFO_THREAD_STATIC_INFO_NATIVEAOT;

    /// C++ `ICorFieldInfo::getArrayOrStringLength` (corinfo.h:2977). `obj`
    /// must be non-null (header precondition). The C++ `-1` sentinel (the
    /// object is neither an array nor a string) becomes `None`.
    fn get_array_or_string_length(&self, obj: ObjectHandle) -> Option<u32>;

    /// C++ `ICorDynamicInfo::getThreadTLSIndex` (corinfo.h:3308). The second
    /// tuple element is the C++ `ppIndirection` out-param (prejit
    /// cookie-table indirection; null → `None`).
    fn get_thread_tls_index(&self) -> (u32, Option<NonNull<c_void>>);

    /// C++ `ICorDynamicInfo::getAddrOfCaptureThreadGlobal` (corinfo.h:3312).
    /// The returned pointer is EE-owned; the C++ null becomes `None`. The
    /// second element is the `ppIndirection` out-param, as in
    /// [`FieldQueries::get_thread_tls_index`].
    fn get_addr_of_capture_thread_global(&self) -> (Option<NonNull<i32>>, Option<NonNull<c_void>>);

    /// C++ `ICorDynamicInfo::getStaticFieldContent` (corinfo.h:3445).
    /// `buffer` receives the field's value (`buffer.len()` is the C++
    /// `bufferSize`); `value_offset` is the C++ parameter that defaults to
    /// 0. Returns `true` iff the constant value was available and copied
    /// into `buffer`.
    fn get_static_field_content(
        &self,
        field: FieldHandle,
        buffer: &mut [u8],
        value_offset: i32,
        ignore_movable_objects: bool,
    ) -> bool;

    /// C++ `ICorDynamicInfo::getStaticFieldCurrentClass` (corinfo.h:3471).
    /// The null class handle becomes `None`; the bool is the C++
    /// `pIsSpeculative` out-param (the type may still change over time).
    fn get_static_field_current_class(&self, field: FieldHandle) -> (Option<ClassHandle>, bool);

    /// C++ `ICorDynamicInfo::getFieldThreadLocalStoreID` (corinfo.h:3497).
    /// `field` must refer to a thread-local-store static (header
    /// precondition). The second element is the `ppIndirection` out-param,
    /// as in [`FieldQueries::get_thread_tls_index`].
    fn get_field_thread_local_store_id(&self, field: FieldHandle) -> (u32, Option<NonNull<c_void>>);
}

extern "C" {
    fn rokajit_ee_print_field_name(
        info: *mut ffi::ICorJitInfo,
        field: ffi::CORINFO_FIELD_HANDLE,
        buffer: *mut c_char,
        buffer_size: usize,
        p_required_buffer_size: *mut usize,
    ) -> usize;
    fn rokajit_ee_get_field_class(
        info: *mut ffi::ICorJitInfo,
        field: ffi::CORINFO_FIELD_HANDLE,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_field_type(
        info: *mut ffi::ICorJitInfo,
        field: ffi::CORINFO_FIELD_HANDLE,
        struct_type: *mut ffi::CORINFO_CLASS_HANDLE,
        field_owner_hint: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoType;
    fn rokajit_ee_get_field_offset(
        info: *mut ffi::ICorJitInfo,
        field: ffi::CORINFO_FIELD_HANDLE,
    ) -> u32;
    fn rokajit_ee_get_field_info(
        info: *mut ffi::ICorJitInfo,
        p_resolved_token: *mut CORINFO_RESOLVED_TOKEN,
        caller_handle: ffi::CORINFO_METHOD_HANDLE,
        flags: ffi::CORINFO_ACCESS_FLAGS,
        p_result: *mut CORINFO_FIELD_INFO,
    );
    fn rokajit_ee_get_thread_local_field_info(
        info: *mut ffi::ICorJitInfo,
        field: ffi::CORINFO_FIELD_HANDLE,
        is_gc_type: bool,
    ) -> u32;
    fn rokajit_ee_get_thread_local_static_blocks_info(
        info: *mut ffi::ICorJitInfo,
        p_info: *mut CORINFO_THREAD_STATIC_BLOCKS_INFO,
    );
    fn rokajit_ee_get_thread_local_static_info_native_aot(
        info: *mut ffi::ICorJitInfo,
        p_info: *mut CORINFO_THREAD_STATIC_INFO_NATIVEAOT,
    );
    fn rokajit_ee_is_field_static(
        info: *mut ffi::ICorJitInfo,
        fld_hnd: ffi::CORINFO_FIELD_HANDLE,
    ) -> bool;
    fn rokajit_ee_get_array_or_string_length(
        info: *mut ffi::ICorJitInfo,
        obj_hnd: ffi::CORINFO_OBJECT_HANDLE,
    ) -> c_int;
    fn rokajit_ee_get_thread_tls_index(
        info: *mut ffi::ICorJitInfo,
        pp_indirection: *mut *mut c_void,
    ) -> u32;
    fn rokajit_ee_get_addr_of_capture_thread_global(
        info: *mut ffi::ICorJitInfo,
        pp_indirection: *mut *mut c_void,
    ) -> *mut i32;
    fn rokajit_ee_get_static_field_content(
        info: *mut ffi::ICorJitInfo,
        field: ffi::CORINFO_FIELD_HANDLE,
        buffer: *mut u8,
        buffer_size: c_int,
        value_offset: c_int,
        ignore_movable_objects: bool,
    ) -> bool;
    fn rokajit_ee_get_static_field_current_class(
        info: *mut ffi::ICorJitInfo,
        field: ffi::CORINFO_FIELD_HANDLE,
        p_is_speculative: *mut bool,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_field_thread_local_store_id(
        info: *mut ffi::ICorJitInfo,
        field: ffi::CORINFO_FIELD_HANDLE,
        pp_indirection: *mut *mut c_void,
    ) -> u32;
}

impl FieldQueries for GasketEeInfo {
    fn get_field_type(&self, field: FieldHandle) -> (CorInfoType, Option<ClassHandle>) {
        let mut struct_type: ffi::CORINFO_CLASS_HANDLE = std::ptr::null_mut();
        let raw = unsafe {
            rokajit_ee_get_field_type(
                self.comp_raw(),
                field.as_raw(),
                &mut struct_type,
                std::ptr::null_mut(),
            )
        };
        let ty = CorInfoType::from_raw(raw)
            .expect("EE contract: getFieldType returns a valid CorInfoType");
        (ty, ClassHandle::from_raw(struct_type))
    }

    fn get_field_offset(&self, field: FieldHandle) -> u32 {
        unsafe { rokajit_ee_get_field_offset(self.comp_raw(), field.as_raw()) }
    }

    fn is_field_static(&self, field: FieldHandle) -> bool {
        unsafe { rokajit_ee_is_field_static(self.comp_raw(), field.as_raw()) }
    }

    fn get_field_info(
        &self,
        token: &mut CORINFO_RESOLVED_TOKEN,
        caller: MethodHandle,
        flags: u32,
    ) -> CORINFO_FIELD_INFO {
        zeroed_out(|result| unsafe {
            rokajit_ee_get_field_info(self.comp_raw(), token, caller.as_raw(), flags, result);
        })
    }

    fn print_field_name(&self, field: FieldHandle) -> String {
        print_object_string(|buf, size, required| unsafe {
            rokajit_ee_print_field_name(self.comp_raw(), field.as_raw(), buf.cast(), size, required)
        })
    }

    fn get_field_class(&self, field: FieldHandle) -> ClassHandle {
        let raw = unsafe { rokajit_ee_get_field_class(self.comp_raw(), field.as_raw()) };
        ClassHandle::from_raw(raw).expect("EE contract: getFieldClass never returns null")
    }

    fn get_thread_local_field_info(&self, field: FieldHandle, is_gc_type: bool) -> u32 {
        unsafe { rokajit_ee_get_thread_local_field_info(self.comp_raw(), field.as_raw(), is_gc_type) }
    }

    fn get_thread_local_static_blocks_info(&self) -> CORINFO_THREAD_STATIC_BLOCKS_INFO {
        zeroed_out(|info| unsafe {
            rokajit_ee_get_thread_local_static_blocks_info(self.comp_raw(), info)
        })
    }

    fn get_thread_local_static_info_native_aot(&self) -> CORINFO_THREAD_STATIC_INFO_NATIVEAOT {
        zeroed_out(|info| unsafe {
            rokajit_ee_get_thread_local_static_info_native_aot(self.comp_raw(), info)
        })
    }

    fn get_array_or_string_length(&self, obj: ObjectHandle) -> Option<u32> {
        let len = unsafe { rokajit_ee_get_array_or_string_length(self.comp_raw(), obj.as_raw()) };
        if len < 0 { None } else { Some(len as u32) }
    }

    fn get_thread_tls_index(&self) -> (u32, Option<NonNull<c_void>>) {
        let mut indirection: *mut c_void = std::ptr::null_mut();
        let index = unsafe { rokajit_ee_get_thread_tls_index(self.comp_raw(), &mut indirection) };
        (index, NonNull::new(indirection))
    }

    fn get_addr_of_capture_thread_global(&self) -> (Option<NonNull<i32>>, Option<NonNull<c_void>>) {
        let mut indirection: *mut c_void = std::ptr::null_mut();
        let addr =
            unsafe { rokajit_ee_get_addr_of_capture_thread_global(self.comp_raw(), &mut indirection) };
        (NonNull::new(addr), NonNull::new(indirection))
    }

    fn get_static_field_content(
        &self,
        field: FieldHandle,
        buffer: &mut [u8],
        value_offset: i32,
        ignore_movable_objects: bool,
    ) -> bool {
        unsafe {
            rokajit_ee_get_static_field_content(
                self.comp_raw(),
                field.as_raw(),
                buffer.as_mut_ptr(),
                buffer.len() as c_int,
                value_offset,
                ignore_movable_objects,
            )
        }
    }

    fn get_static_field_current_class(&self, field: FieldHandle) -> (Option<ClassHandle>, bool) {
        let mut is_speculative: bool = false;
        let raw = unsafe {
            rokajit_ee_get_static_field_current_class(
                self.comp_raw(),
                field.as_raw(),
                &mut is_speculative,
            )
        };
        (ClassHandle::from_raw(raw), is_speculative)
    }

    fn get_field_thread_local_store_id(&self, field: FieldHandle) -> (u32, Option<NonNull<c_void>>) {
        let mut indirection: *mut c_void = std::ptr::null_mut();
        let id = unsafe {
            rokajit_ee_get_field_thread_local_store_id(
                self.comp_raw(),
                field.as_raw(),
                &mut indirection,
            )
        };
        (id, NonNull::new(indirection))
    }
}
