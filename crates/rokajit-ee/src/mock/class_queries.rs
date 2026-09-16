use std::ffi::{c_void, CStr};
use std::ptr::NonNull;

use rokajit_ffi as ffi;
use rokajit_ffi::{CORINFO_CONST_LOOKUP, CORINFO_HELPER_DESC, CORINFO_RESOLVED_TOKEN};

use super::MockEe;
use crate::ee_info::ClassQueries;
use crate::enums::{
    ClassAttribs, CorInfoArrayIntrinsic, CorInfoClassId, CorInfoInitClassResult,
    CorInfoIsAccessAllowedResult, CorInfoType, CorInfoWasmType, GetTypeLayoutResult,
    TypeCompareState,
};
use crate::handles::{
    ClassHandle, ContextHandle, FieldHandle, MethodHandle, ObjectHandle, WasmTypeSymbolHandle,
};

impl ClassQueries for MockEe {
    fn as_cor_info_type(&self, cls: ClassHandle) -> CorInfoType {
        if let Some(&ty) = self.class_cor_info_types.get(&(cls.as_raw() as usize)) {
            return ty;
        }
        if self.classes.contains_key(&(cls.as_raw() as usize)) {
            CorInfoType::ValueClass
        } else {
            CorInfoType::Class
        }
    }

    fn is_value_class(&self, cls: ClassHandle) -> bool {
        self.classes.contains_key(&(cls.as_raw() as usize))
    }

    fn get_class_attribs(&self, _cls: ClassHandle) -> ClassAttribs {
        self.class_attribs
    }

    fn get_class_size(&self, cls: ClassHandle) -> u32 {
        self.classes
            .get(&(cls.as_raw() as usize))
            .map_or(8, |c| c.size)
    }

    fn get_type_for_primitive_numeric_class(&self, _cls: ClassHandle) -> Option<CorInfoType> {
        None
    }

    fn get_class_name_from_metadata(&self, cls: ClassHandle) -> Option<(String, Option<String>)> {
        // Canned (name, namespace) answers, keyed by the class handle's
        // raw value; absent handles answer "no metadata name".
        self.class_names.get(&(cls.as_raw() as usize)).cloned()
    }

    fn get_type_instantiation_argument(
        &self,
        _cls: ClassHandle,
        _index: u32,
    ) -> Option<ClassHandle> {
        None
    }

    fn get_method_instantiation_argument(
        &self,
        _ftn: MethodHandle,
        _index: u32,
    ) -> Option<ClassHandle> {
        None
    }

    fn print_class_name(&self, _cls: ClassHandle) -> String {
        String::new()
    }

    fn get_class_assembly_name(&self, _cls: ClassHandle) -> Option<String> {
        None
    }

    fn long_lifetime_malloc(&self, sz: usize) -> Option<NonNull<u8>> {
        // Fake it with a mock-lifetime buffer; long_lifetime_free is a no-op.
        Some(self.fake_alloc(sz))
    }

    fn long_lifetime_free(&self, _obj: NonNull<u8>) {}

    fn get_is_class_inited_flag_address(
        &self,
        _cls: ClassHandle,
    ) -> Option<(CORINFO_CONST_LOOKUP, i32)> {
        None
    }

    fn get_class_static_dynamic_info(&self, _cls: ClassHandle) -> Option<NonNull<u8>> {
        NonNull::new(self.static_dynamic_info? as *mut u8)
    }

    fn get_class_thread_static_dynamic_info(&self, _cls: ClassHandle) -> Option<NonNull<u8>> {
        NonNull::new(self.thread_static_dynamic_info? as *mut u8)
    }

    fn get_static_base_address(
        &self,
        _cls: ClassHandle,
        _is_gc: bool,
    ) -> Option<CORINFO_CONST_LOOKUP> {
        None
    }

    fn get_heap_class_size(&self, _cls: ClassHandle) -> u32 {
        8
    }

    fn can_allocate_on_stack(&self, _cls: ClassHandle) -> bool {
        false
    }

    fn get_class_alignment_requirement(&self, cls: ClassHandle, _double_align_hint: bool) -> u32 {
        self.classes
            .get(&(cls.as_raw() as usize))
            .map_or(8, |c| c.align)
    }

    fn get_class_gc_layout(&self, cls: ClassHandle, gc_ptrs: &mut [u8]) -> u32 {
        let Some(class) = self.classes.get(&(cls.as_raw() as usize)) else {
            return 0;
        };
        for slot in gc_ptrs.iter_mut() {
            *slot = ffi::CorInfoGCType_TYPE_GC_NONE as u8;
        }
        for &(offset, is_byref) in &class.gc_cells {
            let idx = (offset / 8) as usize;
            if let Some(slot) = gc_ptrs.get_mut(idx) {
                *slot = if is_byref {
                    ffi::CorInfoGCType_TYPE_GC_BYREF as u8
                } else {
                    ffi::CorInfoGCType_TYPE_GC_REF as u8
                };
            }
        }
        class.gc_cells.len() as u32
    }

    fn get_class_num_instance_fields(&self, _cls: ClassHandle) -> u32 {
        0
    }

    fn get_field_in_class(&self, _cls_hnd: ClassHandle, _num: i32) -> FieldHandle {
        // Non-null stand-in; the mock never dereferences handles.
        FieldHandle(1usize as ffi::CORINFO_FIELD_HANDLE)
    }

    fn get_type_layout(
        &self,
        _type_hnd: ClassHandle,
        _tree_nodes: &mut [ffi::CORINFO_TYPE_LAYOUT_NODE],
    ) -> (GetTypeLayoutResult, usize) {
        (GetTypeLayoutResult::Failure, 0)
    }

    fn check_method_modifier(
        &self,
        _h_method: MethodHandle,
        _modifier: &CStr,
        _optional: bool,
    ) -> bool {
        false
    }

    fn get_runtime_type_pointer(&self, _cls: ClassHandle) -> Option<ObjectHandle> {
        None
    }

    fn is_object_immutable(&self, _obj_ptr: ObjectHandle) -> bool {
        false
    }

    fn get_string_char(&self, _str_obj: ObjectHandle, _index: i32) -> Option<u16> {
        None
    }

    fn get_object_type(&self, obj_ptr: ObjectHandle) -> ClassHandle {
        // Reuse the object handle's address as a stable, non-null stand-in.
        ClassHandle(obj_ptr.0 as ffi::CORINFO_CLASS_HANDLE)
    }

    fn init_class(
        &self,
        field: Option<FieldHandle>,
        method: Option<MethodHandle>,
        _context: ContextHandle,
    ) -> CorInfoInitClassResult {
        // The method-prolog query (no field, no method — the 10.8
        // entry-cctor fix) has its own canned verdict, NOT_REQUIRED by
        // default; field/newobj-triggered queries keep the shared one.
        if field.is_none() && method.is_none() {
            return self
                .prolog_init_class
                .unwrap_or(CorInfoInitClassResult::EMPTY);
        }
        // The canned verdict; `CorInfoInitClassResult::EMPTY` is
        // NOT_REQUIRED (bit value 0) — the default.
        self.init_class_result
    }

    fn class_must_be_loaded_before_code_is_run(&self, cls: ClassHandle) {
        self.sink_log
            .borrow_mut()
            .push(format!("class_must_be_loaded_before_code_is_run({cls:?})"));
    }

    fn get_builtin_class(&self, class_id: CorInfoClassId) -> Option<ClassHandle> {
        self.builtin_classes.get(&class_id.to_raw()).copied()
    }

    fn get_type_for_primitive_value_class(&self, cls: ClassHandle) -> Option<CorInfoType> {
        // Canned through `class_cor_info_types` like `as_cor_info_type`:
        // a primitive answer there IS the primitive-value-class verdict.
        match self.class_cor_info_types.get(&(cls.as_raw() as usize)) {
            Some(&ty) if ty != CorInfoType::ValueClass && ty != CorInfoType::Class => Some(ty),
            _ => None,
        }
    }

    fn can_cast(&self, _child: ClassHandle, _parent: ClassHandle) -> bool {
        false
    }

    fn compare_types_for_cast(
        &self,
        _from_class: ClassHandle,
        _to_class: ClassHandle,
    ) -> TypeCompareState {
        TypeCompareState::May
    }

    fn compare_types_for_equality(
        &self,
        _cls1: ClassHandle,
        _cls2: ClassHandle,
    ) -> TypeCompareState {
        TypeCompareState::May
    }

    fn is_more_specific_type(&self, _cls1: ClassHandle, _cls2: ClassHandle) -> bool {
        false
    }

    fn is_exact_type(&self, _cls: ClassHandle) -> bool {
        false
    }

    fn is_generic_type(&self, _cls: ClassHandle) -> TypeCompareState {
        TypeCompareState::May
    }

    fn is_nullable_type(&self, _cls: ClassHandle) -> TypeCompareState {
        TypeCompareState::May
    }

    fn is_enum(&self, _cls: ClassHandle) -> (TypeCompareState, Option<ClassHandle>) {
        (TypeCompareState::MustNot, None)
    }

    fn get_parent_type(&self, _cls: ClassHandle) -> Option<ClassHandle> {
        None
    }

    fn get_child_type(&self, _cls_hnd: ClassHandle) -> (CorInfoType, Option<ClassHandle>) {
        (CorInfoType::Undef, None)
    }

    fn is_sd_array(&self, cls: ClassHandle) -> bool {
        // The canned happy path is an SZ array; designated handles can
        // the gate's false answer.
        !self.non_sd_arrays.contains(&(cls.as_raw() as usize))
    }

    fn get_array_rank(&self, _cls: ClassHandle) -> u32 {
        self.array_rank
    }

    fn get_array_intrinsic_id(&self, _ftn: MethodHandle) -> CorInfoArrayIntrinsic {
        CorInfoArrayIntrinsic::Illegal
    }

    fn get_array_initialization_data(
        &self,
        _field: FieldHandle,
        _size: u32,
    ) -> Option<NonNull<u8>> {
        None
    }

    fn can_access_class(
        &self,
        _p_resolved_token: &CORINFO_RESOLVED_TOKEN,
        _caller_handle: MethodHandle,
    ) -> (CorInfoIsAccessAllowedResult, CORINFO_HELPER_DESC) {
        (CorInfoIsAccessAllowedResult::Allowed, unsafe {
            std::mem::zeroed()
        })
    }

    fn get_system_v_amd64_pass_struct_in_register_descriptor(
        &self,
        struct_hnd: ClassHandle,
    ) -> Option<ffi::SYSTEMV_AMD64_CORINFO_STRUCT_REG_PASSING_DESCRIPTOR> {
        self.classes
            .get(&(struct_hnd.as_raw() as usize))
            .and_then(|c| c.sysv)
    }

    fn get_swift_lowering(&self, _struct_hnd: ClassHandle) -> ffi::CORINFO_SWIFT_LOWERING {
        unsafe { std::mem::zeroed() }
    }

    fn get_fp_struct_lowering(&self, _struct_hnd: ClassHandle) -> ffi::CORINFO_FPSTRUCT_LOWERING {
        unsafe { std::mem::zeroed() }
    }

    fn get_wasm_lowering(&self, _struct_hnd: ClassHandle) -> CorInfoWasmType {
        CorInfoWasmType::Void
    }

    fn get_address_alignment(&self, _address: *mut c_void) -> u32 {
        // Minimum alignment; trivially true for any address.
        1
    }

    fn get_object_content(
        &self,
        _obj: ObjectHandle,
        _buffer: &mut [u8],
        _value_offset: i32,
    ) -> bool {
        false
    }

    fn get_wasm_type_symbol(&self, _types: &mut [CorInfoWasmType]) -> Option<WasmTypeSymbolHandle> {
        None
    }
}
