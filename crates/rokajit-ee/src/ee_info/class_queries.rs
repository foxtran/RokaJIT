//! Class (type) metadata queries (C++ `ICorClassInfo`, corinfo.h:2479).

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::ptr::NonNull;

use rokajit_ffi as ffi;
use rokajit_ffi::{CORINFO_CONST_LOOKUP, CORINFO_HELPER_DESC, CORINFO_RESOLVED_TOKEN};

use super::wrap::{print_object_string, zeroed_out};
use super::GasketEeInfo;
use crate::enums::{
    ClassAttribs, CorInfoArrayIntrinsic, CorInfoClassId, CorInfoInitClassResult,
    CorInfoIsAccessAllowedResult, CorInfoType, CorInfoWasmType, GetTypeLayoutResult,
    TypeCompareState,
};
use crate::handles::{
    ClassHandle, ContextHandle, FieldHandle, MethodHandle, ObjectHandle, WasmTypeSymbolHandle,
};

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

/// Class (type) metadata queries (C++ `ICorClassInfo`).
pub trait ClassQueries {
    /// C++ `ICorClassInfo::asCorInfoType` (corinfo.h:2483).
    fn as_cor_info_type(&self, cls: ClassHandle) -> CorInfoType;

    /// C++ `ICorClassInfo::isValueClass` (corinfo.h:2519).
    fn is_value_class(&self, cls: ClassHandle) -> bool;

    /// C++ `ICorClassInfo::getClassAttribs` (corinfo.h:2522).
    fn get_class_attribs(&self, cls: ClassHandle) -> ClassAttribs;

    /// C++ `ICorClassInfo::getClassSize` (corinfo.h:2559).
    fn get_class_size(&self, cls: ClassHandle) -> u32;

    /// C++ `ICorClassInfo::getTypeForPrimitiveNumericClass`
    /// (corinfo.h:2809). The C++ `CORINFO_TYPE_UNDEF` sentinel becomes
    /// `None`.
    fn get_type_for_primitive_numeric_class(&self, cls: ClassHandle) -> Option<CorInfoType>;

    /// C++ `ICorClassInfo::getClassNameFromMetadata` (corinfo.h:2489).
    /// `None` when the class has no metadata name (the C++ nullptr).
    /// On success the namespace is the second tuple element (`None` when
    /// the EE reports no namespace). The wrapper copies both strings; the
    /// C++ storage is EE-lifetime, so there is nothing to free.
    fn get_class_name_from_metadata(&self, cls: ClassHandle) -> Option<(String, Option<String>)>;

    /// C++ `ICorClassInfo::getTypeInstantiationArgument` (corinfo.h:2496).
    /// `None` when the EE returns a null handle.
    fn get_type_instantiation_argument(
        &self,
        cls: ClassHandle,
        index: u32,
    ) -> Option<ClassHandle>;

    /// C++ `ICorClassInfo::getMethodInstantiationArgument` (corinfo.h:2503).
    /// `None` when the EE returns a null handle.
    fn get_method_instantiation_argument(
        &self,
        ftn: MethodHandle,
        index: u32,
    ) -> Option<ClassHandle>;

    /// C++ `ICorClassInfo::printClassName` (corinfo.h:2511), the
    /// `printObjectDescription` contract (corinfo.h:2444): the wrapper makes
    /// the two calls (size query, then copy) and returns the full,
    /// untruncated UTF-8 name.
    fn print_class_name(&self, cls: ClassHandle) -> String;

    /// C++ `ICorClassInfo::getClassAssemblyName` (corinfo.h:2527). `None`
    /// when the class has no assembly name (the C++ nullptr). The wrapper
    /// copies the string; the C++ storage is EE-lifetime, nothing to free.
    fn get_class_assembly_name(&self, cls: ClassHandle) -> Option<String>;

    /// C++ `ICorStaticInfo::LongLifetimeMalloc` (corinfo.h:2535).
    /// Process-lifetime allocation; `None` on OOM (the C++ nullptr). Every
    /// block handed out must be returned with
    /// [`ClassQueries::long_lifetime_free`]; the EE does not run destructors.
    fn long_lifetime_malloc(&self, sz: usize) -> Option<NonNull<u8>>;

    /// C++ `ICorStaticInfo::LongLifetimeFree` (corinfo.h:2536).
    fn long_lifetime_free(&self, obj: NonNull<u8>);

    /// C++ `ICorStaticInfo::getIsClassInitedFlagAddress` (corinfo.h:2538).
    /// `None` when the C++ returns false; otherwise the lookup and the
    /// additional `offset` out-param.
    fn get_is_class_inited_flag_address(
        &self,
        cls: ClassHandle,
    ) -> Option<(CORINFO_CONST_LOOKUP, i32)>;

    /// C++ `ICorStaticInfo::getClassStaticDynamicInfo` (corinfo.h:2544).
    /// `None` when the class has no dynamic statics info (the C++ nullptr).
    fn get_class_static_dynamic_info(&self, cls: ClassHandle) -> Option<NonNull<u8>>;

    /// C++ `ICorStaticInfo::getClassThreadStaticDynamicInfo`
    /// (corinfo.h:2548). `None` on the C++ nullptr.
    fn get_class_thread_static_dynamic_info(&self, cls: ClassHandle) -> Option<NonNull<u8>>;

    /// C++ `ICorStaticInfo::getStaticBaseAddress` (corinfo.h:2552). `None`
    /// when the C++ returns false.
    fn get_static_base_address(
        &self,
        cls: ClassHandle,
        is_gc: bool,
    ) -> Option<CORINFO_CONST_LOOKUP>;

    /// C++ `ICorClassInfo::getHeapClassSize` (corinfo.h:2564).
    fn get_heap_class_size(&self, cls: ClassHandle) -> u32;

    /// C++ `ICorClassInfo::canAllocateOnStack` (corinfo.h:2568).
    fn can_allocate_on_stack(&self, cls: ClassHandle) -> bool;

    /// C++ `ICorClassInfo::getClassAlignmentRequirement` (corinfo.h:2572).
    /// `double_align_hint` is the C++ `fDoubleAlignHint` (defaulted to false
    /// in the header; always explicit here).
    fn get_class_alignment_requirement(&self, cls: ClassHandle, double_align_hint: bool) -> u32;

    /// C++ `ICorClassInfo::getClassGClayout` (corinfo.h:2586). `gc_ptrs` is
    /// caller-allocated and must have `get_class_size(cls) /
    /// TARGET_POINTER_SIZE` entries for value classes, resp.
    /// `get_heap_class_size(cls) / TARGET_POINTER_SIZE` for reference types.
    /// Returns the number of GC pointers in the object.
    fn get_class_gc_layout(&self, cls: ClassHandle, gc_ptrs: &mut [u8]) -> u32;

    /// C++ `ICorClassInfo::getClassNumInstanceFields` (corinfo.h:2592).
    fn get_class_num_instance_fields(&self, cls: ClassHandle) -> u32;

    /// C++ `ICorClassInfo::getFieldInClass` (corinfo.h:2596). `num` indexes
    /// `[0, get_class_num_instance_fields(cls))`.
    fn get_field_in_class(&self, cls_hnd: ClassHandle, num: i32) -> FieldHandle;

    /// C++ `ICorClassInfo::getTypeLayout` (corinfo.h:2635). On entry
    /// `tree_nodes.len()` is the buffer capacity; returns the result and the
    /// number of nodes written (the C++ `numTreeNodes` in/out). `Partial`
    /// means the tree was truncated to fit. Field information (except GC
    /// pointers) is a hint only — see the header remarks.
    fn get_type_layout(
        &self,
        type_hnd: ClassHandle,
        tree_nodes: &mut [ffi::CORINFO_TYPE_LAYOUT_NODE],
    ) -> (GetTypeLayoutResult, usize);

    /// C++ `ICorClassInfo::checkMethodModifier` (corinfo.h:2640). `modifier`
    /// is a NUL-terminated metadata modifier name (e.g. `c"IsVolatile"`).
    fn check_method_modifier(
        &self,
        h_method: MethodHandle,
        modifier: &CStr,
        optional: bool,
    ) -> bool;

    /// C++ `ICorClassInfo::getRuntimeTypePointer` (corinfo.h:2707). `None`
    /// when the EE has no frozen RuntimeType object for the class (the C++
    /// nullptr).
    fn get_runtime_type_pointer(&self, cls: ClassHandle) -> Option<ObjectHandle>;

    /// C++ `ICorClassInfo::isObjectImmutable` (corinfo.h:2720).
    fn is_object_immutable(&self, obj_ptr: ObjectHandle) -> bool;

    /// C++ `ICorClassInfo::getStringChar` (corinfo.h:2736). `None` when the
    /// handle is not a string or `index` is out of bounds (the C++ false).
    fn get_string_char(&self, str_obj: ObjectHandle, index: i32) -> Option<u16>;

    /// C++ `ICorClassInfo::getObjectType` (corinfo.h:2751). Infallible: the
    /// EE always knows a frozen object's type.
    fn get_object_type(&self, obj_ptr: ObjectHandle) -> ClassHandle;

    /// C++ `ICorClassInfo::initClass` (corinfo.h:2775). `field: None` asks
    /// about the cctor trigger in the method prolog; `method: None` means
    /// the method being compiled (both per the header comment).
    fn init_class(
        &self,
        field: Option<FieldHandle>,
        method: Option<MethodHandle>,
        context: ContextHandle,
    ) -> CorInfoInitClassResult;

    /// C++ `ICorClassInfo::classMustBeLoadedBeforeCodeIsRun`
    /// (corinfo.h:2793). Records the load dependency; no return value.
    fn class_must_be_loaded_before_code_is_run(&self, cls: ClassHandle);

    /// C++ `ICorClassInfo::getBuiltinClass` (corinfo.h:2798). `None` when
    /// the EE returns a null handle for the id.
    fn get_builtin_class(&self, class_id: CorInfoClassId) -> Option<ClassHandle>;

    /// C++ `ICorClassInfo::getTypeForPrimitiveValueClass` (corinfo.h:2803).
    /// The C++ `CORINFO_TYPE_UNDEF` sentinel becomes `None`.
    fn get_type_for_primitive_value_class(&self, cls: ClassHandle) -> Option<CorInfoType>;

    /// C++ `ICorClassInfo::canCast` (corinfo.h:2815).
    fn can_cast(&self, child: ClassHandle, parent: ClassHandle) -> bool;

    /// C++ `ICorClassInfo::compareTypesForCast` (corinfo.h:2822).
    fn compare_types_for_cast(
        &self,
        from_class: ClassHandle,
        to_class: ClassHandle,
    ) -> TypeCompareState;

    /// C++ `ICorClassInfo::compareTypesForEquality` (corinfo.h:2829).
    fn compare_types_for_equality(
        &self,
        cls1: ClassHandle,
        cls2: ClassHandle,
    ) -> TypeCompareState;

    /// C++ `ICorClassInfo::isMoreSpecificType` (corinfo.h:2839). An
    /// optimization hint only; no correctness implications.
    fn is_more_specific_type(&self, cls1: ClassHandle, cls2: ClassHandle) -> bool;

    /// C++ `ICorClassInfo::isExactType` (corinfo.h:2845).
    fn is_exact_type(&self, cls: ClassHandle) -> bool;

    /// C++ `ICorClassInfo::isGenericType` (corinfo.h:2850).
    fn is_generic_type(&self, cls: ClassHandle) -> TypeCompareState;

    /// C++ `ICorClassInfo::isNullableType` (corinfo.h:2855).
    fn is_nullable_type(&self, cls: ClassHandle) -> TypeCompareState;

    /// C++ `ICorClassInfo::isEnum` (corinfo.h:2864). The second tuple
    /// element is the C++ `underlyingType` out-param (`None` when the EE
    /// leaves it null, e.g. when a runtime check is required).
    fn is_enum(&self, cls: ClassHandle) -> (TypeCompareState, Option<ClassHandle>);

    /// C++ `ICorClassInfo::getParentType` (corinfo.h:2872). `None` when the
    /// C++ returns 0, i.e. `cls` is `System.Object`.
    fn get_parent_type(&self, cls: ClassHandle) -> Option<ClassHandle>;

    /// C++ `ICorClassInfo::getChildType` (corinfo.h:2880). The second tuple
    /// element is the C++ `clsRet` out-param: set (`Some`) when the child
    /// type is not primitive.
    fn get_child_type(&self, cls_hnd: ClassHandle) -> (CorInfoType, Option<ClassHandle>);

    /// C++ `ICorClassInfo::isSDArray` (corinfo.h:2886).
    fn is_sd_array(&self, cls: ClassHandle) -> bool;

    /// C++ `ICorClassInfo::getArrayRank` (corinfo.h:2891).
    fn get_array_rank(&self, cls: ClassHandle) -> u32;

    /// C++ `ICorClassInfo::getArrayIntrinsicID` (corinfo.h:2896).
    fn get_array_intrinsic_id(&self, ftn: MethodHandle) -> CorInfoArrayIntrinsic;

    /// C++ `ICorClassInfo::getArrayInitializationData` (corinfo.h:2901).
    /// `None` when there is no static initialization blob (the C++ nullptr).
    /// The returned buffer is EE-owned, `size` bytes.
    fn get_array_initialization_data(&self, field: FieldHandle, size: u32)
        -> Option<NonNull<u8>>;

    /// C++ `ICorClassInfo::canAccessClass` (corinfo.h:2907). The
    /// `CORINFO_HELPER_DESC` is the C++ `pAccessHelper` out-param; it is
    /// meaningful only when the verdict is not `Allowed`.
    fn can_access_class(
        &self,
        p_resolved_token: &CORINFO_RESOLVED_TOKEN,
        caller_handle: MethodHandle,
    ) -> (CorInfoIsAccessAllowedResult, CORINFO_HELPER_DESC);

    /// C++ `ICorStaticInfo::getSystemVAmd64PassStructInRegisterDescriptor`
    /// (corinfo.h:3244). Only valid on a System V target; `None` when the
    /// C++ returns false.
    fn get_system_v_amd64_pass_struct_in_register_descriptor(
        &self,
        struct_hnd: ClassHandle,
    ) -> Option<ffi::SYSTEMV_AMD64_CORINFO_STRUCT_REG_PASSING_DESCRIPTOR>;

    /// C++ `ICorStaticInfo::getSwiftLowering` (corinfo.h:3250).
    fn get_swift_lowering(&self, struct_hnd: ClassHandle) -> ffi::CORINFO_SWIFT_LOWERING;

    /// C++ `ICorStaticInfo::getFpStructLowering` (corinfo.h:3254).
    fn get_fp_struct_lowering(&self, struct_hnd: ClassHandle)
        -> ffi::CORINFO_FPSTRUCT_LOWERING;

    /// C++ `ICorStaticInfo::getWasmLowering` (corinfo.h:3258).
    /// [`CorInfoWasmType::Void`] means the struct must be passed/returned by
    /// reference.
    fn get_wasm_lowering(&self, struct_hnd: ClassHandle) -> CorInfoWasmType;

    /// C++ `ICorStaticInfo::getAddressAlignment` (corinfo.h:3265).
    /// Guaranteed alignment, in bytes, of the data at `address` (a
    /// relocation target such as a static, RVA, or frozen-data blob).
    fn get_address_alignment(&self, address: *mut c_void) -> u32;

    /// C++ `ICorDynamicInfo::getObjectContent` (corinfo.h:3453). Copies the
    /// frozen object's bytes into `buffer` at `value_offset`; returns false
    /// when the content was unavailable.
    fn get_object_content(
        &self,
        obj: ObjectHandle,
        buffer: &mut [u8],
        value_offset: i32,
    ) -> bool;

    /// C++ `ICorDynamicInfo::getWasmTypeSymbol` (corinfo.h:3557). `types`
    /// describes the signature; `None` when the EE returns a null symbol
    /// handle.
    fn get_wasm_type_symbol(
        &self,
        types: &mut [CorInfoWasmType],
    ) -> Option<WasmTypeSymbolHandle>;
}

// ---------------------------------------------------------------------------
// The gasket-backed implementation
// ---------------------------------------------------------------------------

extern "C" {
    fn rokajit_ee_as_cor_info_type(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoType;
    fn rokajit_ee_get_class_name_from_metadata(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
        namespace_name: *mut *const c_char,
    ) -> *const c_char;
    fn rokajit_ee_get_type_instantiation_argument(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
        index: c_uint,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_method_instantiation_argument(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        index: c_uint,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_print_class_name(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
        buffer: *mut c_char,
        buffer_size: usize,
        p_required_buffer_size: *mut usize,
    ) -> usize;
    fn rokajit_ee_is_value_class(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> bool;
    fn rokajit_ee_get_class_attribs(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> u32;
    fn rokajit_ee_get_class_assembly_name(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> *const c_char;
    fn rokajit_ee_long_lifetime_malloc(info: *mut ffi::ICorJitInfo, sz: usize) -> *mut c_void;
    fn rokajit_ee_long_lifetime_free(info: *mut ffi::ICorJitInfo, obj: *mut c_void);
    fn rokajit_ee_get_is_class_inited_flag_address(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
        addr: *mut CORINFO_CONST_LOOKUP,
        offset: *mut c_int,
    ) -> bool;
    fn rokajit_ee_get_class_static_dynamic_info(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> *mut c_void;
    fn rokajit_ee_get_class_thread_static_dynamic_info(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> *mut c_void;
    fn rokajit_ee_get_static_base_address(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
        is_gc: bool,
        addr: *mut CORINFO_CONST_LOOKUP,
    ) -> bool;
    fn rokajit_ee_get_class_size(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> c_uint;
    fn rokajit_ee_get_heap_class_size(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> c_uint;
    fn rokajit_ee_can_allocate_on_stack(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> bool;
    fn rokajit_ee_get_class_alignment_requirement(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
        f_double_align_hint: bool,
    ) -> c_uint;
    fn rokajit_ee_get_class_gc_layout(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
        gc_ptrs: *mut u8,
    ) -> c_uint;
    fn rokajit_ee_get_class_num_instance_fields(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> c_uint;
    fn rokajit_ee_get_field_in_class(
        info: *mut ffi::ICorJitInfo,
        cls_hnd: ffi::CORINFO_CLASS_HANDLE,
        num: i32,
    ) -> ffi::CORINFO_FIELD_HANDLE;
    fn rokajit_ee_get_type_layout(
        info: *mut ffi::ICorJitInfo,
        type_hnd: ffi::CORINFO_CLASS_HANDLE,
        tree_nodes: *mut ffi::CORINFO_TYPE_LAYOUT_NODE,
        num_tree_nodes: *mut usize,
    ) -> ffi::GetTypeLayoutResult;
    fn rokajit_ee_check_method_modifier(
        info: *mut ffi::ICorJitInfo,
        h_method: ffi::CORINFO_METHOD_HANDLE,
        modifier: *const c_char,
        f_optional: bool,
    ) -> bool;
    fn rokajit_ee_get_runtime_type_pointer(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CORINFO_OBJECT_HANDLE;
    fn rokajit_ee_is_object_immutable(
        info: *mut ffi::ICorJitInfo,
        obj_ptr: ffi::CORINFO_OBJECT_HANDLE,
    ) -> bool;
    fn rokajit_ee_get_string_char(
        info: *mut ffi::ICorJitInfo,
        str_obj: ffi::CORINFO_OBJECT_HANDLE,
        index: c_int,
        value: *mut u16,
    ) -> bool;
    fn rokajit_ee_get_object_type(
        info: *mut ffi::ICorJitInfo,
        obj_ptr: ffi::CORINFO_OBJECT_HANDLE,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_init_class(
        info: *mut ffi::ICorJitInfo,
        field: ffi::CORINFO_FIELD_HANDLE,
        method: ffi::CORINFO_METHOD_HANDLE,
        context: ffi::CORINFO_CONTEXT_HANDLE,
    ) -> ffi::CorInfoInitClassResult;
    fn rokajit_ee_class_must_be_loaded_before_code_is_run(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    );
    fn rokajit_ee_get_builtin_class(
        info: *mut ffi::ICorJitInfo,
        class_id: ffi::CorInfoClassId,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_type_for_primitive_value_class(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoType;
    fn rokajit_ee_get_type_for_primitive_numeric_class(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoType;
    fn rokajit_ee_can_cast(
        info: *mut ffi::ICorJitInfo,
        child: ffi::CORINFO_CLASS_HANDLE,
        parent: ffi::CORINFO_CLASS_HANDLE,
    ) -> bool;
    fn rokajit_ee_compare_types_for_cast(
        info: *mut ffi::ICorJitInfo,
        from_class: ffi::CORINFO_CLASS_HANDLE,
        to_class: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::TypeCompareState;
    fn rokajit_ee_compare_types_for_equality(
        info: *mut ffi::ICorJitInfo,
        cls1: ffi::CORINFO_CLASS_HANDLE,
        cls2: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::TypeCompareState;
    fn rokajit_ee_is_more_specific_type(
        info: *mut ffi::ICorJitInfo,
        cls1: ffi::CORINFO_CLASS_HANDLE,
        cls2: ffi::CORINFO_CLASS_HANDLE,
    ) -> bool;
    fn rokajit_ee_is_exact_type(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> bool;
    fn rokajit_ee_is_generic_type(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::TypeCompareState;
    fn rokajit_ee_is_nullable_type(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::TypeCompareState;
    fn rokajit_ee_is_enum(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
        underlying_type: *mut ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::TypeCompareState;
    fn rokajit_ee_get_parent_type(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_child_type(
        info: *mut ffi::ICorJitInfo,
        cls_hnd: ffi::CORINFO_CLASS_HANDLE,
        cls_ret: *mut ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoType;
    fn rokajit_ee_is_sd_array(info: *mut ffi::ICorJitInfo, cls: ffi::CORINFO_CLASS_HANDLE)
        -> bool;
    fn rokajit_ee_get_array_rank(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> c_uint;
    fn rokajit_ee_get_array_intrinsic_id(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
    ) -> ffi::CorInfoArrayIntrinsic;
    fn rokajit_ee_get_array_initialization_data(
        info: *mut ffi::ICorJitInfo,
        field: ffi::CORINFO_FIELD_HANDLE,
        size: u32,
    ) -> *mut c_void;
    fn rokajit_ee_can_access_class(
        info: *mut ffi::ICorJitInfo,
        p_resolved_token: *mut CORINFO_RESOLVED_TOKEN,
        caller_handle: ffi::CORINFO_METHOD_HANDLE,
        p_access_helper: *mut CORINFO_HELPER_DESC,
    ) -> ffi::CorInfoIsAccessAllowedResult;
    fn rokajit_ee_get_system_v_amd64_pass_struct_in_register_descriptor(
        info: *mut ffi::ICorJitInfo,
        struct_hnd: ffi::CORINFO_CLASS_HANDLE,
        struct_pass_in_reg_desc_ptr: *mut ffi::SYSTEMV_AMD64_CORINFO_STRUCT_REG_PASSING_DESCRIPTOR,
    ) -> bool;
    fn rokajit_ee_get_swift_lowering(
        info: *mut ffi::ICorJitInfo,
        struct_hnd: ffi::CORINFO_CLASS_HANDLE,
        p_lowering: *mut ffi::CORINFO_SWIFT_LOWERING,
    );
    fn rokajit_ee_get_fp_struct_lowering(
        info: *mut ffi::ICorJitInfo,
        struct_hnd: ffi::CORINFO_CLASS_HANDLE,
        p_lowering: *mut ffi::CORINFO_FPSTRUCT_LOWERING,
    );
    fn rokajit_ee_get_wasm_lowering(
        info: *mut ffi::ICorJitInfo,
        struct_hnd: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoWasmType;
    fn rokajit_ee_get_address_alignment(
        info: *mut ffi::ICorJitInfo,
        address: *mut c_void,
    ) -> u32;
    fn rokajit_ee_get_object_content(
        info: *mut ffi::ICorJitInfo,
        obj: ffi::CORINFO_OBJECT_HANDLE,
        buffer: *mut u8,
        buffer_size: c_int,
        value_offset: c_int,
    ) -> bool;
    fn rokajit_ee_get_wasm_type_symbol(
        info: *mut ffi::ICorJitInfo,
        types: *mut ffi::CorInfoWasmType,
        types_size: usize,
    ) -> ffi::CORINFO_WASM_TYPE_SYMBOL_HANDLE;
}

/// Copies an EE-lifetime C string into an owned `String`. Null-checked by
/// the callers before this is invoked.
unsafe fn copy_c_str(raw: *const c_char) -> String {
    unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned()
}

impl ClassQueries for GasketEeInfo {
    fn as_cor_info_type(&self, cls: ClassHandle) -> CorInfoType {
        let raw = unsafe { rokajit_ee_as_cor_info_type(self.comp_raw(), cls.as_raw()) };
        // A value outside the enum means the headers grew; Undef is the
        // conservative fallback.
        CorInfoType::from_raw(raw).unwrap_or(CorInfoType::Undef)
    }

    fn is_value_class(&self, cls: ClassHandle) -> bool {
        unsafe { rokajit_ee_is_value_class(self.comp_raw(), cls.as_raw()) }
    }

    fn get_class_attribs(&self, cls: ClassHandle) -> ClassAttribs {
        let raw = unsafe { rokajit_ee_get_class_attribs(self.comp_raw(), cls.as_raw()) };
        ClassAttribs::from_raw(raw)
    }

    fn get_class_size(&self, cls: ClassHandle) -> u32 {
        unsafe { rokajit_ee_get_class_size(self.comp_raw(), cls.as_raw()) }
    }

    fn get_type_for_primitive_numeric_class(&self, cls: ClassHandle) -> Option<CorInfoType> {
        let raw = unsafe {
            rokajit_ee_get_type_for_primitive_numeric_class(self.comp_raw(), cls.as_raw())
        };
        CorInfoType::from_raw(raw).filter(|&t| t != CorInfoType::Undef)
    }

    fn get_class_name_from_metadata(&self, cls: ClassHandle) -> Option<(String, Option<String>)> {
        let mut namespace: *const c_char = std::ptr::null();
        let raw = unsafe {
            rokajit_ee_get_class_name_from_metadata(self.comp_raw(), cls.as_raw(), &mut namespace)
        };
        if raw.is_null() {
            return None;
        }
        let name = unsafe { copy_c_str(raw) };
        let namespace = if namespace.is_null() {
            None
        } else {
            Some(unsafe { copy_c_str(namespace) })
        };
        Some((name, namespace))
    }

    fn get_type_instantiation_argument(
        &self,
        cls: ClassHandle,
        index: u32,
    ) -> Option<ClassHandle> {
        let raw = unsafe {
            rokajit_ee_get_type_instantiation_argument(self.comp_raw(), cls.as_raw(), index)
        };
        ClassHandle::from_raw(raw)
    }

    fn get_method_instantiation_argument(
        &self,
        ftn: MethodHandle,
        index: u32,
    ) -> Option<ClassHandle> {
        let raw = unsafe {
            rokajit_ee_get_method_instantiation_argument(self.comp_raw(), ftn.as_raw(), index)
        };
        ClassHandle::from_raw(raw)
    }

    fn print_class_name(&self, cls: ClassHandle) -> String {
        print_object_string(|buf, size, required| unsafe {
            rokajit_ee_print_class_name(self.comp_raw(), cls.as_raw(), buf.cast(), size, required)
        })
    }

    fn get_class_assembly_name(&self, cls: ClassHandle) -> Option<String> {
        let raw = unsafe { rokajit_ee_get_class_assembly_name(self.comp_raw(), cls.as_raw()) };
        if raw.is_null() {
            None
        } else {
            Some(unsafe { copy_c_str(raw) })
        }
    }

    fn long_lifetime_malloc(&self, sz: usize) -> Option<NonNull<u8>> {
        NonNull::new(unsafe { rokajit_ee_long_lifetime_malloc(self.comp_raw(), sz) } as *mut u8)
    }

    fn long_lifetime_free(&self, obj: NonNull<u8>) {
        unsafe { rokajit_ee_long_lifetime_free(self.comp_raw(), obj.as_ptr() as *mut c_void) }
    }

    fn get_is_class_inited_flag_address(
        &self,
        cls: ClassHandle,
    ) -> Option<(CORINFO_CONST_LOOKUP, i32)> {
        let mut offset: i32 = 0;
        let mut ok = false;
        let addr = zeroed_out(|addr| {
            ok = unsafe {
                rokajit_ee_get_is_class_inited_flag_address(
                    self.comp_raw(),
                    cls.as_raw(),
                    addr,
                    &mut offset,
                )
            };
        });
        if ok { Some((addr, offset)) } else { None }
    }

    fn get_class_static_dynamic_info(&self, cls: ClassHandle) -> Option<NonNull<u8>> {
        NonNull::new(unsafe {
            rokajit_ee_get_class_static_dynamic_info(self.comp_raw(), cls.as_raw())
        } as *mut u8)
    }

    fn get_class_thread_static_dynamic_info(&self, cls: ClassHandle) -> Option<NonNull<u8>> {
        NonNull::new(unsafe {
            rokajit_ee_get_class_thread_static_dynamic_info(self.comp_raw(), cls.as_raw())
        } as *mut u8)
    }

    fn get_static_base_address(
        &self,
        cls: ClassHandle,
        is_gc: bool,
    ) -> Option<CORINFO_CONST_LOOKUP> {
        let mut ok = false;
        let addr = zeroed_out(|addr| {
            ok = unsafe {
                rokajit_ee_get_static_base_address(self.comp_raw(), cls.as_raw(), is_gc, addr)
            };
        });
        if ok { Some(addr) } else { None }
    }

    fn get_heap_class_size(&self, cls: ClassHandle) -> u32 {
        unsafe { rokajit_ee_get_heap_class_size(self.comp_raw(), cls.as_raw()) }
    }

    fn can_allocate_on_stack(&self, cls: ClassHandle) -> bool {
        unsafe { rokajit_ee_can_allocate_on_stack(self.comp_raw(), cls.as_raw()) }
    }

    fn get_class_alignment_requirement(&self, cls: ClassHandle, double_align_hint: bool) -> u32 {
        unsafe {
            rokajit_ee_get_class_alignment_requirement(
                self.comp_raw(),
                cls.as_raw(),
                double_align_hint,
            )
        }
    }

    fn get_class_gc_layout(&self, cls: ClassHandle, gc_ptrs: &mut [u8]) -> u32 {
        unsafe {
            rokajit_ee_get_class_gc_layout(self.comp_raw(), cls.as_raw(), gc_ptrs.as_mut_ptr())
        }
    }

    fn get_class_num_instance_fields(&self, cls: ClassHandle) -> u32 {
        unsafe { rokajit_ee_get_class_num_instance_fields(self.comp_raw(), cls.as_raw()) }
    }

    fn get_field_in_class(&self, cls_hnd: ClassHandle, num: i32) -> FieldHandle {
        // Infallible for in-range `num`; a conforming EE never returns null.
        let raw = unsafe { rokajit_ee_get_field_in_class(self.comp_raw(), cls_hnd.as_raw(), num) };
        FieldHandle::from_raw(raw)
            .expect("EE returned null from getFieldInClass for an in-range index")
    }

    fn get_type_layout(
        &self,
        type_hnd: ClassHandle,
        tree_nodes: &mut [ffi::CORINFO_TYPE_LAYOUT_NODE],
    ) -> (GetTypeLayoutResult, usize) {
        let mut written: usize = tree_nodes.len();
        let raw = unsafe {
            rokajit_ee_get_type_layout(
                self.comp_raw(),
                type_hnd.as_raw(),
                tree_nodes.as_mut_ptr(),
                &mut written,
            )
        };
        (
            GetTypeLayoutResult::from_raw(raw).unwrap_or(GetTypeLayoutResult::Failure),
            written,
        )
    }

    fn check_method_modifier(
        &self,
        h_method: MethodHandle,
        modifier: &CStr,
        optional: bool,
    ) -> bool {
        unsafe {
            rokajit_ee_check_method_modifier(
                self.comp_raw(),
                h_method.as_raw(),
                modifier.as_ptr(),
                optional,
            )
        }
    }

    fn get_runtime_type_pointer(&self, cls: ClassHandle) -> Option<ObjectHandle> {
        let raw = unsafe { rokajit_ee_get_runtime_type_pointer(self.comp_raw(), cls.as_raw()) };
        ObjectHandle::from_raw(raw)
    }

    fn is_object_immutable(&self, obj_ptr: ObjectHandle) -> bool {
        unsafe { rokajit_ee_is_object_immutable(self.comp_raw(), obj_ptr.as_raw()) }
    }

    fn get_string_char(&self, str_obj: ObjectHandle, index: i32) -> Option<u16> {
        let mut value: u16 = 0;
        let ok = unsafe {
            rokajit_ee_get_string_char(self.comp_raw(), str_obj.as_raw(), index, &mut value)
        };
        if ok { Some(value) } else { None }
    }

    fn get_object_type(&self, obj_ptr: ObjectHandle) -> ClassHandle {
        let raw = unsafe { rokajit_ee_get_object_type(self.comp_raw(), obj_ptr.as_raw()) };
        ClassHandle::from_raw(raw).expect("EE returned null from getObjectType")
    }

    fn init_class(
        &self,
        field: Option<FieldHandle>,
        method: Option<MethodHandle>,
        context: ContextHandle,
    ) -> CorInfoInitClassResult {
        let field = field.map_or(std::ptr::null_mut(), |f| f.as_raw());
        let method = method.map_or(std::ptr::null_mut(), |m| m.as_raw());
        let raw = unsafe { rokajit_ee_init_class(self.comp_raw(), field, method, context.as_raw()) };
        CorInfoInitClassResult::from_raw(raw)
    }

    fn class_must_be_loaded_before_code_is_run(&self, cls: ClassHandle) {
        unsafe {
            rokajit_ee_class_must_be_loaded_before_code_is_run(self.comp_raw(), cls.as_raw())
        }
    }

    fn get_builtin_class(&self, class_id: CorInfoClassId) -> Option<ClassHandle> {
        let raw = unsafe { rokajit_ee_get_builtin_class(self.comp_raw(), class_id.to_raw()) };
        ClassHandle::from_raw(raw)
    }

    fn get_type_for_primitive_value_class(&self, cls: ClassHandle) -> Option<CorInfoType> {
        let raw = unsafe {
            rokajit_ee_get_type_for_primitive_value_class(self.comp_raw(), cls.as_raw())
        };
        CorInfoType::from_raw(raw).filter(|&t| t != CorInfoType::Undef)
    }

    fn can_cast(&self, child: ClassHandle, parent: ClassHandle) -> bool {
        unsafe { rokajit_ee_can_cast(self.comp_raw(), child.as_raw(), parent.as_raw()) }
    }

    fn compare_types_for_cast(
        &self,
        from_class: ClassHandle,
        to_class: ClassHandle,
    ) -> TypeCompareState {
        let raw = unsafe {
            rokajit_ee_compare_types_for_cast(self.comp_raw(), from_class.as_raw(), to_class.as_raw())
        };
        // Unknown values mean the headers grew; May (runtime check) is the
        // conservative fallback.
        TypeCompareState::from_raw(raw).unwrap_or(TypeCompareState::May)
    }

    fn compare_types_for_equality(
        &self,
        cls1: ClassHandle,
        cls2: ClassHandle,
    ) -> TypeCompareState {
        let raw = unsafe {
            rokajit_ee_compare_types_for_equality(self.comp_raw(), cls1.as_raw(), cls2.as_raw())
        };
        TypeCompareState::from_raw(raw).unwrap_or(TypeCompareState::May)
    }

    fn is_more_specific_type(&self, cls1: ClassHandle, cls2: ClassHandle) -> bool {
        unsafe { rokajit_ee_is_more_specific_type(self.comp_raw(), cls1.as_raw(), cls2.as_raw()) }
    }

    fn is_exact_type(&self, cls: ClassHandle) -> bool {
        unsafe { rokajit_ee_is_exact_type(self.comp_raw(), cls.as_raw()) }
    }

    fn is_generic_type(&self, cls: ClassHandle) -> TypeCompareState {
        let raw = unsafe { rokajit_ee_is_generic_type(self.comp_raw(), cls.as_raw()) };
        TypeCompareState::from_raw(raw).unwrap_or(TypeCompareState::May)
    }

    fn is_nullable_type(&self, cls: ClassHandle) -> TypeCompareState {
        let raw = unsafe { rokajit_ee_is_nullable_type(self.comp_raw(), cls.as_raw()) };
        TypeCompareState::from_raw(raw).unwrap_or(TypeCompareState::May)
    }

    fn is_enum(&self, cls: ClassHandle) -> (TypeCompareState, Option<ClassHandle>) {
        let mut underlying: ffi::CORINFO_CLASS_HANDLE = std::ptr::null_mut();
        let raw = unsafe { rokajit_ee_is_enum(self.comp_raw(), cls.as_raw(), &mut underlying) };
        (
            TypeCompareState::from_raw(raw).unwrap_or(TypeCompareState::May),
            ClassHandle::from_raw(underlying),
        )
    }

    fn get_parent_type(&self, cls: ClassHandle) -> Option<ClassHandle> {
        let raw = unsafe { rokajit_ee_get_parent_type(self.comp_raw(), cls.as_raw()) };
        ClassHandle::from_raw(raw)
    }

    fn get_child_type(&self, cls_hnd: ClassHandle) -> (CorInfoType, Option<ClassHandle>) {
        let mut cls_ret: ffi::CORINFO_CLASS_HANDLE = std::ptr::null_mut();
        let raw = unsafe {
            rokajit_ee_get_child_type(self.comp_raw(), cls_hnd.as_raw(), &mut cls_ret)
        };
        (
            CorInfoType::from_raw(raw).unwrap_or(CorInfoType::Undef),
            ClassHandle::from_raw(cls_ret),
        )
    }

    fn is_sd_array(&self, cls: ClassHandle) -> bool {
        unsafe { rokajit_ee_is_sd_array(self.comp_raw(), cls.as_raw()) }
    }

    fn get_array_rank(&self, cls: ClassHandle) -> u32 {
        unsafe { rokajit_ee_get_array_rank(self.comp_raw(), cls.as_raw()) }
    }

    fn get_array_intrinsic_id(&self, ftn: MethodHandle) -> CorInfoArrayIntrinsic {
        let raw = unsafe { rokajit_ee_get_array_intrinsic_id(self.comp_raw(), ftn.as_raw()) };
        CorInfoArrayIntrinsic::from_raw(raw).unwrap_or(CorInfoArrayIntrinsic::Illegal)
    }

    fn get_array_initialization_data(
        &self,
        field: FieldHandle,
        size: u32,
    ) -> Option<NonNull<u8>> {
        NonNull::new(unsafe {
            rokajit_ee_get_array_initialization_data(self.comp_raw(), field.as_raw(), size)
        } as *mut u8)
    }

    fn can_access_class(
        &self,
        p_resolved_token: &CORINFO_RESOLVED_TOKEN,
        caller_handle: MethodHandle,
    ) -> (CorInfoIsAccessAllowedResult, CORINFO_HELPER_DESC) {
        let mut raw: ffi::CorInfoIsAccessAllowedResult = 0;
        let helper = zeroed_out(|helper| {
            raw = unsafe {
                rokajit_ee_can_access_class(
                    self.comp_raw(),
                    // The C++ signature is non-const, but canAccessClass does not
                    // mutate the token.
                    p_resolved_token as *const CORINFO_RESOLVED_TOKEN as *mut CORINFO_RESOLVED_TOKEN,
                    caller_handle.as_raw(),
                    helper,
                )
            };
        });
        (
            // Illegal is the conservative fallback for out-of-enum values.
            CorInfoIsAccessAllowedResult::from_raw(raw)
                .unwrap_or(CorInfoIsAccessAllowedResult::Illegal),
            helper,
        )
    }

    fn get_system_v_amd64_pass_struct_in_register_descriptor(
        &self,
        struct_hnd: ClassHandle,
    ) -> Option<ffi::SYSTEMV_AMD64_CORINFO_STRUCT_REG_PASSING_DESCRIPTOR> {
        // Zeroed matches the C++ default constructor (Initialize()).
        let mut ok = false;
        let desc = zeroed_out(|desc| {
            ok = unsafe {
                rokajit_ee_get_system_v_amd64_pass_struct_in_register_descriptor(
                    self.comp_raw(),
                    struct_hnd.as_raw(),
                    desc,
                )
            };
        });
        if ok { Some(desc) } else { None }
    }

    fn get_swift_lowering(&self, struct_hnd: ClassHandle) -> ffi::CORINFO_SWIFT_LOWERING {
        zeroed_out(|lowering| unsafe {
            rokajit_ee_get_swift_lowering(self.comp_raw(), struct_hnd.as_raw(), lowering)
        })
    }

    fn get_fp_struct_lowering(
        &self,
        struct_hnd: ClassHandle,
    ) -> ffi::CORINFO_FPSTRUCT_LOWERING {
        zeroed_out(|lowering| unsafe {
            rokajit_ee_get_fp_struct_lowering(self.comp_raw(), struct_hnd.as_raw(), lowering)
        })
    }

    fn get_wasm_lowering(&self, struct_hnd: ClassHandle) -> CorInfoWasmType {
        let raw = unsafe { rokajit_ee_get_wasm_lowering(self.comp_raw(), struct_hnd.as_raw()) };
        // Void (pass by reference) is the conservative fallback.
        CorInfoWasmType::from_raw(raw).unwrap_or(CorInfoWasmType::Void)
    }

    fn get_address_alignment(&self, address: *mut c_void) -> u32 {
        unsafe { rokajit_ee_get_address_alignment(self.comp_raw(), address) }
    }

    fn get_object_content(
        &self,
        obj: ObjectHandle,
        buffer: &mut [u8],
        value_offset: i32,
    ) -> bool {
        unsafe {
            rokajit_ee_get_object_content(
                self.comp_raw(),
                obj.as_raw(),
                buffer.as_mut_ptr(),
                buffer.len() as c_int,
                value_offset,
            )
        }
    }

    fn get_wasm_type_symbol(
        &self,
        types: &mut [CorInfoWasmType],
    ) -> Option<WasmTypeSymbolHandle> {
        // CorInfoWasmType is #[repr(u32)], layout-identical to the bindgen
        // c_uint alias the forwarder expects.
        let raw = unsafe {
            rokajit_ee_get_wasm_type_symbol(
                self.comp_raw(),
                types.as_mut_ptr() as *mut ffi::CorInfoWasmType,
                types.len(),
            )
        };
        WasmTypeSymbolHandle::from_raw(raw)
    }
}
