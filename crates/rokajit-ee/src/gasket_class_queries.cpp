#include "gasket_common.h"

// Class queries group (C++ ICorClassInfo + the class-flavored members of
// ICorStaticInfo/ICorDynamicInfo). One logic-free forwarder per method;
// signatures copied verbatim from runtime/src/coreclr/inc/corinfo.h.

extern "C" CorInfoType rokajit_ee_as_cor_info_type(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->asCorInfoType(cls); }); }

extern "C" const char* rokajit_ee_get_class_name_from_metadata(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls,
    const char**            namespaceName)
{ return rokajit_ee_trap(info, [&] { return info->getClassNameFromMetadata(cls, namespaceName); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_type_instantiation_argument(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls,
    unsigned                index)
{ return rokajit_ee_trap(info, [&] { return info->getTypeInstantiationArgument(cls, index); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_method_instantiation_argument(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn,
    unsigned                index)
{ return rokajit_ee_trap(info, [&] { return info->getMethodInstantiationArgument(ftn, index); }); }

extern "C" size_t rokajit_ee_print_class_name(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls,
    char*                   buffer,
    size_t                  bufferSize,
    size_t*                 pRequiredBufferSize)
{ return rokajit_ee_trap(info, [&] { return info->printClassName(cls, buffer, bufferSize, pRequiredBufferSize); }); }

extern "C" bool rokajit_ee_is_value_class(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->isValueClass(cls); }); }

extern "C" uint32_t rokajit_ee_get_class_attribs(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getClassAttribs(cls); }); }

extern "C" const char* rokajit_ee_get_class_assembly_name(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getClassAssemblyName(cls); }); }

extern "C" void* rokajit_ee_long_lifetime_malloc(
    ICorJitInfo*            info,
    size_t                  sz)
{ return rokajit_ee_trap(info, [&] { return info->LongLifetimeMalloc(sz); }); }

extern "C" void rokajit_ee_long_lifetime_free(
    ICorJitInfo*            info,
    void*                   obj)
{ return rokajit_ee_trap(info, [&] { return info->LongLifetimeFree(obj); }); }

extern "C" bool rokajit_ee_get_is_class_inited_flag_address(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls,
    CORINFO_CONST_LOOKUP*   addr,
    int*                    offset)
{ return rokajit_ee_trap(info, [&] { return info->getIsClassInitedFlagAddress(cls, addr, offset); }); }

extern "C" void* rokajit_ee_get_class_static_dynamic_info(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getClassStaticDynamicInfo(cls); }); }

extern "C" void* rokajit_ee_get_class_thread_static_dynamic_info(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getClassThreadStaticDynamicInfo(cls); }); }

extern "C" bool rokajit_ee_get_static_base_address(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls,
    bool                    isGc,
    CORINFO_CONST_LOOKUP*   addr)
{ return rokajit_ee_trap(info, [&] { return info->getStaticBaseAddress(cls, isGc, addr); }); }

extern "C" unsigned rokajit_ee_get_class_size(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getClassSize(cls); }); }

extern "C" unsigned rokajit_ee_get_heap_class_size(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getHeapClassSize(cls); }); }

extern "C" bool rokajit_ee_can_allocate_on_stack(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->canAllocateOnStack(cls); }); }

extern "C" unsigned rokajit_ee_get_class_alignment_requirement(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls,
    bool                    fDoubleAlignHint)
{ return rokajit_ee_trap(info, [&] { return info->getClassAlignmentRequirement(cls, fDoubleAlignHint); }); }

extern "C" unsigned rokajit_ee_get_class_gc_layout(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls,
    uint8_t*                gcPtrs)
{ return rokajit_ee_trap(info, [&] { return info->getClassGClayout(cls, gcPtrs); }); }

extern "C" unsigned rokajit_ee_get_class_num_instance_fields(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getClassNumInstanceFields(cls); }); }

extern "C" CORINFO_FIELD_HANDLE rokajit_ee_get_field_in_class(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    clsHnd,
    int32_t                 num)
{ return rokajit_ee_trap(info, [&] { return info->getFieldInClass(clsHnd, num); }); }

extern "C" GetTypeLayoutResult rokajit_ee_get_type_layout(
    ICorJitInfo*                info,
    CORINFO_CLASS_HANDLE        typeHnd,
    CORINFO_TYPE_LAYOUT_NODE*   treeNodes,
    size_t*                     numTreeNodes)
{ return rokajit_ee_trap(info, [&] { return info->getTypeLayout(typeHnd, treeNodes, numTreeNodes); }); }

extern "C" bool rokajit_ee_check_method_modifier(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   hMethod,
    const char*             modifier,
    bool                    fOptional)
{ return rokajit_ee_trap(info, [&] { return info->checkMethodModifier(hMethod, modifier, fOptional); }); }

extern "C" CORINFO_OBJECT_HANDLE rokajit_ee_get_runtime_type_pointer(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getRuntimeTypePointer(cls); }); }

extern "C" bool rokajit_ee_is_object_immutable(
    ICorJitInfo*            info,
    CORINFO_OBJECT_HANDLE   objPtr)
{ return rokajit_ee_trap(info, [&] { return info->isObjectImmutable(objPtr); }); }

extern "C" bool rokajit_ee_get_string_char(
    ICorJitInfo*            info,
    CORINFO_OBJECT_HANDLE   strObj,
    int                     index,
    uint16_t*               value)
{ return rokajit_ee_trap(info, [&] { return info->getStringChar(strObj, index, value); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_object_type(
    ICorJitInfo*            info,
    CORINFO_OBJECT_HANDLE   objPtr)
{ return rokajit_ee_trap(info, [&] { return info->getObjectType(objPtr); }); }

extern "C" CorInfoInitClassResult rokajit_ee_init_class(
    ICorJitInfo*            info,
    CORINFO_FIELD_HANDLE    field,
    CORINFO_METHOD_HANDLE   method,
    CORINFO_CONTEXT_HANDLE  context)
{ return rokajit_ee_trap(info, [&] { return info->initClass(field, method, context); }); }

extern "C" void rokajit_ee_class_must_be_loaded_before_code_is_run(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->classMustBeLoadedBeforeCodeIsRun(cls); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_builtin_class(
    ICorJitInfo*            info,
    CorInfoClassId          classId)
{ return rokajit_ee_trap(info, [&] { return info->getBuiltinClass(classId); }); }

extern "C" CorInfoType rokajit_ee_get_type_for_primitive_value_class(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getTypeForPrimitiveValueClass(cls); }); }

extern "C" CorInfoType rokajit_ee_get_type_for_primitive_numeric_class(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getTypeForPrimitiveNumericClass(cls); }); }

extern "C" bool rokajit_ee_can_cast(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    child,
    CORINFO_CLASS_HANDLE    parent)
{ return rokajit_ee_trap(info, [&] { return info->canCast(child, parent); }); }

extern "C" TypeCompareState rokajit_ee_compare_types_for_cast(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    fromClass,
    CORINFO_CLASS_HANDLE    toClass)
{ return rokajit_ee_trap(info, [&] { return info->compareTypesForCast(fromClass, toClass); }); }

extern "C" TypeCompareState rokajit_ee_compare_types_for_equality(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls1,
    CORINFO_CLASS_HANDLE    cls2)
{ return rokajit_ee_trap(info, [&] { return info->compareTypesForEquality(cls1, cls2); }); }

extern "C" bool rokajit_ee_is_more_specific_type(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls1,
    CORINFO_CLASS_HANDLE    cls2)
{ return rokajit_ee_trap(info, [&] { return info->isMoreSpecificType(cls1, cls2); }); }

extern "C" bool rokajit_ee_is_exact_type(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->isExactType(cls); }); }

extern "C" TypeCompareState rokajit_ee_is_generic_type(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->isGenericType(cls); }); }

extern "C" TypeCompareState rokajit_ee_is_nullable_type(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->isNullableType(cls); }); }

extern "C" TypeCompareState rokajit_ee_is_enum(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls,
    CORINFO_CLASS_HANDLE*   underlyingType)
{ return rokajit_ee_trap(info, [&] { return info->isEnum(cls, underlyingType); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_parent_type(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getParentType(cls); }); }

extern "C" CorInfoType rokajit_ee_get_child_type(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    clsHnd,
    CORINFO_CLASS_HANDLE*   clsRet)
{ return rokajit_ee_trap(info, [&] { return info->getChildType(clsHnd, clsRet); }); }

extern "C" bool rokajit_ee_is_sd_array(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->isSDArray(cls); }); }

extern "C" unsigned rokajit_ee_get_array_rank(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getArrayRank(cls); }); }

extern "C" CorInfoArrayIntrinsic rokajit_ee_get_array_intrinsic_id(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn)
{ return rokajit_ee_trap(info, [&] { return info->getArrayIntrinsicID(ftn); }); }

extern "C" void* rokajit_ee_get_array_initialization_data(
    ICorJitInfo*            info,
    CORINFO_FIELD_HANDLE    field,
    uint32_t                size)
{ return rokajit_ee_trap(info, [&] { return info->getArrayInitializationData(field, size); }); }

extern "C" CorInfoIsAccessAllowedResult rokajit_ee_can_access_class(
    ICorJitInfo*            info,
    CORINFO_RESOLVED_TOKEN* pResolvedToken,
    CORINFO_METHOD_HANDLE   callerHandle,
    CORINFO_HELPER_DESC*    pAccessHelper)
{ return rokajit_ee_trap(info, [&] { return info->canAccessClass(pResolvedToken, callerHandle, pAccessHelper); }); }

extern "C" bool rokajit_ee_get_system_v_amd64_pass_struct_in_register_descriptor(
    ICorJitInfo*                                            info,
    CORINFO_CLASS_HANDLE                                    structHnd,
    SYSTEMV_AMD64_CORINFO_STRUCT_REG_PASSING_DESCRIPTOR*    structPassInRegDescPtr)
{ return rokajit_ee_trap(info, [&] { return info->getSystemVAmd64PassStructInRegisterDescriptor(structHnd, structPassInRegDescPtr); }); }

extern "C" void rokajit_ee_get_swift_lowering(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    structHnd,
    CORINFO_SWIFT_LOWERING* pLowering)
{ return rokajit_ee_trap(info, [&] { return info->getSwiftLowering(structHnd, pLowering); }); }

extern "C" void rokajit_ee_get_fp_struct_lowering(
    ICorJitInfo*                info,
    CORINFO_CLASS_HANDLE        structHnd,
    CORINFO_FPSTRUCT_LOWERING*  pLowering)
{ return rokajit_ee_trap(info, [&] { return info->getFpStructLowering(structHnd, pLowering); }); }

extern "C" CorInfoWasmType rokajit_ee_get_wasm_lowering(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    structHnd)
{ return rokajit_ee_trap(info, [&] { return info->getWasmLowering(structHnd); }); }

extern "C" uint32_t rokajit_ee_get_address_alignment(
    ICorJitInfo*    info,
    void*           address)
{ return rokajit_ee_trap(info, [&] { return info->getAddressAlignment(address); }); }

extern "C" bool rokajit_ee_get_object_content(
    ICorJitInfo*            info,
    CORINFO_OBJECT_HANDLE   obj,
    uint8_t*                buffer,
    int                     bufferSize,
    int                     valueOffset)
{ return rokajit_ee_trap(info, [&] { return info->getObjectContent(obj, buffer, bufferSize, valueOffset); }); }

extern "C" CORINFO_WASM_TYPE_SYMBOL_HANDLE rokajit_ee_get_wasm_type_symbol(
    ICorJitInfo*        info,
    CorInfoWasmType*    types,
    size_t              typesSize)
{ return rokajit_ee_trap(info, [&] { return info->getWasmTypeSymbol(types, typesSize); }); }
