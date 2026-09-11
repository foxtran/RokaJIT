#include "gasket_common.h"

extern "C" size_t rokajit_ee_print_field_name(
    ICorJitInfo*          info,
    CORINFO_FIELD_HANDLE  field,
    char*                 buffer,
    size_t                bufferSize,
    size_t*               pRequiredBufferSize)
{ return rokajit_ee_trap(info, [&] { return info->printFieldName(field, buffer, bufferSize, pRequiredBufferSize); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_field_class(
    ICorJitInfo*          info,
    CORINFO_FIELD_HANDLE  field)
{ return rokajit_ee_trap(info, [&] { return info->getFieldClass(field); }); }

extern "C" CorInfoType rokajit_ee_get_field_type(
    ICorJitInfo*          info,
    CORINFO_FIELD_HANDLE  field,
    CORINFO_CLASS_HANDLE* structType,
    CORINFO_CLASS_HANDLE  fieldOwnerHint)
{ return rokajit_ee_trap(info, [&] { return info->getFieldType(field, structType, fieldOwnerHint); }); }

extern "C" unsigned rokajit_ee_get_field_offset(
    ICorJitInfo*          info,
    CORINFO_FIELD_HANDLE  field)
{ return rokajit_ee_trap(info, [&] { return info->getFieldOffset(field); }); }

extern "C" void rokajit_ee_get_field_info(
    ICorJitInfo*           info,
    CORINFO_RESOLVED_TOKEN* pResolvedToken,
    CORINFO_METHOD_HANDLE  callerHandle,
    CORINFO_ACCESS_FLAGS   flags,
    CORINFO_FIELD_INFO*    pResult)
{ return rokajit_ee_trap(info, [&] { return info->getFieldInfo(pResolvedToken, callerHandle, flags, pResult); }); }

extern "C" uint32_t rokajit_ee_get_thread_local_field_info(
    ICorJitInfo*          info,
    CORINFO_FIELD_HANDLE  field,
    bool                  isGCType)
{ return rokajit_ee_trap(info, [&] { return info->getThreadLocalFieldInfo(field, isGCType); }); }

extern "C" void rokajit_ee_get_thread_local_static_blocks_info(
    ICorJitInfo*                        info,
    CORINFO_THREAD_STATIC_BLOCKS_INFO*  pInfo)
{ return rokajit_ee_trap(info, [&] { return info->getThreadLocalStaticBlocksInfo(pInfo); }); }

extern "C" void rokajit_ee_get_thread_local_static_info_native_aot(
    ICorJitInfo*                            info,
    CORINFO_THREAD_STATIC_INFO_NATIVEAOT*   pInfo)
{ return rokajit_ee_trap(info, [&] { return info->getThreadLocalStaticInfo_NativeAOT(pInfo); }); }

extern "C" bool rokajit_ee_is_field_static(
    ICorJitInfo*          info,
    CORINFO_FIELD_HANDLE  fldHnd)
{ return rokajit_ee_trap(info, [&] { return info->isFieldStatic(fldHnd); }); }

extern "C" int rokajit_ee_get_array_or_string_length(
    ICorJitInfo*           info,
    CORINFO_OBJECT_HANDLE  objHnd)
{ return rokajit_ee_trap(info, [&] { return info->getArrayOrStringLength(objHnd); }); }

extern "C" uint32_t rokajit_ee_get_thread_tls_index(
    ICorJitInfo*  info,
    void**        ppIndirection)
{ return rokajit_ee_trap(info, [&] { return info->getThreadTLSIndex(ppIndirection); }); }

extern "C" int32_t* rokajit_ee_get_addr_of_capture_thread_global(
    ICorJitInfo*  info,
    void**        ppIndirection)
{ return rokajit_ee_trap(info, [&] { return info->getAddrOfCaptureThreadGlobal(ppIndirection); }); }

extern "C" bool rokajit_ee_get_static_field_content(
    ICorJitInfo*          info,
    CORINFO_FIELD_HANDLE  field,
    uint8_t*              buffer,
    int                   bufferSize,
    int                   valueOffset,
    bool                  ignoreMovableObjects)
{ return rokajit_ee_trap(info, [&] { return info->getStaticFieldContent(field, buffer, bufferSize, valueOffset, ignoreMovableObjects); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_static_field_current_class(
    ICorJitInfo*          info,
    CORINFO_FIELD_HANDLE  field,
    bool*                 pIsSpeculative)
{ return rokajit_ee_trap(info, [&] { return info->getStaticFieldCurrentClass(field, pIsSpeculative); }); }

extern "C" uint32_t rokajit_ee_get_field_thread_local_store_id(
    ICorJitInfo*          info,
    CORINFO_FIELD_HANDLE  field,
    void**                ppIndirection)
{ return rokajit_ee_trap(info, [&] { return info->getFieldThreadLocalStoreID(field, ppIndirection); }); }
