#include "gasket_common.h"

extern "C" void rokajit_ee_get_boundaries(
    ICorJitInfo*                 info,
    CORINFO_METHOD_HANDLE        ftn,
    unsigned int*                cILOffsets,
    uint32_t**                   pILOffsets,
    ICorDebugInfo::BoundaryTypes* implicitBoundaries)
{ return rokajit_ee_trap(info, [&] { return info->getBoundaries(ftn, cILOffsets, pILOffsets, implicitBoundaries); }); }

extern "C" void rokajit_ee_set_boundaries(
    ICorJitInfo*                  info,
    CORINFO_METHOD_HANDLE         ftn,
    uint32_t                      cMap,
    ICorDebugInfo::OffsetMapping* pMap)
{ return rokajit_ee_trap(info, [&] { return info->setBoundaries(ftn, cMap, pMap); }); }

extern "C" void rokajit_ee_get_vars(
    ICorJitInfo*               info,
    CORINFO_METHOD_HANDLE      ftn,
    uint32_t*                  cVars,
    ICorDebugInfo::ILVarInfo** vars,
    bool*                      extendOthers)
{ return rokajit_ee_trap(info, [&] { return info->getVars(ftn, cVars, vars, extendOthers); }); }

extern "C" void rokajit_ee_set_vars(
    ICorJitInfo*                   info,
    CORINFO_METHOD_HANDLE          ftn,
    uint32_t                       cVars,
    ICorDebugInfo::NativeVarInfo*  vars)
{ return rokajit_ee_trap(info, [&] { return info->setVars(ftn, cVars, vars); }); }

extern "C" void rokajit_ee_report_rich_mappings(
    ICorJitInfo*                      info,
    ICorDebugInfo::InlineTreeNode*    inlineTreeNodes,
    uint32_t                          numInlineTreeNodes,
    ICorDebugInfo::RichOffsetMapping* mappings,
    uint32_t                          numMappings)
{ return rokajit_ee_trap(info, [&] { return info->reportRichMappings(inlineTreeNodes, numInlineTreeNodes, mappings, numMappings); }); }

extern "C" void rokajit_ee_report_async_debug_info(
    ICorJitInfo*                            info,
    ICorDebugInfo::AsyncInfo*               asyncInfo,
    ICorDebugInfo::AsyncSuspensionPoint*    suspensionPoints,
    ICorDebugInfo::AsyncContinuationVarInfo* vars,
    uint32_t                                numVars)
{ return rokajit_ee_trap(info, [&] { return info->reportAsyncDebugInfo(asyncInfo, suspensionPoints, vars, numVars); }); }

extern "C" void rokajit_ee_report_metadata(
    ICorJitInfo* info,
    const char*  key,
    const void*  value,
    size_t       length)
{ return rokajit_ee_trap(info, [&] { return info->reportMetadata(key, value, length); }); }

extern "C" void* rokajit_ee_allocate_array(
    ICorJitInfo* info,
    size_t       cBytes)
{ return rokajit_ee_trap(info, [&] { return info->allocateArray(cBytes); }); }

extern "C" void rokajit_ee_free_array(
    ICorJitInfo* info,
    void*        array)
{ return rokajit_ee_trap(info, [&] { return info->freeArray(array); }); }

extern "C" CORINFO_JUST_MY_CODE_HANDLE rokajit_ee_get_just_my_code_handle(
    ICorJitInfo*                 info,
    CORINFO_METHOD_HANDLE        method,
    CORINFO_JUST_MY_CODE_HANDLE** ppIndirection)
{ return rokajit_ee_trap(info, [&] { return info->getJustMyCodeHandle(method, ppIndirection); }); }

extern "C" void rokajit_ee_get_profiling_handle(
    ICorJitInfo* info,
    bool*        pbHookFunction,
    void**       pProfilerHandle,
    bool*        pbIndirectedHandles)
{ return rokajit_ee_trap(info, [&] { return info->GetProfilingHandle(pbHookFunction, pProfilerHandle, pbIndirectedHandles); }); }

extern "C" void rokajit_ee_method_compile_complete(
    ICorJitInfo*         info,
    CORINFO_METHOD_HANDLE methHnd)
{ return rokajit_ee_trap(info, [&] { return info->MethodCompileComplete(methHnd); }); }
