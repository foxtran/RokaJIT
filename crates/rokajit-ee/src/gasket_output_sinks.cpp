#include "gasket_common.h"

extern "C" uint32_t rokajit_ee_get_jit_flags(
    ICorJitInfo*    info,
    CORJIT_FLAGS*   flags,
    uint32_t        sizeInBytes)
{ return rokajit_ee_trap(info, [&] { return info->getJitFlags(flags, sizeInBytes); }); }

extern "C" void rokajit_ee_alloc_mem(
    ICorJitInfo*    info,
    AllocMemArgs*   pArgs)
{ return rokajit_ee_trap(info, [&] { return info->allocMem(pArgs); }); }

extern "C" void rokajit_ee_reserve_unwind_info(
    ICorJitInfo*    info,
    bool            isFunclet,
    bool            isColdCode,
    uint32_t        unwindSize)
{ return rokajit_ee_trap(info, [&] { return info->reserveUnwindInfo(isFunclet, isColdCode, unwindSize); }); }

extern "C" void rokajit_ee_alloc_unwind_info(
    ICorJitInfo*    info,
    uint8_t*        pHotCode,
    uint8_t*        pColdCode,
    uint32_t        startOffset,
    uint32_t        endOffset,
    uint32_t        unwindSize,
    uint8_t*        pUnwindBlock,
    CorJitFuncKind  funcKind)
{ return rokajit_ee_trap(info, [&] { return info->allocUnwindInfo(pHotCode, pColdCode, startOffset, endOffset,
                              unwindSize, pUnwindBlock, funcKind); }); }

extern "C" void* rokajit_ee_alloc_gc_info(
    ICorJitInfo*    info,
    size_t          size)
{ return rokajit_ee_trap(info, [&] { return info->allocGCInfo(size); }); }

extern "C" void rokajit_ee_set_eh_count(
    ICorJitInfo*    info,
    unsigned        cEH)
{ return rokajit_ee_trap(info, [&] { return info->setEHcount(cEH); }); }

extern "C" void rokajit_ee_set_eh_info(
    ICorJitInfo*                info,
    unsigned                    EHnumber,
    const CORINFO_EH_CLAUSE*    clause)
{ return rokajit_ee_trap(info, [&] { return info->setEHinfo(EHnumber, clause); }); }

extern "C" bool rokajit_ee_log_msg(
    ICorJitInfo*    info,
    unsigned        level,
    const char*     fmt,
    va_list         args)
{ return rokajit_ee_trap(info, [&] { return info->logMsg(level, fmt, args); }); }

extern "C" int rokajit_ee_do_assert(
    ICorJitInfo*    info,
    const char*     szFile,
    int             iLine,
    const char*     szExpr)
{ return rokajit_ee_trap(info, [&] { return info->doAssert(szFile, iLine, szExpr); }); }

extern "C" void rokajit_ee_report_fatal_error(
    ICorJitInfo*    info,
    CorJitResult    result)
{ return rokajit_ee_trap(info, [&] { return info->reportFatalError(result); }); }

extern "C" void rokajit_ee_record_call_site(
    ICorJitInfo*            info,
    uint32_t                instrOffset,
    CORINFO_SIG_INFO*       callSig,
    CORINFO_METHOD_HANDLE   methodHandle)
{ return rokajit_ee_trap(info, [&] { return info->recordCallSite(instrOffset, callSig, methodHandle); }); }
