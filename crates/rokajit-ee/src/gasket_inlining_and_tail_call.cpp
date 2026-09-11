#include "gasket_common.h"

extern "C" CorInfoInline rokajit_ee_can_inline(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   callerHnd,
    CORINFO_METHOD_HANDLE   calleeHnd)
{ return rokajit_ee_trap(info, [&] { return info->canInline(callerHnd, calleeHnd); }); }

extern "C" void rokajit_ee_begin_inlining(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   inlinerHnd,
    CORINFO_METHOD_HANDLE   inlineeHnd)
{ return rokajit_ee_trap(info, [&] { return info->beginInlining(inlinerHnd, inlineeHnd); }); }

extern "C" void rokajit_ee_report_inlining_decision(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   inlinerHnd,
    CORINFO_METHOD_HANDLE   inlineeHnd,
    CorInfoInline           inlineResult,
    const char*             reason)
{ return rokajit_ee_trap(info, [&] { return info->reportInliningDecision(inlinerHnd, inlineeHnd, inlineResult, reason); }); }

extern "C" bool rokajit_ee_can_tail_call(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   callerHnd,
    CORINFO_METHOD_HANDLE   declaredCalleeHnd,
    CORINFO_METHOD_HANDLE   exactCalleeHnd,
    bool                    fIsTailPrefix)
{ return rokajit_ee_trap(info, [&] { return info->canTailCall(callerHnd, declaredCalleeHnd, exactCalleeHnd, fIsTailPrefix); }); }

extern "C" void rokajit_ee_report_tail_call_decision(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   callerHnd,
    CORINFO_METHOD_HANDLE   calleeHnd,
    bool                    fIsTailPrefix,
    CorInfoTailCall         tailCallResult,
    const char*             reason)
{ return rokajit_ee_trap(info, [&] { return info->reportTailCallDecision(callerHnd, calleeHnd, fIsTailPrefix, tailCallResult, reason); }); }

extern "C" bool rokajit_ee_get_tail_call_helpers(
    ICorJitInfo*                        info,
    CORINFO_RESOLVED_TOKEN*             callToken,
    CORINFO_SIG_INFO*                   sig,
    CORINFO_GET_TAILCALL_HELPERS_FLAGS  flags,
    CORINFO_TAILCALL_HELPERS*           pResult)
{ return rokajit_ee_trap(info, [&] { return info->getTailCallHelpers(callToken, sig, flags, pResult); }); }

extern "C" void rokajit_ee_update_entry_point_for_tail_call(
    ICorJitInfo*            info,
    CORINFO_CONST_LOOKUP*   entryPoint)
{ return rokajit_ee_trap(info, [&] { return info->updateEntryPointForTailCall(entryPoint); }); }
