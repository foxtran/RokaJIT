// Helpers group: runtime-helper selection (CorInfoHelpFunc queries), EE info
// blobs, error-trap runners, and entry-point/lookup queries. One logic-free
// forwarder per C++ method; the Rust side lives in ee_info/helpers.rs.

#include "gasket_common.h"

extern "C" CorInfoHelpFunc rokajit_ee_get_new_helper(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    classHandle,
    bool*                   pHasSideEffects)
{ return rokajit_ee_trap(info, [&] { return info->getNewHelper(classHandle, pHasSideEffects); }); }

extern "C" CorInfoHelpFunc rokajit_ee_get_new_arr_helper(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    arrayCls)
{ return rokajit_ee_trap(info, [&] { return info->getNewArrHelper(arrayCls); }); }

extern "C" CorInfoHelpFunc rokajit_ee_get_casting_helper(
    ICorJitInfo*            info,
    CORINFO_RESOLVED_TOKEN* pResolvedToken,
    bool                    fThrowing)
{ return rokajit_ee_trap(info, [&] { return info->getCastingHelper(pResolvedToken, fThrowing); }); }

extern "C" CorInfoHelpFunc rokajit_ee_get_shared_cctor_helper(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    clsHnd)
{ return rokajit_ee_trap(info, [&] { return info->getSharedCCtorHelper(clsHnd); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_type_for_box(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getTypeForBox(cls); }); }

extern "C" CorInfoHelpFunc rokajit_ee_get_box_helper(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getBoxHelper(cls); }); }

extern "C" CorInfoHelpFunc rokajit_ee_get_un_box_helper(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    cls)
{ return rokajit_ee_trap(info, [&] { return info->getUnBoxHelper(cls); }); }

extern "C" bool rokajit_ee_get_ready_to_run_helper(
    ICorJitInfo*            info,
    CORINFO_RESOLVED_TOKEN* pResolvedToken,
    CorInfoHelpFunc         id,
    CORINFO_METHOD_HANDLE   callerHandle,
    CORINFO_CONST_LOOKUP*   pLookup)
{ return rokajit_ee_trap(info, [&] { return info->getReadyToRunHelper(pResolvedToken, id, callerHandle, pLookup); }); }

extern "C" void rokajit_ee_get_ready_to_run_delegate_ctor_helper(
    ICorJitInfo*            info,
    CORINFO_RESOLVED_TOKEN* pTargetMethod,
    mdToken                 targetConstraint,
    CORINFO_CLASS_HANDLE    delegateType,
    CORINFO_METHOD_HANDLE   callerHandle,
    CORINFO_LOOKUP*         pLookup)
{ return rokajit_ee_trap(info, [&] { return info->getReadyToRunDelegateCtorHelper(pTargetMethod, targetConstraint, delegateType, callerHandle, pLookup); }); }

extern "C" bool rokajit_ee_run_with_error_trap(
    ICorJitInfo*                    info,
    ICorStaticInfo::errorTrapFunction function,
    void*                           parameter)
{ return rokajit_ee_trap(info, [&] { return info->runWithErrorTrap(function, parameter); }); }

extern "C" bool rokajit_ee_run_with_spmi_error_trap(
    ICorJitInfo*                    info,
    ICorStaticInfo::errorTrapFunction function,
    void*                           parameter)
{ return rokajit_ee_trap(info, [&] { return info->runWithSPMIErrorTrap(function, parameter); }); }

extern "C" void rokajit_ee_get_ee_info(
    ICorJitInfo*            info,
    CORINFO_EE_INFO*        pEEInfoOut)
{ return rokajit_ee_trap(info, [&] { return info->getEEInfo(pEEInfoOut); }); }

extern "C" void rokajit_ee_get_wasm_well_known_globals(
    ICorJitInfo*                    info,
    CORINFO_WASM_WELLKNOWN_GLOBALS* pWellKnownGlobalsOut)
{ return rokajit_ee_trap(info, [&] { return info->getWasmWellKnownGlobals(pWellKnownGlobalsOut); }); }

extern "C" void rokajit_ee_get_helper_ftn(
    ICorJitInfo*            info,
    CorInfoHelpFunc         ftnNum,
    CORINFO_CONST_LOOKUP*   pNativeEntrypoint,
    CORINFO_METHOD_HANDLE*  pMethodHandle)
{ return rokajit_ee_trap(info, [&] { return info->getHelperFtn(ftnNum, pNativeEntrypoint, pMethodHandle); }); }

extern "C" void rokajit_ee_get_function_entry_point(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn,
    CORINFO_CONST_LOOKUP*   pResult,
    CORINFO_ACCESS_FLAGS    accessFlags)
{ return rokajit_ee_trap(info, [&] { return info->getFunctionEntryPoint(ftn, pResult, accessFlags); }); }

extern "C" void rokajit_ee_get_function_fixed_entry_point(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn,
    bool                    isUnsafeFunctionPointer,
    CORINFO_CONST_LOOKUP*   pResult)
{ return rokajit_ee_trap(info, [&] { return info->getFunctionFixedEntryPoint(ftn, isUnsafeFunctionPointer, pResult); }); }

extern "C" void rokajit_ee_get_address_of_p_invoke_target(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   method,
    CORINFO_CONST_LOOKUP*   pLookup)
{ return rokajit_ee_trap(info, [&] { return info->getAddressOfPInvokeTarget(method, pLookup); }); }

extern "C" CORINFO_METHOD_HANDLE rokajit_ee_get_delegate_ctor(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   methHnd,
    CORINFO_CLASS_HANDLE    clsHnd,
    CORINFO_METHOD_HANDLE   targetMethodHnd,
    DelegateCtorArgs*       pCtorData)
{ return rokajit_ee_trap(info, [&] { return info->GetDelegateCtor(methHnd, clsHnd, targetMethodHnd, pCtorData); }); }

extern "C" bool rokajit_ee_notify_instruction_set_usage(
    ICorJitInfo*            info,
    CORINFO_InstructionSet  instructionSet,
    bool                    supportEnabled)
{ return rokajit_ee_trap(info, [&] { return info->notifyInstructionSetUsage(instructionSet, supportEnabled); }); }

extern "C" CORINFO_METHOD_HANDLE rokajit_ee_get_special_copy_helper(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    type)
{ return rokajit_ee_trap(info, [&] { return info->getSpecialCopyHelper(type); }); }
