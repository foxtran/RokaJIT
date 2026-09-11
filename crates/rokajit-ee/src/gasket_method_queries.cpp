#include "gasket_common.h"

extern "C" uint32_t rokajit_ee_get_method_attribs(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn)
{ return rokajit_ee_trap(info, [&] { return info->getMethodAttribs(ftn); }); }

extern "C" void rokajit_ee_get_method_sig(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn,
    CORINFO_SIG_INFO*       sig,
    CORINFO_CLASS_HANDLE    memberParent)
{ return rokajit_ee_trap(info, [&] { return info->getMethodSig(ftn, sig, memberParent); }); }

extern "C" bool rokajit_ee_is_intrinsic(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn)
{ return rokajit_ee_trap(info, [&] { return info->isIntrinsic(ftn); }); }

extern "C" bool rokajit_ee_can_value_class_instance_pointer_escape(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn)
{ return rokajit_ee_trap(info, [&] { return info->canValueClassInstancePointerEscape(ftn); }); }

extern "C" bool rokajit_ee_notify_method_info_usage(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn)
{ return rokajit_ee_trap(info, [&] { return info->notifyMethodInfoUsage(ftn); }); }

extern "C" void rokajit_ee_set_method_attribs(
    ICorJitInfo*                info,
    CORINFO_METHOD_HANDLE       ftn,
    CorInfoMethodRuntimeFlags   attribs)
{ return rokajit_ee_trap(info, [&] { return info->setMethodAttribs(ftn, attribs); }); }

extern "C" bool rokajit_ee_get_method_info(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn,
    CORINFO_METHOD_INFO*    methodInfo,
    CORINFO_CONTEXT_HANDLE  context)
{ return rokajit_ee_trap(info, [&] { return info->getMethodInfo(ftn, methodInfo, context); }); }

extern "C" bool rokajit_ee_have_same_method_definition(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   meth1Hnd,
    CORINFO_METHOD_HANDLE   meth2Hnd)
{ return rokajit_ee_trap(info, [&] { return info->haveSameMethodDefinition(meth1Hnd, meth2Hnd); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_type_definition(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    type)
{ return rokajit_ee_trap(info, [&] { return info->getTypeDefinition(type); }); }

extern "C" void rokajit_ee_get_eh_info(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn,
    unsigned                EHnumber,
    CORINFO_EH_CLAUSE*      clause)
{ return rokajit_ee_trap(info, [&] { return info->getEHinfo(ftn, EHnumber, clause); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_method_class(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   method)
{ return rokajit_ee_trap(info, [&] { return info->getMethodClass(method); }); }

extern "C" void rokajit_ee_get_method_vtable_offset(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   method,
    unsigned*               offsetOfIndirection,
    unsigned*               offsetAfterIndirection,
    bool*                   isRelative)
{ return rokajit_ee_trap(info, [&] { return info->getMethodVTableOffset(method, offsetOfIndirection, offsetAfterIndirection, isRelative); }); }

extern "C" bool rokajit_ee_resolve_virtual_method(
    ICorJitInfo*                    info,
    CORINFO_DEVIRTUALIZATION_INFO*  devirtInfo)
{ return rokajit_ee_trap(info, [&] { return info->resolveVirtualMethod(devirtInfo); }); }

extern "C" CORINFO_METHOD_HANDLE rokajit_ee_get_async_other_variant(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn,
    bool*                   variantIsThunk)
{ return rokajit_ee_trap(info, [&] { return info->getAsyncOtherVariant(ftn, variantIsThunk); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_default_comparer_class(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    elemType)
{ return rokajit_ee_trap(info, [&] { return info->getDefaultComparerClass(elemType); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_default_equality_comparer_class(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    elemType)
{ return rokajit_ee_trap(info, [&] { return info->getDefaultEqualityComparerClass(elemType); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_sz_array_helper_enumerator_class(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    elemType)
{ return rokajit_ee_trap(info, [&] { return info->getSZArrayHelperEnumeratorClass(elemType); }); }

extern "C" void rokajit_ee_expand_raw_handle_intrinsic(
    ICorJitInfo*                    info,
    CORINFO_RESOLVED_TOKEN*         pResolvedToken,
    CORINFO_METHOD_HANDLE           callerHandle,
    CORINFO_GENERICHANDLE_RESULT*   pResult)
{ return rokajit_ee_trap(info, [&] { return info->expandRawHandleIntrinsic(pResolvedToken, callerHandle, pResult); }); }

extern "C" bool rokajit_ee_is_intrinsic_type(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    classHnd)
{ return rokajit_ee_trap(info, [&] { return info->isIntrinsicType(classHnd); }); }

extern "C" CorInfoCallConvExtension rokajit_ee_get_unmanaged_call_conv(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   method,
    CORINFO_SIG_INFO*       callSiteSig,
    bool*                   pSuppressGCTransition)
{ return rokajit_ee_trap(info, [&] { return info->getUnmanagedCallConv(method, callSiteSig, pSuppressGCTransition); }); }

extern "C" bool rokajit_ee_p_invoke_marshaling_required(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   method,
    CORINFO_SIG_INFO*       callSiteSig)
{ return rokajit_ee_trap(info, [&] { return info->pInvokeMarshalingRequired(method, callSiteSig); }); }

extern "C" bool rokajit_ee_satisfies_method_constraints(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    parent,
    CORINFO_METHOD_HANDLE   method)
{ return rokajit_ee_trap(info, [&] { return info->satisfiesMethodConstraints(parent, method); }); }

extern "C" void rokajit_ee_method_must_be_loaded_before_code_is_run(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   method)
{ return rokajit_ee_trap(info, [&] { return info->methodMustBeLoadedBeforeCodeIsRun(method); }); }

extern "C" void rokajit_ee_get_gs_cookie(
    ICorJitInfo*    info,
    GSCookie*       pCookieVal,
    GSCookie**      ppCookieVal)
{ return rokajit_ee_trap(info, [&] { return info->getGSCookie(pCookieVal, ppCookieVal); }); }

extern "C" void rokajit_ee_set_patchpoint_info(
    ICorJitInfo*    info,
    PatchpointInfo* patchpointInfo)
{ return rokajit_ee_trap(info, [&] { return info->setPatchpointInfo(patchpointInfo); }); }

extern "C" PatchpointInfo* rokajit_ee_get_osr_info(
    ICorJitInfo*    info,
    unsigned*       ilOffset)
{ return rokajit_ee_trap(info, [&] { return info->getOSRInfo(ilOffset); }); }

extern "C" void rokajit_ee_get_async_info(
    ICorJitInfo*        info,
    CORINFO_ASYNC_INFO* pAsyncInfoOut)
{ return rokajit_ee_trap(info, [&] { return info->getAsyncInfo(pAsyncInfoOut); }); }

extern "C" CORINFO_METHOD_HANDLE rokajit_ee_get_await_return_call(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   callerHandle,
    CORINFO_CONTEXT_HANDLE* contextHandle,
    CORINFO_LOOKUP*         instArg)
{ return rokajit_ee_trap(info, [&] { return info->getAwaitReturnCall(callerHandle, contextHandle, instArg); }); }

extern "C" CORINFO_METHOD_HANDLE rokajit_ee_get_await_awaiter_in_continuation_call(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   callerHandle,
    CORINFO_RESOLVED_TOKEN* pResolvedToken,
    bool                    isUnsafe,
    CORINFO_CONTEXT_HANDLE* contextHandle,
    CORINFO_LOOKUP*         instArg)
{ return rokajit_ee_trap(info, [&] { return info->getAwaitAwaiterInContinuationCall(callerHandle, pResolvedToken, isUnsafe, contextHandle, instArg); }); }

extern "C" mdMethodDef rokajit_ee_get_method_def_from_method(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   hMethod)
{ return rokajit_ee_trap(info, [&] { return info->getMethodDefFromMethod(hMethod); }); }

extern "C" size_t rokajit_ee_print_method_name(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn,
    char*                   buffer,
    size_t                  bufferSize,
    size_t*                 pRequiredBufferSize)
{ return rokajit_ee_trap(info, [&] { return info->printMethodName(ftn, buffer, bufferSize, pRequiredBufferSize); }); }

extern "C" const char* rokajit_ee_get_method_name_from_metadata(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn,
    const char**            className,
    const char**            namespaceName,
    const char**            enclosingClassNames,
    size_t                  maxEnclosingClassNames)
{ return rokajit_ee_trap(info, [&] { return info->getMethodNameFromMetadata(ftn, className, namespaceName, enclosingClassNames, maxEnclosingClassNames); }); }

extern "C" unsigned rokajit_ee_get_method_hash(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn)
{ return rokajit_ee_trap(info, [&] { return info->getMethodHash(ftn); }); }

extern "C" CORINFO_METHOD_HANDLE rokajit_ee_get_async_resumption_stub(
    ICorJitInfo*    info,
    void**          entryPoint)
{ return rokajit_ee_trap(info, [&] { return info->getAsyncResumptionStub(entryPoint); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_continuation_type(
    ICorJitInfo*    info,
    size_t          dataSize,
    bool*           objRefs,
    size_t          objRefsSize)
{ return rokajit_ee_trap(info, [&] { return info->getContinuationType(dataSize, objRefs, objRefsSize); }); }
