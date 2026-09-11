// Tokens & signatures group — extern "C" forwarders into ICorJitInfo for the
// Rust `TokensAndSignatures` wrappers. Logic-free: one virtual call plus the
// frozen exception trap per forwarder.

#include "gasket_common.h"

extern "C" void rokajit_ee_resolve_token(
    ICorJitInfo*            info,
    CORINFO_RESOLVED_TOKEN* pResolvedToken)
{ return rokajit_ee_trap(info, [&] { return info->resolveToken(pResolvedToken); }); }

extern "C" void rokajit_ee_find_sig(
    ICorJitInfo*            info,
    CORINFO_MODULE_HANDLE   module,
    unsigned                sigTOK,
    CORINFO_CONTEXT_HANDLE  context,
    CORINFO_SIG_INFO*       sig)
{ return rokajit_ee_trap(info, [&] { return info->findSig(module, sigTOK, context, sig); }); }

extern "C" void rokajit_ee_find_call_site_sig(
    ICorJitInfo*            info,
    CORINFO_MODULE_HANDLE   module,
    unsigned                methTOK,
    CORINFO_CONTEXT_HANDLE  context,
    CORINFO_SIG_INFO*       sig)
{ return rokajit_ee_trap(info, [&] { return info->findCallSiteSig(module, methTOK, context, sig); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_token_type_as_handle(
    ICorJitInfo*            info,
    CORINFO_RESOLVED_TOKEN* pResolvedToken)
{ return rokajit_ee_trap(info, [&] { return info->getTokenTypeAsHandle(pResolvedToken); }); }

extern "C" int rokajit_ee_get_string_literal(
    ICorJitInfo*            info,
    CORINFO_MODULE_HANDLE   module,
    unsigned                metaTOK,
    char16_t*               buffer,
    int                     bufferSize,
    int                     startIndex)
{ return rokajit_ee_trap(info, [&] { return info->getStringLiteral(module, metaTOK, buffer, bufferSize, startIndex); }); }

extern "C" size_t rokajit_ee_print_object_description(
    ICorJitInfo*            info,
    CORINFO_OBJECT_HANDLE   handle,
    char*                   buffer,
    size_t                  bufferSize,
    size_t*                 pRequiredBufferSize)
{ return rokajit_ee_trap(info, [&] { return info->printObjectDescription(handle, buffer, bufferSize, pRequiredBufferSize); }); }

extern "C" CORINFO_ARG_LIST_HANDLE rokajit_ee_get_arg_next(
    ICorJitInfo*            info,
    CORINFO_ARG_LIST_HANDLE args)
{ return rokajit_ee_trap(info, [&] { return info->getArgNext(args); }); }

extern "C" CorInfoTypeWithMod rokajit_ee_get_arg_type(
    ICorJitInfo*            info,
    CORINFO_SIG_INFO*       sig,
    CORINFO_ARG_LIST_HANDLE args,
    CORINFO_CLASS_HANDLE*   vcTypeRet)
{ return rokajit_ee_trap(info, [&] { return info->getArgType(sig, args, vcTypeRet); }); }

extern "C" int rokajit_ee_get_exact_classes(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    baseType,
    int                     maxExactClasses,
    CORINFO_CLASS_HANDLE*   exactClsRet)
{ return rokajit_ee_trap(info, [&] { return info->getExactClasses(baseType, maxExactClasses, exactClsRet); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_get_arg_class(
    ICorJitInfo*            info,
    CORINFO_SIG_INFO*       sig,
    CORINFO_ARG_LIST_HANDLE args)
{ return rokajit_ee_trap(info, [&] { return info->getArgClass(sig, args); }); }

extern "C" CorInfoHFAElemType rokajit_ee_get_hfa_type(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    hClass)
{ return rokajit_ee_trap(info, [&] { return info->getHFAType(hClass); }); }

extern "C" CORINFO_MODULE_HANDLE rokajit_ee_embed_module_handle(
    ICorJitInfo*            info,
    CORINFO_MODULE_HANDLE   handle,
    void**                  ppIndirection)
{ return rokajit_ee_trap(info, [&] { return info->embedModuleHandle(handle, ppIndirection); }); }

extern "C" CORINFO_CLASS_HANDLE rokajit_ee_embed_class_handle(
    ICorJitInfo*            info,
    CORINFO_CLASS_HANDLE    handle,
    void**                  ppIndirection)
{ return rokajit_ee_trap(info, [&] { return info->embedClassHandle(handle, ppIndirection); }); }

extern "C" CORINFO_METHOD_HANDLE rokajit_ee_embed_method_handle(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   handle,
    void**                  ppIndirection)
{ return rokajit_ee_trap(info, [&] { return info->embedMethodHandle(handle, ppIndirection); }); }

extern "C" CORINFO_FIELD_HANDLE rokajit_ee_embed_field_handle(
    ICorJitInfo*            info,
    CORINFO_FIELD_HANDLE    handle,
    void**                  ppIndirection)
{ return rokajit_ee_trap(info, [&] { return info->embedFieldHandle(handle, ppIndirection); }); }

extern "C" void rokajit_ee_embed_generic_handle(
    ICorJitInfo*                  info,
    CORINFO_RESOLVED_TOKEN*       pResolvedToken,
    bool                          fEmbedParent,
    CORINFO_METHOD_HANDLE         callerHandle,
    CORINFO_GENERICHANDLE_RESULT* pResult)
{ return rokajit_ee_trap(info, [&] { return info->embedGenericHandle(pResolvedToken, fEmbedParent, callerHandle, pResult); }); }

extern "C" void rokajit_ee_get_location_of_this_type(
    ICorJitInfo*          info,
    CORINFO_METHOD_HANDLE context,
    CORINFO_LOOKUP_KIND*  pLookupKind)
{ return rokajit_ee_trap(info, [&] { return info->getLocationOfThisType(context, pLookupKind); }); }

extern "C" void* rokajit_ee_get_cookie_for_interpreter_calli_sig(
    ICorJitInfo*      info,
    CORINFO_SIG_INFO* szMetaSig)
{ return rokajit_ee_trap(info, [&] { return info->GetCookieForInterpreterCalliSig(szMetaSig); }); }

extern "C" void rokajit_ee_get_call_info(
    ICorJitInfo*            info,
    CORINFO_RESOLVED_TOKEN* pResolvedToken,
    CORINFO_RESOLVED_TOKEN* pConstrainedResolvedToken,
    CORINFO_METHOD_HANDLE   callerHandle,
    CORINFO_CALLINFO_FLAGS  flags,
    CORINFO_CALL_INFO*      pResult)
{ return rokajit_ee_trap(info, [&] { return info->getCallInfo(pResolvedToken, pConstrainedResolvedToken, callerHandle, flags, pResult); }); }

extern "C" CORINFO_VARARGS_HANDLE rokajit_ee_get_var_args_handle(
    ICorJitInfo*          info,
    CORINFO_SIG_INFO*     pSig,
    CORINFO_METHOD_HANDLE methHnd,
    void**                ppIndirection)
{ return rokajit_ee_trap(info, [&] { return info->getVarArgsHandle(pSig, methHnd, ppIndirection); }); }

extern "C" InfoAccessType rokajit_ee_construct_string_literal(
    ICorJitInfo*          info,
    CORINFO_MODULE_HANDLE module,
    mdToken               metaTok,
    void**                ppValue)
{ return rokajit_ee_trap(info, [&] { return info->constructStringLiteral(module, metaTok, ppValue); }); }

extern "C" InfoAccessType rokajit_ee_empty_string_literal(
    ICorJitInfo* info,
    void**       ppValue)
{ return rokajit_ee_trap(info, [&] { return info->emptyStringLiteral(ppValue); }); }

extern "C" bool rokajit_ee_convert_pinvoke_calli_to_call(
    ICorJitInfo*            info,
    CORINFO_RESOLVED_TOKEN* pResolvedToken,
    bool                    fMustConvert)
{ return rokajit_ee_trap(info, [&] { return info->convertPInvokeCalliToCall(pResolvedToken, fMustConvert); }); }
