#include "gasket_common.h"

extern "C" JITINTERFACE_HRESULT rokajit_ee_get_pgo_instrumentation_results(
    ICorJitInfo*                            info,
    CORINFO_METHOD_HANDLE                   ftnHnd,
    ICorJitInfo::PgoInstrumentationSchema** pSchema,
    uint32_t*                               pCountSchemaItems,
    uint8_t**                               pInstrumentationData,
    ICorJitInfo::PgoSource*                 pPgoSource,
    bool*                                   pDynamicPgo)
{ return rokajit_ee_trap(info, [&] { return info->getPgoInstrumentationResults(ftnHnd, pSchema, pCountSchemaItems, pInstrumentationData, pPgoSource, pDynamicPgo); }); }

extern "C" JITINTERFACE_HRESULT rokajit_ee_alloc_pgo_instrumentation_by_schema(
    ICorJitInfo*                            info,
    CORINFO_METHOD_HANDLE                   ftnHnd,
    ICorJitInfo::PgoInstrumentationSchema*  pSchema,
    uint32_t                                countSchemaItems,
    uint8_t**                               pInstrumentationData)
{ return rokajit_ee_trap(info, [&] { return info->allocPgoInstrumentationBySchema(ftnHnd, pSchema, countSchemaItems, pInstrumentationData); }); }

extern "C" void rokajit_ee_record_wasm_managed_call_sig(
    ICorJitInfo*        info,
    CORINFO_SIG_INFO*   callSig)
{ return rokajit_ee_trap(info, [&] { return info->recordWasmManagedCallSig(callSig); }); }
