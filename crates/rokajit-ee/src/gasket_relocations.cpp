#include "gasket_common.h"

extern "C" void rokajit_ee_record_relocation(
    ICorJitInfo*    info,
    void*           location,
    void*           locationRW,
    void*           target,
    CorInfoReloc    fRelocType,
    int32_t         addlDelta)
{ return rokajit_ee_trap(info, [&] { return info->recordRelocation(location, locationRW, target, fRelocType, addlDelta); }); }

extern "C" CorInfoReloc rokajit_ee_get_reloc_type_hint(
    ICorJitInfo*    info,
    void*           target)
{ return rokajit_ee_trap(info, [&] { return info->getRelocTypeHint(target); }); }

extern "C" uint32_t rokajit_ee_get_expected_target_architecture(
    ICorJitInfo*    info)
{ return rokajit_ee_trap(info, [&] { return info->getExpectedTargetArchitecture(); }); }
