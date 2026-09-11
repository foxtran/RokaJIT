#include "gasket_common.h"

extern "C" uint32_t rokajit_ee_get_jit_flags(
    ICorJitInfo*    info,
    CORJIT_FLAGS*   flags,
    uint32_t        sizeInBytes)
{
    return info->getJitFlags(flags, sizeInBytes);
}
