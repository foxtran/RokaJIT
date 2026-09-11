#include "gasket_common.h"

extern "C" uint32_t rokajit_ee_get_method_attribs(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn)
{
    return info->getMethodAttribs(ftn);
}

extern "C" void rokajit_ee_get_method_sig(
    ICorJitInfo*            info,
    CORINFO_METHOD_HANDLE   ftn,
    CORINFO_SIG_INFO*       sig,
    CORINFO_CLASS_HANDLE    memberParent)
{
    info->getMethodSig(ftn, sig, memberParent);
}
