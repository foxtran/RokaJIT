// RokaJIT C++ ABI gasket — the only C++ in the project.
//
// Owns every piece of code whose correctness depends on C++ vtable layout:
// our ICorJitCompiler implementation and the jitStartup/getJit exports the
// EE looks up, plus extern "C" forwarders into the EE's ICorJitInfo (the
// pattern for Phase 1's inbound wrappers). clang++ lays out the vtables and
// resolves the virtual calls using the authoritative headers, so no ABI
// transcription happens by hand.
//
// This file contains NO logic beyond forwarding. All behavior lives in Rust
// (rokajit crate).

// <cstddef>/<cstdint> first: the CoreCLR headers use size_t & co. without
// including the headers that define them (same reason the bindgen wrapper
// header leads with these).
#include <cstddef>
#include <cstdint>

#include "corjit.h"
#include "corjithost.h"
#include "jiteeversionguid.h"

// ---------------------------------------------------------------------------
// Rust entry points (defined in the rokajit crate).
// ---------------------------------------------------------------------------

extern "C" CorJitResult rokajit_compile_method(
    ICorJitInfo*            comp,
    CORINFO_METHOD_INFO*    info,
    unsigned                flags,
    uint8_t**               nativeEntry,
    uint32_t*               nativeSizeOfCode);

extern "C" void rokajit_set_target_os(CORINFO_OS os);

extern "C" void rokajit_process_shutdown_work(ICorStaticInfo* info);

// ---------------------------------------------------------------------------
// ICorJitCompiler implementation: pure forwarding into Rust.
// ---------------------------------------------------------------------------

class RokaJitCompiler final : public ICorJitCompiler
{
public:
    CorJitResult compileMethod(
        ICorJitInfo*            comp,
        CORINFO_METHOD_INFO*    info,
        unsigned                flags,
        uint8_t**               nativeEntry,
        uint32_t*               nativeSizeOfCode) override
    {
        return rokajit_compile_method(comp, info, flags, nativeEntry, nativeSizeOfCode);
    }

    void ProcessShutdownWork(ICorStaticInfo* info) override
    {
        rokajit_process_shutdown_work(info);
    }

    void getVersionIdentifier(GUID* versionIdentifier) override
    {
        // The C++ side can see the constexpr constant directly — no linking
        // problem here, unlike in bindgen-generated Rust.
        *versionIdentifier = JITEEVersionIdentifier;
    }

    void setTargetOS(CORINFO_OS os) override
    {
        rokajit_set_target_os(os);
    }
};

// ---------------------------------------------------------------------------
// Exports, following runtime/src/coreclr/jit/ee_il_dll.cpp:41/195.
// ---------------------------------------------------------------------------

static ICorJitHost*    g_jitHost        = nullptr;
static bool            g_jitInitialized = false;
static RokaJitCompiler g_compiler;

#if defined(__GNUC__)
#define ROKAJIT_EXPORT __attribute__((visibility("default")))
#else
#define ROKAJIT_EXPORT __declspec(dllexport)
#endif

extern "C" ROKAJIT_EXPORT void jitStartup(ICorJitHost* jitHost)
{
    g_jitHost = jitHost;
    g_jitInitialized = true;
}

extern "C" ROKAJIT_EXPORT ICorJitCompiler* getJit()
{
    if (!g_jitInitialized)
    {
        return nullptr;
    }

    return &g_compiler;
}

// ---------------------------------------------------------------------------
// EE forwarders — sample pattern for Phase 1's per-functional-group wrappers.
// One-liners only; Rust's EeInfo wrappers will call these through plain C.
// ---------------------------------------------------------------------------

extern "C" uint32_t rokajit_ee_get_jit_flags(
    ICorJitInfo*    info,
    CORJIT_FLAGS*   flags,
    uint32_t        sizeInBytes)
{
    return info->getJitFlags(flags, sizeInBytes);
}

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
