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

#pragma once

// <cstddef>/<cstdint> first: the CoreCLR headers use size_t & co. without
// including the headers that define them (same reason the bindgen wrapper
// header leads with these).
#include <cstddef>
#include <cstdint>

#include "corjit.h"
#include "corjithost.h"
#include "jiteeversionguid.h"

#include <cstdio>
#include <type_traits>

// ---------------------------------------------------------------------------
// Exception trap (frozen pattern, decisions/2026-09-11-gasket-exception-trap.md):
// a C++ exception must never unwind into Rust, so every gasket forwarder
// routes its single virtual call through one of these. reportFatalError is
// the EE's own fatal channel; a conforming EE does not return from it, so
// the zero fallback in the value case is unreachable. ICorJitHost has no
// error channel, so the host variant logs to stderr and returns the zero
// sentinel instead. Note catch (...) traps C++ exceptions only; a hardware
// fault inside the EE unwinds as an unmanaged (SEH) exception, which no
// C++ handler can catch (observed live in step_05).
// ---------------------------------------------------------------------------

template <typename F>
inline auto rokajit_ee_trap(ICorJitInfo* info, F&& call) -> decltype(call())
{
    using R = decltype(call());
    try
    {
        return call();
    }
    catch (...)
    {
        info->reportFatalError(CORJIT_INTERNALERROR);
        if constexpr (!std::is_void_v<R>)
            return R{};
        else
            return;
    }
}

template <typename F>
inline auto rokajit_host_trap(const char* cppName, F&& call) -> decltype(call())
{
    using R = decltype(call());
    try
    {
        return call();
    }
    catch (...)
    {
        fprintf(stderr, "rokajit: C++ exception escaped ICorJitHost::%s\n", cppName);
        if constexpr (!std::is_void_v<R>)
            return R{};
        else
            return;
    }
}


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

// Defined in rokajit-ee (`ee_info/real.rs`): captures the process-lifetime
// ICorJitHost for GasketEeInfo. Called once from jitStartup below.
extern "C" void rokajit_ee_set_jit_host(ICorJitHost* jitHost);

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
// EE forwarders — sample pattern for Phase 1's per-functional-group wrappers.
// One-liners only; Rust's EeInfo wrappers will call these through plain C.
// ---------------------------------------------------------------------------
