#include "gasket_common.h"

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
    rokajit_ee_set_jit_host(jitHost);
    // Step_06 startup hook: resolve config + unsupported-knob warnings.
    // Rust-to-C++ needs no exception trap (Rust cannot throw C++
    // exceptions); the Rust side guards with catch_unwind per the
    // error-model decision, and its EE queries go through the trapped
    // rokajit_host_* forwarders.
    rokajit_on_startup(jitHost);
}

extern "C" ROKAJIT_EXPORT ICorJitCompiler* getJit()
{
    if (!g_jitInitialized)
    {
        return nullptr;
    }

    return &g_compiler;
}
