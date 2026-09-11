// Host forwarders: one extern "C" one-liner per ICorJitHost virtual
// (corjithost.h:14), named rokajit_host_*. ICorJitHost has no error channel,
// so the exception trap logs to stderr and returns the zero sentinel
// (decisions/2026-09-11-gasket-exception-trap.md).

#include "gasket_common.h"
#include <cstdio>

extern "C" void* rokajit_host_allocate_memory(
    ICorJitHost*    host,
    size_t          size)
{ return rokajit_host_trap("allocateMemory", [&] { return host->allocateMemory(size); }); }

extern "C" void rokajit_host_free_memory(
    ICorJitHost*    host,
    void*           block)
{ return rokajit_host_trap("freeMemory", [&] { return host->freeMemory(block); }); }

extern "C" int rokajit_host_get_int_config_value(
    ICorJitHost*    host,
    const char*     name,
    int             defaultValue)
{ return rokajit_host_trap("getIntConfigValue", [&] { return host->getIntConfigValue(name, defaultValue); }); }

extern "C" const char* rokajit_host_get_string_config_value(
    ICorJitHost*    host,
    const char*     name)
{ return rokajit_host_trap("getStringConfigValue", [&] { return host->getStringConfigValue(name); }); }

extern "C" void rokajit_host_free_string_config_value(
    ICorJitHost*    host,
    const char*     value)
{ return rokajit_host_trap("freeStringConfigValue", [&] { return host->freeStringConfigValue(value); }); }

extern "C" void* rokajit_host_allocate_slab(
    ICorJitHost*    host,
    size_t          size,
    size_t*         pActualSize)
{ return rokajit_host_trap("allocateSlab", [&] { return host->allocateSlab(size, pActualSize); }); }

extern "C" void rokajit_host_free_slab(
    ICorJitHost*    host,
    void*           slab,
    size_t          actualSize)
{ return rokajit_host_trap("freeSlab", [&] { return host->freeSlab(slab, actualSize); }); }
