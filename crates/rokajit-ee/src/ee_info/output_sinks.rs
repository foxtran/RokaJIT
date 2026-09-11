use std::ptr::NonNull;

use rokajit_ffi::{CORINFO_EH_CLAUSE, CORINFO_SIG_INFO, CORJIT_FLAGS};

use crate::enums::{AllocMemFlags, CorJitFuncKind, CorJitResult};
use crate::handles::MethodHandle;

/// One memory chunk requested from the EE (row shape of C++ `AllocMemChunk`
/// in-fields, corjit.h:79).
#[derive(Copy, Clone, Debug)]
pub struct ChunkRequest {
    /// Power-of-two alignment; hot code ≤ 32, cold code = 1, RO data ≤ 64
    /// (corjit.h:81).
    pub alignment: u32,
    pub size: u32,
    pub flags: AllocMemFlags,
}

/// One chunk allocated by the EE (row shape of `AllocMemChunk` out-fields).
/// `executable` is the address the CPU reads; `writable` is the alias to
/// write through (identical on Linux/x64 today, distinct on W^X targets).
/// Both are EE-owned and live as long as the compiled method.
#[derive(Copy, Clone, Debug)]
pub struct AllocatedChunk {
    pub executable: NonNull<u8>,
    pub writable: NonNull<u8>,
}

/// Per-compilation output sinks (C++ `ICorJitInfo`, corjit.h:171).
///
/// Ordering contract, from the header comments: `reserve_unwind_info`
/// before [`OutputSinks::alloc_mem`]; everything else after. `set_eh_count`
/// before any `set_eh_info`. These sinks return `()` because the EE signals
/// allocation failure by longjmp, never by return value.
pub trait OutputSinks {
    /// C++ `ICorJitInfo::getJitFlags` (corjit.h:454): the per-compilation
    /// `CORJIT_FLAGS` (optimization/debug/EH switches). Returned as the
    /// layout-verified bindgen struct.
    fn get_jit_flags(&self) -> CORJIT_FLAGS;

    /// C++ `ICorJitInfo::allocMem` (corjit.h:175). One hot-code chunk, at
    /// most one cold-code chunk, any number of data chunks; result chunks
    /// line up 1:1 with the request.
    fn alloc_mem(&self, request: &[ChunkRequest], xcptns_count: u32) -> Vec<AllocatedChunk>;

    /// C++ `ICorJitInfo::reserveUnwindInfo` (corjit.h:190). Once for the
    /// root function, once per funclet, once per cold-code block.
    fn reserve_unwind_info(&self, is_funclet: bool, is_cold_code: bool, unwind_size: u32);

    /// C++ `ICorJitInfo::allocUnwindInfo` (corjit.h:213).
    fn alloc_unwind_info(
        &self,
        hot_code: NonNull<u8>,
        cold_code: Option<NonNull<u8>>,
        start_offset: u32,
        end_offset: u32,
        unwind: &[u8],
        func_kind: CorJitFuncKind,
    );

    /// C++ `ICorJitInfo::allocGCInfo` (corjit.h:226): EE-owned buffer the
    /// JIT writes the encoded GC info blob into.
    fn alloc_gc_info(&self, size: usize) -> NonNull<u8>;

    /// C++ `ICorJitInfo::setEHcount` (corjit.h:233).
    fn set_eh_count(&self, count: u32);

    /// C++ `ICorJitInfo::setEHinfo` (corjit.h:242). `clause` is the bindgen
    /// `CORINFO_EH_CLAUSE` (corinfo.h:1619) with native (not IL) offsets.
    fn set_eh_info(&self, index: u32, clause: &CORINFO_EH_CLAUSE);

    /// C++ `ICorJitInfo::recordCallSite` (corjit.h:421): associate a native
    /// call site with the sig/handle used to lay it out. Both may be absent
    /// (helper calls, calli) — the C++ nullptrs become `None`.
    fn record_call_site(
        &self,
        instr_offset: u32,
        sig: Option<&CORINFO_SIG_INFO>,
        method: Option<MethodHandle>,
    );

    /// C++ `ICorJitInfo::reportFatalError` (corjit.h:256). Called by the
    /// outermost panic guard when the EE info object is available; see
    /// `docs/error-model.md`.
    fn report_fatal_error(&self, result: CorJitResult);
}
