use std::ffi::{c_char, c_int, c_uint, CString};
use std::ptr::NonNull;

use rokajit_ffi as ffi;
use rokajit_ffi::{CORINFO_EH_CLAUSE, CORINFO_SIG_INFO, CORJIT_FLAGS};

use super::wrap::zeroed_out;
use super::GasketEeInfo;
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

    /// C++ `ICorJitInfo::logMsg` (corjit.h:250). Returns `true` when the EE
    /// logged the message. Levels: 2 = error, 3 = warning, N >= 4 means the
    /// event happens roughly 10^(N-3) times per run.
    ///
    /// `unsafe` because `args` is the C `va_list` state for `fmt`, which
    /// stable Rust cannot construct: the caller must own valid
    /// `__builtin_va_list` storage (a `[__va_list_tag; 1]`) prepared with
    /// C-side varargs machinery and pass a pointer to it. On the C ABI a
    /// `va_list` parameter decays to a pointer to that storage, which is
    /// what the forwarder receives. Present for surface completeness; the
    /// core formats its own diagnostics instead of calling through.
    unsafe fn log_msg(
        &self,
        level: u32,
        fmt: *const c_char,
        args: *mut ffi::__va_list_tag,
    ) -> bool;

    /// C++ `ICorJitInfo::doAssert` (corjit.h:254). Returns `true` when the
    /// EE asks the JIT to retry (DebugBreak), `false` when the assert should
    /// be ignored. The C++ `int` return is a boolean per the header comment.
    fn do_assert(&self, file: &str, line: i32, expr: &str) -> bool;
}

extern "C" {
    fn rokajit_ee_get_jit_flags(
        info: *mut ffi::ICorJitInfo,
        flags: *mut CORJIT_FLAGS,
        size_in_bytes: u32,
    ) -> u32;
    fn rokajit_ee_alloc_mem(info: *mut ffi::ICorJitInfo, p_args: *mut ffi::AllocMemArgs);
    fn rokajit_ee_reserve_unwind_info(
        info: *mut ffi::ICorJitInfo,
        is_funclet: bool,
        is_cold_code: bool,
        unwind_size: u32,
    );
    fn rokajit_ee_alloc_unwind_info(
        info: *mut ffi::ICorJitInfo,
        p_hot_code: *mut u8,
        p_cold_code: *mut u8,
        start_offset: u32,
        end_offset: u32,
        unwind_size: u32,
        p_unwind_block: *mut u8,
        func_kind: ffi::CorJitFuncKind,
    );
    fn rokajit_ee_alloc_gc_info(info: *mut ffi::ICorJitInfo, size: usize) -> *mut u8;
    fn rokajit_ee_set_eh_count(info: *mut ffi::ICorJitInfo, c_eh: c_uint);
    fn rokajit_ee_set_eh_info(
        info: *mut ffi::ICorJitInfo,
        eh_number: c_uint,
        clause: *const CORINFO_EH_CLAUSE,
    );
    fn rokajit_ee_log_msg(
        info: *mut ffi::ICorJitInfo,
        level: c_uint,
        fmt: *const c_char,
        args: *mut ffi::__va_list_tag,
    ) -> bool;
    fn rokajit_ee_do_assert(
        info: *mut ffi::ICorJitInfo,
        sz_file: *const c_char,
        i_line: c_int,
        sz_expr: *const c_char,
    ) -> c_int;
    fn rokajit_ee_report_fatal_error(info: *mut ffi::ICorJitInfo, result: ffi::CorJitResult);
    fn rokajit_ee_record_call_site(
        info: *mut ffi::ICorJitInfo,
        instr_offset: u32,
        call_sig: *mut CORINFO_SIG_INFO,
        method_handle: ffi::CORINFO_METHOD_HANDLE,
    );
}

impl OutputSinks for GasketEeInfo {
    fn get_jit_flags(&self) -> CORJIT_FLAGS {
        zeroed_out(|flags| unsafe {
            rokajit_ee_get_jit_flags(
                self.comp_raw(),
                flags,
                std::mem::size_of::<CORJIT_FLAGS>() as u32,
            );
        })
    }

    fn alloc_mem(&self, request: &[ChunkRequest], xcptns_count: u32) -> Vec<AllocatedChunk> {
        let mut chunks: Vec<ffi::AllocMemChunk> = request
            .iter()
            .map(|r| ffi::AllocMemChunk {
                alignment: r.alignment,
                size: r.size,
                flags: r.flags.to_raw(),
                block: std::ptr::null_mut(),
                blockRW: std::ptr::null_mut(),
            })
            .collect();
        let mut args = ffi::AllocMemArgs {
            chunks: chunks.as_mut_ptr(),
            chunksCount: chunks.len() as c_uint,
            xcptnsCount: xcptns_count,
        };
        unsafe { rokajit_ee_alloc_mem(self.comp_raw(), &mut args) };
        // The EE fills every chunk or throws (jitinterface.cpp
        // CEEJitInfo::allocMem); a null block after a normal return is an EE
        // contract violation, i.e. a bug.
        chunks
            .iter()
            .map(|c| AllocatedChunk {
                executable: NonNull::new(c.block)
                    .expect("EE contract: allocMem fills every chunk or throws"),
                writable: NonNull::new(c.blockRW)
                    .expect("EE contract: allocMem fills every chunk or throws"),
            })
            .collect()
    }

    fn reserve_unwind_info(&self, is_funclet: bool, is_cold_code: bool, unwind_size: u32) {
        unsafe {
            rokajit_ee_reserve_unwind_info(self.comp_raw(), is_funclet, is_cold_code, unwind_size)
        };
    }

    fn alloc_unwind_info(
        &self,
        hot_code: NonNull<u8>,
        cold_code: Option<NonNull<u8>>,
        start_offset: u32,
        end_offset: u32,
        unwind: &[u8],
        func_kind: CorJitFuncKind,
    ) {
        let cold_code = cold_code.map_or(std::ptr::null_mut(), NonNull::as_ptr);
        unsafe {
            rokajit_ee_alloc_unwind_info(
                self.comp_raw(),
                hot_code.as_ptr(),
                cold_code,
                start_offset,
                end_offset,
                unwind.len() as u32,
                unwind.as_ptr() as *mut u8,
                func_kind.to_raw(),
            )
        };
    }

    fn alloc_gc_info(&self, size: usize) -> NonNull<u8> {
        let block = unsafe { rokajit_ee_alloc_gc_info(self.comp_raw(), size) };
        NonNull::new(block).expect("EE contract: allocGCInfo returns a block or throws")
    }

    fn set_eh_count(&self, count: u32) {
        unsafe { rokajit_ee_set_eh_count(self.comp_raw(), count) };
    }

    fn set_eh_info(&self, index: u32, clause: &CORINFO_EH_CLAUSE) {
        unsafe { rokajit_ee_set_eh_info(self.comp_raw(), index, clause) };
    }

    fn record_call_site(
        &self,
        instr_offset: u32,
        sig: Option<&CORINFO_SIG_INFO>,
        method: Option<MethodHandle>,
    ) {
        let sig = sig.map_or(std::ptr::null_mut(), |s| s as *const _ as *mut _);
        let method = method.map_or(std::ptr::null_mut(), MethodHandle::as_raw);
        unsafe { rokajit_ee_record_call_site(self.comp_raw(), instr_offset, sig, method) };
    }

    fn report_fatal_error(&self, result: CorJitResult) {
        unsafe { rokajit_ee_report_fatal_error(self.comp_raw(), result.to_raw()) };
    }

    unsafe fn log_msg(
        &self,
        level: u32,
        fmt: *const c_char,
        args: *mut ffi::__va_list_tag,
    ) -> bool {
        unsafe { rokajit_ee_log_msg(self.comp_raw(), level, fmt, args) }
    }

    fn do_assert(&self, file: &str, line: i32, expr: &str) -> bool {
        // Interior NULs in the caller's strings are a caller bug; degrade to
        // "ignore the assert" rather than panic across an FFI call setup.
        let (Ok(file), Ok(expr)) = (CString::new(file), CString::new(expr)) else {
            return false;
        };
        unsafe { rokajit_ee_do_assert(self.comp_raw(), file.as_ptr(), line, expr.as_ptr()) != 0 }
    }
}
