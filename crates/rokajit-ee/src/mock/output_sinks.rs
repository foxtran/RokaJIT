use std::ffi::c_char;
use std::ptr::NonNull;

use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::{AllocatedChunk, ChunkRequest, OutputSinks};
use crate::enums::{CorJitFuncKind, CorJitResult};
use crate::handles::MethodHandle;

impl OutputSinks for MockEe {
    fn get_jit_flags(&self) -> ffi::CORJIT_FLAGS {
        unsafe { std::mem::zeroed() }
    }

    fn alloc_mem(&self, request: &[ChunkRequest], xcptns_count: u32) -> Vec<AllocatedChunk> {
        self.sink_log
            .borrow_mut()
            .push(format!("alloc_mem({}, xcptns={})", request.len(), xcptns_count));
        request
            .iter()
            .map(|r| {
                let p = self.fake_alloc(r.size as usize);
                AllocatedChunk { executable: p, writable: p }
            })
            .collect()
    }

    fn reserve_unwind_info(&self, is_funclet: bool, is_cold_code: bool, unwind_size: u32) {
        self.sink_log.borrow_mut().push(format!(
            "reserve_unwind_info({is_funclet}, {is_cold_code}, {unwind_size})"
        ));
    }

    fn alloc_unwind_info(
        &self,
        _hot_code: NonNull<u8>,
        _cold_code: Option<NonNull<u8>>,
        _start_offset: u32,
        _end_offset: u32,
        unwind: &[u8],
        func_kind: CorJitFuncKind,
    ) {
        self.sink_log
            .borrow_mut()
            .push(format!("alloc_unwind_info({}, {func_kind:?})", unwind.len()));
    }

    fn alloc_gc_info(&self, size: usize) -> NonNull<u8> {
        self.fake_alloc(size)
    }

    fn set_eh_count(&self, count: u32) {
        self.sink_log.borrow_mut().push(format!("set_eh_count({count})"));
    }

    fn set_eh_info(&self, index: u32, _clause: &ffi::CORINFO_EH_CLAUSE) {
        self.sink_log.borrow_mut().push(format!("set_eh_info({index})"));
    }

    fn record_call_site(
        &self,
        instr_offset: u32,
        _sig: Option<&ffi::CORINFO_SIG_INFO>,
        _method: Option<MethodHandle>,
    ) {
        self.sink_log
            .borrow_mut()
            .push(format!("record_call_site({instr_offset})"));
    }

    fn report_fatal_error(&self, result: CorJitResult) {
        self.sink_log
            .borrow_mut()
            .push(format!("report_fatal_error({result:?})"));
    }

    unsafe fn log_msg(
        &self,
        level: u32,
        _fmt: *const c_char,
        _args: *mut ffi::__va_list_tag,
    ) -> bool {
        self.sink_log.borrow_mut().push(format!("log_msg({level})"));
        // The mock's sink_log IS the log, so logging always "succeeded".
        true
    }

    fn do_assert(&self, file: &str, line: i32, expr: &str) -> bool {
        self.sink_log
            .borrow_mut()
            .push(format!("do_assert({file}:{line}: {expr})"));
        // false = ignore the assert (never asks for a DebugBreak retry).
        false
    }
}
