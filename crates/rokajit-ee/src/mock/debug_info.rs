use std::ffi::CStr;
use std::ptr::NonNull;

use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::{BoundaryMap, DebugInfo, NativeVarInfo};
use crate::handles::{JustMyCodeHandle, MethodHandle, ProfilingHandle};

impl DebugInfo for MockEe {
    fn get_boundaries(&self, _ftn: MethodHandle) -> Vec<u32> {
        Vec::new()
    }

    fn set_boundaries(&self, _ftn: MethodHandle, map: &[BoundaryMap]) {
        self.sink_log
            .borrow_mut()
            .push(format!("set_boundaries({})", map.len()));
    }

    fn get_vars(&self, _ftn: MethodHandle) -> Vec<u32> {
        Vec::new()
    }

    fn set_vars(&self, _ftn: MethodHandle, vars: &[NativeVarInfo]) {
        self.sink_log
            .borrow_mut()
            .push(format!("set_vars({})", vars.len()));
    }

    fn report_rich_mappings(
        &self,
        inline_tree_nodes: &[ffi::ICorDebugInfo_InlineTreeNode],
        mappings: &[ffi::ICorDebugInfo_RichOffsetMapping],
    ) {
        self.sink_log.borrow_mut().push(format!(
            "report_rich_mappings({}, {})",
            inline_tree_nodes.len(),
            mappings.len()
        ));
    }

    fn report_async_debug_info(
        &self,
        async_info: &ffi::ICorDebugInfo_AsyncInfo,
        suspension_points: &[ffi::ICorDebugInfo_AsyncSuspensionPoint],
        vars: &[ffi::ICorDebugInfo_AsyncContinuationVarInfo],
    ) {
        self.sink_log.borrow_mut().push(format!(
            "report_async_debug_info({}, {}, {})",
            async_info.NumSuspensionPoints,
            suspension_points.len(),
            vars.len()
        ));
    }

    fn report_metadata(&self, key: &CStr, value: &[u8]) {
        self.sink_log
            .borrow_mut()
            .push(format!("report_metadata({:?}, {})", key, value.len()));
    }

    fn allocate_array(&self, bytes: usize) -> Option<NonNull<u8>> {
        Some(self.fake_alloc(bytes))
    }

    fn free_array(&self, _array: NonNull<u8>) {
        self.sink_log.borrow_mut().push("free_array".to_string());
    }

    fn get_just_my_code_handle(
        &self,
        _method: MethodHandle,
    ) -> (
        Option<JustMyCodeHandle>,
        Option<NonNull<ffi::CORINFO_JUST_MY_CODE_HANDLE>>,
    ) {
        (None, None)
    }

    fn get_profiling_handle(&self) -> (bool, Option<ProfilingHandle>, bool) {
        (false, None, false)
    }

    fn method_compile_complete(&self, _meth_hnd: MethodHandle) {
        self.sink_log
            .borrow_mut()
            .push("method_compile_complete".to_string());
    }
}
