use rokajit_ffi::{
    CORINFO_CONST_LOOKUP, CORINFO_RESOLVED_TOKEN, CORINFO_SIG_INFO, CORINFO_TAILCALL_HELPERS,
};

use super::MockEe;
use crate::ee_info::InliningAndTailCall;
use crate::enums::{CorInfoInline, GetTailCallHelpersFlags};
use crate::handles::MethodHandle;

impl InliningAndTailCall for MockEe {
    fn can_inline(&self, _caller: MethodHandle, _callee: MethodHandle) -> CorInfoInline {
        CorInfoInline::Never
    }

    fn begin_inlining(&self, inliner: MethodHandle, inlinee: MethodHandle) {
        self.sink_log
            .borrow_mut()
            .push(format!("begin_inlining({inliner:?}, {inlinee:?})"));
    }

    fn report_inlining_decision(
        &self,
        _inliner: MethodHandle,
        _inlinee: MethodHandle,
        _verdict: CorInfoInline,
        _reason: &str,
    ) {
    }

    fn can_tail_call(
        &self,
        _caller: MethodHandle,
        _declared_callee: MethodHandle,
        _exact_callee: MethodHandle,
        _is_tail_prefix: bool,
    ) -> bool {
        false
    }

    fn report_tail_call_decision(
        &self,
        _caller: MethodHandle,
        _callee: MethodHandle,
        _is_tail_prefix: bool,
        _verdict: CorInfoInline,
        _reason: &str,
    ) {
    }

    fn get_tail_call_helpers(
        &self,
        _call_token: Option<&CORINFO_RESOLVED_TOKEN>,
        _sig: &CORINFO_SIG_INFO,
        _flags: GetTailCallHelpersFlags,
    ) -> Option<CORINFO_TAILCALL_HELPERS> {
        // Canned answer: no helper-assisted tail call (the C++ false).
        None
    }

    fn update_entry_point_for_tail_call(&self, _entry_point: &mut CORINFO_CONST_LOOKUP) {
        self.sink_log
            .borrow_mut()
            .push("update_entry_point_for_tail_call()".to_string());
    }
}
