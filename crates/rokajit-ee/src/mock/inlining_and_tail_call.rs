use super::MockEe;
use crate::ee_info::InliningAndTailCall;
use crate::enums::CorInfoInline;
use crate::handles::MethodHandle;

impl InliningAndTailCall for MockEe {
    fn can_inline(&self, _caller: MethodHandle, _callee: MethodHandle) -> CorInfoInline {
        CorInfoInline::Never
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
}
