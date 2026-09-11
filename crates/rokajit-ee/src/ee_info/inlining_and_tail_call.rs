use crate::enums::CorInfoInline;
use crate::handles::MethodHandle;

/// Inline / tail-call decisions (C++ `ICorMethodInfo`, inlining part).
pub trait InliningAndTailCall {
    /// C++ `ICorMethodInfo::canInline` (corinfo.h:2241).
    fn can_inline(&self, caller: MethodHandle, callee: MethodHandle) -> CorInfoInline;

    /// C++ `ICorMethodInfo::reportInliningDecision` (corinfo.h:2254).
    /// Mandatory after every `can_inline` probe unless the compile fails.
    fn report_inlining_decision(
        &self,
        inliner: MethodHandle,
        inlinee: MethodHandle,
        verdict: CorInfoInline,
        reason: &str,
    );

    /// C++ `ICorMethodInfo::canTailCall` (corinfo.h:2263).
    fn can_tail_call(
        &self,
        caller: MethodHandle,
        declared_callee: MethodHandle,
        exact_callee: MethodHandle,
        is_tail_prefix: bool,
    ) -> bool;

    /// C++ `ICorMethodInfo::reportTailCallDecision` (corinfo.h:2273).
    fn report_tail_call_decision(
        &self,
        caller: MethodHandle,
        callee: MethodHandle,
        is_tail_prefix: bool,
        verdict: CorInfoInline,
        reason: &str,
    );
}
