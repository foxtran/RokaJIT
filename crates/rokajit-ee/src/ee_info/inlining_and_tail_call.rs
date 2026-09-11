use std::ffi::{c_char, CString};

use rokajit_ffi as ffi;
use rokajit_ffi::{
    CORINFO_CONST_LOOKUP, CORINFO_RESOLVED_TOKEN, CORINFO_SIG_INFO, CORINFO_TAILCALL_HELPERS,
};

use super::wrap::zeroed_out;
use super::GasketEeInfo;
use crate::enums::{CorInfoInline, GetTailCallHelpersFlags};
use crate::handles::MethodHandle;

/// Inline / tail-call decisions (C++ `ICorMethodInfo`, inlining part).
pub trait InliningAndTailCall {
    /// C++ `ICorMethodInfo::canInline` (corinfo.h:2241).
    fn can_inline(&self, caller: MethodHandle, callee: MethodHandle) -> CorInfoInline;

    /// C++ `ICorMethodInfo::beginInlining` (corinfo.h:2248). Always paired
    /// with a `report_inlining_decision` call unless compilation fails.
    fn begin_inlining(&self, inliner: MethodHandle, inlinee: MethodHandle);

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

    /// C++ `ICorDynamicInfo::getTailCallHelpers` (corinfo.h:3514).
    /// `call_token` is the resolved token for the call; `None` (the C++
    /// nullptr) is legal for `calli`. `None` return = the C++ `false`
    /// (no helper-assisted tail call available for this site).
    fn get_tail_call_helpers(
        &self,
        call_token: Option<&CORINFO_RESOLVED_TOKEN>,
        sig: &CORINFO_SIG_INFO,
        flags: GetTailCallHelpersFlags,
    ) -> Option<CORINFO_TAILCALL_HELPERS>;

    /// C++ `ICorDynamicInfo::updateEntryPointForTailCall` (corinfo.h:3555).
    /// The EE rewrites `entry_point` in place to a tail-callable form (AOT
    /// x64 delay-loaded fast tailcalls).
    fn update_entry_point_for_tail_call(&self, entry_point: &mut CORINFO_CONST_LOOKUP);
}

extern "C" {
    fn rokajit_ee_can_inline(
        info: *mut ffi::ICorJitInfo,
        caller: ffi::CORINFO_METHOD_HANDLE,
        callee: ffi::CORINFO_METHOD_HANDLE,
    ) -> ffi::CorInfoInline;
    fn rokajit_ee_begin_inlining(
        info: *mut ffi::ICorJitInfo,
        inliner: ffi::CORINFO_METHOD_HANDLE,
        inlinee: ffi::CORINFO_METHOD_HANDLE,
    );
    fn rokajit_ee_report_inlining_decision(
        info: *mut ffi::ICorJitInfo,
        inliner: ffi::CORINFO_METHOD_HANDLE,
        inlinee: ffi::CORINFO_METHOD_HANDLE,
        verdict: ffi::CorInfoInline,
        reason: *const c_char,
    );
    fn rokajit_ee_can_tail_call(
        info: *mut ffi::ICorJitInfo,
        caller: ffi::CORINFO_METHOD_HANDLE,
        declared_callee: ffi::CORINFO_METHOD_HANDLE,
        exact_callee: ffi::CORINFO_METHOD_HANDLE,
        is_tail_prefix: bool,
    ) -> bool;
    fn rokajit_ee_report_tail_call_decision(
        info: *mut ffi::ICorJitInfo,
        caller: ffi::CORINFO_METHOD_HANDLE,
        callee: ffi::CORINFO_METHOD_HANDLE,
        is_tail_prefix: bool,
        verdict: ffi::CorInfoTailCall,
        reason: *const c_char,
    );
    fn rokajit_ee_get_tail_call_helpers(
        info: *mut ffi::ICorJitInfo,
        call_token: *mut ffi::CORINFO_RESOLVED_TOKEN,
        sig: *mut ffi::CORINFO_SIG_INFO,
        flags: ffi::CORINFO_GET_TAILCALL_HELPERS_FLAGS,
        result: *mut ffi::CORINFO_TAILCALL_HELPERS,
    ) -> bool;
    fn rokajit_ee_update_entry_point_for_tail_call(
        info: *mut ffi::ICorJitInfo,
        entry_point: *mut ffi::CORINFO_CONST_LOOKUP,
    );
}

impl InliningAndTailCall for GasketEeInfo {
    fn can_inline(&self, caller: MethodHandle, callee: MethodHandle) -> CorInfoInline {
        let raw =
            unsafe { rokajit_ee_can_inline(self.comp_raw(), caller.as_raw(), callee.as_raw()) };
        // A conforming EE only returns the six defined values; an
        // out-of-enum value can only come from headers that grew variants.
        // All failure verdicts are negative, so Fail is the safe fallback.
        CorInfoInline::from_raw(raw).unwrap_or(CorInfoInline::Fail)
    }

    fn begin_inlining(&self, inliner: MethodHandle, inlinee: MethodHandle) {
        unsafe { rokajit_ee_begin_inlining(self.comp_raw(), inliner.as_raw(), inlinee.as_raw()) };
    }

    fn report_inlining_decision(
        &self,
        inliner: MethodHandle,
        inlinee: MethodHandle,
        verdict: CorInfoInline,
        reason: &str,
    ) {
        // Reason strings are plain diagnostics; interior NULs degrade to the
        // empty string rather than failing the report.
        let reason = CString::new(reason).unwrap_or_default();
        unsafe {
            rokajit_ee_report_inlining_decision(
                self.comp_raw(),
                inliner.as_raw(),
                inlinee.as_raw(),
                verdict.to_raw(),
                reason.as_ptr(),
            )
        };
    }

    fn can_tail_call(
        &self,
        caller: MethodHandle,
        declared_callee: MethodHandle,
        exact_callee: MethodHandle,
        is_tail_prefix: bool,
    ) -> bool {
        unsafe {
            rokajit_ee_can_tail_call(
                self.comp_raw(),
                caller.as_raw(),
                declared_callee.as_raw(),
                exact_callee.as_raw(),
                is_tail_prefix,
            )
        }
    }

    fn report_tail_call_decision(
        &self,
        caller: MethodHandle,
        callee: MethodHandle,
        is_tail_prefix: bool,
        verdict: CorInfoInline,
        reason: &str,
    ) {
        // step_02 froze `verdict` as CorInfoInline, but C++ takes
        // CorInfoTailCall (corinfo.h:2273). Both are i32 enums whose success
        // values coincide numerically (0/1/2) and whose failures are
        // negative, so the raw value passes through unchanged. Recorded as
        // a step_02 contract contradiction for the orchestrator.
        let reason = CString::new(reason).unwrap_or_default();
        unsafe {
            rokajit_ee_report_tail_call_decision(
                self.comp_raw(),
                caller.as_raw(),
                callee.as_raw(),
                is_tail_prefix,
                verdict.to_raw(),
                reason.as_ptr(),
            )
        };
    }

    fn get_tail_call_helpers(
        &self,
        call_token: Option<&CORINFO_RESOLVED_TOKEN>,
        sig: &CORINFO_SIG_INFO,
        flags: GetTailCallHelpersFlags,
    ) -> Option<CORINFO_TAILCALL_HELPERS> {
        let call_token = call_token.map_or(std::ptr::null_mut(), |token| {
            token as *const CORINFO_RESOLVED_TOKEN as *mut CORINFO_RESOLVED_TOKEN
        });
        let mut ok = false;
        let result = zeroed_out(|result| {
            ok = unsafe {
                rokajit_ee_get_tail_call_helpers(
                    self.comp_raw(),
                    call_token,
                    sig as *const CORINFO_SIG_INFO as *mut CORINFO_SIG_INFO,
                    flags.to_raw(),
                    result,
                )
            };
        });
        // C++ false = no helper-assisted tail call for this site.
        ok.then_some(result)
    }

    fn update_entry_point_for_tail_call(&self, entry_point: &mut CORINFO_CONST_LOOKUP) {
        unsafe { rokajit_ee_update_entry_point_for_tail_call(self.comp_raw(), entry_point) };
    }
}
