use std::ffi::c_void;

use rokajit_ffi as ffi;
use rokajit_ffi::{
    DelegateCtorArgs, CORINFO_CONST_LOOKUP, CORINFO_EE_INFO, CORINFO_LOOKUP,
    CORINFO_RESOLVED_TOKEN, CORINFO_WASM_WELLKNOWN_GLOBALS,
};

use super::wrap::zeroed_out;
use super::GasketEeInfo;
use crate::enums::{AccessFlags, CorInfoHelpFunc, InstructionSet};
use crate::handles::{ClassHandle, MethodHandle};

/// Where to call an EE helper (result of
/// [`Helpers::get_helper_ftn`]).
///
/// `entrypoint` is the EE's access description (C++ `CORINFO_CONST_LOOKUP`,
/// corinfo.h:1040): either a direct address or a handle to indirect
/// through, per its `accessType` field.
pub struct HelperTarget {
    pub entrypoint: CORINFO_CONST_LOOKUP,
    /// The managed method implementing the helper, when it has one (the
    /// C++ `pMethodHandle` out-param; nullptr → `None`).
    pub method: Option<MethodHandle>,
}

/// Runtime helper selection (C++ `ICorDynamicInfo`, helper part).
pub trait Helpers {
    /// C++ `ICorDynamicInfo::getHelperFtn` (corinfo.h:3317).
    fn get_helper_ftn(&self, id: CorInfoHelpFunc) -> HelperTarget;

    /// C++ `ICorClassInfo::getNewHelper` (corinfo.h:2658). The C++
    /// `pHasSideEffects` out-param is folded into the return:
    /// `(helper, has_side_effects)`; `None` when the C++ `fHasSideEffects`
    /// in/out was null — see the header comment for the distinction.
    fn get_new_helper(
        &self,
        token: &CORINFO_RESOLVED_TOKEN,
        caller: MethodHandle,
    ) -> (CorInfoHelpFunc, Option<bool>);

    /// C++ `ICorClassInfo::getCastingHelper` (corinfo.h:2669).
    fn get_casting_helper(&self, token: &CORINFO_RESOLVED_TOKEN, throwing: bool)
        -> CorInfoHelpFunc;

    /// C++ `ICorClassInfo::getBoxHelper` (corinfo.h:2687).
    fn get_box_helper(&self, cls: ClassHandle) -> CorInfoHelpFunc;

    /// C++ `ICorDynamicInfo::getFunctionEntryPoint` (corinfo.h:3326).
    fn get_function_entry_point(&self, ftn: MethodHandle) -> CORINFO_CONST_LOOKUP;

    /// C++ `ICorClassInfo::getNewArrHelper` (corinfo.h:2664): the 1-D array
    /// allocation helper optimized for `array_cls`.
    fn get_new_arr_helper(&self, array_cls: ClassHandle) -> CorInfoHelpFunc;

    /// C++ `ICorClassInfo::getSharedCCtorHelper` (corinfo.h:2675): the
    /// helper that triggers `cls`'s static constructor.
    fn get_shared_cctor_helper(&self, cls: ClassHandle) -> CorInfoHelpFunc;

    /// C++ `ICorClassInfo::getTypeForBox` (corinfo.h:2680): the type actually
    /// produced by boxing `cls` (boxing `Nullable<T>` yields boxed `T`).
    /// Infallible — the EE always has an answer.
    fn get_type_for_box(&self, cls: ClassHandle) -> ClassHandle;

    /// C++ `ICorClassInfo::getUnBoxHelper` (corinfo.h:2703). Note the C++
    /// `helperCopies` copy-vs-pointer distinction described in the header
    /// comment is a request-level contract, not a parameter of this call.
    fn get_un_box_helper(&self, cls: ClassHandle) -> CorInfoHelpFunc;

    /// C++ `ICorDynamicInfo::getReadyToRunHelper` (corinfo.h:2755). `None`
    /// when the C++ bool result is false — no ReadyToRun helper exists for
    /// this lookup and `pLookup` is not meaningful.
    fn get_ready_to_run_helper(
        &self,
        token: &CORINFO_RESOLVED_TOKEN,
        id: CorInfoHelpFunc,
        caller: MethodHandle,
    ) -> Option<CORINFO_CONST_LOOKUP>;

    /// C++ `ICorDynamicInfo::getReadyToRunDelegateCtorHelper`
    /// (corinfo.h:2762).
    fn get_ready_to_run_delegate_ctor_helper(
        &self,
        target_method: &CORINFO_RESOLVED_TOKEN,
        target_constraint: ffi::mdToken,
        delegate_type: ClassHandle,
        caller: MethodHandle,
    ) -> CORINFO_LOOKUP;

    /// C++ `ICorStaticInfo::runWithErrorTrap` (corinfo.h:3137): runs
    /// `function(parameter)` under the EE's exception trap. Returns `true`
    /// when `function` completed, `false` when it threw (the EE caught the
    /// exception). `function` must be callable from C++ with C ABI and must
    /// itself never let a Rust panic escape.
    fn run_with_error_trap(
        &self,
        function: extern "C" fn(*mut c_void),
        parameter: *mut c_void,
    ) -> bool;

    /// C++ `ICorStaticInfo::runWithSPMIErrorTrap` (corinfo.h:3146): like
    /// [`Helpers::run_with_error_trap`], but the trap also recognizes
    /// SuperPMI exceptions.
    fn run_with_spmi_error_trap(
        &self,
        function: extern "C" fn(*mut c_void),
        parameter: *mut c_void,
    ) -> bool;

    /// C++ `ICorStaticInfo::getEEInfo` (corinfo.h:3159): EE-internal data
    /// structure layout, returned as the bindgen `CORINFO_EE_INFO`.
    fn get_ee_info(&self) -> CORINFO_EE_INFO;

    /// C++ `ICorStaticInfo::getWasmWellKnownGlobals` (corinfo.h:3271). Only
    /// meaningful when targeting wasm; on other targets the EE fills zeros.
    fn get_wasm_well_known_globals(&self) -> CORINFO_WASM_WELLKNOWN_GLOBALS;

    /// C++ `ICorDynamicInfo::getFunctionFixedEntryPoint` (corinfo.h:3335):
    /// a directly callable, multi-callable entry point for `ftn`.
    fn get_function_fixed_entry_point(
        &self,
        ftn: MethodHandle,
        is_unsafe_function_pointer: bool,
    ) -> CORINFO_CONST_LOOKUP;

    /// C++ `ICorDynamicInfo::getAddressOfPInvokeTarget` (corinfo.h:3389).
    /// The result may be a fixup area for late-bound PInvoke calls.
    fn get_address_of_p_invoke_target(&self, method: MethodHandle) -> CORINFO_CONST_LOOKUP;

    /// C++ `ICorDynamicInfo::GetDelegateCtor` (corinfo.h:3502). `None` when
    /// the EE finds no usable delegate ctor (the C++ null return).
    fn get_delegate_ctor(
        &self,
        meth: MethodHandle,
        cls: ClassHandle,
        target_method: MethodHandle,
        ctor_data: &DelegateCtorArgs,
    ) -> Option<MethodHandle>;

    /// C++ `ICorDynamicInfo::notifyInstructionSetUsage` (corinfo.h:3545):
    /// tells the EE the method will (`support_enabled` = true) or will not
    /// use `instruction_set`. Returns `true` when the instruction set is
    /// supported unconditionally (no runtime check needed).
    fn notify_instruction_set_usage(
        &self,
        instruction_set: InstructionSet,
        support_enabled: bool,
    ) -> bool;

    /// C++ `ICorDynamicInfo::getSpecialCopyHelper` (corinfo.h:3562). `None`
    /// when the type has no special copy helper (the C++ null return — the
    /// common answer).
    fn get_special_copy_helper(&self, cls: ClassHandle) -> Option<MethodHandle>;
}

extern "C" {
    fn rokajit_ee_get_new_helper(
        info: *mut ffi::ICorJitInfo,
        class_handle: ffi::CORINFO_CLASS_HANDLE,
        p_has_side_effects: *mut bool,
    ) -> ffi::CorInfoHelpFunc;
    fn rokajit_ee_get_new_arr_helper(
        info: *mut ffi::ICorJitInfo,
        array_cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoHelpFunc;
    fn rokajit_ee_get_casting_helper(
        info: *mut ffi::ICorJitInfo,
        p_resolved_token: *mut CORINFO_RESOLVED_TOKEN,
        f_throwing: bool,
    ) -> ffi::CorInfoHelpFunc;
    fn rokajit_ee_get_shared_cctor_helper(
        info: *mut ffi::ICorJitInfo,
        cls_hnd: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoHelpFunc;
    fn rokajit_ee_get_type_for_box(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_box_helper(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoHelpFunc;
    fn rokajit_ee_get_un_box_helper(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoHelpFunc;
    fn rokajit_ee_get_ready_to_run_helper(
        info: *mut ffi::ICorJitInfo,
        p_resolved_token: *mut CORINFO_RESOLVED_TOKEN,
        id: ffi::CorInfoHelpFunc,
        caller_handle: ffi::CORINFO_METHOD_HANDLE,
        p_lookup: *mut CORINFO_CONST_LOOKUP,
    ) -> bool;
    fn rokajit_ee_get_ready_to_run_delegate_ctor_helper(
        info: *mut ffi::ICorJitInfo,
        p_target_method: *mut CORINFO_RESOLVED_TOKEN,
        target_constraint: ffi::mdToken,
        delegate_type: ffi::CORINFO_CLASS_HANDLE,
        caller_handle: ffi::CORINFO_METHOD_HANDLE,
        p_lookup: *mut CORINFO_LOOKUP,
    );
    fn rokajit_ee_run_with_error_trap(
        info: *mut ffi::ICorJitInfo,
        function: Option<extern "C" fn(*mut c_void)>,
        parameter: *mut c_void,
    ) -> bool;
    fn rokajit_ee_run_with_spmi_error_trap(
        info: *mut ffi::ICorJitInfo,
        function: Option<extern "C" fn(*mut c_void)>,
        parameter: *mut c_void,
    ) -> bool;
    fn rokajit_ee_get_ee_info(info: *mut ffi::ICorJitInfo, p_ee_info_out: *mut CORINFO_EE_INFO);
    fn rokajit_ee_get_wasm_well_known_globals(
        info: *mut ffi::ICorJitInfo,
        p_well_known_globals_out: *mut CORINFO_WASM_WELLKNOWN_GLOBALS,
    );
    fn rokajit_ee_get_helper_ftn(
        info: *mut ffi::ICorJitInfo,
        ftn_num: ffi::CorInfoHelpFunc,
        p_native_entrypoint: *mut CORINFO_CONST_LOOKUP,
        p_method_handle: *mut ffi::CORINFO_METHOD_HANDLE,
    );
    fn rokajit_ee_get_function_entry_point(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        p_result: *mut CORINFO_CONST_LOOKUP,
        access_flags: ffi::CORINFO_ACCESS_FLAGS,
    );
    fn rokajit_ee_get_function_fixed_entry_point(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        is_unsafe_function_pointer: bool,
        p_result: *mut CORINFO_CONST_LOOKUP,
    );
    fn rokajit_ee_get_address_of_p_invoke_target(
        info: *mut ffi::ICorJitInfo,
        method: ffi::CORINFO_METHOD_HANDLE,
        p_lookup: *mut CORINFO_CONST_LOOKUP,
    );
    fn rokajit_ee_get_delegate_ctor(
        info: *mut ffi::ICorJitInfo,
        meth_hnd: ffi::CORINFO_METHOD_HANDLE,
        cls_hnd: ffi::CORINFO_CLASS_HANDLE,
        target_method_hnd: ffi::CORINFO_METHOD_HANDLE,
        p_ctor_data: *mut DelegateCtorArgs,
    ) -> ffi::CORINFO_METHOD_HANDLE;
    fn rokajit_ee_notify_instruction_set_usage(
        info: *mut ffi::ICorJitInfo,
        instruction_set: ffi::CORINFO_InstructionSet,
        support_enabled: bool,
    ) -> bool;
    fn rokajit_ee_get_special_copy_helper(
        info: *mut ffi::ICorJitInfo,
        cls: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CORINFO_METHOD_HANDLE;
}

impl Helpers for GasketEeInfo {
    fn get_helper_ftn(&self, id: CorInfoHelpFunc) -> HelperTarget {
        let mut method: ffi::CORINFO_METHOD_HANDLE = std::ptr::null_mut();
        let entrypoint = zeroed_out(|entrypoint| unsafe {
            rokajit_ee_get_helper_ftn(self.comp_raw(), id.to_raw(), entrypoint, &mut method)
        });
        HelperTarget {
            entrypoint,
            method: MethodHandle::from_raw(method),
        }
    }

    fn get_new_helper(
        &self,
        token: &CORINFO_RESOLVED_TOKEN,
        _caller: MethodHandle,
    ) -> (CorInfoHelpFunc, Option<bool>) {
        // CONTRADICTION (flagged for the orchestrator): the step_02 frozen
        // signature above takes a resolved token + caller, but the real C++
        // `getNewHelper` (corinfo.h:2658) takes a `CORINFO_CLASS_HANDLE`.
        // The wrapper passes the token's `hClass` (what RyuJIT passes for
        // allocation sites) and ignores `caller`. `pHasSideEffects` is a
        // pure out-param, so the wrapper always passes a valid pointer and
        // the answer is always `Some`.
        let mut has_side_effects = false;
        let helper = unsafe {
            rokajit_ee_get_new_helper(self.comp_raw(), token.hClass, &mut has_side_effects)
        };
        (CorInfoHelpFunc::from_raw(helper), Some(has_side_effects))
    }

    fn get_casting_helper(
        &self,
        token: &CORINFO_RESOLVED_TOKEN,
        throwing: bool,
    ) -> CorInfoHelpFunc {
        let token = token as *const CORINFO_RESOLVED_TOKEN as *mut CORINFO_RESOLVED_TOKEN;
        let helper = unsafe { rokajit_ee_get_casting_helper(self.comp_raw(), token, throwing) };
        CorInfoHelpFunc::from_raw(helper)
    }

    fn get_box_helper(&self, cls: ClassHandle) -> CorInfoHelpFunc {
        let helper = unsafe { rokajit_ee_get_box_helper(self.comp_raw(), cls.as_raw()) };
        CorInfoHelpFunc::from_raw(helper)
    }

    fn get_function_entry_point(&self, ftn: MethodHandle) -> CORINFO_CONST_LOOKUP {
        zeroed_out(|result| unsafe {
            rokajit_ee_get_function_entry_point(
                self.comp_raw(),
                ftn.as_raw(),
                result,
                AccessFlags::EMPTY.to_raw(),
            )
        })
    }

    fn get_new_arr_helper(&self, array_cls: ClassHandle) -> CorInfoHelpFunc {
        let helper = unsafe { rokajit_ee_get_new_arr_helper(self.comp_raw(), array_cls.as_raw()) };
        CorInfoHelpFunc::from_raw(helper)
    }

    fn get_shared_cctor_helper(&self, cls: ClassHandle) -> CorInfoHelpFunc {
        let helper = unsafe { rokajit_ee_get_shared_cctor_helper(self.comp_raw(), cls.as_raw()) };
        CorInfoHelpFunc::from_raw(helper)
    }

    fn get_type_for_box(&self, cls: ClassHandle) -> ClassHandle {
        let boxed = unsafe { rokajit_ee_get_type_for_box(self.comp_raw(), cls.as_raw()) };
        ClassHandle::from_raw(boxed).expect("EE contract: getTypeForBox never returns null")
    }

    fn get_un_box_helper(&self, cls: ClassHandle) -> CorInfoHelpFunc {
        let helper = unsafe { rokajit_ee_get_un_box_helper(self.comp_raw(), cls.as_raw()) };
        CorInfoHelpFunc::from_raw(helper)
    }

    fn get_ready_to_run_helper(
        &self,
        token: &CORINFO_RESOLVED_TOKEN,
        id: CorInfoHelpFunc,
        caller: MethodHandle,
    ) -> Option<CORINFO_CONST_LOOKUP> {
        let token = token as *const CORINFO_RESOLVED_TOKEN as *mut CORINFO_RESOLVED_TOKEN;
        let mut found = false;
        let lookup = zeroed_out(|lookup| {
            found = unsafe {
                rokajit_ee_get_ready_to_run_helper(
                    self.comp_raw(),
                    token,
                    id.to_raw(),
                    caller.as_raw(),
                    lookup,
                )
            };
        });
        found.then_some(lookup)
    }

    fn get_ready_to_run_delegate_ctor_helper(
        &self,
        target_method: &CORINFO_RESOLVED_TOKEN,
        target_constraint: ffi::mdToken,
        delegate_type: ClassHandle,
        caller: MethodHandle,
    ) -> CORINFO_LOOKUP {
        let target_method =
            target_method as *const CORINFO_RESOLVED_TOKEN as *mut CORINFO_RESOLVED_TOKEN;
        zeroed_out(|lookup| unsafe {
            rokajit_ee_get_ready_to_run_delegate_ctor_helper(
                self.comp_raw(),
                target_method,
                target_constraint,
                delegate_type.as_raw(),
                caller.as_raw(),
                lookup,
            )
        })
    }

    // `parameter` is an opaque pass-through the EE forwards verbatim to
    // `function` — both are supplied by the caller, and the EE never
    // dereferences it itself, so the wrapper carries no safety burden of
    // its own. The unsafe contract lives at the callback's definition.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn run_with_error_trap(
        &self,
        function: extern "C" fn(*mut c_void),
        parameter: *mut c_void,
    ) -> bool {
        unsafe { rokajit_ee_run_with_error_trap(self.comp_raw(), Some(function), parameter) }
    }

    // Same opaque pass-through guarantee as `run_with_error_trap`.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn run_with_spmi_error_trap(
        &self,
        function: extern "C" fn(*mut c_void),
        parameter: *mut c_void,
    ) -> bool {
        unsafe { rokajit_ee_run_with_spmi_error_trap(self.comp_raw(), Some(function), parameter) }
    }

    fn get_ee_info(&self) -> CORINFO_EE_INFO {
        zeroed_out(|info| unsafe { rokajit_ee_get_ee_info(self.comp_raw(), info) })
    }

    fn get_wasm_well_known_globals(&self) -> CORINFO_WASM_WELLKNOWN_GLOBALS {
        zeroed_out(|globals| unsafe {
            rokajit_ee_get_wasm_well_known_globals(self.comp_raw(), globals)
        })
    }

    fn get_function_fixed_entry_point(
        &self,
        ftn: MethodHandle,
        is_unsafe_function_pointer: bool,
    ) -> CORINFO_CONST_LOOKUP {
        zeroed_out(|result| unsafe {
            rokajit_ee_get_function_fixed_entry_point(
                self.comp_raw(),
                ftn.as_raw(),
                is_unsafe_function_pointer,
                result,
            )
        })
    }

    fn get_address_of_p_invoke_target(&self, method: MethodHandle) -> CORINFO_CONST_LOOKUP {
        zeroed_out(|lookup| unsafe {
            rokajit_ee_get_address_of_p_invoke_target(self.comp_raw(), method.as_raw(), lookup)
        })
    }

    fn get_delegate_ctor(
        &self,
        meth: MethodHandle,
        cls: ClassHandle,
        target_method: MethodHandle,
        ctor_data: &DelegateCtorArgs,
    ) -> Option<MethodHandle> {
        // The C++ parameter is non-const; copy the caller's struct so an EE
        // that writes back into it cannot mutate shared state.
        let mut ctor_data = *ctor_data;
        let ctor = unsafe {
            rokajit_ee_get_delegate_ctor(
                self.comp_raw(),
                meth.as_raw(),
                cls.as_raw(),
                target_method.as_raw(),
                &mut ctor_data,
            )
        };
        MethodHandle::from_raw(ctor)
    }

    fn notify_instruction_set_usage(
        &self,
        instruction_set: InstructionSet,
        support_enabled: bool,
    ) -> bool {
        unsafe {
            rokajit_ee_notify_instruction_set_usage(
                self.comp_raw(),
                instruction_set.to_raw(),
                support_enabled,
            )
        }
    }

    fn get_special_copy_helper(&self, cls: ClassHandle) -> Option<MethodHandle> {
        let helper = unsafe { rokajit_ee_get_special_copy_helper(self.comp_raw(), cls.as_raw()) };
        MethodHandle::from_raw(helper)
    }
}
