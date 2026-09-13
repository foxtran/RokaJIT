use std::ffi::c_void;

use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::{HelperTarget, Helpers};
use crate::enums::{CorInfoHelpFunc, InstructionSet};
use crate::handles::{ClassHandle, MethodHandle};

impl Helpers for MockEe {
    fn get_helper_ftn(&self, _id: CorInfoHelpFunc) -> HelperTarget {
        HelperTarget {
            entrypoint: unsafe { std::mem::zeroed() },
            method: None,
        }
    }

    fn get_new_helper(
        &self,
        _token: &ffi::CORINFO_RESOLVED_TOKEN,
        _caller: MethodHandle,
    ) -> (CorInfoHelpFunc, Option<bool>) {
        (
            self.new_helper.unwrap_or(CorInfoHelpFunc::NEWFAST),
            Some(true),
        )
    }

    fn get_casting_helper(
        &self,
        _token: &ffi::CORINFO_RESOLVED_TOKEN,
        throwing: bool,
    ) -> CorInfoHelpFunc {
        if throwing {
            CorInfoHelpFunc::CHKCASTANY
        } else {
            CorInfoHelpFunc::ISINSTANCEOFANY
        }
    }

    fn get_box_helper(&self, _cls: ClassHandle) -> CorInfoHelpFunc {
        CorInfoHelpFunc::BOX
    }

    fn get_function_entry_point(&self, ftn: MethodHandle) -> ffi::CORINFO_CONST_LOOKUP {
        let mut lookup: ffi::CORINFO_CONST_LOOKUP = crate::ee_info::wrap::zeroed_out(|_| ());
        if let Some(&addr) = self.entry_points.get(&(ftn.as_raw() as usize)) {
            lookup.__bindgen_anon_1.addr = addr as *mut c_void;
        } else if let Some(&slot) = self.entry_point_slots.get(&(ftn.as_raw() as usize)) {
            // Not-yet-compiled callee: the entry-point slot (precode target
            // slot), IAT_PVALUE (jitinterface.cpp getFunctionEntryPoint).
            lookup.accessType = ffi::InfoAccessType_IAT_PVALUE;
            lookup.__bindgen_anon_1.addr = slot as *mut c_void;
        }
        lookup
    }

    fn get_new_arr_helper(&self, _array_cls: ClassHandle) -> CorInfoHelpFunc {
        CorInfoHelpFunc::NEWARR_1_PTR
    }

    fn get_shared_cctor_helper(&self, _cls: ClassHandle) -> CorInfoHelpFunc {
        CorInfoHelpFunc::INITCLASS
    }

    fn get_type_for_box(&self, cls: ClassHandle) -> ClassHandle {
        // No Nullable<T> in the canned world: boxing is the identity.
        cls
    }

    fn get_un_box_helper(&self, _cls: ClassHandle) -> CorInfoHelpFunc {
        CorInfoHelpFunc::UNBOX
    }

    fn get_ready_to_run_helper(
        &self,
        _token: &ffi::CORINFO_RESOLVED_TOKEN,
        _id: CorInfoHelpFunc,
        _caller: MethodHandle,
    ) -> Option<ffi::CORINFO_CONST_LOOKUP> {
        // Canned EE: no ReadyToRun helpers.
        None
    }

    fn get_ready_to_run_delegate_ctor_helper(
        &self,
        _target_method: &ffi::CORINFO_RESOLVED_TOKEN,
        _target_constraint: ffi::mdToken,
        _delegate_type: ClassHandle,
        _caller: MethodHandle,
    ) -> ffi::CORINFO_LOOKUP {
        unsafe { std::mem::zeroed() }
    }

    fn run_with_error_trap(
        &self,
        function: extern "C" fn(*mut c_void),
        parameter: *mut c_void,
    ) -> bool {
        // The canned trap is no trap at all: run the callback directly and
        // report success. A panicking callback is a caller bug (the real EE
        // contract forbids unwinding through it too).
        function(parameter);
        true
    }

    fn run_with_spmi_error_trap(
        &self,
        function: extern "C" fn(*mut c_void),
        parameter: *mut c_void,
    ) -> bool {
        function(parameter);
        true
    }

    fn get_ee_info(&self) -> ffi::CORINFO_EE_INFO {
        unsafe { std::mem::zeroed() }
    }

    fn get_wasm_well_known_globals(&self) -> ffi::CORINFO_WASM_WELLKNOWN_GLOBALS {
        unsafe { std::mem::zeroed() }
    }

    fn get_function_fixed_entry_point(
        &self,
        _ftn: MethodHandle,
        _is_unsafe_function_pointer: bool,
    ) -> ffi::CORINFO_CONST_LOOKUP {
        unsafe { std::mem::zeroed() }
    }

    fn get_address_of_p_invoke_target(&self, _method: MethodHandle) -> ffi::CORINFO_CONST_LOOKUP {
        unsafe { std::mem::zeroed() }
    }

    fn get_delegate_ctor(
        &self,
        _meth: MethodHandle,
        _cls: ClassHandle,
        _target_method: MethodHandle,
        _ctor_data: &ffi::DelegateCtorArgs,
    ) -> Option<MethodHandle> {
        // Canned EE: no usable delegate ctor.
        None
    }

    fn notify_instruction_set_usage(
        &self,
        _instruction_set: InstructionSet,
        _support_enabled: bool,
    ) -> bool {
        // Conservative canned answer: nothing is supported unconditionally.
        false
    }

    fn get_special_copy_helper(&self, _cls: ClassHandle) -> Option<MethodHandle> {
        // Canned EE: no special copy helpers (the common real answer).
        None
    }
}
