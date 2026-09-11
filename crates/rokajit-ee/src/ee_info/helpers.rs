use rokajit_ffi::{CORINFO_CONST_LOOKUP, CORINFO_RESOLVED_TOKEN};

use crate::enums::CorInfoHelpFunc;
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
    fn get_casting_helper(
        &self,
        token: &CORINFO_RESOLVED_TOKEN,
        throwing: bool,
    ) -> CorInfoHelpFunc;

    /// C++ `ICorClassInfo::getBoxHelper` (corinfo.h:2687).
    fn get_box_helper(&self, cls: ClassHandle) -> CorInfoHelpFunc;

    /// C++ `ICorDynamicInfo::getFunctionEntryPoint` (corinfo.h:3326).
    fn get_function_entry_point(&self, ftn: MethodHandle) -> CORINFO_CONST_LOOKUP;
}
