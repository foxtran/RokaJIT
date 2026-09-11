use rokajit_ffi::{CORINFO_CALL_INFO, CORINFO_RESOLVED_TOKEN, CORINFO_SIG_INFO};

use crate::enums::{CallInfoFlags, CorInfoType};
use crate::handles::{ArgListHandle, ClassHandle, MethodHandle};

/// Token resolution and signature walking (C++ `ICorModuleInfo` +
/// `ICorSigInfo`).
///
/// **This is the trait the IL importer resolves tokens through**: every
/// metadata token in the IL stream goes through
/// [`TokensAndSignatures::resolve_token`] once, and the signature of the
/// resolved method/field is then walked with `get_arg_type`/`get_arg_next`.
pub trait TokensAndSignatures {
    /// C++ `ICorStaticInfo::resolveToken` (corinfo.h:2406). The caller fills
    /// in `tokenScope`/`token`/`tokenType` (the `[In]` fields); the EE fills
    /// the handles and spec signatures. In/out by mutation, exactly like the
    /// C++.
    fn resolve_token(&self, token: &mut CORINFO_RESOLVED_TOKEN);

    /// C++ `ICorSigInfo::getArgNext` (corinfo.h:3093). The end-of-list
    /// nullptr becomes `None`.
    fn get_arg_next(&self, args: ArgListHandle) -> Option<ArgListHandle>;

    /// C++ `ICorSigInfo::getArgType` (corinfo.h:3106). The `vcTypeRet`
    /// out-param (set for value-class args) becomes the second tuple
    /// element. `CorInfoTypeWithMod` modifiers (modreq/modopt) are dropped;
    /// no consumer needs them yet.
    fn get_arg_type(
        &self,
        sig: &CORINFO_SIG_INFO,
        args: ArgListHandle,
    ) -> (CorInfoType, Option<ClassHandle>);

    /// C++ `ICorSigInfo::getArgClass` (corinfo.h:3122). The nullptr failure
    /// sentinel becomes `None`.
    fn get_arg_class(
        &self,
        sig: &CORINFO_SIG_INFO,
        args: ArgListHandle,
    ) -> Option<ClassHandle>;

    /// C++ `ICorDynamicInfo::getCallInfo` (corinfo.h:3415): the EE's verdict
    /// on how to perform a call to a resolved token.
    fn get_call_info(
        &self,
        token: &mut CORINFO_RESOLVED_TOKEN,
        constrained: Option<&CORINFO_RESOLVED_TOKEN>,
        caller: MethodHandle,
        flags: CallInfoFlags,
    ) -> CORINFO_CALL_INFO;
}
