use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::TokensAndSignatures;
use crate::enums::{CallInfoFlags, CorInfoType};
use crate::handles::{ArgListHandle, ClassHandle, MethodHandle};

impl TokensAndSignatures for MockEe {
    fn resolve_token(&self, _token: &mut ffi::CORINFO_RESOLVED_TOKEN) {}

    fn get_arg_next(&self, _args: ArgListHandle) -> Option<ArgListHandle> {
        None
    }

    fn get_arg_type(
        &self,
        _sig: &ffi::CORINFO_SIG_INFO,
        _args: ArgListHandle,
    ) -> (CorInfoType, Option<ClassHandle>) {
        (CorInfoType::Int, None)
    }

    fn get_arg_class(
        &self,
        _sig: &ffi::CORINFO_SIG_INFO,
        _args: ArgListHandle,
    ) -> Option<ClassHandle> {
        None
    }

    fn get_call_info(
        &self,
        _token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        _constrained: Option<&ffi::CORINFO_RESOLVED_TOKEN>,
        _caller: MethodHandle,
        _flags: CallInfoFlags,
    ) -> ffi::CORINFO_CALL_INFO {
        unsafe { std::mem::zeroed() }
    }
}
