use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::MethodQueries;
use crate::enums::MethodAttribs;
use crate::handles::{ClassHandle, MethodHandle};

impl MethodQueries for MockEe {
    fn get_method_attribs(&self, _ftn: MethodHandle) -> MethodAttribs {
        self.method_attribs
    }

    fn get_method_sig(
        &self,
        _ftn: MethodHandle,
        _member_parent: Option<ClassHandle>,
    ) -> ffi::CORINFO_SIG_INFO {
        // Zeroed mirror struct: all handles null, retType = UNDEF.
        unsafe { std::mem::zeroed() }
    }

    fn get_method_class(&self, ftn: MethodHandle) -> ClassHandle {
        // The mock has one class; reuse the method handle's address as a
        // stable, non-null stand-in.
        ClassHandle(ftn.0 as ffi::CORINFO_CLASS_HANDLE)
    }

    fn get_eh_info(&self, _ftn: MethodHandle, _index: u32) -> ffi::CORINFO_EH_CLAUSE {
        unsafe { std::mem::zeroed() }
    }

    fn get_method_hash(&self, _ftn: MethodHandle) -> u32 {
        0
    }

    fn method_must_be_loaded_before_code_is_run(&self, _ftn: MethodHandle) {}

    fn get_method_name_from_metadata(&self, _ftn: MethodHandle) -> Option<String> {
        self.method_name.clone()
    }
}
