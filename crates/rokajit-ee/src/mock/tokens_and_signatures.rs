use std::ffi::c_void;
use std::ptr::NonNull;

use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::TokensAndSignatures;
use crate::enums::{CallInfoFlags, CorInfoHFAElemType, CorInfoType, InfoAccessType};
use crate::handles::{
    ArgListHandle, ClassHandle, ContextHandle, FieldHandle, MethodHandle, ModuleHandle,
    ObjectHandle, VarArgsHandle,
};

impl TokensAndSignatures for MockEe {
    fn resolve_token(&self, token: &mut ffi::CORINFO_RESOLVED_TOKEN) {
        if let Some(method) = self.methods.get(&token.token) {
            token.hMethod = method.handle.as_raw();
            // One mock class per method: reuse the method handle's address.
            token.hClass = method.handle.as_raw() as ffi::CORINFO_CLASS_HANDLE;
        }
    }

    fn find_sig(
        &self,
        _module: ModuleHandle,
        _sig_tok: u32,
        _context: Option<ContextHandle>,
    ) -> ffi::CORINFO_SIG_INFO {
        unsafe { std::mem::zeroed() }
    }

    fn find_call_site_sig(
        &self,
        _module: ModuleHandle,
        _meth_tok: u32,
        _context: Option<ContextHandle>,
    ) -> ffi::CORINFO_SIG_INFO {
        unsafe { std::mem::zeroed() }
    }

    fn get_token_type_as_handle(
        &self,
        _token: &ffi::CORINFO_RESOLVED_TOKEN,
    ) -> Option<ClassHandle> {
        None
    }

    fn get_string_literal(
        &self,
        _module: ModuleHandle,
        _meta_tok: u32,
        _start_index: i32,
    ) -> Option<String> {
        None
    }

    fn print_object_description(&self, _handle: ObjectHandle) -> String {
        String::new()
    }

    fn get_arg_next(&self, args: ArgListHandle) -> Option<ArgListHandle> {
        let (list, index) = Self::decode_cursor(args);
        let next = index + 1;
        if next < self.arg_lists[list].len() {
            Self::cursor(list, next)
        } else {
            None
        }
    }

    fn get_arg_type(
        &self,
        _sig: &ffi::CORINFO_SIG_INFO,
        args: ArgListHandle,
    ) -> (CorInfoType, Option<ClassHandle>) {
        let (list, index) = Self::decode_cursor(args);
        (self.arg_lists[list][index], None)
    }

    fn get_exact_classes(
        &self,
        _base_type: ClassHandle,
        _max_exact_classes: i32,
    ) -> Option<Vec<ClassHandle>> {
        None
    }

    fn get_arg_class(
        &self,
        _sig: &ffi::CORINFO_SIG_INFO,
        _args: ArgListHandle,
    ) -> Option<ClassHandle> {
        None
    }

    fn get_hfa_type(&self, _class: ClassHandle) -> CorInfoHFAElemType {
        CorInfoHFAElemType::None
    }

    fn embed_module_handle(
        &self,
        handle: ModuleHandle,
    ) -> (Option<ModuleHandle>, Option<NonNull<c_void>>) {
        // Echo the input handle; never indirected.
        (Some(handle), None)
    }

    fn embed_class_handle(
        &self,
        handle: ClassHandle,
    ) -> (Option<ClassHandle>, Option<NonNull<c_void>>) {
        (Some(handle), None)
    }

    fn embed_method_handle(
        &self,
        handle: MethodHandle,
    ) -> (Option<MethodHandle>, Option<NonNull<c_void>>) {
        (Some(handle), None)
    }

    fn embed_field_handle(
        &self,
        handle: FieldHandle,
    ) -> (Option<FieldHandle>, Option<NonNull<c_void>>) {
        (Some(handle), None)
    }

    fn embed_generic_handle(
        &self,
        _token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        _embed_parent: bool,
        _caller: MethodHandle,
    ) -> ffi::CORINFO_GENERICHANDLE_RESULT {
        unsafe { std::mem::zeroed() }
    }

    fn get_location_of_this_type(&self, _context: MethodHandle) -> ffi::CORINFO_LOOKUP_KIND {
        // needsRuntimeLookup = false.
        unsafe { std::mem::zeroed() }
    }

    fn get_cookie_for_interpreter_calli_sig(
        &self,
        _sig: &ffi::CORINFO_SIG_INFO,
    ) -> Option<NonNull<c_void>> {
        None
    }

    fn get_call_info(
        &self,
        token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        _constrained: Option<&ffi::CORINFO_RESOLVED_TOKEN>,
        _caller: MethodHandle,
        _flags: CallInfoFlags,
    ) -> ffi::CORINFO_CALL_INFO {
        let mut info: ffi::CORINFO_CALL_INFO = unsafe { std::mem::zeroed() };
        if let Some(method) = self.methods.get(&token.token) {
            info.hMethod = method.handle.as_raw();
            info.kind = ffi::CORINFO_CALL_KIND_CORINFO_CALL;
            info.sig = self.method_sig_info(method);
        }
        info
    }

    fn get_var_args_handle(
        &self,
        _sig: &ffi::CORINFO_SIG_INFO,
        _meth: MethodHandle,
    ) -> (Option<VarArgsHandle>, Option<NonNull<c_void>>) {
        (None, None)
    }

    fn construct_string_literal(
        &self,
        _module: ModuleHandle,
        _meta_tok: ffi::mdToken,
    ) -> (InfoAccessType, Option<NonNull<c_void>>) {
        (InfoAccessType::Value, None)
    }

    fn empty_string_literal(&self) -> (InfoAccessType, Option<NonNull<c_void>>) {
        (InfoAccessType::Value, None)
    }

    fn convert_pinvoke_calli_to_call(
        &self,
        _token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        _must_convert: bool,
    ) -> bool {
        false
    }
}
