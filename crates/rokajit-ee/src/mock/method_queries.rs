use std::ffi::c_void;
use std::ptr::NonNull;

use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::MethodQueries;
use crate::enums::{CorInfoCallConvExtension, MethodAttribs, MethodRuntimeFlags};
use crate::handles::{ClassHandle, ContextHandle, MethodHandle};

impl MethodQueries for MockEe {
    fn get_method_attribs(&self, _ftn: MethodHandle) -> MethodAttribs {
        self.method_attribs
    }

    fn get_method_sig(
        &self,
        ftn: MethodHandle,
        _member_parent: Option<ClassHandle>,
    ) -> ffi::CORINFO_SIG_INFO {
        match self.methods.values().find(|m| m.handle == ftn) {
            Some(method) => self.method_sig_info(method),
            // Zeroed mirror struct: all handles null, retType = UNDEF.
            None => unsafe { std::mem::zeroed() },
        }
    }

    fn get_method_class(&self, ftn: MethodHandle) -> ClassHandle {
        // Overrides (struct instance methods, step_10.9) come from
        // `method_classes`; otherwise reuse the method handle's address as
        // a stable, non-null stand-in (never a registered value class).
        if let Some(&class) = self.method_classes.get(&(ftn.as_raw() as usize)) {
            return class;
        }
        ClassHandle(ftn.0 as ffi::CORINFO_CLASS_HANDLE)
    }

    fn get_eh_info(&self, _ftn: MethodHandle, index: u32) -> ffi::CORINFO_EH_CLAUSE {
        // Canned clauses by index (10.6); absent index = zeroed clause.
        self.eh_clauses
            .get(index as usize)
            .copied()
            .unwrap_or_else(|| unsafe { std::mem::zeroed() })
    }

    fn get_method_hash(&self, _ftn: MethodHandle) -> u32 {
        0
    }

    fn method_must_be_loaded_before_code_is_run(&self, _ftn: MethodHandle) {}

    fn get_method_name_from_metadata(&self, _ftn: MethodHandle) -> Option<String> {
        self.method_name.clone()
    }

    fn is_intrinsic(&self, _ftn: MethodHandle) -> bool {
        false
    }

    fn can_value_class_instance_pointer_escape(&self, _ftn: MethodHandle) -> bool {
        // Conservative answer.
        true
    }

    fn notify_method_info_usage(&self, _ftn: MethodHandle) -> bool {
        true
    }

    fn set_method_attribs(&self, ftn: MethodHandle, attribs: MethodRuntimeFlags) {
        self.sink_log
            .borrow_mut()
            .push(format!("set_method_attribs({ftn:?}, {attribs:?})"));
    }

    fn get_method_info(
        &self,
        _ftn: MethodHandle,
        _context: Option<ContextHandle>,
    ) -> Option<ffi::CORINFO_METHOD_INFO> {
        None
    }

    fn have_same_method_definition(&self, meth1: MethodHandle, meth2: MethodHandle) -> bool {
        meth1 == meth2
    }

    fn get_type_definition(&self, ty: ClassHandle) -> ClassHandle {
        // The header contract returns the input for an unconstructed generic.
        ty
    }

    fn get_method_vtable_offset(&self, method: MethodHandle) -> (u32, u32, bool) {
        // Canned per-method verdicts (step_10.12); the default is an
        // ordinary mid-table slot with no chunk indirection
        // (`CORINFO_VIRTUALCALL_NO_CHUNK`, corinfo.h:1416).
        self.vtable_offsets
            .get(&(method.as_raw() as usize))
            .copied()
            .unwrap_or((ffi::CORINFO_VIRTUALCALL_NO_CHUNK, 0x28, false))
    }

    fn resolve_virtual_method(&self, _info: &mut ffi::CORINFO_DEVIRTUALIZATION_INFO) -> bool {
        false
    }

    fn get_async_other_variant(&self, _ftn: MethodHandle) -> Option<(MethodHandle, bool)> {
        None
    }

    fn get_default_comparer_class(&self, _elem_type: ClassHandle) -> Option<ClassHandle> {
        None
    }

    fn get_default_equality_comparer_class(&self, _elem_type: ClassHandle) -> Option<ClassHandle> {
        None
    }

    fn get_sz_array_helper_enumerator_class(&self, _elem_type: ClassHandle) -> Option<ClassHandle> {
        None
    }

    fn expand_raw_handle_intrinsic(
        &self,
        _resolved_token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        _caller: MethodHandle,
    ) -> ffi::CORINFO_GENERICHANDLE_RESULT {
        unsafe { std::mem::zeroed() }
    }

    fn is_intrinsic_type(&self, _class_hnd: ClassHandle) -> bool {
        false
    }

    fn get_unmanaged_call_conv(
        &self,
        _method: Option<MethodHandle>,
        _call_site_sig: Option<&ffi::CORINFO_SIG_INFO>,
    ) -> (CorInfoCallConvExtension, bool) {
        (CorInfoCallConvExtension::Managed, false)
    }

    fn p_invoke_marshaling_required(
        &self,
        _method: Option<MethodHandle>,
        _call_site_sig: Option<&ffi::CORINFO_SIG_INFO>,
    ) -> bool {
        false
    }

    fn satisfies_method_constraints(&self, _parent: ClassHandle, _method: MethodHandle) -> bool {
        true
    }

    fn get_gs_cookie(&self) -> (usize, Option<NonNull<usize>>) {
        (0, None)
    }

    fn set_patchpoint_info(&self, patchpoint_info: Option<NonNull<ffi::PatchpointInfo>>) {
        self.sink_log
            .borrow_mut()
            .push(format!("set_patchpoint_info({patchpoint_info:?})"));
    }

    fn get_osr_info(&self) -> Option<(NonNull<ffi::PatchpointInfo>, u32)> {
        None
    }

    fn get_async_info(&self) -> ffi::CORINFO_ASYNC_INFO {
        unsafe { std::mem::zeroed() }
    }

    fn get_await_return_call(
        &self,
        _caller: MethodHandle,
    ) -> Option<(MethodHandle, Option<ContextHandle>, ffi::CORINFO_LOOKUP)> {
        None
    }

    fn get_await_awaiter_in_continuation_call(
        &self,
        _caller: MethodHandle,
        _resolved_token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        _is_unsafe: bool,
    ) -> Option<(MethodHandle, Option<ContextHandle>, ffi::CORINFO_LOOKUP)> {
        None
    }

    fn get_method_def_from_method(&self, _method: MethodHandle) -> Option<u32> {
        // The dynamic-method answer (C++ mdMethodDefNil).
        None
    }

    fn print_method_name(&self, _ftn: MethodHandle) -> String {
        self.method_name
            .clone()
            .unwrap_or_else(|| "mockMethod".into())
    }

    fn get_async_resumption_stub(&self) -> Option<(MethodHandle, NonNull<c_void>)> {
        None
    }

    fn get_continuation_type(&self, _data_size: usize, _obj_refs: &[bool]) -> Option<ClassHandle> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ee_info::MethodQueries;

    fn handle(raw: usize) -> MethodHandle {
        MethodHandle::from_raw(raw as *mut u8 as _).unwrap()
    }

    /// A canned clause comes back by index, verbatim; the union member is
    /// the bindgen struct's (what the trait returns IS the EE's shape).
    #[test]
    fn canned_eh_clauses_are_returned_by_index() {
        let ee = MockEe {
            eh_clauses: vec![
                {
                    let mut c: ffi::CORINFO_EH_CLAUSE = unsafe { std::mem::zeroed() };
                    c.Flags = 0; // typed catch
                    c.TryOffset = 0;
                    c.TryLength = 10;
                    c.HandlerOffset = 12;
                    c.HandlerLength = 6;
                    c.__bindgen_anon_1.ClassToken = 0x0200_0042;
                    c
                },
                {
                    let mut c: ffi::CORINFO_EH_CLAUSE = unsafe { std::mem::zeroed() };
                    c.Flags = ffi::CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_FINALLY;
                    c.TryOffset = 20;
                    c.TryLength = 8;
                    c.HandlerOffset = 30;
                    c.HandlerLength = 4;
                    c
                },
            ],
            ..MockEe::default()
        };
        let first = ee.get_eh_info(handle(1), 0);
        assert_eq!(first.Flags, 0);
        assert_eq!((first.TryOffset, first.TryLength), (0, 10));
        // SAFETY: ClassToken was the union member written above.
        assert_eq!(unsafe { first.__bindgen_anon_1.ClassToken }, 0x0200_0042);
        let second = ee.get_eh_info(handle(1), 1);
        assert_eq!(
            second.Flags,
            ffi::CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_FINALLY
        );
        assert_eq!((second.HandlerOffset, second.HandlerLength), (30, 4));
    }

    /// No canned clauses: the old behavior, a zeroed clause.
    #[test]
    fn eh_clauses_default_to_zeroed() {
        let ee = MockEe::default();
        let clause = ee.get_eh_info(handle(1), 0);
        assert_eq!(clause.Flags, 0);
        assert_eq!(clause.TryOffset, 0);
        // SAFETY: a zeroed union reads as zero through either member.
        assert_eq!(unsafe { clause.__bindgen_anon_1.ClassToken }, 0);
    }
}
