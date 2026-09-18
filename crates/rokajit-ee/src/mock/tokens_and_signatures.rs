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
        if let Some(field) = self.fields.get(&token.token) {
            token.hField = field.handle.as_raw();
            // The declaring class: the field's value class when it is a
            // struct field, else the field handle's address (step_10.4's
            // one-mock-class-per-field stand-in).
            token.hClass = match field.value_class {
                Some(c) => c.as_raw(),
                None => field.handle.as_raw() as ffi::CORINFO_CLASS_HANDLE,
            };
        }
        if let Some(&class) = self.class_tokens.get(&token.token) {
            token.hClass = class.as_raw();
        }
    }

    fn find_sig(
        &self,
        _module: ModuleHandle,
        sig_tok: u32,
        _context: Option<ContextHandle>,
    ) -> ffi::CORINFO_SIG_INFO {
        // A registered calli callsite signature (step_10.12); anything
        // else stays zeroed.
        if let Some((sig, arg_list)) = self.calli_sigs.get(&sig_tok) {
            let call_conv = if sig.has_this {
                ffi::CorInfoCallConv_CORINFO_CALLCONV_HASTHIS
            } else {
                ffi::CorInfoCallConv_CORINFO_CALLCONV_DEFAULT
            };
            let call_conv = self
                .calli_sig_convs
                .get(&sig_tok)
                .map_or(call_conv, |&conv| call_conv | conv);
            return self.build_sig_info(call_conv, sig.ret, sig.ret_class, *arg_list);
        }
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
        self.token_type_class
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
    ) -> (CorInfoType, Option<ClassHandle>, bool) {
        let (list, index) = Self::decode_cursor(args);
        let arg = self.arg_lists[list][index];
        (arg.ty, arg.class, arg.pinned)
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
        token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        embed_parent: bool,
        _caller: MethodHandle,
    ) -> ffi::CORINFO_GENERICHANDLE_RESULT {
        // A canned direct embedding (step_10.10): token-deterministic
        // handle constant, no runtime lookup, handle kind from the
        // resolved token's handles — or CLASS when the caller asked for
        // the parent (embed_parent: the resolved method/field's owning
        // class, corinfo.h's embedGenericHandle contract). The `embed_*`
        // flags switch to the rejection forms (runtime lookup /
        // indirection cell).
        let mut result: ffi::CORINFO_GENERICHANDLE_RESULT = unsafe { std::mem::zeroed() };
        result.handleType = if embed_parent {
            ffi::CorInfoGenericHandleType_CORINFO_HANDLETYPE_CLASS
        } else if !token.hMethod.is_null() {
            ffi::CorInfoGenericHandleType_CORINFO_HANDLETYPE_METHOD
        } else if !token.hField.is_null() {
            ffi::CorInfoGenericHandleType_CORINFO_HANDLETYPE_FIELD
        } else {
            ffi::CorInfoGenericHandleType_CORINFO_HANDLETYPE_CLASS
        };
        // A canned full lookup answer (step_11.3B's runtime-lookup
        // fixtures) wins over the flag-driven rejection forms.
        if let Some(lookup) = &self.embed_lookup {
            result.lookup = *lookup;
            return result;
        }
        let handle = (0x7A7A_0000usize + token.token as usize) as ffi::CORINFO_GENERIC_HANDLE;
        result.compileTimeHandle = handle;
        result.lookup.lookupKind.needsRuntimeLookup = self.embed_runtime_lookup;
        let mut const_lookup: ffi::CORINFO_CONST_LOOKUP = unsafe { std::mem::zeroed() };
        if self.embed_indirection {
            const_lookup.accessType = ffi::InfoAccessType_IAT_PVALUE;
            const_lookup.__bindgen_anon_1.addr = handle as *mut c_void;
        } else {
            const_lookup.accessType = ffi::InfoAccessType_IAT_VALUE;
            const_lookup.__bindgen_anon_1.handle = handle;
        }
        result.lookup.__bindgen_anon_1.constLookup = const_lookup;
        result
    }

    fn get_location_of_this_type(&self, _context: MethodHandle) -> ffi::CORINFO_LOOKUP_KIND {
        // The default cans needsRuntimeLookup = false.
        self.this_type_lookup
            .unwrap_or_else(|| unsafe { std::mem::zeroed() })
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
        constrained: Option<&ffi::CORINFO_RESOLVED_TOKEN>,
        _caller: MethodHandle,
        flags: CallInfoFlags,
    ) -> ffi::CORINFO_CALL_INFO {
        self.call_info_flags.borrow_mut().push(flags);
        self.constrained_seen
            .borrow_mut()
            .push(constrained.map_or(0, |t| t.token));
        let mut info: ffi::CORINFO_CALL_INFO = unsafe { std::mem::zeroed() };
        if let Some(method) = self.methods.get(&token.token) {
            info.hMethod = method.handle.as_raw();
            // Per-token kind overrides first (step_10.12's stub/ldvirtftn
            // fallback paths), then the step_10.4 set (a vtable dispatch).
            info.kind = if let Some(kind) = self.call_kinds.get(&token.token) {
                *kind
            } else if self.non_direct_calls.contains(&token.token) {
                ffi::CORINFO_CALL_KIND_CORINFO_VIRTUALCALL_VTABLE
            } else {
                ffi::CORINFO_CALL_KIND_CORINFO_CALL
            };
            info.sig = self.method_sig_info(method);
            // corinfo.h:1343: a CODE_POINTER verdict invalidates hMethod
            // (the importer's intrinsic name-match then falls back to the
            // resolved token's method).
            if info.kind == ffi::CORINFO_CALL_KIND_CORINFO_CALL_CODE_POINTER {
                info.hMethod = std::ptr::null_mut();
            }
        }
        // A canned one-class method instantiation on the call sig (the
        // GetArrayDataReference<T> fixture).
        if let Some(cell) = self.call_meth_inst.get(&token.token) {
            info.sig.sigInst.methInstCount = 1;
            info.sig.sigInst.methInst =
                std::ptr::from_ref::<ffi::CORINFO_CLASS_HANDLE>(cell) as *mut _;
        }
        // Canned constrained-call this transforms (step_11.3C).
        if let Some(&transform) = self.this_transforms.get(&token.token) {
            info.thisTransform = transform;
        }
        // Canned generics-context answers (step_11.3B).
        if let Some(&(context, needs_lookup)) = self.call_contexts.get(&token.token) {
            info.contextHandle = context as ffi::CORINFO_CONTEXT_HANDLE;
            info.exactContextNeedsRuntimeLookup = needs_lookup;
        }
        if let Some(lookup) = self.call_code_pointer_lookups.get(&token.token) {
            info.__bindgen_anon_1.codePointerLookup = *lookup;
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
        meta_tok: ffi::mdToken,
    ) -> (InfoAccessType, Option<NonNull<c_void>>) {
        // A canned frozen object: deterministic per token (identical
        // literals get identical references, like the real EE's interning),
        // never dereferenced by tests.
        if self.string_literal_cell {
            let cell = (0xCE11_5A00usize + meta_tok as usize) as *mut c_void;
            return (InfoAccessType::PValue, NonNull::new(cell));
        }
        let ptr = (0x5AFE_0000usize + meta_tok as usize) as *mut c_void;
        (InfoAccessType::Value, NonNull::new(ptr))
    }

    fn empty_string_literal(&self) -> (InfoAccessType, Option<NonNull<c_void>>) {
        if self.string_literal_cell {
            return (
                InfoAccessType::PValue,
                NonNull::new(0xCE11_E5A0 as *mut c_void),
            );
        }
        (
            InfoAccessType::Value,
            NonNull::new(0x5AFE_E5A0 as *mut c_void),
        )
    }

    fn convert_pinvoke_calli_to_call(
        &self,
        _token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        _must_convert: bool,
    ) -> bool {
        false
    }
}
