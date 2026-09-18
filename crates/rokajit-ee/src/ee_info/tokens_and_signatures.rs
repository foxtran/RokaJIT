use std::ffi::{c_int, c_void};
use std::ptr::NonNull;

use rokajit_ffi as ffi;
use rokajit_ffi::{CORINFO_CALL_INFO, CORINFO_RESOLVED_TOKEN, CORINFO_SIG_INFO};

use super::wrap::{print_object_string, zeroed_out};
use super::GasketEeInfo;
use crate::enums::{CallInfoFlags, CorInfoHFAElemType, CorInfoType, InfoAccessType};
use crate::handles::{
    ArgListHandle, ClassHandle, ContextHandle, FieldHandle, MethodHandle, ModuleHandle,
    ObjectHandle, VarArgsHandle,
};

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
    /// C++. **`tokenType` is a mandatory IN hint** (RyuJIT always sets it,
    /// importer.cpp:70): a zero tokenType hard-faults the EE instead of
    /// producing a catchable error (verified live in step_05).
    fn resolve_token(&self, token: &mut CORINFO_RESOLVED_TOKEN);

    /// C++ `ICorStaticInfo::findSig` (corinfo.h:2409): the signature stored
    /// under a `StandaloneSig` token in `module`, instantiated for `context`
    /// (`None` = null context, for signatures with no generic variables).
    fn find_sig(
        &self,
        module: ModuleHandle,
        sig_tok: u32,
        context: Option<ContextHandle>,
    ) -> CORINFO_SIG_INFO;

    /// C++ `ICorStaticInfo::findCallSiteSig` (corinfo.h:2419). For varargs
    /// the call-site signature can differ from the definition signature; this
    /// fetches the call-site one for the method token `meth_tok`.
    fn find_call_site_sig(
        &self,
        module: ModuleHandle,
        meth_tok: u32,
        context: Option<ContextHandle>,
    ) -> CORINFO_SIG_INFO;

    /// C++ `ICorStaticInfo::getTokenTypeAsHandle` (corinfo.h:2426): the class
    /// of the resolved token (owning type for field/method tokens, the type
    /// itself for type tokens). The C++ nullptr becomes `None`.
    fn get_token_type_as_handle(&self, token: &CORINFO_RESOLVED_TOKEN) -> Option<ClassHandle>;

    /// C++ `ICorStaticInfo::getStringLiteral` (corinfo.h:2434). Returns the
    /// (sub)string starting at `start_index` for the `ldstr` token
    /// `meta_tok` in `module`; the C++ `-1` ("input is incorrect") becomes
    /// `None`. The wrapper copies the UTF-16 buffer into an owned `String`
    /// (invalid UTF-16 degrades to U+FFFD); the caller-provided C++ buffer
    /// leaves no EE-side obligation.
    fn get_string_literal(
        &self,
        module: ModuleHandle,
        meta_tok: u32,
        start_index: i32,
    ) -> Option<String>;

    /// C++ `ICorStaticInfo::printObjectDescription` (corinfo.h:2468): a
    /// textual UTF-8 rendering of a frozen object, for diagnostics only
    /// (JitDisasm). The wrapper does the size-query-then-fetch dance and
    /// copies the bytes into an owned `String` (lossy on invalid UTF-8).
    fn print_object_description(&self, handle: ObjectHandle) -> String;

    /// C++ `ICorSigInfo::getArgNext` (corinfo.h:3093). Advances one
    /// signature element. **The walk must be bounded by
    /// `CORINFO_SIG_INFO::numArgs`**: the CoreCLR EE never returns nullptr
    /// here (CEEInfo::getArgNext, jitinterface.cpp:9724), so `None` is not
    /// an end-of-list signal on the real EE — stepping past `numArgs`
    /// walks off the signature blob (verified live in step_05).
    fn get_arg_next(&self, args: ArgListHandle) -> Option<ArgListHandle>;

    /// C++ `ICorSigInfo::getArgType` (corinfo.h:3106). The `vcTypeRet`
    /// out-param (set for value-class args) becomes the second tuple
    /// element. `CorInfoTypeWithMod` modifiers (modreq/modopt) are dropped
    /// except `CORINFO_TYPE_MOD_PINNED` (corinfo.h:634 — a `fixed` local's
    /// marker, RyuJIT's `lvPinned` read at lclvars.cpp:240), reported as
    /// the third tuple element: the GC-info pinned slot flag depends on it
    /// (an unpinned `fixed` buffer moves under a native write —
    /// thread-race.cs's residual crash).
    fn get_arg_type(
        &self,
        sig: &CORINFO_SIG_INFO,
        args: ArgListHandle,
    ) -> (CorInfoType, Option<ClassHandle>, bool);

    /// C++ `ICorArgInfo::getExactClasses` (corinfo.h:3115): the exact loaded
    /// classes deriving from `base_type`, up to `max_exact_classes`. The C++
    /// `-1` ("more than max, or more may load later") becomes `None`.
    fn get_exact_classes(
        &self,
        base_type: ClassHandle,
        max_exact_classes: i32,
    ) -> Option<Vec<ClassHandle>>;

    /// C++ `ICorSigInfo::getArgClass` (corinfo.h:3122). The nullptr failure
    /// sentinel becomes `None`.
    fn get_arg_class(&self, sig: &CORINFO_SIG_INFO, args: ArgListHandle) -> Option<ClassHandle>;

    /// C++ `ICorArgInfo::getHFAType` (corinfo.h:3128): the HFA element kind
    /// of the valuetype `class_`, or [`CorInfoHFAElemType::None`] if it is
    /// not an HFA.
    fn get_hfa_type(&self, class_: ClassHandle) -> CorInfoHFAElemType;

    /// C++ `ICorDynamicInfo::embedModuleHandle` (corinfo.h:3341). Returns
    /// the embeddable handle plus the `*ppIndirection` value (`Some` when the
    /// handle must be reached through that indirection cell at runtime).
    fn embed_module_handle(
        &self,
        handle: ModuleHandle,
    ) -> (Option<ModuleHandle>, Option<NonNull<c_void>>);

    /// C++ `ICorDynamicInfo::embedClassHandle` (corinfo.h:3346). Same
    /// indirection contract as [`TokensAndSignatures::embed_module_handle`].
    fn embed_class_handle(
        &self,
        handle: ClassHandle,
    ) -> (Option<ClassHandle>, Option<NonNull<c_void>>);

    /// C++ `ICorDynamicInfo::embedMethodHandle` (corinfo.h:3351). Same
    /// indirection contract as [`TokensAndSignatures::embed_module_handle`].
    fn embed_method_handle(
        &self,
        handle: MethodHandle,
    ) -> (Option<MethodHandle>, Option<NonNull<c_void>>);

    /// C++ `ICorDynamicInfo::embedFieldHandle` (corinfo.h:3356). Same
    /// indirection contract as [`TokensAndSignatures::embed_module_handle`].
    fn embed_field_handle(
        &self,
        handle: FieldHandle,
    ) -> (Option<FieldHandle>, Option<NonNull<c_void>>);

    /// C++ `ICorDynamicInfo::embedGenericHandle` (corinfo.h:3368): embed the
    /// handle for a resolved token, describing any runtime lookup in the
    /// returned mirror struct. `embed_parent` = the C++ `fEmbedParent`
    /// (embed the parent type handle of a field/method handle instead).
    fn embed_generic_handle(
        &self,
        token: &mut CORINFO_RESOLVED_TOKEN,
        embed_parent: bool,
        caller: MethodHandle,
    ) -> ffi::CORINFO_GENERICHANDLE_RESULT;

    /// C++ `ICorDynamicInfo::getLocationOfThisType` (corinfo.h:3382): how
    /// code shared across generic instantiations locates the exact enclosing
    /// type (for running the `.cctor`).
    fn get_location_of_this_type(&self, context: MethodHandle) -> ffi::CORINFO_LOOKUP_KIND;

    /// C++ `ICorDynamicInfo::GetCookieForInterpreterCalliSig`
    /// (corinfo.h:3395): the cookie to pass to `INTOP_CALLI` in the
    /// interpreter. The C++ nullptr becomes `None`. The cookie is an
    /// opaque EE-owned pointer with no handle newtype, so it crosses as
    /// `NonNull<c_void>`.
    fn get_cookie_for_interpreter_calli_sig(
        &self,
        sig: &CORINFO_SIG_INFO,
    ) -> Option<NonNull<c_void>>;

    /// C++ `ICorDynamicInfo::getCallInfo` (corinfo.h:3415): the EE's verdict
    /// on how to perform a call to a resolved token.
    fn get_call_info(
        &self,
        token: &mut CORINFO_RESOLVED_TOKEN,
        constrained: Option<&CORINFO_RESOLVED_TOKEN>,
        caller: MethodHandle,
        flags: CallInfoFlags,
    ) -> CORINFO_CALL_INFO;

    /// C++ `ICorDynamicInfo::getVarArgsHandle` (corinfo.h:3477): register a
    /// vararg signature and get its VM cookie, plus the `*ppIndirection`
    /// value.
    fn get_var_args_handle(
        &self,
        sig: &CORINFO_SIG_INFO,
        meth: MethodHandle,
    ) -> (Option<VarArgsHandle>, Option<NonNull<c_void>>);

    /// C++ `ICorDynamicInfo::constructStringLiteral` (corinfo.h:3484):
    /// allocate the string object for an `ldstr` token and describe how to
    /// reach it (`IAT_VALUE` = `ppValue` holds the object reference
    /// directly). The second tuple element is the `*ppValue` out-param.
    fn construct_string_literal(
        &self,
        module: ModuleHandle,
        meta_tok: ffi::mdToken,
    ) -> (InfoAccessType, Option<NonNull<c_void>>);

    /// C++ `ICorDynamicInfo::emptyStringLiteral` (corinfo.h:3490): the same
    /// for the empty string.
    fn empty_string_literal(&self) -> (InfoAccessType, Option<NonNull<c_void>>);

    /// C++ `ICorDynamicInfo::convertPInvokeCalliToCall` (corinfo.h:3539):
    /// optionally rewrite a P/Invoke `calli` into a regular method call for
    /// argument marshalling. On `true`, `token.hMethod`/`token.hClass` name
    /// the replacement. `must_convert` = the C++ `fMustConvert`.
    fn convert_pinvoke_calli_to_call(
        &self,
        token: &mut CORINFO_RESOLVED_TOKEN,
        must_convert: bool,
    ) -> bool;
}

extern "C" {
    fn rokajit_ee_resolve_token(
        info: *mut ffi::ICorJitInfo,
        resolved_token: *mut CORINFO_RESOLVED_TOKEN,
    );
    fn rokajit_ee_find_sig(
        info: *mut ffi::ICorJitInfo,
        module: ffi::CORINFO_MODULE_HANDLE,
        sig_tok: u32,
        context: ffi::CORINFO_CONTEXT_HANDLE,
        sig: *mut CORINFO_SIG_INFO,
    );
    fn rokajit_ee_find_call_site_sig(
        info: *mut ffi::ICorJitInfo,
        module: ffi::CORINFO_MODULE_HANDLE,
        meth_tok: u32,
        context: ffi::CORINFO_CONTEXT_HANDLE,
        sig: *mut CORINFO_SIG_INFO,
    );
    fn rokajit_ee_get_token_type_as_handle(
        info: *mut ffi::ICorJitInfo,
        resolved_token: *mut CORINFO_RESOLVED_TOKEN,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_string_literal(
        info: *mut ffi::ICorJitInfo,
        module: ffi::CORINFO_MODULE_HANDLE,
        meta_tok: u32,
        buffer: *mut u16,
        buffer_size: c_int,
        start_index: c_int,
    ) -> c_int;
    fn rokajit_ee_print_object_description(
        info: *mut ffi::ICorJitInfo,
        handle: ffi::CORINFO_OBJECT_HANDLE,
        buffer: *mut std::ffi::c_char,
        buffer_size: usize,
        required_buffer_size: *mut usize,
    ) -> usize;
    fn rokajit_ee_get_arg_next(
        info: *mut ffi::ICorJitInfo,
        args: ffi::CORINFO_ARG_LIST_HANDLE,
    ) -> ffi::CORINFO_ARG_LIST_HANDLE;
    fn rokajit_ee_get_arg_type(
        info: *mut ffi::ICorJitInfo,
        sig: *mut CORINFO_SIG_INFO,
        args: ffi::CORINFO_ARG_LIST_HANDLE,
        vc_type_ret: *mut ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoTypeWithMod;
    fn rokajit_ee_get_exact_classes(
        info: *mut ffi::ICorJitInfo,
        base_type: ffi::CORINFO_CLASS_HANDLE,
        max_exact_classes: c_int,
        exact_cls_ret: *mut ffi::CORINFO_CLASS_HANDLE,
    ) -> c_int;
    fn rokajit_ee_get_arg_class(
        info: *mut ffi::ICorJitInfo,
        sig: *mut CORINFO_SIG_INFO,
        args: ffi::CORINFO_ARG_LIST_HANDLE,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_hfa_type(
        info: *mut ffi::ICorJitInfo,
        class: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CorInfoHFAElemType;
    fn rokajit_ee_embed_module_handle(
        info: *mut ffi::ICorJitInfo,
        handle: ffi::CORINFO_MODULE_HANDLE,
        indirection: *mut *mut c_void,
    ) -> ffi::CORINFO_MODULE_HANDLE;
    fn rokajit_ee_embed_class_handle(
        info: *mut ffi::ICorJitInfo,
        handle: ffi::CORINFO_CLASS_HANDLE,
        indirection: *mut *mut c_void,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_embed_method_handle(
        info: *mut ffi::ICorJitInfo,
        handle: ffi::CORINFO_METHOD_HANDLE,
        indirection: *mut *mut c_void,
    ) -> ffi::CORINFO_METHOD_HANDLE;
    fn rokajit_ee_embed_field_handle(
        info: *mut ffi::ICorJitInfo,
        handle: ffi::CORINFO_FIELD_HANDLE,
        indirection: *mut *mut c_void,
    ) -> ffi::CORINFO_FIELD_HANDLE;
    fn rokajit_ee_embed_generic_handle(
        info: *mut ffi::ICorJitInfo,
        resolved_token: *mut CORINFO_RESOLVED_TOKEN,
        embed_parent: bool,
        caller_handle: ffi::CORINFO_METHOD_HANDLE,
        result: *mut ffi::CORINFO_GENERICHANDLE_RESULT,
    );
    fn rokajit_ee_get_location_of_this_type(
        info: *mut ffi::ICorJitInfo,
        context: ffi::CORINFO_METHOD_HANDLE,
        lookup_kind: *mut ffi::CORINFO_LOOKUP_KIND,
    );
    fn rokajit_ee_get_cookie_for_interpreter_calli_sig(
        info: *mut ffi::ICorJitInfo,
        meta_sig: *mut CORINFO_SIG_INFO,
    ) -> *mut c_void;
    fn rokajit_ee_get_call_info(
        info: *mut ffi::ICorJitInfo,
        resolved_token: *mut CORINFO_RESOLVED_TOKEN,
        constrained_resolved_token: *mut CORINFO_RESOLVED_TOKEN,
        caller_handle: ffi::CORINFO_METHOD_HANDLE,
        flags: ffi::CORINFO_CALLINFO_FLAGS,
        result: *mut CORINFO_CALL_INFO,
    );
    fn rokajit_ee_get_var_args_handle(
        info: *mut ffi::ICorJitInfo,
        sig: *mut CORINFO_SIG_INFO,
        meth_hnd: ffi::CORINFO_METHOD_HANDLE,
        indirection: *mut *mut c_void,
    ) -> ffi::CORINFO_VARARGS_HANDLE;
    fn rokajit_ee_construct_string_literal(
        info: *mut ffi::ICorJitInfo,
        module: ffi::CORINFO_MODULE_HANDLE,
        meta_tok: ffi::mdToken,
        value: *mut *mut c_void,
    ) -> ffi::InfoAccessType;
    fn rokajit_ee_empty_string_literal(
        info: *mut ffi::ICorJitInfo,
        value: *mut *mut c_void,
    ) -> ffi::InfoAccessType;
    fn rokajit_ee_convert_pinvoke_calli_to_call(
        info: *mut ffi::ICorJitInfo,
        resolved_token: *mut CORINFO_RESOLVED_TOKEN,
        must_convert: bool,
    ) -> bool;
}

impl TokensAndSignatures for GasketEeInfo {
    fn resolve_token(&self, token: &mut CORINFO_RESOLVED_TOKEN) {
        unsafe { rokajit_ee_resolve_token(self.comp_raw(), token) };
    }

    fn find_sig(
        &self,
        module: ModuleHandle,
        sig_tok: u32,
        context: Option<ContextHandle>,
    ) -> CORINFO_SIG_INFO {
        let context = context.map_or(std::ptr::null_mut(), |c| c.as_raw());
        zeroed_out(|sig| unsafe {
            rokajit_ee_find_sig(self.comp_raw(), module.as_raw(), sig_tok, context, sig)
        })
    }

    fn find_call_site_sig(
        &self,
        module: ModuleHandle,
        meth_tok: u32,
        context: Option<ContextHandle>,
    ) -> CORINFO_SIG_INFO {
        let context = context.map_or(std::ptr::null_mut(), |c| c.as_raw());
        zeroed_out(|sig| unsafe {
            rokajit_ee_find_call_site_sig(self.comp_raw(), module.as_raw(), meth_tok, context, sig)
        })
    }

    fn get_token_type_as_handle(&self, token: &CORINFO_RESOLVED_TOKEN) -> Option<ClassHandle> {
        // C++ takes a mutable pointer for an `[IN]` parameter; the EE does
        // not write through it.
        let token = token as *const CORINFO_RESOLVED_TOKEN as *mut CORINFO_RESOLVED_TOKEN;
        let raw = unsafe { rokajit_ee_get_token_type_as_handle(self.comp_raw(), token) };
        ClassHandle::from_raw(raw)
    }

    fn get_string_literal(
        &self,
        module: ModuleHandle,
        meta_tok: u32,
        start_index: i32,
    ) -> Option<String> {
        // The header does not allow a null buffer for the size query (unlike
        // printObjectDescription), so probe with a stack buffer and only
        // re-issue with an exact-size buffer when the literal does not fit.
        const STACK_CAP: usize = 256;
        let mut stack_buf = [0u16; STACK_CAP];
        let len = unsafe {
            rokajit_ee_get_string_literal(
                self.comp_raw(),
                module.as_raw(),
                meta_tok,
                stack_buf.as_mut_ptr(),
                STACK_CAP as c_int,
                start_index,
            )
        };
        if len < 0 {
            return None;
        }
        let len = len as usize;
        if len <= STACK_CAP {
            return Some(String::from_utf16_lossy(&stack_buf[..len]));
        }
        let mut buf = vec![0u16; len];
        let len = unsafe {
            rokajit_ee_get_string_literal(
                self.comp_raw(),
                module.as_raw(),
                meta_tok,
                buf.as_mut_ptr(),
                buf.len() as c_int,
                start_index,
            )
        };
        if len < 0 {
            return None;
        }
        let len = (len as usize).min(buf.len());
        Some(String::from_utf16_lossy(&buf[..len]))
    }

    fn print_object_description(&self, handle: ObjectHandle) -> String {
        print_object_string(|buf, size, required| unsafe {
            rokajit_ee_print_object_description(
                self.comp_raw(),
                handle.as_raw(),
                buf.cast(),
                size,
                required,
            )
        })
    }

    fn get_arg_next(&self, args: ArgListHandle) -> Option<ArgListHandle> {
        let raw = unsafe { rokajit_ee_get_arg_next(self.comp_raw(), args.as_raw()) };
        ArgListHandle::from_raw(raw)
    }

    fn get_arg_type(
        &self,
        sig: &CORINFO_SIG_INFO,
        args: ArgListHandle,
    ) -> (CorInfoType, Option<ClassHandle>, bool) {
        let sig = sig as *const CORINFO_SIG_INFO as *mut CORINFO_SIG_INFO;
        let mut vc_type: ffi::CORINFO_CLASS_HANDLE = std::ptr::null_mut();
        let raw =
            unsafe { rokajit_ee_get_arg_type(self.comp_raw(), sig, args.as_raw(), &mut vc_type) };
        // Strip the CorInfoTypeWithMod modifiers (the trait contract drops
        // them); the mask leaves a plain CorInfoType. `MOD_PINNED`
        // survives separately.
        let ty = CorInfoType::from_raw(raw & ffi::CorInfoTypeWithMod_CORINFO_TYPE_MASK)
            .unwrap_or(CorInfoType::Undef);
        let pinned = raw & ffi::CorInfoTypeWithMod_CORINFO_TYPE_MOD_PINNED != 0;
        (ty, ClassHandle::from_raw(vc_type), pinned)
    }

    fn get_exact_classes(
        &self,
        base_type: ClassHandle,
        max_exact_classes: i32,
    ) -> Option<Vec<ClassHandle>> {
        // Caller-owned buffer; the EE writes up to `max_exact_classes`
        // entries and returns the count, or -1 (no free obligation).
        let mut buf: Vec<ffi::CORINFO_CLASS_HANDLE> =
            vec![std::ptr::null_mut(); max_exact_classes.max(0) as usize];
        let count = unsafe {
            rokajit_ee_get_exact_classes(
                self.comp_raw(),
                base_type.as_raw(),
                max_exact_classes,
                buf.as_mut_ptr(),
            )
        };
        if count < 0 {
            return None;
        }
        buf.truncate(count as usize);
        buf.into_iter().map(ClassHandle::from_raw).collect()
    }

    fn get_arg_class(&self, sig: &CORINFO_SIG_INFO, args: ArgListHandle) -> Option<ClassHandle> {
        let sig = sig as *const CORINFO_SIG_INFO as *mut CORINFO_SIG_INFO;
        let raw = unsafe { rokajit_ee_get_arg_class(self.comp_raw(), sig, args.as_raw()) };
        ClassHandle::from_raw(raw)
    }

    fn get_hfa_type(&self, class_: ClassHandle) -> CorInfoHFAElemType {
        let raw = unsafe { rokajit_ee_get_hfa_type(self.comp_raw(), class_.as_raw()) };
        CorInfoHFAElemType::from_raw(raw).unwrap_or(CorInfoHFAElemType::None)
    }

    fn embed_module_handle(
        &self,
        handle: ModuleHandle,
    ) -> (Option<ModuleHandle>, Option<NonNull<c_void>>) {
        let mut indirection: *mut c_void = std::ptr::null_mut();
        let raw = unsafe {
            rokajit_ee_embed_module_handle(self.comp_raw(), handle.as_raw(), &mut indirection)
        };
        (ModuleHandle::from_raw(raw), NonNull::new(indirection))
    }

    fn embed_class_handle(
        &self,
        handle: ClassHandle,
    ) -> (Option<ClassHandle>, Option<NonNull<c_void>>) {
        let mut indirection: *mut c_void = std::ptr::null_mut();
        let raw = unsafe {
            rokajit_ee_embed_class_handle(self.comp_raw(), handle.as_raw(), &mut indirection)
        };
        (ClassHandle::from_raw(raw), NonNull::new(indirection))
    }

    fn embed_method_handle(
        &self,
        handle: MethodHandle,
    ) -> (Option<MethodHandle>, Option<NonNull<c_void>>) {
        let mut indirection: *mut c_void = std::ptr::null_mut();
        let raw = unsafe {
            rokajit_ee_embed_method_handle(self.comp_raw(), handle.as_raw(), &mut indirection)
        };
        (MethodHandle::from_raw(raw), NonNull::new(indirection))
    }

    fn embed_field_handle(
        &self,
        handle: FieldHandle,
    ) -> (Option<FieldHandle>, Option<NonNull<c_void>>) {
        let mut indirection: *mut c_void = std::ptr::null_mut();
        let raw = unsafe {
            rokajit_ee_embed_field_handle(self.comp_raw(), handle.as_raw(), &mut indirection)
        };
        (FieldHandle::from_raw(raw), NonNull::new(indirection))
    }

    fn embed_generic_handle(
        &self,
        token: &mut CORINFO_RESOLVED_TOKEN,
        embed_parent: bool,
        caller: MethodHandle,
    ) -> ffi::CORINFO_GENERICHANDLE_RESULT {
        zeroed_out(|result| unsafe {
            rokajit_ee_embed_generic_handle(
                self.comp_raw(),
                token,
                embed_parent,
                caller.as_raw(),
                result,
            )
        })
    }

    fn get_location_of_this_type(&self, context: MethodHandle) -> ffi::CORINFO_LOOKUP_KIND {
        zeroed_out(|lookup_kind| unsafe {
            rokajit_ee_get_location_of_this_type(self.comp_raw(), context.as_raw(), lookup_kind)
        })
    }

    fn get_cookie_for_interpreter_calli_sig(
        &self,
        sig: &CORINFO_SIG_INFO,
    ) -> Option<NonNull<c_void>> {
        let sig = sig as *const CORINFO_SIG_INFO as *mut CORINFO_SIG_INFO;
        let raw = unsafe { rokajit_ee_get_cookie_for_interpreter_calli_sig(self.comp_raw(), sig) };
        NonNull::new(raw)
    }

    fn get_call_info(
        &self,
        token: &mut CORINFO_RESOLVED_TOKEN,
        constrained: Option<&CORINFO_RESOLVED_TOKEN>,
        caller: MethodHandle,
        flags: CallInfoFlags,
    ) -> CORINFO_CALL_INFO {
        let constrained = constrained.map_or(std::ptr::null_mut(), |t| {
            t as *const CORINFO_RESOLVED_TOKEN as *mut CORINFO_RESOLVED_TOKEN
        });
        zeroed_out(|result| unsafe {
            rokajit_ee_get_call_info(
                self.comp_raw(),
                token,
                constrained,
                caller.as_raw(),
                flags.to_raw(),
                result,
            )
        })
    }

    fn get_var_args_handle(
        &self,
        sig: &CORINFO_SIG_INFO,
        meth: MethodHandle,
    ) -> (Option<VarArgsHandle>, Option<NonNull<c_void>>) {
        let sig = sig as *const CORINFO_SIG_INFO as *mut CORINFO_SIG_INFO;
        let mut indirection: *mut c_void = std::ptr::null_mut();
        let raw = unsafe {
            rokajit_ee_get_var_args_handle(self.comp_raw(), sig, meth.as_raw(), &mut indirection)
        };
        (VarArgsHandle::from_raw(raw), NonNull::new(indirection))
    }

    fn construct_string_literal(
        &self,
        module: ModuleHandle,
        meta_tok: ffi::mdToken,
    ) -> (InfoAccessType, Option<NonNull<c_void>>) {
        let mut value: *mut c_void = std::ptr::null_mut();
        let raw = unsafe {
            rokajit_ee_construct_string_literal(
                self.comp_raw(),
                module.as_raw(),
                meta_tok,
                &mut value,
            )
        };
        let access = InfoAccessType::from_raw(raw).unwrap_or(InfoAccessType::Value);
        (access, NonNull::new(value))
    }

    fn empty_string_literal(&self) -> (InfoAccessType, Option<NonNull<c_void>>) {
        let mut value: *mut c_void = std::ptr::null_mut();
        let raw = unsafe { rokajit_ee_empty_string_literal(self.comp_raw(), &mut value) };
        let access = InfoAccessType::from_raw(raw).unwrap_or(InfoAccessType::Value);
        (access, NonNull::new(value))
    }

    fn convert_pinvoke_calli_to_call(
        &self,
        token: &mut CORINFO_RESOLVED_TOKEN,
        must_convert: bool,
    ) -> bool {
        unsafe { rokajit_ee_convert_pinvoke_calli_to_call(self.comp_raw(), token, must_convert) }
    }
}
