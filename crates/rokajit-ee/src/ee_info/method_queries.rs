use std::ffi::{c_char, c_uint, c_void};
use std::ptr::NonNull;

use rokajit_ffi as ffi;
use rokajit_ffi::{CORINFO_EH_CLAUSE, CORINFO_SIG_INFO};

use super::wrap::{print_object_string, zeroed_out};
use super::GasketEeInfo;
use crate::enums::{CorInfoCallConvExtension, MethodAttribs, MethodRuntimeFlags};
use crate::handles::{ClassHandle, ContextHandle, MethodHandle};

/// `mdMethodDefNil` (corhdr.h:1535) — a `#define`, so bindgen did not emit
/// it. `getMethodDefFromMethod` returns it for dynamic methods.
const MD_METHOD_DEF_NIL: u32 = 0x0600_0000;

/// Method metadata queries (C++ `ICorMethodInfo`).
pub trait MethodQueries {
    /// C++ `ICorMethodInfo::getMethodAttribs` (corinfo.h:2166).
    fn get_method_attribs(&self, ftn: MethodHandle) -> MethodAttribs;

    /// C++ `ICorMethodInfo::getMethodSig` (corinfo.h:2180). `member_parent`
    /// is the C++ `memberParent` out-of-scope hint; `None` = null.
    fn get_method_sig(
        &self,
        ftn: MethodHandle,
        member_parent: Option<ClassHandle>,
    ) -> CORINFO_SIG_INFO;

    /// C++ `ICorMethodInfo::getMethodClass` (corinfo.h:2287). Infallible: the
    /// EE always knows a method's owning class.
    fn get_method_class(&self, ftn: MethodHandle) -> ClassHandle;

    /// C++ `ICorMethodInfo::getEHinfo` (corinfo.h:2280). `index` runs over
    /// `CORINFO_METHOD_INFO::EHcount`.
    fn get_eh_info(&self, ftn: MethodHandle, index: u32) -> CORINFO_EH_CLAUSE;

    /// C++ `ICorMethodInfo::getMethodHash` (corinfo.h:3239) — debug/range
    /// knobs only.
    fn get_method_hash(&self, ftn: MethodHandle) -> u32;

    /// C++ `ICorMethodInfo::methodMustBeLoadedBeforeCodeIsRun`
    /// (corinfo.h:2376).
    fn method_must_be_loaded_before_code_is_run(&self, ftn: MethodHandle);

    /// C++ `ICorMethodInfo::getMethodNameFromMetadata` (corinfo.h:3228).
    /// `None` when the method has no metadata name (the C++ nullptr). The
    /// wrapper copies the string; the C++ storage is EE-lifetime, so there
    /// is nothing to free. The step_02 surface keeps only the method name;
    /// the class/namespace/enclosing-class out-params are passed null slots.
    fn get_method_name_from_metadata(&self, ftn: MethodHandle) -> Option<String>;

    /// C++ `ICorStaticInfo::isIntrinsic` (corinfo.h:2140). Fast equivalent of
    /// testing the `CORINFO_FLG_INTRINSIC` bit of `get_method_attribs`.
    fn is_intrinsic(&self, ftn: MethodHandle) -> bool;

    /// C++ `ICorStaticInfo::canValueClassInstancePointerEscape`
    /// (corinfo.h:2150). `false` means the EE guarantees (ECMA-335 aug.
    /// III.1.7.7) the value-type `this` pointer does not escape `ftn`;
    /// `true` is the conservative answer.
    fn can_value_class_instance_pointer_escape(&self, ftn: MethodHandle) -> bool;

    /// C++ `ICorStaticInfo::notifyMethodInfoUsage` (corinfo.h:2163). `false`
    /// = the JIT may not rely on the method's MethodInfo in the current
    /// method; the info may change.
    fn notify_method_info_usage(&self, ftn: MethodHandle) -> bool;

    /// C++ `ICorStaticInfo::setMethodAttribs` (corinfo.h:2171): sets private
    /// JIT flags, later retrievable through `get_method_attribs`.
    fn set_method_attribs(&self, ftn: MethodHandle, attribs: MethodRuntimeFlags);

    /// C++ `ICorStaticInfo::getMethodInfo` (corinfo.h:2195). The C++ `false`
    /// (method is not IL, or otherwise unavailable) becomes `None`.
    /// `context` is the C++ `context` in-param; `None` = the C++ default
    /// NULL. Only usable on methods known to be IL (header note at
    /// corinfo.h:2186).
    fn get_method_info(
        &self,
        ftn: MethodHandle,
        context: Option<ContextHandle>,
    ) -> Option<ffi::CORINFO_METHOD_INFO>;

    /// C++ `ICorStaticInfo::haveSameMethodDefinition` (corinfo.h:2216).
    /// E.g. `Foo<int>` and `Foo<uint>` have different handles but share the
    /// same method definition.
    fn have_same_method_definition(&self, meth1: MethodHandle, meth2: MethodHandle) -> bool;

    /// C++ `ICorStaticInfo::getTypeDefinition` (corinfo.h:2234): the
    /// unconstructed generic type definition (`Foo<int>` → `Foo<>`).
    /// Infallible per the contract (returns the input for an unconstructed
    /// generic); precondition — the header says to call this only when the
    /// input is in fact a generic type. A null return is an EE contract
    /// violation (the wrapper `expect`s).
    fn get_type_definition(&self, ty: ClassHandle) -> ClassHandle;

    /// C++ `ICorStaticInfo::getMethodVTableOffset` (corinfo.h:2293). Tuple:
    /// `(offset_of_indirection, offset_after_indirection, is_relative)`.
    fn get_method_vtable_offset(&self, method: MethodHandle) -> (u32, u32, bool);

    /// C++ `ICorStaticInfo::resolveVirtualMethod` (corinfo.h:2305). In/out
    /// `CORINFO_DEVIRTUALIZATION_INFO` exactly as C++; `false` =
    /// devirtualization not possible (`devirtualizedMethod` stays null).
    fn resolve_virtual_method(&self, info: &mut ffi::CORINFO_DEVIRTUALIZATION_INFO) -> bool;

    /// C++ `ICorStaticInfo::getAsyncOtherVariant` (corinfo.h:2312). The null
    /// handle (method has no async variant) becomes `None`; the `bool` is
    /// the C++ `variantIsThunk` out-param (the returned method is a
    /// VM-provided thunk).
    fn get_async_other_variant(&self, ftn: MethodHandle) -> Option<(MethodHandle, bool)>;

    /// C++ `ICorStaticInfo::getDefaultComparerClass` (corinfo.h:2319). The
    /// null handle (type can't be determined exactly) becomes `None`.
    fn get_default_comparer_class(&self, elem_type: ClassHandle) -> Option<ClassHandle>;

    /// C++ `ICorStaticInfo::getDefaultEqualityComparerClass`
    /// (corinfo.h:2325). The null handle (type can't be determined exactly)
    /// becomes `None`.
    fn get_default_equality_comparer_class(&self, elem_type: ClassHandle) -> Option<ClassHandle>;

    /// C++ `ICorStaticInfo::getSZArrayHelperEnumeratorClass`
    /// (corinfo.h:2331). The null handle (type can't be determined exactly)
    /// becomes `None`.
    fn get_sz_array_helper_enumerator_class(&self, elem_type: ClassHandle) -> Option<ClassHandle>;

    /// C++ `ICorStaticInfo::expandRawHandleIntrinsic` (corinfo.h:2341).
    /// `resolved_token` is in/out exactly as C++; the result is the C++
    /// `pResult` out-param, returned by value.
    fn expand_raw_handle_intrinsic(
        &self,
        resolved_token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        caller: MethodHandle,
    ) -> ffi::CORINFO_GENERICHANDLE_RESULT;

    /// C++ `ICorStaticInfo::isIntrinsicType` (corinfo.h:2348). The C++
    /// declaration carries a default implementation returning `false`; the
    /// CoreCLR EE overrides it.
    fn is_intrinsic_type(&self, class_hnd: ClassHandle) -> bool;

    /// C++ `ICorStaticInfo::getUnmanagedCallConv` (corinfo.h:2356). `method`
    /// `None` covers the C++ null case (function pointer with the
    /// `CORINFO_CALLCONV_UNMANAGED` calling convention); `call_site_sig` is
    /// required in that case. Tuple: `(call_conv, suppress_gc_transition)`.
    /// An EE value outside the pinned header set is header drift — the
    /// wrapper `expect`s (a bug, never an expected answer).
    fn get_unmanaged_call_conv(
        &self,
        method: Option<MethodHandle>,
        call_site_sig: Option<&CORINFO_SIG_INFO>,
    ) -> (CorInfoCallConvExtension, bool);

    /// C++ `ICorStaticInfo::pInvokeMarshalingRequired` (corinfo.h:2364).
    /// `method` `None` = calli (C++ `method == 0`); `call_site_sig` is only
    /// needed for the varargs/calli case.
    fn p_invoke_marshaling_required(
        &self,
        method: Option<MethodHandle>,
        call_site_sig: Option<&CORINFO_SIG_INFO>,
    ) -> bool;

    /// C++ `ICorStaticInfo::satisfiesMethodConstraints` (corinfo.h:2370).
    /// `parent` is the exact parent of `method`.
    fn satisfies_method_constraints(&self, parent: ClassHandle, method: MethodHandle) -> bool;

    /// C++ `ICorStaticInfo::getGSCookie` (corinfo.h:2382): the /GS guard
    /// cookie. Exactly one of the tuple elements is meaningful: the constant
    /// cookie value (plain JIT) or a pointer to the cookie location (Ngen);
    /// the C++ null `ppCookieVal` becomes `None`.
    fn get_gs_cookie(&self) -> (usize, Option<NonNull<usize>>);

    /// C++ `ICorStaticInfo::setPatchpointInfo` (corinfo.h:2388).
    /// `PatchpointInfo` is opaque across the interface (corinfo.h:985) and
    /// JIT-owned; `None` passes the C++ nullptr (clears the info).
    fn set_patchpoint_info(&self, patchpoint_info: Option<NonNull<ffi::PatchpointInfo>>);

    /// C++ `ICorStaticInfo::getOSRInfo` (corinfo.h:2393). The null
    /// `PatchpointInfo*` (no OSR entry for this method) becomes `None`; the
    /// `u32` is the IL offset of the OSR entry point.
    fn get_osr_info(&self) -> Option<(NonNull<ffi::PatchpointInfo>, u32)>;

    /// C++ `ICorStaticInfo::getAsyncInfo` (corinfo.h:3163): EE-internal
    /// async helper handles, returned as the bindgen mirror struct.
    fn get_async_info(&self) -> ffi::CORINFO_ASYNC_INFO;

    /// C++ `ICorStaticInfo::getAwaitReturnCall` (corinfo.h:3176). The null
    /// handle becomes `None`. Tuple: `(await_call, context, inst_arg)` —
    /// `context` is the inlining context exactly as `getCallInfo` would
    /// report it (may be absent), `inst_arg` the instantiation argument.
    fn get_await_return_call(
        &self,
        caller: MethodHandle,
    ) -> Option<(MethodHandle, Option<ContextHandle>, ffi::CORINFO_LOOKUP)>;

    /// C++ `ICorStaticInfo::getAwaitAwaiterInContinuationCall`
    /// (corinfo.h:3191). `resolved_token` is the
    /// `AsyncHelpers.AwaitAwaiter`/`UnsafeAwaitAwaiter` call-site token being
    /// replaced (in/out as C++); `is_unsafe` selects the unsafe variant.
    /// NULL (transformation cannot be performed) becomes `None`; the tuple
    /// is `(call, context, inst_arg)` as in `get_await_return_call`.
    fn get_await_awaiter_in_continuation_call(
        &self,
        caller: MethodHandle,
        resolved_token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        is_unsafe: bool,
    ) -> Option<(MethodHandle, Option<ContextHandle>, ffi::CORINFO_LOOKUP)>;

    /// C++ `ICorStaticInfo::getMethodDefFromMethod` (corinfo.h:3207). Debug
    /// use only. `mdMethodDefNil` (dynamic methods) becomes `None`.
    fn get_method_def_from_method(&self, method: MethodHandle) -> Option<u32>;

    /// C++ `ICorStaticInfo::printMethodName` (corinfo.h:3214). Like
    /// `get_method_name_from_metadata` but also produces names for
    /// metadata-less functions. The wrapper performs the C++ two-call sizing
    /// dance (nullptr buffer queries the required size, corinfo.h:2463) and
    /// copies into an owned `String`; nothing to free on the C++ side.
    fn print_method_name(&self, ftn: MethodHandle) -> String;

    /// C++ `ICorDynamicInfo::getAsyncResumptionStub` (corinfo.h:3528).
    /// Tuple: `(stub_method, entry_point)`. A null stub handle becomes
    /// `None`.
    fn get_async_resumption_stub(&self) -> Option<(MethodHandle, NonNull<c_void>)>;

    /// C++ `ICorDynamicInfo::getContinuationType` (corinfo.h:3530).
    /// `obj_refs` is the C++ IN array marking which pointer-sized slots of
    /// the continuation data hold GC references (`objRefsSize` = slice
    /// length). A null class handle becomes `None`.
    fn get_continuation_type(&self, data_size: usize, obj_refs: &[bool]) -> Option<ClassHandle>;
}

extern "C" {
    fn rokajit_ee_get_method_attribs(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
    ) -> u32;
    fn rokajit_ee_get_method_sig(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        sig: *mut CORINFO_SIG_INFO,
        member_parent: ffi::CORINFO_CLASS_HANDLE,
    );
    fn rokajit_ee_is_intrinsic(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
    ) -> bool;
    fn rokajit_ee_can_value_class_instance_pointer_escape(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
    ) -> bool;
    fn rokajit_ee_notify_method_info_usage(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
    ) -> bool;
    fn rokajit_ee_set_method_attribs(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        attribs: ffi::CorInfoMethodRuntimeFlags,
    );
    fn rokajit_ee_get_method_info(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        method_info: *mut ffi::CORINFO_METHOD_INFO,
        context: ffi::CORINFO_CONTEXT_HANDLE,
    ) -> bool;
    fn rokajit_ee_have_same_method_definition(
        info: *mut ffi::ICorJitInfo,
        meth1_hnd: ffi::CORINFO_METHOD_HANDLE,
        meth2_hnd: ffi::CORINFO_METHOD_HANDLE,
    ) -> bool;
    fn rokajit_ee_get_type_definition(
        info: *mut ffi::ICorJitInfo,
        ty: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_eh_info(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        eh_number: c_uint,
        clause: *mut CORINFO_EH_CLAUSE,
    );
    fn rokajit_ee_get_method_class(
        info: *mut ffi::ICorJitInfo,
        method: ffi::CORINFO_METHOD_HANDLE,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_method_vtable_offset(
        info: *mut ffi::ICorJitInfo,
        method: ffi::CORINFO_METHOD_HANDLE,
        offset_of_indirection: *mut c_uint,
        offset_after_indirection: *mut c_uint,
        is_relative: *mut bool,
    );
    fn rokajit_ee_resolve_virtual_method(
        info: *mut ffi::ICorJitInfo,
        devirt_info: *mut ffi::CORINFO_DEVIRTUALIZATION_INFO,
    ) -> bool;
    fn rokajit_ee_get_async_other_variant(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        variant_is_thunk: *mut bool,
    ) -> ffi::CORINFO_METHOD_HANDLE;
    fn rokajit_ee_get_default_comparer_class(
        info: *mut ffi::ICorJitInfo,
        elem_type: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_default_equality_comparer_class(
        info: *mut ffi::ICorJitInfo,
        elem_type: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_get_sz_array_helper_enumerator_class(
        info: *mut ffi::ICorJitInfo,
        elem_type: ffi::CORINFO_CLASS_HANDLE,
    ) -> ffi::CORINFO_CLASS_HANDLE;
    fn rokajit_ee_expand_raw_handle_intrinsic(
        info: *mut ffi::ICorJitInfo,
        p_resolved_token: *mut ffi::CORINFO_RESOLVED_TOKEN,
        caller_handle: ffi::CORINFO_METHOD_HANDLE,
        p_result: *mut ffi::CORINFO_GENERICHANDLE_RESULT,
    );
    fn rokajit_ee_is_intrinsic_type(
        info: *mut ffi::ICorJitInfo,
        class_hnd: ffi::CORINFO_CLASS_HANDLE,
    ) -> bool;
    fn rokajit_ee_get_unmanaged_call_conv(
        info: *mut ffi::ICorJitInfo,
        method: ffi::CORINFO_METHOD_HANDLE,
        call_site_sig: *mut CORINFO_SIG_INFO,
        p_suppress_gc_transition: *mut bool,
    ) -> ffi::CorInfoCallConvExtension;
    fn rokajit_ee_p_invoke_marshaling_required(
        info: *mut ffi::ICorJitInfo,
        method: ffi::CORINFO_METHOD_HANDLE,
        call_site_sig: *mut CORINFO_SIG_INFO,
    ) -> bool;
    fn rokajit_ee_satisfies_method_constraints(
        info: *mut ffi::ICorJitInfo,
        parent: ffi::CORINFO_CLASS_HANDLE,
        method: ffi::CORINFO_METHOD_HANDLE,
    ) -> bool;
    fn rokajit_ee_method_must_be_loaded_before_code_is_run(
        info: *mut ffi::ICorJitInfo,
        method: ffi::CORINFO_METHOD_HANDLE,
    );
    fn rokajit_ee_get_gs_cookie(
        info: *mut ffi::ICorJitInfo,
        p_cookie_val: *mut ffi::GSCookie,
        pp_cookie_val: *mut *mut ffi::GSCookie,
    );
    fn rokajit_ee_set_patchpoint_info(
        info: *mut ffi::ICorJitInfo,
        patchpoint_info: *mut ffi::PatchpointInfo,
    );
    fn rokajit_ee_get_osr_info(
        info: *mut ffi::ICorJitInfo,
        il_offset: *mut c_uint,
    ) -> *mut ffi::PatchpointInfo;
    fn rokajit_ee_get_async_info(
        info: *mut ffi::ICorJitInfo,
        p_async_info_out: *mut ffi::CORINFO_ASYNC_INFO,
    );
    fn rokajit_ee_get_await_return_call(
        info: *mut ffi::ICorJitInfo,
        caller_handle: ffi::CORINFO_METHOD_HANDLE,
        context_handle: *mut ffi::CORINFO_CONTEXT_HANDLE,
        inst_arg: *mut ffi::CORINFO_LOOKUP,
    ) -> ffi::CORINFO_METHOD_HANDLE;
    fn rokajit_ee_get_await_awaiter_in_continuation_call(
        info: *mut ffi::ICorJitInfo,
        caller_handle: ffi::CORINFO_METHOD_HANDLE,
        p_resolved_token: *mut ffi::CORINFO_RESOLVED_TOKEN,
        is_unsafe: bool,
        context_handle: *mut ffi::CORINFO_CONTEXT_HANDLE,
        inst_arg: *mut ffi::CORINFO_LOOKUP,
    ) -> ffi::CORINFO_METHOD_HANDLE;
    fn rokajit_ee_get_method_def_from_method(
        info: *mut ffi::ICorJitInfo,
        h_method: ffi::CORINFO_METHOD_HANDLE,
    ) -> ffi::mdMethodDef;
    fn rokajit_ee_print_method_name(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        buffer: *mut c_char,
        buffer_size: usize,
        p_required_buffer_size: *mut usize,
    ) -> usize;
    fn rokajit_ee_get_method_name_from_metadata(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        class_name: *mut *const c_char,
        namespace_name: *mut *const c_char,
        enclosing_class_names: *mut *const c_char,
        max_enclosing_class_names: usize,
    ) -> *const c_char;
    fn rokajit_ee_get_method_hash(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
    ) -> c_uint;
    fn rokajit_ee_get_async_resumption_stub(
        info: *mut ffi::ICorJitInfo,
        entry_point: *mut *mut c_void,
    ) -> ffi::CORINFO_METHOD_HANDLE;
    fn rokajit_ee_get_continuation_type(
        info: *mut ffi::ICorJitInfo,
        data_size: usize,
        obj_refs: *const bool,
        obj_refs_size: usize,
    ) -> ffi::CORINFO_CLASS_HANDLE;
}

impl MethodQueries for GasketEeInfo {
    fn get_method_attribs(&self, ftn: MethodHandle) -> MethodAttribs {
        let raw = unsafe { rokajit_ee_get_method_attribs(self.comp_raw(), ftn.as_raw()) };
        MethodAttribs::from_raw(raw)
    }

    fn get_method_sig(
        &self,
        ftn: MethodHandle,
        member_parent: Option<ClassHandle>,
    ) -> CORINFO_SIG_INFO {
        let member_parent = member_parent.map_or(std::ptr::null_mut(), ClassHandle::as_raw);
        zeroed_out(|sig| unsafe {
            rokajit_ee_get_method_sig(self.comp_raw(), ftn.as_raw(), sig, member_parent)
        })
    }

    fn get_method_class(&self, ftn: MethodHandle) -> ClassHandle {
        let raw = unsafe { rokajit_ee_get_method_class(self.comp_raw(), ftn.as_raw()) };
        ClassHandle::from_raw(raw).expect("EE contract: getMethodClass always returns a class")
    }

    fn get_eh_info(&self, ftn: MethodHandle, index: u32) -> CORINFO_EH_CLAUSE {
        zeroed_out(|clause| unsafe {
            rokajit_ee_get_eh_info(self.comp_raw(), ftn.as_raw(), index, clause)
        })
    }

    fn get_method_hash(&self, ftn: MethodHandle) -> u32 {
        unsafe { rokajit_ee_get_method_hash(self.comp_raw(), ftn.as_raw()) }
    }

    fn method_must_be_loaded_before_code_is_run(&self, ftn: MethodHandle) {
        unsafe {
            rokajit_ee_method_must_be_loaded_before_code_is_run(self.comp_raw(), ftn.as_raw())
        };
    }

    fn get_method_name_from_metadata(&self, ftn: MethodHandle) -> Option<String> {
        // The step_02 surface keeps only the method name; RyuJIT passes
        // nullptr for the class/namespace/enclosing-class out-params it does
        // not want.
        let name = unsafe {
            rokajit_ee_get_method_name_from_metadata(
                self.comp_raw(),
                ftn.as_raw(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if name.is_null() {
            return None;
        }
        // EE-lifetime storage: copy, nothing to free.
        Some(
            unsafe { std::ffi::CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    fn is_intrinsic(&self, ftn: MethodHandle) -> bool {
        unsafe { rokajit_ee_is_intrinsic(self.comp_raw(), ftn.as_raw()) }
    }

    fn can_value_class_instance_pointer_escape(&self, ftn: MethodHandle) -> bool {
        unsafe { rokajit_ee_can_value_class_instance_pointer_escape(self.comp_raw(), ftn.as_raw()) }
    }

    fn notify_method_info_usage(&self, ftn: MethodHandle) -> bool {
        unsafe { rokajit_ee_notify_method_info_usage(self.comp_raw(), ftn.as_raw()) }
    }

    fn set_method_attribs(&self, ftn: MethodHandle, attribs: MethodRuntimeFlags) {
        unsafe { rokajit_ee_set_method_attribs(self.comp_raw(), ftn.as_raw(), attribs.to_raw()) };
    }

    fn get_method_info(
        &self,
        ftn: MethodHandle,
        context: Option<ContextHandle>,
    ) -> Option<ffi::CORINFO_METHOD_INFO> {
        let context = context.map_or(std::ptr::null_mut(), ContextHandle::as_raw);
        let mut ok = false;
        let method_info = zeroed_out(|method_info| {
            ok = unsafe {
                rokajit_ee_get_method_info(self.comp_raw(), ftn.as_raw(), method_info, context)
            };
        });
        if ok {
            Some(method_info)
        } else {
            None
        }
    }

    fn have_same_method_definition(&self, meth1: MethodHandle, meth2: MethodHandle) -> bool {
        unsafe {
            rokajit_ee_have_same_method_definition(self.comp_raw(), meth1.as_raw(), meth2.as_raw())
        }
    }

    fn get_type_definition(&self, ty: ClassHandle) -> ClassHandle {
        let raw = unsafe { rokajit_ee_get_type_definition(self.comp_raw(), ty.as_raw()) };
        ClassHandle::from_raw(raw).expect("EE contract: getTypeDefinition returns a valid handle")
    }

    fn get_method_vtable_offset(&self, method: MethodHandle) -> (u32, u32, bool) {
        let mut offset_of_indirection: c_uint = 0;
        let mut offset_after_indirection: c_uint = 0;
        let mut is_relative = false;
        unsafe {
            rokajit_ee_get_method_vtable_offset(
                self.comp_raw(),
                method.as_raw(),
                &mut offset_of_indirection,
                &mut offset_after_indirection,
                &mut is_relative,
            )
        };
        (offset_of_indirection, offset_after_indirection, is_relative)
    }

    fn resolve_virtual_method(&self, info: &mut ffi::CORINFO_DEVIRTUALIZATION_INFO) -> bool {
        unsafe { rokajit_ee_resolve_virtual_method(self.comp_raw(), info) }
    }

    fn get_async_other_variant(&self, ftn: MethodHandle) -> Option<(MethodHandle, bool)> {
        let mut variant_is_thunk = false;
        let raw = unsafe {
            rokajit_ee_get_async_other_variant(self.comp_raw(), ftn.as_raw(), &mut variant_is_thunk)
        };
        MethodHandle::from_raw(raw).map(|m| (m, variant_is_thunk))
    }

    fn get_default_comparer_class(&self, elem_type: ClassHandle) -> Option<ClassHandle> {
        let raw =
            unsafe { rokajit_ee_get_default_comparer_class(self.comp_raw(), elem_type.as_raw()) };
        ClassHandle::from_raw(raw)
    }

    fn get_default_equality_comparer_class(&self, elem_type: ClassHandle) -> Option<ClassHandle> {
        let raw = unsafe {
            rokajit_ee_get_default_equality_comparer_class(self.comp_raw(), elem_type.as_raw())
        };
        ClassHandle::from_raw(raw)
    }

    fn get_sz_array_helper_enumerator_class(&self, elem_type: ClassHandle) -> Option<ClassHandle> {
        let raw = unsafe {
            rokajit_ee_get_sz_array_helper_enumerator_class(self.comp_raw(), elem_type.as_raw())
        };
        ClassHandle::from_raw(raw)
    }

    fn expand_raw_handle_intrinsic(
        &self,
        resolved_token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        caller: MethodHandle,
    ) -> ffi::CORINFO_GENERICHANDLE_RESULT {
        zeroed_out(|result| unsafe {
            rokajit_ee_expand_raw_handle_intrinsic(
                self.comp_raw(),
                resolved_token,
                caller.as_raw(),
                result,
            )
        })
    }

    fn is_intrinsic_type(&self, class_hnd: ClassHandle) -> bool {
        unsafe { rokajit_ee_is_intrinsic_type(self.comp_raw(), class_hnd.as_raw()) }
    }

    fn get_unmanaged_call_conv(
        &self,
        method: Option<MethodHandle>,
        call_site_sig: Option<&CORINFO_SIG_INFO>,
    ) -> (CorInfoCallConvExtension, bool) {
        let method = method.map_or(std::ptr::null_mut(), MethodHandle::as_raw);
        let sig = call_site_sig.map_or(std::ptr::null_mut(), |s| s as *const _ as *mut _);
        let mut suppress_gc_transition = false;
        let raw = unsafe {
            rokajit_ee_get_unmanaged_call_conv(
                self.comp_raw(),
                method,
                sig,
                &mut suppress_gc_transition,
            )
        };
        let call_conv = CorInfoCallConvExtension::from_raw(raw)
            .expect("EE returned a CorInfoCallConvExtension outside the pinned header set");
        (call_conv, suppress_gc_transition)
    }

    fn p_invoke_marshaling_required(
        &self,
        method: Option<MethodHandle>,
        call_site_sig: Option<&CORINFO_SIG_INFO>,
    ) -> bool {
        let method = method.map_or(std::ptr::null_mut(), MethodHandle::as_raw);
        let sig = call_site_sig.map_or(std::ptr::null_mut(), |s| s as *const _ as *mut _);
        unsafe { rokajit_ee_p_invoke_marshaling_required(self.comp_raw(), method, sig) }
    }

    fn satisfies_method_constraints(&self, parent: ClassHandle, method: MethodHandle) -> bool {
        unsafe {
            rokajit_ee_satisfies_method_constraints(
                self.comp_raw(),
                parent.as_raw(),
                method.as_raw(),
            )
        }
    }

    fn get_gs_cookie(&self) -> (usize, Option<NonNull<usize>>) {
        let mut cookie_val: ffi::GSCookie = 0;
        let mut p_cookie_val: *mut ffi::GSCookie = std::ptr::null_mut();
        unsafe { rokajit_ee_get_gs_cookie(self.comp_raw(), &mut cookie_val, &mut p_cookie_val) };
        (cookie_val, NonNull::new(p_cookie_val))
    }

    fn set_patchpoint_info(&self, patchpoint_info: Option<NonNull<ffi::PatchpointInfo>>) {
        let ptr = patchpoint_info.map_or(std::ptr::null_mut(), NonNull::as_ptr);
        unsafe { rokajit_ee_set_patchpoint_info(self.comp_raw(), ptr) };
    }

    fn get_osr_info(&self) -> Option<(NonNull<ffi::PatchpointInfo>, u32)> {
        let mut il_offset: c_uint = 0;
        let raw = unsafe { rokajit_ee_get_osr_info(self.comp_raw(), &mut il_offset) };
        NonNull::new(raw).map(|p| (p, il_offset))
    }

    fn get_async_info(&self) -> ffi::CORINFO_ASYNC_INFO {
        zeroed_out(|async_info| unsafe { rokajit_ee_get_async_info(self.comp_raw(), async_info) })
    }

    fn get_await_return_call(
        &self,
        caller: MethodHandle,
    ) -> Option<(MethodHandle, Option<ContextHandle>, ffi::CORINFO_LOOKUP)> {
        let mut context: ffi::CORINFO_CONTEXT_HANDLE = std::ptr::null_mut();
        let mut raw = std::ptr::null_mut();
        let inst_arg = zeroed_out(|inst_arg| {
            raw = unsafe {
                rokajit_ee_get_await_return_call(
                    self.comp_raw(),
                    caller.as_raw(),
                    &mut context,
                    inst_arg,
                )
            };
        });
        MethodHandle::from_raw(raw).map(|m| (m, ContextHandle::from_raw(context), inst_arg))
    }

    fn get_await_awaiter_in_continuation_call(
        &self,
        caller: MethodHandle,
        resolved_token: &mut ffi::CORINFO_RESOLVED_TOKEN,
        is_unsafe: bool,
    ) -> Option<(MethodHandle, Option<ContextHandle>, ffi::CORINFO_LOOKUP)> {
        let mut context: ffi::CORINFO_CONTEXT_HANDLE = std::ptr::null_mut();
        let mut raw = std::ptr::null_mut();
        let inst_arg = zeroed_out(|inst_arg| {
            raw = unsafe {
                rokajit_ee_get_await_awaiter_in_continuation_call(
                    self.comp_raw(),
                    caller.as_raw(),
                    resolved_token,
                    is_unsafe,
                    &mut context,
                    inst_arg,
                )
            };
        });
        MethodHandle::from_raw(raw).map(|m| (m, ContextHandle::from_raw(context), inst_arg))
    }

    fn get_method_def_from_method(&self, method: MethodHandle) -> Option<u32> {
        let raw =
            unsafe { rokajit_ee_get_method_def_from_method(self.comp_raw(), method.as_raw()) };
        if raw == MD_METHOD_DEF_NIL {
            None
        } else {
            Some(raw)
        }
    }

    fn print_method_name(&self, ftn: MethodHandle) -> String {
        print_object_string(|buf, size, required| unsafe {
            rokajit_ee_print_method_name(self.comp_raw(), ftn.as_raw(), buf.cast(), size, required)
        })
    }

    fn get_async_resumption_stub(&self) -> Option<(MethodHandle, NonNull<c_void>)> {
        let mut entry_point: *mut c_void = std::ptr::null_mut();
        let raw =
            unsafe { rokajit_ee_get_async_resumption_stub(self.comp_raw(), &mut entry_point) };
        match (MethodHandle::from_raw(raw), NonNull::new(entry_point)) {
            (Some(method), Some(entry)) => Some((method, entry)),
            _ => None,
        }
    }

    fn get_continuation_type(&self, data_size: usize, obj_refs: &[bool]) -> Option<ClassHandle> {
        let raw = unsafe {
            rokajit_ee_get_continuation_type(
                self.comp_raw(),
                data_size,
                obj_refs.as_ptr(),
                obj_refs.len(),
            )
        };
        ClassHandle::from_raw(raw)
    }
}
