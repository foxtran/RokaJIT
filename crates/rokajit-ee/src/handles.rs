//! Handle newtypes for the EE↔JIT interface.
//!
//! bindgen emits the `CORINFO_*_HANDLE` types as raw pointer aliases
//! (`*mut CORINFO_METHOD_STRUCT_`, etc.). Raw pointers give no type
//! separation: a method handle and a class handle are interchangeable at the
//! type level, and null is inexpressible. This module freezes the wrapper
//! policy (see `decisions/2026-09-11-handle-newtyping.md`):
//!
//! - **One `#[repr(transparent)]` newtype per handle kind**, wrapping the
//!   bindgen alias. No cross-kind conversion exists except through
//!   [`as_raw`](MethodHandle::as_raw) at the FFI edge, and there are no
//!   transmutes anywhere.
//! - **Null is `Option<Handle>`**, never a sentinel value inside the
//!   newtype. A bare `MethodHandle` is always non-null. `from_raw` performs
//!   the null check once, at the boundary.
//! - **Diagnostics print the kind and address** (`MethodHandle(0x7f…)`), so
//!   logs and panic messages never show an anonymous pointer.
//! - **Handles are EE-owned and compile-scoped.** They are valid only for
//!   the duration of one `compileMethod` call (the EE may reclaim the
//!   underlying data afterwards). Rust lifetimes cannot express this (the
//!   handles arrive from C++), so it is a documented invariant: nothing may
//!   store a handle beyond the compilation that produced it. Handles are
//!   `Copy` and compare by address (`PartialEq`/`Eq`/`Hash`), which matches
//!   how RyuJIT uses them (identity keys into EE-side tables).
//! - Handles are deliberately **not `Send`/`Sync`** (they contain raw
//!   pointers); EE calls happen on the thread running `compileMethod`.

/// Defines one handle newtype wrapping a bindgen raw-pointer alias.
macro_rules! handle_newtype {
    ($name:ident, $raw:ty, $doc:literal) => {
        #[doc = $doc]
        ///
        /// `#[repr(transparent)]` over the bindgen alias, so passing it to the
        /// gasket's `extern "C"` forwarders is free. Always non-null; a null
        /// handle is represented as `Option` of this type, and `from_raw`
        /// performs the check.
        #[repr(transparent)]
        #[derive(Copy, Clone, PartialEq, Eq, Hash)]
        pub struct $name(pub $raw);

        impl $name {
            /// Wraps a raw handle from the EE, or returns `None` if it is
            /// null. This is the only way to construct the newtype from FFI.
            #[inline]
            pub const fn from_raw(raw: $raw) -> Option<Self> {
                if raw.is_null() {
                    None
                } else {
                    Some(Self(raw))
                }
            }

            /// Unwraps to the raw bindgen alias, for calls into the gasket
            /// forwarders. Never null.
            #[inline]
            pub const fn as_raw(self) -> $raw {
                self.0
            }
        }

        impl ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                write!(f, concat!(stringify!($name), "({:p})"), self.0)
            }
        }
    };
}

handle_newtype!(
    MethodHandle,
    rokajit_ffi::CORINFO_METHOD_HANDLE,
    "Opaque EE reference to a method (`CORINFO_METHOD_HANDLE`)."
);
handle_newtype!(
    ClassHandle,
    rokajit_ffi::CORINFO_CLASS_HANDLE,
    "Opaque EE reference to a class / type (`CORINFO_CLASS_HANDLE`)."
);
handle_newtype!(
    FieldHandle,
    rokajit_ffi::CORINFO_FIELD_HANDLE,
    "Opaque EE reference to a field (`CORINFO_FIELD_HANDLE`)."
);
handle_newtype!(
    ModuleHandle,
    rokajit_ffi::CORINFO_MODULE_HANDLE,
    "Opaque EE reference to a module (`CORINFO_MODULE_HANDLE`)."
);
handle_newtype!(
    ContextHandle,
    rokajit_ffi::CORINFO_CONTEXT_HANDLE,
    "Opaque EE reference to a generic context (`CORINFO_CONTEXT_HANDLE`)."
);

impl ContextHandle {
    /// Builds a class context (`MAKE_CLASSCONTEXT`, corinfo.h:1030): a
    /// `CORINFO_CONTEXT_HANDLE` is a method or class handle tagged in its
    /// low bit (`CORINFO_CONTEXTFLAGS_CLASS` = 0x01, corinfo.h:1025), not
    /// the bare handle. Passing an untagged class handle makes the EE take
    /// the method branch of `GetTypeFromContext` and dereference the
    /// MethodTable as a MethodDesc — a crash, not an error.
    pub fn from_class(class: ClassHandle) -> Self {
        const CORINFO_CONTEXTFLAGS_CLASS: usize = 0x01;
        Self(
            (class.as_raw() as usize | CORINFO_CONTEXTFLAGS_CLASS)
                as rokajit_ffi::CORINFO_CONTEXT_HANDLE,
        )
    }
}
handle_newtype!(
    ObjectHandle,
    rokajit_ffi::CORINFO_OBJECT_HANDLE,
    "Opaque EE reference to a frozen object (`CORINFO_OBJECT_HANDLE`)."
);
handle_newtype!(
    ArgListHandle,
    rokajit_ffi::CORINFO_ARG_LIST_HANDLE,
    "Signature-argument cursor (`CORINFO_ARG_LIST_HANDLE`), advanced with \
     `getArgNext`."
);
handle_newtype!(
    GenericHandle,
    rokajit_ffi::CORINFO_GENERIC_HANDLE,
    "A handle of statically-unknown kind, embedded into code \
     (`CORINFO_GENERIC_HANDLE`); its kind is given by a \
     `CorInfoGenericHandleType`."
);
handle_newtype!(
    ProfilingHandle,
    rokajit_ffi::CORINFO_PROFILING_HANDLE,
    "Opaque EE reference used by the profiling API (`CORINFO_PROFILING_HANDLE`)."
);
handle_newtype!(
    JustMyCodeHandle,
    rokajit_ffi::CORINFO_JUST_MY_CODE_HANDLE,
    "Just-my-code token (`CORINFO_JUST_MY_CODE_HANDLE`)."
);
handle_newtype!(
    VarArgsHandle,
    rokajit_ffi::CORINFO_VARARGS_HANDLE,
    "VM cookie for a vararg signature (`CORINFO_VARARGS_HANDLE`), returned \
     by `getVarArgsHandle`. Added at step_04 integration: the frozen list \
     of ten kinds predated the tokens/sig group's surface."
);
handle_newtype!(
    WasmTypeSymbolHandle,
    rokajit_ffi::CORINFO_WASM_TYPE_SYMBOL_HANDLE,
    "Opaque EE reference to a wasm type symbol \
     (`CORINFO_WASM_TYPE_SYMBOL_HANDLE`, corinfo.h:1000)."
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_is_none() {
        assert_eq!(MethodHandle::from_raw(std::ptr::null_mut()), None);
    }

    #[test]
    fn non_null_round_trips() {
        let mut cell = 0u8;
        let raw = &mut cell as *mut u8 as rokajit_ffi::CORINFO_METHOD_HANDLE;
        let h = MethodHandle::from_raw(raw).unwrap();
        assert_eq!(h.as_raw(), raw);
        // Kind-tagged diagnostic printing.
        assert!(format!("{h:?}").starts_with("MethodHandle(0x"));
    }

    #[test]
    fn transparent_layout() {
        assert_eq!(
            std::mem::size_of::<ClassHandle>(),
            std::mem::size_of::<rokajit_ffi::CORINFO_CLASS_HANDLE>()
        );
    }
}
