//! `MockEe` — an `EeInfo` implementation with canned answers, proving the
//! trait is implementable without a live EE (and serving as the test double
//! for compiler-core unit tests). Test-only.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;

use rokajit_ffi as ffi;

// Only the in-module tests use the trait surface (`EeInfo`, `ChunkRequest`,
// `AllocMemFlags`); the mock bodies reach them through `super::MockEe`.
#[cfg(test)]
use crate::ee_info::*;
use crate::enums::*;
use crate::handles::*;

mod class_queries;
mod debug_info;
mod field_queries;
mod helpers;
mod host;
mod inlining_and_tail_call;
mod method_queries;
mod output_sinks;
mod pgo;
mod relocations;
mod tokens_and_signatures;

/// A canned method signature — the stack-relevant shape only (importer
/// fixtures; step_07.2).
#[derive(Clone)]
pub struct MockSig {
    pub ret: CorInfoType,
    pub args: Vec<CorInfoType>,
    pub has_this: bool,
}

/// A canned method the mock resolves metadata tokens to.
pub struct MockMethod {
    /// The fake handle `resolve_token`/`get_call_info` hand back.
    pub handle: MethodHandle,
    pub sig: MockSig,
    /// Index into the mock's arg-list table (drives sig-cursor walking).
    arg_list: usize,
}

/// Canned EE. Every query returns the stored/default value; output sinks
/// record what they were handed so tests can assert on the flow.
#[derive(Default)]
pub struct MockEe {
    pub method_attribs: MethodAttribs,
    pub class_attribs: ClassAttribs,
    pub method_name: Option<String>,
    /// Canned methods for `resolve_token`/`get_call_info`/`get_method_sig`,
    /// keyed by metadata token.
    pub methods: HashMap<u32, MockMethod>,
    /// Canned directly-callable entry points for `get_function_entry_point`
    /// (step_07.5 codegen tests), keyed by the method handle's raw value.
    /// Absent handles get a zeroed lookup (`IAT_VALUE`, null address).
    pub entry_points: HashMap<usize, usize>,
    /// Canned entry-point *slots* for `get_function_entry_point` (step_07.7
    /// codegen tests of the IAT_PVALUE indirect-call form), keyed like
    /// [`Self::entry_points`] and consulted after it.
    pub entry_point_slots: HashMap<usize, usize>,
    /// Registered signature argument lists. Fake `ArgListHandle` cursors
    /// encode `(list, index)` — the mock never dereferences handles.
    arg_lists: Vec<Vec<CorInfoType>>,
    /// Sink calls observed, newest last, as "(kind, detail)" strings.
    pub sink_log: RefCell<Vec<String>>,
    /// Buffers handed out by the fake `alloc_mem`/`alloc_gc_info`, kept
    /// alive for the mock's lifetime.
    buffers: RefCell<Vec<Box<[u8]>>>,
}

// Cursors pack a (list, index) pair into the opaque handle value; the +1
// keeps the first cursor of each list non-null.
const CURSOR_INDEX_BITS: usize = 20;

impl MockEe {
    fn cursor_raw(list: usize, index: usize) -> ffi::CORINFO_ARG_LIST_HANDLE {
        (((list + 1) << CURSOR_INDEX_BITS) | index) as ffi::CORINFO_ARG_LIST_HANDLE
    }

    fn cursor(list: usize, index: usize) -> Option<ArgListHandle> {
        ArgListHandle::from_raw(Self::cursor_raw(list, index))
    }

    fn decode_cursor(cursor: ArgListHandle) -> (usize, usize) {
        let raw = cursor.as_raw() as usize;
        (
            (raw >> CURSOR_INDEX_BITS) - 1,
            raw & ((1 << CURSOR_INDEX_BITS) - 1),
        )
    }

    /// Builds the `CORINFO_SIG_INFO` mirror over an already-registered
    /// argument list.
    fn build_sig_info(
        &self,
        call_conv: ffi::CorInfoCallConv,
        ret: CorInfoType,
        arg_list: usize,
    ) -> ffi::CORINFO_SIG_INFO {
        let mut sig: ffi::CORINFO_SIG_INFO = unsafe { std::mem::zeroed() };
        sig.callConv = call_conv;
        sig.set_retType(ret.to_raw());
        sig.set_numArgs(self.arg_lists[arg_list].len() as u32);
        sig.args = Self::cursor_raw(arg_list, 0);
        sig
    }

    /// The `CORINFO_SIG_INFO` for a registered method.
    fn method_sig_info(&self, method: &MockMethod) -> ffi::CORINFO_SIG_INFO {
        let call_conv = if method.sig.has_this {
            ffi::CorInfoCallConv_CORINFO_CALLCONV_HASTHIS
        } else {
            ffi::CorInfoCallConv_CORINFO_CALLCONV_DEFAULT
        };
        self.build_sig_info(call_conv, method.sig.ret, method.arg_list)
    }

    /// Registers a canned method under `token`; returns its fake handle.
    pub fn add_method(&mut self, token: u32, sig: MockSig) -> MethodHandle {
        let arg_list = self.arg_lists.len();
        self.arg_lists.push(sig.args.clone());
        // Non-null stand-in; the mock never dereferences handles.
        let raw = (0x1000 + 0x10 * self.methods.len()) as ffi::CORINFO_METHOD_HANDLE;
        let handle = MethodHandle::from_raw(raw).expect("fake handle is non-null");
        self.methods.insert(
            token,
            MockMethod {
                handle,
                sig,
                arg_list,
            },
        );
        handle
    }

    /// Builds the argument-signature mirror for a method the mock doesn't
    /// resolve tokens to — e.g. the entry method's `MethodInfo::args`.
    pub fn make_method_sig(&mut self, sig: &MockSig) -> ffi::CORINFO_SIG_INFO {
        let arg_list = self.arg_lists.len();
        self.arg_lists.push(sig.args.clone());
        let method = MockMethod {
            handle: MethodHandle::from_raw(1usize as ffi::CORINFO_METHOD_HANDLE)
                .expect("fake handle is non-null"),
            sig: sig.clone(),
            arg_list,
        };
        self.method_sig_info(&method)
    }

    /// Builds the locals-signature mirror (`CORINFO_CALLCONV_LOCAL_SIG`)
    /// for `MethodInfo::locals` — the only source of IL local types.
    pub fn make_locals_sig(&mut self, locals: &[CorInfoType]) -> ffi::CORINFO_SIG_INFO {
        let arg_list = self.arg_lists.len();
        self.arg_lists.push(locals.to_vec());
        self.build_sig_info(
            ffi::CorInfoCallConv_CORINFO_CALLCONV_LOCAL_SIG,
            CorInfoType::Void,
            arg_list,
        )
    }
}

impl MockEe {
    fn fake_alloc(&self, size: usize) -> NonNull<u8> {
        let mut buf = vec![0u8; size.max(1)].into_boxed_slice();
        let ptr = NonNull::new(buf.as_mut_ptr()).unwrap();
        self.buffers.borrow_mut().push(buf);
        ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_method_handle() -> MethodHandle {
        // Non-null stand-in; the mock never dereferences handles.
        MethodHandle::from_raw(1usize as ffi::CORINFO_METHOD_HANDLE).unwrap()
    }

    /// The acceptance test: `MockEe` instantiates and the trait is usable
    /// through the `EeInfo` supertrait with no EE anywhere.
    #[test]
    fn mock_ee_implements_ee_info() {
        let ee = MockEe {
            method_name: Some("Fib".into()),
            ..MockEe::default()
        };
        // Exercise it as the core would: through the composed supertrait.
        fn core_view(ee: &dyn EeInfo, ftn: MethodHandle) -> (MethodAttribs, Option<String>) {
            (
                ee.get_method_attribs(ftn),
                ee.get_method_name_from_metadata(ftn),
            )
        }
        let (attribs, name) = core_view(&ee, fake_method_handle());
        assert_eq!(attribs, MethodAttribs::EMPTY);
        assert_eq!(name.as_deref(), Some("Fib"));
    }

    #[test]
    fn sinks_record_in_order() {
        let ee = MockEe::default();
        ee.reserve_unwind_info(false, false, 32);
        let chunks = ee.alloc_mem(
            &[ChunkRequest {
                alignment: 16,
                size: 64,
                flags: AllocMemFlags::HOT_CODE,
            }],
            0,
        );
        assert_eq!(chunks.len(), 1);
        ee.set_eh_count(0);
        let log = ee.sink_log.borrow();
        assert_eq!(
            log.as_slice(),
            [
                "reserve_unwind_info(false, false, 32)",
                "alloc_mem(1, xcptns=0)",
                "set_eh_count(0)",
            ]
        );
    }
}
