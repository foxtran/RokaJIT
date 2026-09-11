//! `MockEe` — an `EeInfo` implementation with canned answers, proving the
//! trait is implementable without a live EE (and serving as the test double
//! for compiler-core unit tests). Test-only.

use std::cell::RefCell;
use std::ptr::NonNull;

use rokajit_ffi as ffi;

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

/// Canned EE. Every query returns the stored/default value; output sinks
/// record what they were handed so tests can assert on the flow.
#[derive(Default)]
pub struct MockEe {
    pub method_attribs: MethodAttribs,
    pub class_attribs: ClassAttribs,
    pub method_name: Option<String>,
    /// Sink calls observed, newest last, as "(kind, detail)" strings.
    pub sink_log: RefCell<Vec<String>>,
    /// Buffers handed out by the fake `alloc_mem`/`alloc_gc_info`, kept
    /// alive for the mock's lifetime.
    buffers: RefCell<Vec<Box<[u8]>>>,
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
