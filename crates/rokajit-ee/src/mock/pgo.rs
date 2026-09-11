use std::ptr::NonNull;

use rokajit_ffi as ffi;

use super::MockEe;
use crate::ee_info::{Pgo, PgoResults, PgoSchemaItem};
use crate::handles::MethodHandle;

impl Pgo for MockEe {
    fn get_pgo_instrumentation_results(&self, _ftn: MethodHandle) -> Result<PgoResults, i32> {
        // E_INVALIDARG-ish: the mock has no profile data.
        Err(-2147024809)
    }

    fn alloc_pgo_instrumentation_by_schema(
        &self,
        _ftn: MethodHandle,
        schema: &mut [PgoSchemaItem],
    ) -> Result<NonNull<u8>, i32> {
        for (i, item) in schema.iter_mut().enumerate() {
            item.offset = i * 8;
        }
        Ok(self.fake_alloc(schema.len() * 8))
    }

    fn record_wasm_managed_call_sig(&self, _call_sig: &ffi::CORINFO_SIG_INFO) {
        self.sink_log
            .borrow_mut()
            .push("record_wasm_managed_call_sig()".into());
    }
}
