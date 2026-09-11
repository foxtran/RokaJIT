use std::ptr::NonNull;

use rokajit_ffi as ffi;

use crate::handles::MethodHandle;

/// Where a PGO counter buffer came from (C++ `ICorJitInfo::PgoSource`,
/// corjit.h:364).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[repr(i32)]
pub enum PgoSource {
    Unknown = ffi::ICorJitInfo_PgoSource_Unknown,
    Static = ffi::ICorJitInfo_PgoSource_Static,
    Dynamic = ffi::ICorJitInfo_PgoSource_Dynamic,
    Blend = ffi::ICorJitInfo_PgoSource_Blend,
    Text = ffi::ICorJitInfo_PgoSource_Text,
    Ibc = ffi::ICorJitInfo_PgoSource_IBC,
    Sampling = ffi::ICorJitInfo_PgoSource_Sampling,
    Synthesis = ffi::ICorJitInfo_PgoSource_Synthesis,
}

/// One PGO instrumentation schema row (C++
/// `ICorJitInfo::PgoInstrumentationSchema`, corjit.h:355). `kind`/`other`
/// stay raw: the schema vocabulary (`PgoInstrumentationKind`) is shared
/// with the managed side and is interpreted by the PGO consumer, not the
/// trait.
#[derive(Copy, Clone, Debug)]
pub struct PgoSchemaItem {
    pub offset: usize,
    pub kind: u32,
    pub il_offset: i32,
    pub count: i32,
    pub other: i32,
}

/// Owned copy of what `getPgoInstrumentationResults` returns. The C++
/// pointers "will not remain valid after jit completes" (corjit.h:390), so
/// the wrapper copies schema + data into these `Vec`s immediately.
pub struct PgoResults {
    pub schema: Vec<PgoSchemaItem>,
    pub data: Vec<u8>,
    pub source: PgoSource,
    pub dynamic_pgo: bool,
}

/// Profile data (C++ `ICorJitInfo` PGO methods, corjit.h:387/409).
///
/// These are the only trait calls whose C++ form returns an HRESULT; the
/// raw `JITINTERFACE_HRESULT` is the `Err` payload.
pub trait Pgo {
    /// C++ `ICorJitInfo::getPgoInstrumentationResults` (corjit.h:387).
    fn get_pgo_instrumentation_results(
        &self,
        ftn: MethodHandle,
    ) -> Result<PgoResults, i32>;

    /// C++ `ICorJitInfo::allocPgoInstrumentationBySchema` (corjit.h:409).
    /// The EE fills in `offset` on each schema row; the returned pointer is
    /// the instrumentation buffer (EE-owned).
    fn alloc_pgo_instrumentation_by_schema(
        &self,
        ftn: MethodHandle,
        schema: &mut [PgoSchemaItem],
    ) -> Result<NonNull<u8>, i32>;
}
