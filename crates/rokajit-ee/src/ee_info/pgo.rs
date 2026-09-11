use std::ptr::NonNull;

use rokajit_ffi as ffi;

use super::GasketEeInfo;
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

// Hand-expansion of the `ffi_enum!` conversion impl (the `crate::enums`
// macros are not exported, and `PgoSource` itself stays in this group file).
// Small closed enum per the frozen enum policy.
impl PgoSource {
    /// Converts a raw ABI value, returning `None` for values not in the
    /// enum (i.e. the headers grew new variants).
    #[inline]
    pub fn from_raw(raw: i32) -> Option<Self> {
        if raw == ffi::ICorJitInfo_PgoSource_Unknown {
            return Some(Self::Unknown);
        }
        if raw == ffi::ICorJitInfo_PgoSource_Static {
            return Some(Self::Static);
        }
        if raw == ffi::ICorJitInfo_PgoSource_Dynamic {
            return Some(Self::Dynamic);
        }
        if raw == ffi::ICorJitInfo_PgoSource_Blend {
            return Some(Self::Blend);
        }
        if raw == ffi::ICorJitInfo_PgoSource_Text {
            return Some(Self::Text);
        }
        if raw == ffi::ICorJitInfo_PgoSource_IBC {
            return Some(Self::Ibc);
        }
        if raw == ffi::ICorJitInfo_PgoSource_Sampling {
            return Some(Self::Sampling);
        }
        if raw == ffi::ICorJitInfo_PgoSource_Synthesis {
            return Some(Self::Synthesis);
        }
        None
    }

    /// The raw ABI value, for calls into the gasket forwarders.
    #[inline]
    pub const fn to_raw(self) -> i32 {
        self as i32
    }
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

    /// C++ `ICorJitInfo::recordWasmManagedCallSig` (corjit.h:429). Sink
    /// style: no failure channel. A no-op on all targets except ReadyToRun
    /// Wasm compilation.
    fn record_wasm_managed_call_sig(&self, call_sig: &ffi::CORINFO_SIG_INFO);
}

extern "C" {
    fn rokajit_ee_get_pgo_instrumentation_results(
        info: *mut ffi::ICorJitInfo,
        ftn_hnd: ffi::CORINFO_METHOD_HANDLE,
        p_schema: *mut *mut ffi::ICorJitInfo_PgoInstrumentationSchema,
        p_count_schema_items: *mut u32,
        p_instrumentation_data: *mut *mut u8,
        p_pgo_source: *mut ffi::ICorJitInfo_PgoSource,
        p_dynamic_pgo: *mut bool,
    ) -> ffi::JITINTERFACE_HRESULT;
    fn rokajit_ee_alloc_pgo_instrumentation_by_schema(
        info: *mut ffi::ICorJitInfo,
        ftn_hnd: ffi::CORINFO_METHOD_HANDLE,
        p_schema: *mut ffi::ICorJitInfo_PgoInstrumentationSchema,
        count_schema_items: u32,
        p_instrumentation_data: *mut *mut u8,
    ) -> ffi::JITINTERFACE_HRESULT;
    fn rokajit_ee_record_wasm_managed_call_sig(
        info: *mut ffi::ICorJitInfo,
        call_sig: *mut ffi::CORINFO_SIG_INFO,
    );
}

/// Size of one schema row's data entry, mirroring the EE's
/// `InstrumentationKindToSize` (pgo_formatprocessing.h:34): the low nibble
/// (`MarshalMask`) selects the payload width. Unknown nibbles count as 0,
/// exactly like the EE's release-build fallback.
fn instrumentation_kind_size(kind: u32) -> usize {
    match kind & 0xF {
        1 => 4,                                      // FourByte
        2 => 8,                                      // EightByte
        3 | 4 => std::mem::size_of::<usize>(),       // TypeHandle / MethodHandle
        _ => 0,                                      // None / unknown
    }
}

impl Pgo for GasketEeInfo {
    fn get_pgo_instrumentation_results(
        &self,
        ftn: MethodHandle,
    ) -> Result<PgoResults, i32> {
        let mut schema_ptr: *mut ffi::ICorJitInfo_PgoInstrumentationSchema =
            std::ptr::null_mut();
        let mut count_schema_items: u32 = 0;
        let mut data_ptr: *mut u8 = std::ptr::null_mut();
        let mut pgo_source: ffi::ICorJitInfo_PgoSource = ffi::ICorJitInfo_PgoSource_Unknown;
        let mut dynamic_pgo = false;
        let hr = unsafe {
            rokajit_ee_get_pgo_instrumentation_results(
                self.comp_raw(),
                ftn.as_raw(),
                &mut schema_ptr,
                &mut count_schema_items,
                &mut data_ptr,
                &mut pgo_source,
                &mut dynamic_pgo,
            )
        };
        if hr < 0 {
            // `dynamic_pgo` is valid even on failure, but the step_02-frozen
            // `Result` surface has nowhere to carry it — it is dropped here.
            return Err(hr);
        }

        let schema: Vec<PgoSchemaItem> = if count_schema_items == 0 {
            Vec::new()
        } else {
            assert!(
                !schema_ptr.is_null(),
                "EE contract: getPgoInstrumentationResults returned {count_schema_items} items but a null schema"
            );
            unsafe { std::slice::from_raw_parts(schema_ptr, count_schema_items as usize) }
                .iter()
                .map(|raw| PgoSchemaItem {
                    offset: raw.Offset,
                    kind: raw.InstrumentationKind as u32,
                    il_offset: raw.ILOffset,
                    count: raw.Count,
                    other: raw.Other,
                })
                .collect()
        };

        // The buffer length is not returned; the EE sizes it from the schema
        // (vm/pgo.cpp:729): the highest `Offset + Count * entry size`.
        let data_len = schema
            .iter()
            .map(|item| {
                item.offset.saturating_add(
                    (item.count.max(0) as usize).saturating_mul(instrumentation_kind_size(item.kind)),
                )
            })
            .max()
            .unwrap_or(0);
        let data: Vec<u8> = if data_len == 0 {
            Vec::new()
        } else {
            assert!(
                !data_ptr.is_null(),
                "EE contract: getPgoInstrumentationResults schema implies {data_len} data bytes but a null buffer"
            );
            unsafe { std::slice::from_raw_parts(data_ptr, data_len) }.to_vec()
        };

        let source = PgoSource::from_raw(pgo_source)
            .expect("EE returned a PgoSource outside the pinned header set");
        Ok(PgoResults {
            schema,
            data,
            source,
            dynamic_pgo,
        })
    }

    fn alloc_pgo_instrumentation_by_schema(
        &self,
        ftn: MethodHandle,
        schema: &mut [PgoSchemaItem],
    ) -> Result<NonNull<u8>, i32> {
        let mut raw_schema: Vec<ffi::ICorJitInfo_PgoInstrumentationSchema> = schema
            .iter()
            .map(|item| ffi::ICorJitInfo_PgoInstrumentationSchema {
                Offset: item.offset,
                InstrumentationKind: item.kind as ffi::ICorJitInfo_PgoInstrumentationKind,
                ILOffset: item.il_offset,
                Count: item.count,
                Other: item.other,
            })
            .collect();
        let mut data_ptr: *mut u8 = std::ptr::null_mut();
        let hr = unsafe {
            rokajit_ee_alloc_pgo_instrumentation_by_schema(
                self.comp_raw(),
                ftn.as_raw(),
                raw_schema.as_mut_ptr(),
                schema.len() as u32,
                &mut data_ptr,
            )
        };
        if hr < 0 {
            return Err(hr);
        }
        // The VM fills in `Offset` on each row; copy the rows back.
        for (item, raw) in schema.iter_mut().zip(raw_schema.iter()) {
            item.offset = raw.Offset;
            item.kind = raw.InstrumentationKind as u32;
            item.il_offset = raw.ILOffset;
            item.count = raw.Count;
            item.other = raw.Other;
        }
        Ok(NonNull::new(data_ptr)
            .expect("EE contract: allocPgoInstrumentationBySchema succeeded but returned a null buffer"))
    }

    fn record_wasm_managed_call_sig(&self, call_sig: &ffi::CORINFO_SIG_INFO) {
        // IN parameter; the cast to *mut satisfies the C++ signature.
        let call_sig = call_sig as *const _ as *mut _;
        unsafe { rokajit_ee_record_wasm_managed_call_sig(self.comp_raw(), call_sig) };
    }
}
