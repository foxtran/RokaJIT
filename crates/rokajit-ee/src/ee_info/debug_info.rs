use std::ffi::CStr;
use std::mem::size_of;
use std::ptr::NonNull;

use rokajit_ffi as ffi;

use crate::handles::{JustMyCodeHandle, MethodHandle, ProfilingHandle};

/// One IL→native offset pair for the debugger (the row shape of C++
/// `ICorDebugInfo::setBoundaries`, corinfo.h:3007; the C++ array element is
/// `ICorDebugInfo::OffsetMapping` in cordebuginfo.h).
pub struct BoundaryMap {
    pub il_offset: u32,
    pub native_offset: u32,
    /// C++ `ICorDebugInfo::SourceTypes` word; kept raw until the debug
    /// consumer lands.
    pub source: u32,
}

/// One local-variable home for the debugger (row shape of C++
/// `ICorDebugInfo::setVars`, corinfo.h:3034; element type
/// `ICorDebugInfo::NativeVarInfo` in cordebuginfo.h). Location fields stay
/// as the raw `vlType`-tagged words until the debug consumer lands.
pub struct NativeVarInfo {
    pub start_offset: u32,
    pub end_offset: u32,
    pub var_number: u32,
    pub loc_type: u32,
    pub loc_words: [u32; 3],
}

/// Debug info flow. `get_*` queries are hints from the EE; `set_*` are
/// output sinks the JIT fills after emission. Buffers the EE hands back via
/// `get_*` are freed by the wrapper (`freeArray`); the trait surface only
/// ever sees owned `Vec`s.
pub trait DebugInfo {
    /// C++ `ICorDebugInfo::getBoundaries` (corinfo.h:2992): interesting IL
    /// offsets for breakpoints.
    fn get_boundaries(&self, ftn: MethodHandle) -> Vec<u32>;

    /// C++ `ICorDebugInfo::setBoundaries` (corinfo.h:3007).
    fn set_boundaries(&self, ftn: MethodHandle, map: &[BoundaryMap]);

    /// C++ `ICorDebugInfo::getVars` (corinfo.h:3022): IL local/arg homes.
    fn get_vars(&self, ftn: MethodHandle) -> Vec<u32>;

    /// C++ `ICorDebugInfo::setVars` (corinfo.h:3034).
    fn set_vars(&self, ftn: MethodHandle, vars: &[NativeVarInfo]);

    /// C++ `ICorDebugInfo::reportRichMappings` (corinfo.h:3044): the inline
    /// tree plus rich IL→native mappings. The wrapper copies both slices
    /// into `allocateArray` memory; ownership transfers to the EE with the
    /// call.
    fn report_rich_mappings(
        &self,
        inline_tree_nodes: &[ffi::ICorDebugInfo_InlineTreeNode],
        mappings: &[ffi::ICorDebugInfo_RichOffsetMapping],
    );

    /// C++ `ICorDebugInfo::reportAsyncDebugInfo` (corinfo.h:3054).
    /// `async_info` (the `NumSuspensionPoints` header word) is borrowed for
    /// the call only; the two arrays are copied into `allocateArray` memory
    /// and ownership transfers to the EE. `suspension_points.len()` must
    /// equal `async_info.NumSuspensionPoints`.
    fn report_async_debug_info(
        &self,
        async_info: &ffi::ICorDebugInfo_AsyncInfo,
        suspension_points: &[ffi::ICorDebugInfo_AsyncSuspensionPoint],
        vars: &[ffi::ICorDebugInfo_AsyncContinuationVarInfo],
    );

    /// C++ `ICorStaticInfo::reportMetadata` (corinfo.h:3063): metrics sink.
    /// `key` is a short static string (the JIT passes literals);
    /// `value` is opaque bytes the EE copies before returning.
    fn report_metadata(&self, key: &CStr, value: &[u8]);

    /// C++ `ICorDebugInfo::allocateArray` (corinfo.h:3073): EE-heap buffer
    /// for data handed back through `set_boundaries`/`set_vars`/
    /// `report_rich_mappings`/`report_async_debug_info`. A conforming EE
    /// reports OOM by exception/longjmp and never returns null; `None`
    /// covers the pathological null sentinel (only reachable via the
    /// gasket's exception trap).
    fn allocate_array(&self, bytes: usize) -> Option<NonNull<u8>>;

    /// C++ `ICorDebugInfo::freeArray` (corinfo.h:3081): release an
    /// EE-allocated buffer (`get_boundaries`/`get_vars` results, or an
    /// `allocate_array` buffer that was never handed to the EE).
    fn free_array(&self, array: NonNull<u8>);

    /// C++ `ICorDynamicInfo::getJustMyCodeHandle` (corinfo.h:3400). The EE
    /// answers through exactly one channel: `(Some(handle), None)` is a
    /// direct handle, `(None, Some(ptr))` is an indirection to a
    /// process-lifetime handle cell; `(None, None)` means the method is not
    /// JustMyCode-instrumented (RyuJIT asserts `!dbgHandle || !pDbgHandle`,
    /// flowgraph.cpp:2614).
    fn get_just_my_code_handle(
        &self,
        method: MethodHandle,
    ) -> (
        Option<JustMyCodeHandle>,
        Option<NonNull<ffi::CORINFO_JUST_MY_CODE_HANDLE>>,
    );

    /// C++ `ICorDynamicInfo::GetProfilingHandle` (corinfo.h:3408). Returns
    /// `(hook_function, handle, indirected_handles)`: whether the profiler
    /// wants the enter/leave hook called, the process-unique profiling
    /// handle (native IP or descriptor address; `None` when not profiling),
    /// and whether the handle is indirected.
    fn get_profiling_handle(&self) -> (bool, Option<ProfilingHandle>, bool);

    /// C++ `ICorDynamicInfo::MethodCompileComplete` (corinfo.h:3509):
    /// notification sink fired once per completed compilation.
    fn method_compile_complete(&self, meth_hnd: MethodHandle);
}

extern "C" {
    fn rokajit_ee_get_boundaries(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        c_il_offsets: *mut u32,
        p_il_offsets: *mut *mut u32,
        implicit_boundaries: *mut ffi::ICorDebugInfo_BoundaryTypes,
    );
    fn rokajit_ee_set_boundaries(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        c_map: u32,
        p_map: *mut ffi::ICorDebugInfo_OffsetMapping,
    );
    fn rokajit_ee_get_vars(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        c_vars: *mut u32,
        vars: *mut *mut ffi::ICorDebugInfo_ILVarInfo,
        extend_others: *mut bool,
    );
    fn rokajit_ee_set_vars(
        info: *mut ffi::ICorJitInfo,
        ftn: ffi::CORINFO_METHOD_HANDLE,
        c_vars: u32,
        vars: *mut ffi::ICorDebugInfo_NativeVarInfo,
    );
    fn rokajit_ee_report_rich_mappings(
        info: *mut ffi::ICorJitInfo,
        inline_tree_nodes: *mut ffi::ICorDebugInfo_InlineTreeNode,
        num_inline_tree_nodes: u32,
        mappings: *mut ffi::ICorDebugInfo_RichOffsetMapping,
        num_mappings: u32,
    );
    fn rokajit_ee_report_async_debug_info(
        info: *mut ffi::ICorJitInfo,
        async_info: *mut ffi::ICorDebugInfo_AsyncInfo,
        suspension_points: *mut ffi::ICorDebugInfo_AsyncSuspensionPoint,
        vars: *mut ffi::ICorDebugInfo_AsyncContinuationVarInfo,
        num_vars: u32,
    );
    fn rokajit_ee_report_metadata(
        info: *mut ffi::ICorJitInfo,
        key: *const std::os::raw::c_char,
        value: *const std::os::raw::c_void,
        length: usize,
    );
    fn rokajit_ee_allocate_array(
        info: *mut ffi::ICorJitInfo,
        c_bytes: usize,
    ) -> *mut std::os::raw::c_void;
    fn rokajit_ee_free_array(info: *mut ffi::ICorJitInfo, array: *mut std::os::raw::c_void);
    fn rokajit_ee_get_just_my_code_handle(
        info: *mut ffi::ICorJitInfo,
        method: ffi::CORINFO_METHOD_HANDLE,
        pp_indirection: *mut *mut ffi::CORINFO_JUST_MY_CODE_HANDLE,
    ) -> ffi::CORINFO_JUST_MY_CODE_HANDLE;
    fn rokajit_ee_get_profiling_handle(
        info: *mut ffi::ICorJitInfo,
        pb_hook_function: *mut bool,
        p_profiler_handle: *mut *mut std::os::raw::c_void,
        pb_indirected_handles: *mut bool,
    );
    fn rokajit_ee_method_compile_complete(
        info: *mut ffi::ICorJitInfo,
        meth_hnd: ffi::CORINFO_METHOD_HANDLE,
    );
}

impl DebugInfo for super::GasketEeInfo {
    fn get_boundaries(&self, ftn: MethodHandle) -> Vec<u32> {
        unsafe {
            let mut count = 0u32;
            let mut offsets: *mut u32 = std::ptr::null_mut();
            // The implicitBoundaries out-param is not surfaced by the
            // step_02 trait signature; the EE still wants a slot to write.
            let mut implicit_boundaries: ffi::ICorDebugInfo_BoundaryTypes = 0;
            rokajit_ee_get_boundaries(
                self.comp_raw(),
                ftn.as_raw(),
                &mut count,
                &mut offsets,
                &mut implicit_boundaries,
            );
            if offsets.is_null() {
                return Vec::new();
            }
            let result = std::slice::from_raw_parts(offsets, count as usize).to_vec();
            // C++ obligation (corinfo.h:2996): "jit MUST free with freeArray".
            rokajit_ee_free_array(self.comp_raw(), offsets.cast());
            result
        }
    }

    fn set_boundaries(&self, ftn: MethodHandle, map: &[BoundaryMap]) {
        unsafe {
            let count = map.len() as u32;
            // C++ contract (corinfo.h:3011): pMap is "jit allocated with
            // allocateArray, EE frees".
            let buf = if map.is_empty() {
                std::ptr::null_mut()
            } else {
                match self.allocate_array(map.len() * size_of::<ffi::ICorDebugInfo_OffsetMapping>())
                {
                    Some(p) => p.as_ptr().cast::<ffi::ICorDebugInfo_OffsetMapping>(),
                    None => return,
                }
            };
            for (i, m) in map.iter().enumerate() {
                buf.add(i).write(ffi::ICorDebugInfo_OffsetMapping {
                    nativeOffset: m.native_offset,
                    ilOffset: m.il_offset,
                    source: m.source,
                });
            }
            rokajit_ee_set_boundaries(self.comp_raw(), ftn.as_raw(), count, buf);
        }
    }

    fn get_vars(&self, ftn: MethodHandle) -> Vec<u32> {
        unsafe {
            let mut count = 0u32;
            let mut vars: *mut ffi::ICorDebugInfo_ILVarInfo = std::ptr::null_mut();
            // extendOthers is not surfaced by the step_02 trait signature.
            let mut extend_others = false;
            rokajit_ee_get_vars(
                self.comp_raw(),
                ftn.as_raw(),
                &mut count,
                &mut vars,
                &mut extend_others,
            );
            if vars.is_null() {
                return Vec::new();
            }
            // The step_02 surface keeps only the var numbers; the IL scope
            // extents ride along in the C++ array but have no Rust home yet.
            let result = std::slice::from_raw_parts(vars, count as usize)
                .iter()
                .map(|v| v.varNumber)
                .collect();
            // C++ obligation (corinfo.h:3026): "jit MUST free with freeArray".
            rokajit_ee_free_array(self.comp_raw(), vars.cast());
            result
        }
    }

    fn set_vars(&self, ftn: MethodHandle, vars: &[NativeVarInfo]) {
        unsafe {
            let count = vars.len() as u32;
            // C++ contract (corinfo.h:3038): "jit allocated with
            // allocateArray, EE frees". RyuJIT still calls setVars with a
            // null buffer when the count is zero (ee_il_dll.cpp eeSetLVdone).
            let buf = if vars.is_empty() {
                std::ptr::null_mut()
            } else {
                match self
                    .allocate_array(vars.len() * size_of::<ffi::ICorDebugInfo_NativeVarInfo>())
                {
                    Some(p) => p.as_ptr().cast::<ffi::ICorDebugInfo_NativeVarInfo>(),
                    None => return,
                }
            };
            for (i, v) in vars.iter().enumerate() {
                // The Rust mirror keeps the location as a vlType tag plus
                // the union's three raw words; reassemble the bindgen
                // VarLoc by writing the words over the union storage (the
                // union is 12 bytes, alignment 4 — see the bindgen layout
                // assertions). No transmutes anywhere.
                let mut loc: ffi::ICorDebugInfo_VarLoc = std::mem::zeroed();
                loc.vlType = v.loc_type;
                let words = (&mut loc.__bindgen_anon_1
                    as *mut ffi::ICorDebugInfo_VarLoc__bindgen_ty_1)
                    .cast::<u32>();
                words.add(0).write(v.loc_words[0]);
                words.add(1).write(v.loc_words[1]);
                words.add(2).write(v.loc_words[2]);
                buf.add(i).write(ffi::ICorDebugInfo_NativeVarInfo {
                    startOffset: v.start_offset,
                    endOffset: v.end_offset,
                    // Only async methods use this field (scopeinfo.cpp:1973);
                    // ordinary rows get 0 (scopeinfo.cpp:2172).
                    callReturnValueILOffset: 0,
                    varNumber: v.var_number,
                    loc,
                });
            }
            rokajit_ee_set_vars(self.comp_raw(), ftn.as_raw(), count, buf);
        }
    }

    fn report_rich_mappings(
        &self,
        inline_tree_nodes: &[ffi::ICorDebugInfo_InlineTreeNode],
        mappings: &[ffi::ICorDebugInfo_RichOffsetMapping],
    ) {
        unsafe {
            // C++ contract (corinfo.h:3042): both arrays are allocateArray
            // memory, ownership transferred to the EE by this call.
            let nodes = match copy_to_ee_array(self, inline_tree_nodes) {
                Some(p) => p,
                None => return,
            };
            let maps = match copy_to_ee_array(self, mappings) {
                Some(p) => p,
                None => return,
            };
            rokajit_ee_report_rich_mappings(
                self.comp_raw(),
                nodes,
                inline_tree_nodes.len() as u32,
                maps,
                mappings.len() as u32,
            );
        }
    }

    fn report_async_debug_info(
        &self,
        async_info: &ffi::ICorDebugInfo_AsyncInfo,
        suspension_points: &[ffi::ICorDebugInfo_AsyncSuspensionPoint],
        vars: &[ffi::ICorDebugInfo_AsyncContinuationVarInfo],
    ) {
        unsafe {
            // asyncInfo stays a borrowed pointer (RyuJIT passes a
            // stack local, codegencommon.cpp:7258); only the two arrays
            // transfer ownership.
            let susp = match copy_to_ee_array(self, suspension_points) {
                Some(p) => p,
                None => return,
            };
            let host_vars = match copy_to_ee_array(self, vars) {
                Some(p) => p,
                None => return,
            };
            rokajit_ee_report_async_debug_info(
                self.comp_raw(),
                async_info as *const _ as *mut _,
                susp,
                host_vars,
                vars.len() as u32,
            );
        }
    }

    fn report_metadata(&self, key: &CStr, value: &[u8]) {
        unsafe {
            rokajit_ee_report_metadata(
                self.comp_raw(),
                key.as_ptr(),
                value.as_ptr().cast(),
                value.len(),
            );
        }
    }

    fn allocate_array(&self, bytes: usize) -> Option<NonNull<u8>> {
        unsafe { NonNull::new(rokajit_ee_allocate_array(self.comp_raw(), bytes).cast::<u8>()) }
    }

    fn free_array(&self, array: NonNull<u8>) {
        unsafe { rokajit_ee_free_array(self.comp_raw(), array.as_ptr().cast()) }
    }

    fn get_just_my_code_handle(
        &self,
        method: MethodHandle,
    ) -> (
        Option<JustMyCodeHandle>,
        Option<NonNull<ffi::CORINFO_JUST_MY_CODE_HANDLE>>,
    ) {
        unsafe {
            let mut indirection: *mut ffi::CORINFO_JUST_MY_CODE_HANDLE = std::ptr::null_mut();
            let raw = rokajit_ee_get_just_my_code_handle(
                self.comp_raw(),
                method.as_raw(),
                &mut indirection,
            );
            (JustMyCodeHandle::from_raw(raw), NonNull::new(indirection))
        }
    }

    fn get_profiling_handle(&self) -> (bool, Option<ProfilingHandle>, bool) {
        unsafe {
            let mut hook_function = false;
            let mut handle: *mut std::os::raw::c_void = std::ptr::null_mut();
            let mut indirected_handles = false;
            rokajit_ee_get_profiling_handle(
                self.comp_raw(),
                &mut hook_function,
                &mut handle,
                &mut indirected_handles,
            );
            (
                hook_function,
                ProfilingHandle::from_raw(handle.cast()),
                indirected_handles,
            )
        }
    }

    fn method_compile_complete(&self, meth_hnd: MethodHandle) {
        unsafe { rokajit_ee_method_compile_complete(self.comp_raw(), meth_hnd.as_raw()) }
    }
}

/// Copy a slice into EE-allocated memory (`allocateArray`), for the
/// sink-style reports whose contract is "JIT allocates with allocateArray,
/// EE frees". Empty slices map to null (count 0); `None` is the
/// allocateArray OOM sentinel.
fn copy_to_ee_array<T: Copy>(ee: &super::GasketEeInfo, items: &[T]) -> Option<*mut T> {
    if items.is_empty() {
        return Some(std::ptr::null_mut());
    }
    let buf = ee
        .allocate_array(std::mem::size_of_val(items))?
        .as_ptr()
        .cast::<T>();
    unsafe { std::ptr::copy_nonoverlapping(items.as_ptr(), buf, items.len()) };
    Some(buf)
}
