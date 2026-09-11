use crate::handles::MethodHandle;

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
}
