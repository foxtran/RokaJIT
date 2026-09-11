use rokajit_ffi::{CORINFO_FIELD_INFO, CORINFO_RESOLVED_TOKEN};

use crate::enums::CorInfoType;
use crate::handles::{ClassHandle, FieldHandle, MethodHandle};

/// Field metadata queries (C++ `ICorFieldInfo`).
pub trait FieldQueries {
    /// C++ `ICorFieldInfo::getFieldType` (corinfo.h:2939). The C++
    /// `structType` out-param (set for value-class fields) becomes the
    /// second tuple element. The `fieldOwnerHint` parameter is not exposed
    /// (RyuJIT passes null outside exotic diagnostics paths).
    fn get_field_type(&self, field: FieldHandle) -> (CorInfoType, Option<ClassHandle>);

    /// C++ `ICorFieldInfo::getFieldOffset` (corinfo.h:2946).
    fn get_field_offset(&self, field: FieldHandle) -> u32;

    /// C++ `ICorFieldInfo::isFieldStatic` (corinfo.h:2974).
    fn is_field_static(&self, field: FieldHandle) -> bool;

    /// C++ `ICorFieldInfo::getFieldInfo` (corinfo.h:2950): how to access a
    /// resolved field. `flags` is the raw `CORINFO_ACCESS_FLAGS` word
    /// (corinfo.h:622); it gets a flag newtype when the first consumer
    /// lands.
    fn get_field_info(
        &self,
        token: &mut CORINFO_RESOLVED_TOKEN,
        caller: MethodHandle,
        flags: u32,
    ) -> CORINFO_FIELD_INFO;
}
