use crate::enums::{ClassAttribs, CorInfoType};
use crate::handles::ClassHandle;

/// Class (type) metadata queries (C++ `ICorClassInfo`).
pub trait ClassQueries {
    /// C++ `ICorClassInfo::asCorInfoType` (corinfo.h:2483).
    fn as_cor_info_type(&self, cls: ClassHandle) -> CorInfoType;

    /// C++ `ICorClassInfo::isValueClass` (corinfo.h:2519).
    fn is_value_class(&self, cls: ClassHandle) -> bool;

    /// C++ `ICorClassInfo::getClassAttribs` (corinfo.h:2522).
    fn get_class_attribs(&self, cls: ClassHandle) -> ClassAttribs;

    /// C++ `ICorClassInfo::getClassSize` (corinfo.h:2559).
    fn get_class_size(&self, cls: ClassHandle) -> u32;

    /// C++ `ICorClassInfo::getTypeForPrimitiveNumericClass`
    /// (corinfo.h:2809). The C++ `CORINFO_TYPE_UNDEF` sentinel becomes
    /// `None`.
    fn get_type_for_primitive_numeric_class(&self, cls: ClassHandle) -> Option<CorInfoType>;
}
