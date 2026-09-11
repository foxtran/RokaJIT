use rokajit_ffi::{CORINFO_EH_CLAUSE, CORINFO_SIG_INFO};

use crate::enums::MethodAttribs;
use crate::handles::{ClassHandle, MethodHandle};

/// Method metadata queries (C++ `ICorMethodInfo`).
pub trait MethodQueries {
    /// C++ `ICorMethodInfo::getMethodAttribs` (corinfo.h:2166).
    fn get_method_attribs(&self, ftn: MethodHandle) -> MethodAttribs;

    /// C++ `ICorMethodInfo::getMethodSig` (corinfo.h:2180). `member_parent`
    /// is the C++ `memberParent` out-of-scope hint; `None` = null.
    fn get_method_sig(
        &self,
        ftn: MethodHandle,
        member_parent: Option<ClassHandle>,
    ) -> CORINFO_SIG_INFO;

    /// C++ `ICorMethodInfo::getMethodClass` (corinfo.h:2287). Infallible: the
    /// EE always knows a method's owning class.
    fn get_method_class(&self, ftn: MethodHandle) -> ClassHandle;

    /// C++ `ICorMethodInfo::getEHinfo` (corinfo.h:2280). `index` runs over
    /// `CORINFO_METHOD_INFO::EHcount`.
    fn get_eh_info(&self, ftn: MethodHandle, index: u32) -> CORINFO_EH_CLAUSE;

    /// C++ `ICorMethodInfo::getMethodHash` (corinfo.h:3239) — debug/range
    /// knobs only.
    fn get_method_hash(&self, ftn: MethodHandle) -> u32;

    /// C++ `ICorMethodInfo::methodMustBeLoadedBeforeCodeIsRun`
    /// (corinfo.h:2376).
    fn method_must_be_loaded_before_code_is_run(&self, ftn: MethodHandle);

    /// C++ `ICorMethodInfo::getMethodNameFromMetadata` (corinfo.h:3228).
    /// `None` when the method has no metadata name (the C++ nullptr). The
    /// wrapper copies the string; the C++ storage is EE-lifetime, so there
    /// is nothing to free.
    fn get_method_name_from_metadata(&self, ftn: MethodHandle) -> Option<String>;
}
