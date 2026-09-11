//! The `EeHost` trait — the safe Rust view of `ICorJitHost`
//! (corjithost.h:14): process-lifetime services handed to `jitStartup`,
//! independent of any single compilation.

/// C++ `ICorJitHost` (corjithost.h:14). The host object outlives the JIT
/// ("lives at least as long as the JIT itself", corjithost.h:12), so this
/// trait carries no lifetime parameter.
///
/// Ownership rule (frozen): **strings cross as owned `String`s**. C++
/// `getStringConfigValue` returns a pointer the caller must hand back to
/// `freeStringConfigValue` (corjithost.h:34-40); the wrapper copies to a
/// `String` and frees the original before returning, so the core never
/// holds EE-allocated memory.
pub trait EeHost {
    /// C++ `ICorJitHost::getIntConfigValue` (corjithost.h:24).
    fn get_int_config_value(&self, name: &str, default: i32) -> i32;

    /// C++ `ICorJitHost::getStringConfigValue` (corjithost.h:30), with the
    /// result copied out and the original freed via `freeStringConfigValue`
    /// inside the wrapper. `None` = the C++ nullptr (no value configured).
    fn get_string_config_value(&self, name: &str) -> Option<String>;
}
