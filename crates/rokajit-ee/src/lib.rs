//! `rokajit-ee` — the safe(ish) Rust side of the EE↔JIT boundary, plus the
//! carrier for the C++ ABI gasket.
//!
//! # The gasket
//!
//! All vtable-shaped code is C++ (`src/gasket_*.cpp`, built by `build.rs` via
//! the `cc` crate) so that vtable layout is correct by construction
//! (`decisions/2026-09-11-cpp-abi-gasket.md`). The gasket is logic-free: it
//! forwards `ICorJitCompiler` calls into the `rokajit` crate's
//! `extern "C"` entry points, and exposes `rokajit_ee_*` forwarders into
//! the EE's `ICorJitInfo` for the Rust wrappers below.
//!
//! # The safe surface
//!
//! The compiler core (`rokajit`) never sees a raw C pointer. It consumes:
//!
//! - [`ee_info::EeInfo`] — the safe trait view of the 183-method
//!   `ICorJitInfo` chain, split into per-functional-group sub-traits;
//! - [`host::EeHost`] — the safe view of `ICorJitHost` (config access);
//! - [`handles`] — `#[repr(transparent)]` newtypes over the bindgen
//!   `CORINFO_*_HANDLE` pointer aliases;
//! - [`enums`] — idiomatic Rust enums/flag newtypes over the bindgen
//!   C++ enum constants.
//!
//! # Handle newtyping policy (frozen; `decisions/2026-09-11-handle-newtyping.md`)
//!
//! 1. One transparent newtype per handle kind; no cross-kind transmutes.
//! 2. Null handles are `Option<Handle>`; a bare handle value is always
//!    non-null, checked once by `from_raw` at the FFI edge.
//! 3. `Debug` prints kind + address (`MethodHandle(0x7f…)`).
//! 4. Handles are EE-owned and valid only for the duration of one
//!    `compileMethod` call; nothing may store them longer (raw pointers
//!    cannot carry this lifetime, so it is a documented invariant).
//! 5. Handles are `Copy`/`Eq`/`Hash` by address and neither `Send` nor
//!    `Sync`.
//!
//! # What this crate deliberately does not contain
//!
//! No compiler logic, no allocation policy, no caching — all behavior lives
//! in `rokajit`. The concrete `EeInfo` implementation that talks to a real
//! EE through the gasket forwarders is [`ee_info::GasketEeInfo`] (step 04);
//! `MockEe` (test-only) proves the trait implementable without a live EE.

#[macro_use]
pub mod enums;
pub mod handles;
pub mod ee_info;
pub mod host;

#[cfg(test)]
mod mock;
