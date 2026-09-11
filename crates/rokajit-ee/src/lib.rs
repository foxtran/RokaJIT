//! Carrier crate for the C++ ABI gasket.
//!
//! All gasket code is C++ (`src/gasket.cpp`, built by `build.rs` via the
//! `cc` crate) so that vtable layout is correct by construction. There is no
//! Rust API here; the `rokajit` cdylib links the gasket archive and calls
//! its `extern "C"` entry points.
