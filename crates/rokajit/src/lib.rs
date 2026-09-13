//! RokaJIT — a Rust reimplementation of RyuJIT, the CoreCLR JIT compiler.
//!
//! This crate is the compiler **core**: the pipeline, the IR, the generic
//! codegen machinery, and the artifact. It is compiled as an rlib; the
//! `cdylib` that is a drop-in replacement for CoreCLR's `libclrjit.so` is
//! the thin `rokajit-cdy` crate at the top of the workspace, which owns the
//! `extern "C"` entry points the gasket forwards to and wires in the
//! concrete target (`rokajit-x64`) — the core cannot name it (Cargo forbids
//! the dependency cycle; `decisions/2026-09-11-pipeline-and-target-contracts.md`).
//!
//! # Error model (frozen; `decisions/2026-09-11-error-model.md`)
//!
//! - The compiler core fails with [`error::CompileError`]; the FFI edge
//!   (`rokajit-cdy`) maps it to the EE's `CorJitResult` via
//!   [`error::CompileError::to_cor_jit_result`] — the single mapping table.
//! - Panics are bugs, never control flow. `catch_unwind` appears only at
//!   the edge's `extern "C"` entry points. A caught panic is reported to the
//!   EE via `EeInfo::report_fatal_error(CorJitResult::InternalError)` when
//!   the EE info object is available, then surfaces as
//!   `CORJIT_INTERNALERROR`.
//!
//! # Contracts
//!
//! - [`ir`] — the HIR/LIR IR shapes (docs: `RokaJIT-internal/docs/ir-design.md`).
//! - [`artifact`] — what a successful `compileMethod` produces and hands to
//!   the EE's output sinks.
//! - [`pipeline`] — the compilation pipeline: stage boundaries (import →
//!   morph → lower → codegen → metadata) and the [`pipeline::compile`]
//!   driver.
//! - [`metadata`] — the single metadata channel: GC maps, EH clauses,
//!   unwind info, and IL-offset maps all drain through
//!   [`metadata::MetadataBuilder`].
//! - [`target`] — the `Target` trait; everything machine-specific lives
//!   behind it, in backend crates (`rokajit-x64`).
//! - The safe EE surface (`EeInfo`, handles, enums) lives in `rokajit-ee`;
//!   this crate depends on it, never the other way.

pub mod artifact;
pub mod codegen;
pub mod config;
pub mod config_table;
pub mod error;
pub mod import;
pub mod ir;
pub mod lower;
pub mod metadata;
pub mod morph;
pub mod pipeline;
pub mod structs;
pub mod target;
