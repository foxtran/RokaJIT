//! The compilation pipeline — stage boundaries and the driver (frozen;
//! `decisions/2026-09-11-pipeline-and-target-contracts.md`).
//!
//! One `compileMethod` request flows through five stages, each a free
//! function with typed input/output so it is individually testable:
//!
//! ```text
//! MethodInfo ──▶ import ──▶ morph ──▶ lower ──▶ codegen ──▶ build_metadata
//!                  │          │         │          │              │
//!               hir::Method hir::Method lir::Method CodegenOutput MetadataOutput
//! ```
//!
//! | Stage | Step | In → Out |
//! |---|---|---|
//! | [`import`] | 07.2 | [`MethodInfo`] + IL bytes → [`hir::Method`] |
//! | [`morph`] | 07.3 | `hir::Method` → `hir::Method` (call-arg and return normalization; pure IR transform) |
//! | [`lower`] | 07.4 | `hir::Method` → [`lir::Method`], target-parameterized |
//! | [`codegen`] | 07.5 (+ 07.6 encoder) | `lir::Method` → [`CodegenOutput`] |
//! | [`build_metadata`] | 07.7 | [`CodegenOutput`] + `lir::Method` → [`MetadataOutput`] |
//!
//! [`compile`] is the only caller of the stages; it exists so the FFI edge
//! (`lib.rs`) has exactly one safe entry point and so the artifact is
//! assembled from stage outputs in exactly one place.
//!
//! Only the signatures and the data crossing the boundaries are frozen
//! here. Stages land one sub-step at a time; unlanded stages are stubs
//! returning `CompileError::Unsupported` naming their sub-step.

use rokajit_ee::ee_info::EeInfo;
use rokajit_ee::handles::MethodHandle;
use rokajit_ffi::CORINFO_SIG_INFO;

use crate::artifact::{
    CallSite, CodeChunks, CompilationArtifact, DataChunk, EhClause, Relocation, UnwindBlob,
};
use crate::error::{CompileError, CompileResult};
use crate::ir::{hir, lir};
use crate::target::Target;

/// What the pipeline compiles: an owned snapshot of the EE's
/// `CORINFO_METHOD_INFO`, built by the FFI edge (the pointer-chasing copy
/// lives there, where `unsafe` is allowed).
///
/// Signature blobs (`args`, `locals`) cross as the bindgen mirror structs
/// by value, per the frozen `EeInfo` convention; their pointer fields are
/// valid for the duration of the compilation, like handles.
pub struct MethodInfo {
    /// The method being compiled (`CORINFO_METHOD_INFO::ftn`) — the handle
    /// all per-method EE queries (`get_method_attribs`, `get_eh_info`, …)
    /// key on.
    pub ftn: MethodHandle,
    /// The CIL byte stream, copied out of `ILCode`/`ILCodeSize`.
    pub il: Vec<u8>,
    /// `maxStack` from the method header: the evaluation-stack high-water
    /// mark the importer's stack discipline checks against.
    pub max_stack: u32,
    /// `EHcount`: the number of clauses to fetch via
    /// `EeInfo::get_eh_info`. Zero for fib.
    pub eh_count: u32,
    /// `CORINFO_OPT_INIT_LOCALS`: IL locals must be zero-initialized on
    /// entry.
    pub init_locals: bool,
    /// The argument signature (`CORINFO_METHOD_INFO::args`).
    pub args: CORINFO_SIG_INFO,
    /// The locals signature (`CORINFO_METHOD_INFO::locals`) — the only
    /// source of IL local types; no `EeInfo` query returns it.
    pub locals: CORINFO_SIG_INFO,
}

/// The compilation tier (docs/JITs/README.md verdict 2: one compiler with
/// tier knobs — the tiers differ in which passes run, not in which compiler
/// runs). Plumbed through [`compile`] from day one even though only tier 0
/// exists.
///
/// `#[non_exhaustive]`: adding tier 1 (linear-scan regalloc) must not break
/// downstream `match`es.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum Tier {
    /// Winch-style baseline: single pass over LIR, no liveness, round-robin
    /// registers, GC refs frame-resident so root reporting is static
    /// (verdict 3).
    Tier0,
}

/// What [`codegen`] produces: the emitted bytes plus everything
/// [`build_metadata`] needs to encode GC info, unwind, and EH tables. All
/// fields are target-generic vocabulary from [`crate::artifact`].
pub struct CodegenOutput {
    /// Hot (+ optional cold) code chunks, ready for `allocMem`.
    pub code: CodeChunks,
    /// Read-only data chunks (jump tables, constant pools).
    pub ro_data: Vec<DataChunk>,
    /// Relocations recorded during emission; targets are addresses obtained
    /// from the EE during compilation.
    pub relocations: Vec<Relocation>,
    /// Managed call sites recorded during emission. These are also the GC
    /// safepoints: the metadata stage keys GC info off their native
    /// offsets.
    pub call_sites: Vec<CallSite>,
    /// Frame and GC-root facts for the metadata stage.
    pub frame: FrameInfo,
}

/// Target-generic facts about the compiled frame, recorded by codegen.
pub struct FrameInfo {
    /// Frame size in bytes (IL locals + temps + spill area), excluding the
    /// return address and any saved frame pointer.
    pub frame_size: u32,
    /// Frame slots that hold GC pointers. In tier 0 every GC-ref local is
    /// frame-resident for its whole scope (Winch-style), so this one set is
    /// the root set at *every* safepoint; per-safepoint liveness arrives
    /// with tier 1 as a contract extension.
    pub gc_roots: Vec<GcRootSlot>,
}

/// One frame slot holding a GC pointer.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct GcRootSlot {
    /// Byte offset of the slot relative to the frame base defined by the
    /// target's frame contract (x64 tier 0: the frame pointer).
    pub offset: u32,
    /// `true` for an interior pointer ([`crate::ir::Type::ByRef`]); the GC
    /// keeps the containing object alive and updates the pointer on moves.
    pub is_byref: bool,
    /// `true` if the slot is pinned (mirrors [`hir::Local::pinned`]).
    pub pinned: bool,
}

/// What [`build_metadata`] produces: the remaining artifact fields, in the
/// target's EE-facing encodings. The encoding entry points themselves are
/// `Target` extensions frozen by step_07.7 (the "metadata builder API"
/// decision), not here.
pub struct MetadataOutput {
    /// Unwind blobs, one per function fragment (see
    /// [`CompilationArtifact::unwind`]).
    pub unwind: Vec<UnwindBlob>,
    /// The encoded GC info blob (see [`CompilationArtifact::gc_info`]).
    /// Valid even when the method has no GC roots: the EE requires a
    /// parseable minimal encoding for every method.
    pub gc_info: Vec<u8>,
    /// Native-offset EH clauses (see [`CompilationArtifact::eh_clauses`]).
    pub eh_clauses: Vec<EhClause>,
}

/// The pipeline driver: the core's one entry point, called by the FFI edge
/// in `lib.rs`.
///
/// `target` is a parameter — not a global — because the `rokajit` cdylib
/// cannot name a concrete backend (Cargo forbids the rokajit ↔ rokajit-x64
/// dependency cycle), and because a parameter keeps every stage testable
/// against mock targets. Wiring the concrete target into the cdylib is a
/// later integration step (see the decisions file).
pub fn compile(
    info: &MethodInfo,
    ee: &dyn EeInfo,
    target: &dyn Target,
    tier: Tier,
) -> CompileResult<CompilationArtifact> {
    let hir = import(info, ee)?;
    let hir = morph(hir)?;
    let lir = lower(hir, target)?;
    let output = codegen(&lir, ee, target, tier)?;
    let metadata = build_metadata(&output, &lir, target)?;
    Ok(CompilationArtifact {
        code: output.code,
        ro_data: output.ro_data,
        relocations: output.relocations,
        call_sites: output.call_sites,
        unwind: metadata.unwind,
        gc_info: metadata.gc_info,
        eh_clauses: metadata.eh_clauses,
    })
}

/// Stage 1 (step_07.2): CIL → HIR. Resolves every token through
/// `EeInfo::resolve_token`; guarantees the `ir::hir` invariants (well-typed
/// trees, stack-height discipline, control flow only on terminators).
/// Implemented in [`crate::import`].
pub fn import(info: &MethodInfo, ee: &dyn EeInfo) -> CompileResult<hir::Method> {
    crate::import::import(info, ee)
}

/// Stage 2 (step_07.3): morph-lite — call-argument and return
/// normalization. A pure `hir::Method → hir::Method` transform producing a
/// new value (no in-place mutation); it needs no EE queries and no target
/// knowledge. Implemented in [`crate::morph`].
pub fn morph(method: hir::Method) -> CompileResult<hir::Method> {
    crate::morph::morph(method)
}

/// Stage 3 (step_07.4): HIR → LIR lowering. The generic driver lives in
/// [`crate::lower`]; the target parameter gates legality (register-class
/// coverage). Instruction selection is a separate, backend-owned step:
/// backend rule sets (written in `rokajit::lower_rules!`, e.g.
/// `rokajit_x64::lower`) map `lir` statements to machine-instruction
/// descriptors; codegen (07.5) drives them.
pub fn lower(method: hir::Method, target: &dyn Target) -> CompileResult<lir::Method> {
    crate::lower::lower(method, target)
}

/// Stage 4 (step_07.5, encoding via step_07.6's `Target` extension):
/// LIR → machine code. Tier 0 is Winch-style: single pass, no liveness,
/// round-robin registers, GC refs frame-resident. The EE parameter exists
/// because emission resolves call/helper addresses whose results land in
/// [`CodegenOutput::relocations`]. Implemented in [`crate::codegen`].
pub fn codegen(
    method: &lir::Method,
    ee: &dyn EeInfo,
    target: &dyn Target,
    tier: Tier,
) -> CompileResult<CodegenOutput> {
    crate::codegen::codegen(method, ee, target, tier)
}

/// Stage 5 (step_07.7): encode GC info, unwind, and EH tables from the
/// facts codegen recorded, in the target's EE-facing encodings. The stage
/// boundary is frozen here; the encoding API on `Target` is step_07.7's
/// "metadata builder API" decision.
pub fn build_metadata(
    output: &CodegenOutput,
    method: &lir::Method,
    target: &dyn Target,
) -> CompileResult<MetadataOutput> {
    let _ = (output, method, target);
    Err(CompileError::Unsupported(
        "metadata: implemented in step_07.7",
    ))
}
