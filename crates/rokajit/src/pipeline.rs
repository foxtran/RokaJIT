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
//! here. Stages landed one sub-step at a time (07.2–07.7); every stage now
//! delegates to its implementing module.

use rokajit_ee::ee_info::EeInfo;
use rokajit_ee::enums::CorJitFuncKind;
use rokajit_ee::handles::MethodHandle;
use rokajit_ffi::CORINFO_SIG_INFO;

use crate::artifact::{
    CallSite, CodeChunks, CompilationArtifact, DataChunk, EhClause, Relocation, UnwindBlob,
};
use crate::error::CompileResult;
use crate::ir::{hir, lir, GenericsContext};
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
    /// The CorInfoOptions generics bits (corinfo.h:709-715): how a shared
    /// generic body receives its instantiation context — through `this`
    /// (FROM_THIS), or through the hidden context argument as a
    /// MethodDesc* (FROM_METHODDESC) or MethodTable* (FROM_METHODTABLE).
    /// `None` when no CORINFO_GENERICS_CTXT_* bit is set (non-shared
    /// code; step_11.3B).
    pub generics_context: Option<GenericsContext>,
    /// `CORINFO_GENERICS_CTXT_KEEP_ALIVE` (corinfo.h:715): the context
    /// must stay reported (and, for FROM_THIS, alive) for the method's
    /// whole extent.
    pub generics_context_keep_alive: bool,
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
    /// Funclets in emission order; emitted after the main body in the hot
    /// chunk.
    pub funclets: Vec<FuncletInfo>,
    /// EH clauses with native hot-relative offsets, in VM order
    /// (innermost-first; SAMETRY already applied to flags by codegen).
    pub eh_clauses: Vec<EhClause>,
    /// Fully-interruptible ranges `[start, end)`, native hot-relative,
    /// sorted, disjoint. EMPTY = partially interruptible method (the slim
    /// GC-info header).
    pub interruptible_ranges: Vec<(u32, u32)>,
}

/// One funclet (an EH handler body emitted after the main body), described
/// for the unwind/GC-info encoders.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct FuncletInfo {
    /// Native `[start, end)` offsets, hot-chunk-relative.
    pub start_offset: u32,
    pub end_offset: u32,
    /// Length of the funclet prolog (`sub rsp, N`); also the unwind code's
    /// offset.
    pub prolog_len: u8,
    /// Bytes the funclet prolog subtracts from rsp (8-aligned, ≥ 8).
    pub sp_delta: u32,
    /// Always [`CorJitFuncKind::Handler`] today (filters are Unsupported).
    pub kind: CorJitFuncKind,
}

/// Target-generic facts about the compiled frame, recorded by codegen.
pub struct FrameInfo {
    /// Frame size in bytes (IL locals + temps + spill area), excluding the
    /// return address and any saved frame pointer.
    pub frame_size: u32,
    /// The frame's outgoing-argument area in bytes — the maximum
    /// `CallAbi::stack_arg_bytes` over the method's call sites (0 when no
    /// call overflows the register pools). The fat GC header reports it
    /// (`SizeOfStackOutgoingAndScratchArea`).
    pub outgoing_bytes: u32,
    /// Frame slots that hold GC pointers. In tier 0 every GC-ref local is
    /// frame-resident for its whole scope (Winch-style), so this one set is
    /// the root set at *every* safepoint; per-safepoint liveness arrives
    /// with tier 1 as a contract extension.
    pub gc_roots: Vec<GcRootSlot>,
    /// The generics-context slot the GC info reports (step_11.3B), when
    /// the method carries one — its presence forces the fat header.
    pub generics_context: Option<GenericsContextGcInfo>,
}

/// The GC-info encoding facts for a method's generics context
/// (step_11.3B; gcinfoencoder.cpp:936-1046).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct GenericsContextGcInfo {
    /// The context slot's frame offset: the negative of its
    /// bytes-below-rbp — the same convention as the untracked slot
    /// table's `GcStackSlot.SpOffset`.
    pub slot_offset: i32,
    /// Which context it is (the fat header's 2-bit contextParamType:
    /// MT=1, MD=2, THIS=3 — gcinfodecoder.h:241-245).
    pub kind: GenericsContext,
    /// Native code offset after the incoming-argument homing stores: the
    /// point the context slot becomes reportable, encoded as the fat
    /// header's prolog size (`varl_u(normPrologSize - 1)`).
    pub prolog_end: u32,
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
/// target's EE-facing encodings (GC info, unwind — the `Target` extension
/// methods frozen by step_07.7's "metadata builder API" decision) plus the
/// target-independent sections (EH clauses, IL-offset map).
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
    /// The IL→native offset map (see [`CompilationArtifact::il_map`]).
    pub il_map: Vec<crate::artifact::IlMapEntry>,
}

/// The pipeline driver: the core's one entry point, called by the FFI edge
/// in `lib.rs`.
///
/// `target` is a parameter — not a global — because the compiler core
/// cannot name a concrete backend (Cargo forbids the rokajit ↔ rokajit-x64
/// dependency cycle), and because a parameter keeps every stage testable
/// against mock targets. The cdylib edge (`rokajit-cdy`) wires in the
/// concrete `X64Target` (step_07.7).
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
        il_map: metadata.il_map,
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

/// Stage 5 (step_07.7): encode GC info, unwind, EH tables, and the
/// IL-offset map from the facts codegen recorded. All four sections drain
/// through the one metadata channel ([`crate::metadata::MetadataBuilder`]);
/// the EE-facing encodings (GC info, unwind) are `Target` extensions.
/// Implemented in [`crate::metadata`].
pub fn build_metadata(
    output: &CodegenOutput,
    method: &lir::Method,
    target: &dyn Target,
) -> CompileResult<MetadataOutput> {
    crate::metadata::build_metadata(output, method, target)
}
