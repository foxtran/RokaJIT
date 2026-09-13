//! The compilation artifact — what a successful `compileMethod` produces
//! and hands back across the gasket (frozen;
//! `decisions/2026-09-11-compilation-artifact.md`).
//!
//! This is the core↔ee boundary value: the core builds it, the FFI edge in
//! `lib.rs` drains it into the EE's output sinks. Every field exists
//! because a specific sink consumes it — the mapping is:
//!
//! | Artifact field | EE sink (corjit.h) |
//! |---|---|
//! | [`CompilationArtifact::code`] | `allocMem` (chunk sizes/alignments), then the code bytes are copied into the returned chunks |
//! | [`CompilationArtifact::ro_data`] | `allocMem` (`READONLY_DATA` chunks) |
//! | [`CompilationArtifact::unwind`] | `reserveUnwindInfo` (sizes, before `allocMem`), `allocUnwindInfo` |
//! | [`CompilationArtifact::gc_info`] | `allocGCInfo` (size), then the blob is copied in |
//! | [`CompilationArtifact::eh_clauses`] | `setEHcount` (len), `setEHinfo` (each) |
//! | [`CompilationArtifact::il_map`] | `setBoundaries` (when non-empty) |
//! | [`CompilationArtifact::relocations`] | `recordRelocation` (each) |
//! | [`CompilationArtifact::call_sites`] | `recordCallSite` (each) |
//!
//! Everything here is **owned** (`Vec`s, offsets, newtyped handles) — no
//! pointers into EE memory, no borrows of the `EeInfo` object, so the value
//! is freely movable between the core and the edge. Relocation/call-site
//! targets that are EE handles stay handles; addresses obtained from the EE
//! during compilation are plain `usize` values (data, not pointers).

use rokajit_ee::enums::{AllocMemFlags, CorJitFuncKind, EhClauseFlags, RelocType};
use rokajit_ee::handles::MethodHandle;

/// A successful compilation, ready to be drained into the EE's sinks in
/// sink order (see the table above).
pub struct CompilationArtifact {
    /// Code chunks: exactly one hot chunk, at most one cold chunk.
    pub code: CodeChunks,
    /// Read-only data chunks (jump tables, constant pools, GC-string
    /// literals…). Emitted after the code chunks in the `allocMem` request.
    pub ro_data: Vec<DataChunk>,
    /// Unwind blobs, one per function fragment (root + funclets + cold
    /// code), in the target's unwind encoding (x64: `RUNTIME_FUNCTION` +
    /// unwind codes).
    pub unwind: Vec<UnwindBlob>,
    /// The encoded GC info blob (GcInfoEncoder format), consumed verbatim
    /// by `allocGCInfo`. Empty only for methods with no GC pointers
    /// anywhere — and even then the EE wants a valid minimal encoding.
    pub gc_info: Vec<u8>,
    /// Native-offset EH clauses, in `setEHinfo` order (`index` = position
    /// in this vec).
    pub eh_clauses: Vec<EhClause>,
    /// The IL→native offset map for the debugger, in `setBoundaries` order.
    /// Empty means no debug info is emitted (the edge skips the sink call);
    /// codegen does not produce mappings yet — the channel exists and is
    /// drained, the contents arrive with the debug-info step.
    pub il_map: Vec<IlMapEntry>,
    /// Relocations to record, in native-offset order.
    pub relocations: Vec<Relocation>,
    /// Managed call sites to record, in native-offset order.
    pub call_sites: Vec<CallSite>,
}

/// The code chunks handed to `allocMem`.
pub struct CodeChunks {
    pub hot: CodeChunk,
    pub cold: Option<CodeChunk>,
}

/// One emitted code chunk: contents plus the `allocMem` parameters
/// (`AllocMemChunk`, corjit.h:79) it was produced for.
pub struct CodeChunk {
    pub bytes: Vec<u8>,
    /// Hot-code alignment ≤ 32, cold-code alignment = 1 (corjit.h:81).
    pub alignment: u32,
}

/// One read-only data chunk for `allocMem` (`CORJIT_ALLOCMEM_READONLY_DATA`
/// ± `HAS_POINTERS_TO_CODE`).
pub struct DataChunk {
    pub bytes: Vec<u8>,
    /// RO-data alignment ≤ 64 (corjit.h:81).
    pub alignment: u32,
    pub flags: AllocMemFlags,
}

/// The unwind info for one function fragment, consumed by
/// `allocUnwindInfo` (corjit.h:213).
pub struct UnwindBlob {
    /// Which fragment this describes (corjit.h:61).
    pub func_kind: CorJitFuncKind,
    /// Whether `pColdCode` (rather than `pHotCode`) is the base for the
    /// offsets.
    pub is_cold_code: bool,
    /// `[start, end)` native offsets relative to the fragment's chunk.
    pub start_offset: u32,
    pub end_offset: u32,
    /// The target-encoded unwind bytes (`pUnwindBlock`).
    pub bytes: Vec<u8>,
}

/// One EH clause with **native** offsets, drained to `setEHinfo`
/// (corjit.h:242) as a `CORINFO_EH_CLAUSE` (corinfo.h:1619).
///
/// RyuJIT's `genReportEH` repurposes the `CORINFO_EH_CLAUSE` length fields
/// as END offsets (runtime/src/coreclr/jit/codegencommon.cpp:2727-2789:
/// `clause.TryLength = tryEnd; clause.HandlerLength = hndEnd;`), and this
/// artifact follows that: `try_end`/`handler_end` are native END offsets
/// relative to the hot-chunk start, NOT lengths.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct EhClause {
    pub flags: EhClauseFlags,
    pub try_offset: u32,
    /// Native END offset of the try region, hot-chunk-relative (drained to
    /// `TryLength`; see the struct docs).
    pub try_end: u32,
    pub handler_offset: u32,
    /// Native END offset of the handler region, hot-chunk-relative (drained
    /// to `HandlerLength`; see the struct docs).
    pub handler_end: u32,
    /// The `ClassToken`/`FilterOffset` union member.
    pub class_or_filter: ClassTokenOrFilter,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ClassTokenOrFilter {
    /// The raw mdToken from `getEHinfo`'s `CORINFO_EH_CLAUSE::ClassToken`,
    /// carried straight through to `setEHinfo` — the VM resolves and
    /// type-tests it at dispatch; no JIT-side handle query exists.
    ClassToken(u32),
    FilterOffset(u32),
}

/// One IL→native offset pair (the row shape of `setBoundaries`'
/// `ICorDebugInfo::OffsetMapping`, cordebuginfo.h, minus the raw
/// `SourceTypes` word the edge fills with 0).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct IlMapEntry {
    pub il_offset: u32,
    pub native_offset: u32,
}

/// One relocation for `recordRelocation` (corjit.h:435).
pub struct Relocation {
    /// Native offset of the slot within its chunk…
    pub chunk: ChunkRef,
    /// …and the offset itself (the EE computes `location` = chunk base +
    /// offset).
    pub offset: u32,
    /// The absolute target address the slot must end up pointing at,
    /// obtained from the EE during compilation (helper entry points,
    /// string literals, field addresses…).
    pub target: usize,
    pub reloc_type: RelocType,
    pub addl_delta: i32,
}

/// One managed call site for `recordCallSite` (corjit.h:421).
pub struct CallSite {
    /// Native offset of the call instruction within its chunk.
    pub chunk: ChunkRef,
    pub offset: u32,
    /// The call instruction's length in bytes. The GC safepoint for the
    /// call is `offset + size` (the return address — the encoder's
    /// `callSite += callSiteSizes[...]`, gcinfoencoder.cpp:1100); the
    /// target's codegen records it (x64: 5 for `call rel32`, 6 for
    /// `call [rip+rel32]`).
    pub size: u32,
    /// The signature and method used to lay out the call site; both absent
    /// for helper calls and signature-less calli (C++ nullptrs).
    pub sig: Option<crate::ir::CallSig>,
    pub method: Option<MethodHandle>,
}

/// Which `allocMem` chunk an offset refers to.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ChunkRef {
    HotCode,
    ColdCode,
    /// Index into `CompilationArtifact::ro_data`.
    RoData(u32),
}
