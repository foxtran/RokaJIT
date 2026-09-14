//! The metadata channel (step_07.7;
//! `decisions/2026-09-11-metadata-builder-api.md`).
//!
//! Verdict 5 of `docs/JITs/README.md` (Graal infopoints / J9 stack atlas):
//! **one** builder API through which every side table drains — GC maps, EH
//! clauses, unwind info, and IL-offset maps. Codegen records facts
//! ([`CodegenOutput`]); [`pipeline::build_metadata`] feeds them into the
//! builder; [`MetadataBuilder::finish`] renders every section, calling the
//! target for the two EE-facing encodings (GC info, unwind) and producing
//! the target-independent sections (EH clauses, IL-offset map) itself.
//!
//! The channel is the architecture; for fib the contents are minimal: two
//! call-site safepoints, an empty root set, one root-function unwind blob,
//! no EH, no IL map. What must never change is that a new side table is a
//! new section on this builder, never a new ad-hoc channel.

use crate::artifact::{EhClause, IlMapEntry};
use crate::error::{CompileError, CompileResult};
use crate::ir::lir;
use crate::pipeline::{
    CodegenOutput, FuncletInfo, GcRootSlot, GenericsContextGcInfo, MetadataOutput,
};
use crate::target::Target;

/// The GC-info encoder's input, in target-generic vocabulary. The target
/// owns the GcInfoEncoder-format rendering (x64: `rokajit_x64::gcinfo`).
///
/// Offsets are native hot-code offsets. GC-root slot offsets are relative
/// to the frame base defined by the target's frame contract (see
/// [`GcRootSlot`]); how that base is declared to the GC-info decoder (x64:
/// the slim header's stack-base-register bit, rbp) is the target's
/// business.
pub struct GcInfoInput {
    /// Hot-code length in bytes — the TOTAL length (main body + funclets)
    /// when the method has EH.
    pub code_len: u32,
    /// Frame size in bytes, excluding the return address and any saved
    /// frame pointer (same definition as [`crate::pipeline::FrameInfo`]).
    pub frame_size: u32,
    /// The static GC-root set. Tier 0 keeps every GC ref frame-resident,
    /// so this one set is the root set at *every* safepoint.
    pub gc_roots: Vec<GcRootSlot>,
    /// The GC safepoints: native offsets of each managed call's *return
    /// address* (call-site offset + instruction length, both recorded by
    /// codegen in [`crate::artifact::CallSite`]), ascending. Ignored for
    /// fully-interruptible (EH) methods: the fat header carries
    /// NUM_SAFE_POINTS = 0.
    pub safepoints: Vec<u32>,
    /// Fully-interruptible ranges `[start, end)`, native hot-relative,
    /// sorted, disjoint. Non-empty ⇒ EH method: fat header, fully
    /// interruptible, WantsReportOnlyLeaf — and safepoints are NOT emitted.
    /// Empty ⇒ today's slim partially-interruptible encoding.
    pub interruptible_ranges: Vec<(u32, u32)>,
    /// Frame outgoing-argument area in bytes
    /// (`SizeOfStackOutgoingAndScratchArea`; written only in the fat
    /// header). Codegen's frame layout supplies it.
    pub outgoing_area_size: u32,
    /// The generics-context slot to report (step_11.3B); its presence
    /// forces the fat header.
    pub generics_context: Option<GenericsContextGcInfo>,
}

/// The unwind encoder's input, in target-generic vocabulary. The target
/// owns the prolog shape (x64 tier 0: the fixed three-instruction frame
/// contract), so the input carries facts, not instructions.
pub struct UnwindInput {
    /// Frame size in bytes (same definition as [`GcInfoInput::frame_size`]).
    pub frame_size: u32,
    /// MAIN-body code length: the root fragment's `[start, end)` is
    /// `[0, code_len)` — the main body only, NOT including funclets (they
    /// are separate fragments). Cold fragments arrive with their consumers.
    pub code_len: u32,
    /// Funclets in emission order (they follow the main body in the hot
    /// chunk); one unwind blob each, after the root blob.
    pub funclets: Vec<FuncletInfo>,
}

/// The single metadata channel. One builder per compilation; sections are
/// recorded in any order and rendered together by [`Self::finish`].
///
/// The builder is deliberately a plain value with `record_*` methods (not
/// a sink trait): there is exactly one producer (the pipeline's metadata
/// stage) and exactly one renderer (`finish`), and the value shape keeps
/// both unit-testable without a target or an EE.
pub struct MetadataBuilder {
    code_len: u32,
    frame_size: u32,
    gc_roots: Vec<GcRootSlot>,
    safepoints: Vec<u32>,
    eh_clauses: Vec<EhClause>,
    il_map: Vec<IlMapEntry>,
    funclets: Vec<FuncletInfo>,
    interruptible_ranges: Vec<(u32, u32)>,
    outgoing_area_size: u32,
    generics_context: Option<GenericsContextGcInfo>,
}

impl MetadataBuilder {
    /// Start the channel with the frame facts every section needs.
    pub fn new(code_len: u32, frame_size: u32, gc_roots: Vec<GcRootSlot>) -> Self {
        MetadataBuilder {
            code_len,
            frame_size,
            gc_roots,
            safepoints: Vec::new(),
            eh_clauses: Vec::new(),
            il_map: Vec::new(),
            funclets: Vec::new(),
            interruptible_ranges: Vec::new(),
            outgoing_area_size: 0,
            generics_context: None,
        }
    }

    /// Record a managed call site: a GC safepoint (the GC-info section keys
    /// off these) and the future home of deopt data (verdict 5). The
    /// safepoint is the return address: `offset + size`.
    pub fn record_safepoint_call(&mut self, offset: u32, size: u32) {
        self.safepoints.push(offset + size);
    }

    /// Record one EH clause with native offsets (drained to `setEHinfo`).
    pub fn record_eh_clause(&mut self, clause: EhClause) {
        self.eh_clauses.push(clause);
    }

    /// Record the EH shape the GC-info and unwind encoders need (10.6):
    /// the funclets (emission order) and the fully-interruptible ranges
    /// covering the main body and each funclet body. `outgoing_area_size`
    /// is the frame's outgoing-argument area in bytes.
    pub fn record_eh_shape(
        &mut self,
        funclets: Vec<FuncletInfo>,
        interruptible_ranges: Vec<(u32, u32)>,
        outgoing_area_size: u32,
    ) {
        self.funclets = funclets;
        self.interruptible_ranges = interruptible_ranges;
        self.outgoing_area_size = outgoing_area_size;
    }

    /// Record one IL→native offset pair (drained to `setBoundaries` when
    /// non-empty). No caller today: codegen does not yet track
    /// per-statement native offsets (07.4: terminator-derived statements
    /// carry `IL_OFFSET_NONE`).
    pub fn record_il_mapping(&mut self, il_offset: u32, native_offset: u32) {
        self.il_map.push(IlMapEntry {
            il_offset,
            native_offset,
        });
    }

    /// Record the generics-context slot the GC-info fat header reports
    /// (step_11.3B).
    pub fn record_generics_context(&mut self, context: Option<GenericsContextGcInfo>) {
        self.generics_context = context;
    }

    /// Render every section: GC info and unwind through the target's
    /// encoders (the EE-facing formats are target-owned), EH clauses and
    /// the IL-offset map target-independently.
    pub fn finish(self, target: &dyn Target) -> CompileResult<MetadataOutput> {
        let gc_info = target.encode_gc_info(&GcInfoInput {
            // TOTAL code length (main + funclets): funclets live in the hot
            // chunk after the main body.
            code_len: self.code_len,
            frame_size: self.frame_size,
            gc_roots: self.gc_roots,
            safepoints: self.safepoints,
            interruptible_ranges: self.interruptible_ranges,
            outgoing_area_size: self.outgoing_area_size,
            generics_context: self.generics_context,
        })?;
        // The root fragment covers the main body only: funclets follow it
        // in emission order, so the first funclet's start is the main
        // body's end.
        let main_len = self
            .funclets
            .first()
            .map_or(self.code_len, |f| f.start_offset);
        let unwind = target.encode_unwind_info(&UnwindInput {
            frame_size: self.frame_size,
            code_len: main_len,
            funclets: self.funclets,
        })?;
        Ok(MetadataOutput {
            unwind,
            gc_info,
            eh_clauses: self.eh_clauses,
            il_map: self.il_map,
        })
    }
}

/// Stage 5 body (step_07.7): feed codegen's facts into the channel and
/// render. The safepoints are codegen's managed call sites (its
/// [`CodegenOutput::call_sites`] double as the GC safepoints — the frozen
/// pipeline contract). EH clauses, funclets, and interruptible ranges come
/// from codegen as native facts (10.6) and drain through unchanged.
pub fn build_metadata(
    output: &CodegenOutput,
    _method: &lir::Method,
    target: &dyn Target,
) -> CompileResult<MetadataOutput> {
    if output.code.cold.is_some() {
        return Err(CompileError::Unsupported(
            "cold code fragments: a later step",
        ));
    }
    let mut builder = MetadataBuilder::new(
        output.code.hot.bytes.len() as u32,
        output.frame.frame_size,
        output.frame.gc_roots.clone(),
    );
    for site in &output.call_sites {
        builder.record_safepoint_call(site.offset, site.size);
    }
    for clause in &output.eh_clauses {
        builder.record_eh_clause(*clause);
    }
    // outgoing_area_size is codegen's frame fact (FrameInfo carries the
    // frame's outgoing-argument area; the fat GC header reports it).
    builder.record_eh_shape(
        output.funclets.clone(),
        output.interruptible_ranges.clone(),
        output.frame.outgoing_bytes,
    );
    builder.record_generics_context(output.frame.generics_context);
    builder.finish(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{
        CallSite, ChunkRef, ClassTokenOrFilter, CodeChunk, CodeChunks, UnwindBlob,
    };
    use crate::ir::{CallSig, Type};
    use crate::pipeline::FrameInfo;
    use crate::target::{CallAbi, RegClassId};
    use rokajit_ee::enums::{CorJitFuncKind, EhClauseFlags};

    /// A target whose encoders echo their input, so the builder's data
    /// flow is observable without any real encoding.
    struct EchoTarget;

    impl Target for EchoTarget {
        fn pointer_size(&self) -> u8 {
            8
        }
        fn register_classes(&self) -> &'static [crate::target::RegisterClass] {
            &[]
        }
        fn class_of(
            &self,
            _ty: Type,
            _layouts: &crate::structs::StructLayouts,
        ) -> Option<RegClassId> {
            Some(RegClassId(0))
        }
        fn classify_call(
            &self,
            _sig: &CallSig,
            _layouts: &crate::structs::StructLayouts,
        ) -> CompileResult<CallAbi> {
            Err(CompileError::Unsupported("echo target"))
        }
        fn call_site_stack_alignment(&self) -> u32 {
            16
        }
        fn encode_gc_info(&self, input: &GcInfoInput) -> CompileResult<Vec<u8>> {
            // Not an encoding — a structural echo the tests assert against.
            let mut blob = input.code_len.to_le_bytes().to_vec();
            blob.extend(input.frame_size.to_le_bytes());
            blob.extend(input.safepoints.iter().flat_map(|o| o.to_le_bytes()));
            blob.push(input.gc_roots.len() as u8);
            // Per root: the slot offset, then a flag byte (bit 0 = byref,
            // bit 1 = pinned) — offsets and flags, not just the count.
            for root in &input.gc_roots {
                blob.extend(root.offset.to_le_bytes());
                blob.push(u8::from(root.is_byref) | (u8::from(root.pinned) << 1));
            }
            // The EH facts: outgoing area, then the interruptible ranges.
            blob.extend(input.outgoing_area_size.to_le_bytes());
            blob.push(input.interruptible_ranges.len() as u8);
            for (start, end) in &input.interruptible_ranges {
                blob.extend(start.to_le_bytes());
                blob.extend(end.to_le_bytes());
            }
            Ok(blob)
        }
        fn encode_unwind_info(&self, input: &UnwindInput) -> CompileResult<Vec<UnwindBlob>> {
            let mut blobs = vec![UnwindBlob {
                func_kind: CorJitFuncKind::Root,
                is_cold_code: false,
                start_offset: 0,
                end_offset: input.code_len,
                bytes: input.frame_size.to_le_bytes().to_vec(),
            }];
            for f in &input.funclets {
                blobs.push(UnwindBlob {
                    func_kind: f.kind,
                    is_cold_code: false,
                    start_offset: f.start_offset,
                    end_offset: f.end_offset,
                    bytes: vec![f.prolog_len],
                });
            }
            Ok(blobs)
        }
    }

    fn codegen_output(code: &[u8], call_sites: &[u32]) -> CodegenOutput {
        CodegenOutput {
            code: CodeChunks {
                hot: CodeChunk {
                    bytes: code.to_vec(),
                    alignment: 16,
                },
                cold: None,
            },
            ro_data: Vec::new(),
            relocations: Vec::new(),
            call_sites: call_sites
                .iter()
                .map(|&offset| CallSite {
                    chunk: ChunkRef::HotCode,
                    offset,
                    size: 5,
                    sig: None,
                    method: None,
                })
                .collect(),
            frame: FrameInfo {
                frame_size: 32,
                outgoing_bytes: 0,
                gc_roots: Vec::new(),
                generics_context: None,
            },
            funclets: Vec::new(),
            eh_clauses: Vec::new(),
            interruptible_ranges: Vec::new(),
        }
    }

    use crate::structs::StructLayouts;

    fn empty_method() -> lir::Method {
        lir::Method {
            blocks: Vec::new(),
            locals: Vec::new(),
            eh_regions: Vec::new(),
            num_args: 0,
            num_il_locals: 0,
            struct_layouts: StructLayouts::new(),
            generics_context: None,
        }
    }

    /// The channel end to end: call sites become GC safepoints (return
    /// addresses, offset + size), frame facts reach both target encoders,
    /// the artifact sections assemble.
    #[test]
    fn builder_drains_all_sections_through_one_channel() {
        let output = codegen_output(&[0xAA; 73], &[32, 51]);
        let meta = build_metadata(&output, &empty_method(), &EchoTarget).expect("renders");

        assert_eq!(meta.gc_info[0..4], 73u32.to_le_bytes(), "code length");
        assert_eq!(meta.gc_info[4..8], 32u32.to_le_bytes(), "frame size");
        assert_eq!(meta.gc_info[8..12], 37u32.to_le_bytes(), "safepoint 1");
        assert_eq!(meta.gc_info[12..16], 56u32.to_le_bytes(), "safepoint 2");
        assert_eq!(meta.gc_info[16], 0, "no roots");
        assert_eq!(meta.unwind.len(), 1);
        assert_eq!(meta.unwind[0].end_offset, 73);
        assert_eq!(meta.unwind[0].bytes, 32u32.to_le_bytes());
        assert!(meta.eh_clauses.is_empty());
        assert!(meta.il_map.is_empty());
    }

    /// The ref map (step_10.4): every GC-root slot reaches the GC-info
    /// encoder with its offset and flags intact — a ref arg at its slot,
    /// a ref IL local, and a byref temp (an interior pointer).
    #[test]
    fn gc_root_slots_drain_with_offsets_and_flags() {
        let roots = vec![
            GcRootSlot {
                offset: 8,
                is_byref: false,
                pinned: false,
            },
            GcRootSlot {
                offset: 16,
                is_byref: false,
                pinned: false,
            },
            GcRootSlot {
                offset: 24,
                is_byref: true,
                pinned: false,
            },
        ];
        let mut output = codegen_output(&[0xAA; 16], &[8]);
        output.frame.gc_roots = roots.clone();
        let meta = build_metadata(&output, &empty_method(), &EchoTarget).expect("renders");

        // code_len, frame_size, one safepoint (8 + 5), then the roots.
        assert_eq!(meta.gc_info[0..4], 16u32.to_le_bytes());
        assert_eq!(meta.gc_info[4..8], 32u32.to_le_bytes());
        assert_eq!(meta.gc_info[8..12], 13u32.to_le_bytes());
        assert_eq!(meta.gc_info[12], 3, "three roots");
        let mut at = 13;
        for root in &roots {
            assert_eq!(meta.gc_info[at..at + 4], root.offset.to_le_bytes());
            assert_eq!(
                meta.gc_info[at + 4],
                u8::from(root.is_byref) | (u8::from(root.pinned) << 1),
                "flags for the slot at {}",
                root.offset
            );
            at += 5;
        }
        assert_eq!(meta.gc_info.len(), at + 5, "only the EH echo tail remains");
        // The EH echo tail: outgoing area (0), zero interruptible ranges.
        assert_eq!(meta.gc_info[at..at + 4], 0u32.to_le_bytes());
        assert_eq!(meta.gc_info[at + 4], 0);

        // Directly through the builder too (the channel's own API); no
        // safepoints recorded, so the count byte follows the frame size.
        let builder = MetadataBuilder::new(16, 32, roots);
        let meta = builder.finish(&EchoTarget).expect("renders");
        assert_eq!(meta.gc_info[8], 3);
        assert_eq!(meta.gc_info[9..13], 8u32.to_le_bytes());
        assert_eq!(meta.gc_info[13], 0);
        assert_eq!(meta.gc_info[14..18], 16u32.to_le_bytes());
        assert_eq!(meta.gc_info[18], 0);
        assert_eq!(meta.gc_info[19..23], 24u32.to_le_bytes());
        assert_eq!(meta.gc_info[23], 1, "the byref temp's flag");
    }

    /// The EH and IL-map sections drain through the channel: recorded
    /// clauses (raw class token, end offsets) and mappings arrive in the
    /// output unchanged.
    #[test]
    fn eh_and_il_map_sections_drain() {
        let mut builder = MetadataBuilder::new(16, 32, Vec::new());
        builder.record_eh_clause(EhClause {
            flags: EhClauseFlags::FINALLY,
            try_offset: 0,
            try_end: 8,
            handler_offset: 8,
            handler_end: 16,
            class_or_filter: ClassTokenOrFilter::ClassToken(0x0200_0042),
        });
        builder.record_il_mapping(0, 8);
        builder.record_il_mapping(7, 14);
        let meta = builder.finish(&EchoTarget).expect("renders");
        assert_eq!(
            meta.eh_clauses,
            vec![EhClause {
                flags: EhClauseFlags::FINALLY,
                try_offset: 0,
                try_end: 8,
                handler_offset: 8,
                handler_end: 16,
                class_or_filter: ClassTokenOrFilter::ClassToken(0x0200_0042),
            }]
        );
        assert_eq!(
            meta.il_map,
            vec![
                IlMapEntry {
                    il_offset: 0,
                    native_offset: 8
                },
                IlMapEntry {
                    il_offset: 7,
                    native_offset: 14
                },
            ]
        );
    }

    /// The EH path end to end (10.6): codegen's clauses, funclets, and
    /// interruptible ranges reach both target encoders and the artifact
    /// sections; the root unwind blob covers the main body only (it ends
    /// where the first funclet starts), while GC info sees the total code
    /// length.
    #[test]
    fn eh_facts_drain_from_codegen_output() {
        let mut output = codegen_output(&[0xAA; 100], &[32]);
        output.frame.outgoing_bytes = 24;
        output.funclets.push(FuncletInfo {
            start_offset: 80,
            end_offset: 100,
            prolog_len: 4,
            sp_delta: 16,
            kind: CorJitFuncKind::Handler,
        });
        output.interruptible_ranges = vec![(8, 73), (84, 96)];
        output.eh_clauses.push(EhClause {
            flags: EhClauseFlags::EMPTY,
            try_offset: 16,
            try_end: 40,
            handler_offset: 80,
            handler_end: 100,
            class_or_filter: ClassTokenOrFilter::ClassToken(0x0200_0007),
        });
        let meta = build_metadata(&output, &empty_method(), &EchoTarget).expect("renders");

        // GC info: TOTAL code length (main + funclet), then the echo tail
        // (one safepoint 37, no roots) with the ranges verbatim.
        assert_eq!(meta.gc_info[0..4], 100u32.to_le_bytes(), "total length");
        let tail = &meta.gc_info[13..];
        assert_eq!(tail[0..4], 24u32.to_le_bytes(), "the frame outgoing area");
        assert_eq!(tail[4], 2, "two interruptible ranges");
        assert_eq!(tail[5..9], 8u32.to_le_bytes());
        assert_eq!(tail[9..13], 73u32.to_le_bytes());
        assert_eq!(tail[13..17], 84u32.to_le_bytes());
        assert_eq!(tail[17..21], 96u32.to_le_bytes());

        // Unwind: root blob covers [0, 80) — the main body — then the
        // funclet blob, in emission order.
        assert_eq!(meta.unwind.len(), 2);
        assert_eq!(meta.unwind[0].func_kind, CorJitFuncKind::Root);
        assert_eq!(meta.unwind[0].end_offset, 80, "main body length");
        assert_eq!(meta.unwind[1].func_kind, CorJitFuncKind::Handler);
        assert_eq!(
            (meta.unwind[1].start_offset, meta.unwind[1].end_offset),
            (80, 100)
        );

        // The clause drains unchanged (raw token, end offsets).
        assert_eq!(meta.eh_clauses.len(), 1);
        assert_eq!(meta.eh_clauses[0].try_end, 40);
        assert_eq!(
            meta.eh_clauses[0].class_or_filter,
            ClassTokenOrFilter::ClassToken(0x0200_0007)
        );
    }

    /// The default (encoder-less) target answers Unsupported, so a backend
    /// without metadata encoders fails the method gracefully.
    #[test]
    fn target_without_encoders_is_unsupported() {
        struct BareTarget;
        impl Target for BareTarget {
            fn pointer_size(&self) -> u8 {
                8
            }
            fn register_classes(&self) -> &'static [crate::target::RegisterClass] {
                &[]
            }
            fn class_of(
                &self,
                _ty: Type,
                _layouts: &crate::structs::StructLayouts,
            ) -> Option<RegClassId> {
                None
            }
            fn classify_call(
                &self,
                _sig: &CallSig,
                _layouts: &crate::structs::StructLayouts,
            ) -> CompileResult<CallAbi> {
                Err(CompileError::Unsupported("bare"))
            }
            fn call_site_stack_alignment(&self) -> u32 {
                16
            }
        }
        let output = codegen_output(&[0xC3], &[]);
        assert!(matches!(
            build_metadata(&output, &empty_method(), &BareTarget),
            Err(CompileError::Unsupported(_))
        ));
    }

    /// Cold fragments remain later-step material with a named cause, not a
    /// panic or wrong output. (EH regions are no longer gated here: they
    /// arrive as native facts on [`CodegenOutput`], 10.6.)
    #[test]
    fn cold_chunks_are_unsupported() {
        let mut output = codegen_output(&[0xC3], &[]);
        output.code.cold = Some(CodeChunk {
            bytes: vec![0xCC],
            alignment: 1,
        });
        assert!(matches!(
            build_metadata(&output, &empty_method(), &EchoTarget),
            Err(CompileError::Unsupported(_))
        ));
    }
}
