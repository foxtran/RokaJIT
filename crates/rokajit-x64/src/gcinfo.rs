//! The x64 GC-info encoder (step_07.7): renders [`GcInfoInput`] into the
//! blob `allocGCInfo` consumes — the GcInfoEncoder format, GCINFO_VERSION 5
//! with `AMD64GcInfoEncoding` constants (references:
//! `runtime/src/coreclr/gcinfo/gcinfoencoder.cpp` `TGcInfoEncoder::Build`,
//! `runtime/src/coreclr/inc/gcinfotypes.h`, decoder cross-check in
//! `runtime/src/coreclr/vm/gcinfodecoder.cpp`).
//!
//! What the tier-0 frame contract needs — and no more:
//!
//! - **Slim header** (`gcinfoencoder.cpp:945-960`): no vararg, no GS
//!   cookie, no generics context, no reverse-pinvoke frame, no EnC, no
//!   interruptible ranges (tier-0 methods are partially interruptible —
//!   safepoints only), and the stack base register normalizes to 0: every
//!   tier-0 frame is rbp-based, and `NORMALIZE_STACK_BASE_REGISTER(rbp) =
//!   rbp ^ 5 = 0` (`gcinfotypes.h:583`), so the header stays slim with the
//!   SBR bit set. The decoder (`gcinfodecoder.cpp:294-320`) reads exactly
//!   two bits here.//! - **Code length** as `varl_u(code_len, CODE_LENGTH_ENCBASE=8)`;
//!   `NORMALIZE_CODE_LENGTH` is the identity on AMD64.
//! - **Safepoints** (`NUM_SAFE_POINTS_ENCBASE=2`): the managed calls'
//!   return addresses, as recorded by codegen (the encoder's
//!   `callSite += m_pCallSiteSizes[...]`, gcinfoencoder.cpp:1100, already
//!   applied — [`GcInfoInput::safepoints`]). Each offset is written in
//!   `CeilOfLog2(code_len)` bits (`NORMALIZE_CODE_OFFSET` is the identity
//!   on AMD64). EH methods are fully interruptible and carry NO safepoints
//!   (the count varl is 0 — see the fat-header path below).
//! - **Slot table**: no register slots (every GC reference is
//!   frame-resident), no *tracked* stack slots, and one **untracked**
//!   stack slot per GC-root frame slot (gcinfoencoder.cpp:1432-1605).
//!   Untracked = live at every safepoint with no per-safepoint state —
//!   exactly the tier-0 static root set, and what RyuJIT reports when
//!   liveness is off (`noTrackedGCSlots`, gcencode.cpp:4182). The VM
//!   decoder reads nothing past the slot table when there are no tracked
//!   slots (gcinfodecoder.cpp:833: `GetNumTracked() == 0` jumps straight
//!   to `ReportUntracked`), so no live-state/chunk section is emitted.
//!
//! Untracked slot encoding: base (2 bits, `GC_FRAMEREG_REL` — rbp,
//! matching the header's stack-base-register bit), then — for the first
//! slot and for every slot whose PREDECESSOR had any flag bit set —
//! the normalized offset as `varl_s` (`NORMALIZE_STACK_SLOT(x) = x >> 3`;
//! the offset is the negative of the slot's bytes-below-rbp) plus 2 flag
//! bits (interior|pinned); when the predecessor's flags were zero, a
//! bare `varl_u` offset delta instead (gcinfoencoder.cpp:1584-1595 — the
//! decoder branches on the predecessor's flags, gcinfodecoder.cpp:1264,
//! NOT on flag equality). Deltas are unsigned, hence the ascending sort.
//!
//! Bit packing is LSB-first within each byte (`BitStreamWriter::Write`,
//! gcinfoencoder.h: the first bit written lands in bit 0 of the first
//! byte), matching `BitStreamReader` in gcinfodecoder.h.
//!
//! ## The fat header (10.6 EH methods)
//!
//! Non-empty [`GcInfoInput::interruptible_ranges`] selects the
//! fully-interruptible encoding; the slim path above stays byte-identical
//! for empty ranges. Layout (encoder `Build`, gcinfoencoder.cpp:942-1162;
//! decoder read order, gcinfodecoder.cpp:294-411):
//!
//! - Slim bit 1, then the 10-bit fat-flags word
//!   (`GC_INFO_FLAGS_BIT_SIZE`, gcinfodecoder.h:257) with exactly
//!   `GC_INFO_HAS_STACK_BASE_REGISTER` (0x40 — rbp, normalized to 0) and
//!   `GC_INFO_WANTS_REPORT_ONLY_LEAF` (0x80 — RyuJIT sets it for any
//!   method with funclets, gcencode.cpp:3998-4004, to avoid
//!   double-reporting the parent frame) set. The macro
//!   `GCINFO_WRITE_VARL_U(..., ENCBASE, RangeSize)`'s third parameter is a
//!   MEASURE_GCINFO size counter only (gcinfoencoder.cpp:47-103) — it does
//!   not affect the encoding.
//! - Code length (TOTAL: main body + funclets), then the normalized stack
//!   base register (0), then `NORMALIZE_SIZE_OF_STACK_AREA(outgoing) =
//!   outgoing >> 3` (`SIZE_OF_STACK_AREA_ENCBASE=3`; AMD64 has
//!   `HAS_FIXED_STACK_PARAMETER_SCRATCH_AREA`, gcinfotypes.h:620).
//! - `NUM_SAFE_POINTS` varl = 0 (no safepoint offsets follow), then
//!   `NUM_INTERRUPTIBLE_RANGES` (`NUM_INTERRUPTIBLE_RANGES_ENCBASE=1`),
//!   then per range `varl_u(start - last_stop,
//!   INTERRUPTIBLE_RANGE_DELTA1_ENCBASE=6)` and `varl_u(len - 1,
//!   INTERRUPTIBLE_RANGE_DELTA2_ENCBASE=6)` (gcinfoencoder.cpp:1143-1162;
//!   the decoder adds the 1 back, gcinfodecoder.cpp:604).
//! - The untracked slot table, unchanged from the slim path.
//! - With zero tracked slots there are no lifetime transitions, so the
//!   fully-interruptible chunk section collapses to the chunk-pointer-size
//!   varl (`POINTER_SIZE_ENCBASE=3`) encoding 0 — and the encoder's
//!   `numUsedSlots == 0` early exit (gcinfoencoder.cpp:1448) skips even
//!   that when the slot table is empty.
//!
//! ## The generics context (step_11.3B)
//!
//! A reported context ([`GcInfoInput::generics_context`]) also forces the
//! fat header — with or without EH (gcinfoencoder.cpp:940's
//! `hasContextParamType`): the 2-bit `contextParamType` flag field
//! (MT/MD/THIS), then after the code-length varl `varl_u(normPrologSize -
//! 1, NORM_PROLOG_SIZE_ENCBASE=5)` (the context slot becomes reportable at
//! the end of the incoming-argument homing — clr-abi.md:92) and
//! `varl_s(NORMALIZE_STACK_SLOT(slot),
//! GENERICS_INST_CONTEXT_STACK_SLOT_ENCBASE=6)`, then the stack-base
//! register, the stack area, and the rest of the fat/slim tail unchanged.
//! A context WITHOUT EH keeps the partially-interruptible tail (safepoint
//! count, then the zero ranges count, then the fixed-width offsets — the
//! Build order, gcinfoencoder.cpp:1117-1148).
//!
//! Not yet encoded (named causes, later steps): tracked GC-root slots
//! (per-safepoint liveness — a tier-1 contract extension), GS cookie,
//! reverse-pinvoke frame, EnC.

use rokajit::error::{CompileError, CompileResult};
use rokajit::metadata::GcInfoInput;

/// `AMD64GcInfoEncoding::CODE_LENGTH_ENCBASE` (gcinfotypes.h:595).
const CODE_LENGTH_ENCBASE: u32 = 8;
/// `AMD64GcInfoEncoding::NUM_SAFE_POINTS_ENCBASE` (gcinfotypes.h:614).
const NUM_SAFE_POINTS_ENCBASE: u32 = 2;
/// `AMD64GcInfoEncoding::NUM_STACK_SLOTS_ENCBASE` (gcinfotypes.h:603).
const NUM_STACK_SLOTS_ENCBASE: u32 = 2;
/// `AMD64GcInfoEncoding::NUM_UNTRACKED_SLOTS_ENCBASE` (gcinfotypes.h:604).
const NUM_UNTRACKED_SLOTS_ENCBASE: u32 = 1;
/// `AMD64GcInfoEncoding::STACK_SLOT_ENCBASE` (gcinfotypes.h:612).
const STACK_SLOT_ENCBASE: u32 = 6;
/// `AMD64GcInfoEncoding::STACK_SLOT_DELTA_ENCBASE` (gcinfotypes.h:613).
const STACK_SLOT_DELTA_ENCBASE: u32 = 4;
/// `GcStackSlotBase::GC_FRAMEREG_REL` (gcinfotypes.h:74).
const GC_FRAMEREG_REL: u64 = 2;
/// `AMD64GcInfoEncoding::STACK_BASE_REGISTER_ENCBASE` (gcinfotypes.h:598).
const STACK_BASE_REGISTER_ENCBASE: u32 = 3;
/// `AMD64GcInfoEncoding::SIZE_OF_STACK_AREA_ENCBASE` (gcinfotypes.h:599).
const SIZE_OF_STACK_AREA_ENCBASE: u32 = 3;
/// `AMD64GcInfoEncoding::NUM_INTERRUPTIBLE_RANGES_ENCBASE`
/// (gcinfotypes.h:615).
const NUM_INTERRUPTIBLE_RANGES_ENCBASE: u32 = 1;
/// `AMD64GcInfoEncoding::INTERRUPTIBLE_RANGE_DELTA1_ENCBASE`
/// (gcinfotypes.h:608).
const INTERRUPTIBLE_RANGE_DELTA1_ENCBASE: u32 = 6;
/// `AMD64GcInfoEncoding::INTERRUPTIBLE_RANGE_DELTA2_ENCBASE`
/// (gcinfotypes.h:609).
const INTERRUPTIBLE_RANGE_DELTA2_ENCBASE: u32 = 6;
/// `AMD64GcInfoEncoding::POINTER_SIZE_ENCBASE` (gcinfotypes.h:617).
const POINTER_SIZE_ENCBASE: u32 = 3;
/// `AMD64GcInfoEncoding::NORM_PROLOG_SIZE_ENCBASE` (gcinfotypes.h:605).
const NORM_PROLOG_SIZE_ENCBASE: u32 = 5;
/// `AMD64GcInfoEncoding::GENERICS_INST_CONTEXT_STACK_SLOT_ENCBASE`
/// (gcinfotypes.h:592).
const GENERICS_INST_CONTEXT_STACK_SLOT_ENCBASE: u32 = 6;

/// The 2-bit `contextParamType` field of the fat-flags word
/// (gcinfodecoder.h:241-245: MT=0x10, MD=0x20, THIS=0x30 — field values
/// 1/2/3 in bits 4-5).
fn context_param_type_bits(kind: rokajit::ir::GenericsContext) -> u64 {
    match kind {
        rokajit::ir::GenericsContext::MethodTable => 0x10,
        rokajit::ir::GenericsContext::MethodDesc => 0x20,
        rokajit::ir::GenericsContext::This => 0x30,
    }
}

/// The `Target::encode_gc_info` body for x64.
pub fn encode(input: &GcInfoInput) -> CompileResult<Vec<u8>> {
    let mut w = BitWriter::new();
    // Non-empty interruptible ranges ⇒ an EH method; a reported generics
    // context (step_11.3B) also forces the fat header
    // (gcinfoencoder.cpp:940-952's slimHeader condition). The slim path is
    // byte-identical to pre-EH output.
    let fully_interruptible = !input.interruptible_ranges.is_empty();
    let fat = fully_interruptible || input.generics_context.is_some();
    if fat {
        // GC_INFO_HAS_STACK_BASE_REGISTER (0x40): rbp normalizes to 0.
        let mut flags = 0x40u64;
        if fully_interruptible {
            // RyuJIT sets WANTS_REPORT_ONLY_LEAF for any method with
            // funclets (gcencode.cpp:3998-4004) — no double-reporting of
            // the parent frame.
            flags |= 0x80;
        }
        if let Some(ctx) = &input.generics_context {
            flags |= context_param_type_bits(ctx.kind);
        }
        w.write(1, 1);
        w.write(flags, 10);
        w.write_varl_u(input.code_len, CODE_LENGTH_ENCBASE);
        if let Some(ctx) = &input.generics_context {
            // The prolog size bounds where the context slot may be
            // reported (gcinfoencoder.cpp:1004-1017; the slot's homing
            // store lies inside the prolog by construction).
            // NORMALIZE_CODE_OFFSET is the identity on AMD64.
            if ctx.prolog_end == 0 || ctx.prolog_end >= input.code_len {
                return Err(CompileError::Internal(
                    "generics-context prolog end outside the method body",
                ));
            }
            w.write_varl_u(ctx.prolog_end - 1, NORM_PROLOG_SIZE_ENCBASE);
            if ctx.slot_offset % 8 != 0 {
                return Err(CompileError::Internal(
                    "generics-context slot not 8-aligned",
                ));
            }
            // NORMALIZE_STACK_SLOT(x) = x >> 3 (gcinfotypes.h).
            w.write_varl_s(
                i64::from(ctx.slot_offset / 8),
                GENERICS_INST_CONTEXT_STACK_SLOT_ENCBASE,
            );
        }
        // rbp: NORMALIZE_STACK_BASE_REGISTER(5) = 5 ^ 5 = 0.
        w.write_varl_u(0, STACK_BASE_REGISTER_ENCBASE);
        // NORMALIZE_SIZE_OF_STACK_AREA(x) = x >> 3. The fat header always
        // carries it on AMD64 (HAS_FIXED_STACK_PARAMETER_SCRATCH_AREA,
        // gcinfotypes.h:620).
        if !input.outgoing_area_size.is_multiple_of(8) {
            return Err(CompileError::Internal(
                "outgoing argument area not 8-aligned",
            ));
        }
        w.write_varl_u(input.outgoing_area_size >> 3, SIZE_OF_STACK_AREA_ENCBASE);
        // NUM_SAFE_POINTS, then NUM_INTERRUPTIBLE_RANGES, then the
        // safepoint offsets, then the ranges — the encoder's Build order
        // (gcinfoencoder.cpp:1117-1162). Fully interruptible: zero
        // safepoints.
        let mut safepoints = input.safepoints.clone();
        safepoints.sort_unstable();
        if fully_interruptible {
            w.write_varl_u(0, NUM_SAFE_POINTS_ENCBASE);
        } else {
            w.write_varl_u(safepoints.len() as u32, NUM_SAFE_POINTS_ENCBASE);
        }
        w.write_varl_u(
            input.interruptible_ranges.len() as u32,
            NUM_INTERRUPTIBLE_RANGES_ENCBASE,
        );
        if !fully_interruptible {
            let offset_bits = ceil_log2(input.code_len);
            for safepoint in safepoints {
                w.write(u64::from(safepoint), offset_bits);
            }
        }
        let mut last_stop = 0u32;
        for &(start, end) in &input.interruptible_ranges {
            if start < last_stop || end <= start {
                return Err(CompileError::Internal(
                    "interruptible ranges not sorted, disjoint, and non-empty",
                ));
            }
            w.write_varl_u(start - last_stop, INTERRUPTIBLE_RANGE_DELTA1_ENCBASE);
            w.write_varl_u(end - start - 1, INTERRUPTIBLE_RANGE_DELTA2_ENCBASE);
            last_stop = end;
        }
    } else {
        // Slim header: slim-encoding bit 0, then "has stack base register" —
        // set, because the tier-0 frame is rbp-based and GC slot offsets are
        // rbp-relative (rbp normalizes to 0, so no fat header is needed).
        w.write(0, 1);
        w.write(1, 1);
        w.write_varl_u(input.code_len, CODE_LENGTH_ENCBASE);
        let mut safepoints = input.safepoints.clone();
        safepoints.sort_unstable();
        w.write_varl_u(safepoints.len() as u32, NUM_SAFE_POINTS_ENCBASE);
        let offset_bits = ceil_log2(input.code_len);
        for safepoint in safepoints {
            w.write(u64::from(safepoint), offset_bits);
        }
    }
    // Slot table: no register slots, no tracked stack slots, one
    // untracked stack slot per GC root (always-live — the tier-0 static
    // root set). Sort like the EE (gcinfoencoder.cpp:777-795): flagged
    // (interior/pinned) slots first, plain slots last, offsets ascending
    // within a flag group — so delta runs are always plain-flagged. The
    // encoding rule (gcinfoencoder.cpp:1584-1595): the first slot is
    // base + varl_s offset + flags; every later slot is base + varl_s
    // offset + flags when the PREVIOUS slot had any flag bit set
    // (interior/pinned), and a bare unsigned delta when the previous
    // slot's flags were zero — the decoder branches on the previous
    // flags (gcinfodecoder.cpp:1264), NOT on flag equality. Deltas are
    // unsigned, hence offset-descending within the plain run.
    let mut roots = input.gc_roots.clone();
    roots.sort_by_key(|r| {
        let flags = u64::from(r.is_byref) | (u64::from(r.pinned) << 1);
        (std::cmp::Reverse(flags), std::cmp::Reverse(r.offset))
    });
    w.write(0, 1);
    if roots.is_empty() {
        // The encoder's numUsedSlots == 0 early exit
        // (gcinfoencoder.cpp:1448): no chunk-pointer section either.
        w.write(0, 1);
        return Ok(w.finish());
    }
    w.write(1, 1);
    w.write_varl_u(0, NUM_STACK_SLOTS_ENCBASE);
    w.write_varl_u(roots.len() as u32, NUM_UNTRACKED_SLOTS_ENCBASE);
    let mut last_norm = 0i32;
    let mut last_flags = 0u64;
    for (i, root) in roots.iter().enumerate() {
        if root.offset % 8 != 0 {
            return Err(CompileError::Internal("GC-root slot not 8-aligned"));
        }
        // `GcStackSlot.SpOffset` is relative to the frame register
        // (gcinfodecoder.cpp `GetStackSlot`: frame + spOffset); our slots
        // live at [rbp - offset].
        let norm = -((root.offset / 8) as i32);
        let flags = u64::from(root.is_byref) | (u64::from(root.pinned) << 1);
        w.write(GC_FRAMEREG_REL, 2);
        if i == 0 || last_flags != 0 {
            w.write_varl_s(i64::from(norm), STACK_SLOT_ENCBASE);
            w.write(flags, 2);
        } else {
            w.write_varl_u((norm - last_norm) as u32, STACK_SLOT_DELTA_ENCBASE);
        }
        last_norm = norm;
        last_flags = flags;
    }
    if fully_interruptible {
        // No tracked slots ⇒ no lifetime transitions ⇒ every chunk
        // pointer is zero: the section collapses to the pointer-size varl
        // encoding CeilOfLog2(0 + 1) = 0 (gcinfoencoder.cpp:2089-2098).
        w.write_varl_u(0, POINTER_SIZE_ENCBASE);
    }
    Ok(w.finish())
}

/// `CeilOfLog2`: the smallest `k` with `2^k >= x` (0 for x ≤ 1).
fn ceil_log2(x: u32) -> u32 {
    if x <= 1 {
        0
    } else {
        u32::BITS - (x - 1).leading_zeros()
    }
}

/// LSB-first bit stream, matching the EE's `BitStreamWriter`/`BitStreamReader`
/// pair: the first bit written is bit 0 of byte 0.
struct BitWriter {
    bytes: Vec<u8>,
    /// Bits used in the last byte (0 means the vec is empty or full).
    used: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            bytes: Vec::new(),
            used: 0,
        }
    }

    /// Append the low `count` bits of `value`, LSB first.
    fn write(&mut self, value: u64, count: u32) {
        for i in 0..count {
            if self.used == 0 {
                self.bytes.push(0);
            }
            if (value >> i) & 1 == 1 {
                let last = self.bytes.len() - 1;
                self.bytes[last] |= 1 << self.used;
            }
            self.used = (self.used + 1) % 8;
        }
    }

    /// Append `n` as a variable-length unsigned: chunks of `base` bits,
    /// LSB-significant chunk first, each chunk followed by a continuation
    /// bit (`BitStreamWriter::EncodeVarLengthUnsigned`,
    /// gcinfoencoder.cpp:2571).
    fn write_varl_u(&mut self, mut n: u32, base: u32) {
        loop {
            let chunk = n & ((1 << base) - 1);
            n >>= base;
            if n == 0 {
                // Final chunk: the extension bit (bit #base) stays 0.
                self.write(u64::from(chunk), base + 1);
                return;
            }
            self.write(u64::from(chunk | (1 << base)), base + 1);
        }
    }

    /// Append `n` as a variable-length signed: chunks of `base` bits,
    /// LSB-significant chunk first, each chunk followed by a continuation
    /// bit; the top bit of the final chunk is the sign
    /// (`BitStreamWriter::EncodeVarLengthSigned`, gcinfoencoder.cpp:2595).
    fn write_varl_s(&mut self, mut n: i64, base: u32) {
        let num_encodings = 1u64 << base;
        loop {
            let chunk = (n as u64) & (num_encodings - 1);
            let topmost = chunk & (num_encodings >> 1);
            n >>= base; // signed arithmetic shift
            if (topmost != 0 && n == -1) || (topmost == 0 && n == 0) {
                // The topmost bit correctly represents the sign; the
                // extension bit (bit #base) stays 0.
                self.write(chunk, base + 1);
                return;
            }
            self.write(chunk | num_encodings, base + 1);
        }
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    //! Expected bytes are hand-computed bit-by-bit in the tests below and
    //! cross-checked by re-decoding the blob in the VM decoder's field
    //! order (slim bit, SBR bit, varl code length, varl safepoint count,
    //! fixed-width safepoint offsets, two slot-table flag bits).

    use super::*;

    fn input(code_len: u32, safepoints: &[u32]) -> GcInfoInput {
        GcInfoInput {
            code_len,
            frame_size: 32,
            gc_roots: Vec::new(),
            safepoints: safepoints.to_vec(),
            interruptible_ranges: Vec::new(),
            outgoing_area_size: 0,
            generics_context: None,
        }
    }

    /// A decoder for the subset we emit, in the VM's read order.
    struct Reader<'a> {
        bytes: &'a [u8],
        bit: usize,
    }

    impl Reader<'_> {
        fn read(&mut self, count: u32) -> u64 {
            let mut value = 0u64;
            for i in 0..count {
                let b = (self.bytes[self.bit / 8] >> (self.bit % 8)) & 1;
                value |= u64::from(b) << i;
                self.bit += 1;
            }
            value
        }

        fn read_varl_u(&mut self, base: u32) -> u64 {
            let mut value = 0u64;
            let mut shift = 0;
            loop {
                let chunk = self.read(base + 1);
                value |= (chunk & ((1 << base) - 1)) << shift;
                if chunk & (1 << base) == 0 {
                    return value;
                }
                shift += base;
            }
        }

        /// `BitStreamReader::DecodeVarLengthSigned`: the final chunk's top
        /// bit sign-extends.
        fn read_varl_s(&mut self, base: u32) -> i64 {
            let mut value = 0i64;
            let mut shift = 0;
            loop {
                let chunk = self.read(base + 1);
                value |= ((chunk & ((1 << base) - 1)) as i64) << shift;
                if chunk & (1 << base) == 0 {
                    let sbits = 64 - (shift + base);
                    return (value << sbits) >> sbits;
                }
                shift += base;
            }
        }
    }

    /// fib's real shape: 73 code bytes, call return addresses at 37 and 56.
    #[test]
    fn fib_blob_is_byte_exact() {
        let blob = encode(&input(73, &[37, 56])).expect("encodes");
        // Bit stream (LSB-first; see the module docs for the field order):
        //   [0] slim=0, [1] sbr=1
        //   [2..10]  varl8(73): 73 = 64+8+1 → 1,0,0,1,0,0,1,0 + ext 0
        //   [11..13] varl2(2): 0,1 + ext 0
        //   [14..20] safepoint 37: 1,0,1,0,0,1,0
        //   [21..27] safepoint 56: 0,0,0,1,1,1,0
        //   [28] no register slots, [29] no stack/untracked slots
        assert_eq!(blob, [0x26, 0x51, 0x09, 0x07]);
    }

    #[test]
    fn fib_blob_decodes_back() {
        let blob = encode(&input(73, &[37, 56])).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(1), 0, "slim header");
        assert_eq!(r.read(1), 1, "rbp stack base register");
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 73);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 2);
        assert_eq!(r.read(7), 37, "first call's return address");
        assert_eq!(r.read(7), 56, "second call's return address");
        assert_eq!(r.read(1), 0, "no register slots");
        assert_eq!(r.read(1), 0, "no stack/untracked slots");
        assert_eq!(r.bit, 30, "30 bits = 4 bytes, 2 padding");
    }

    /// No calls: the header plus an empty safepoint vector and slot table.
    #[test]
    fn leaf_method_blob() {
        let blob = encode(&input(9, &[])).expect("encodes");
        // 0,1 | varl8(9): 1,0,0,1,0,0,0,0,ext 0 | varl2(0): 0,0,0 | 0 | 0 = 16 bits
        assert_eq!(blob, [0x26, 0x00]);
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(1), 0);
        assert_eq!(r.read(1), 1);
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 9);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 0);
        assert_eq!(r.read(1), 0);
        assert_eq!(r.read(1), 0);
    }

    /// A code length past the first varl chunk boundary exercises the
    /// continuation bit (300 = 44 + 1<<8: chunk 44+ext, then chunk 1).
    #[test]
    fn code_length_varl_continuation() {
        let blob = encode(&input(300, &[])).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(1), 0);
        assert_eq!(r.read(1), 1);
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 300);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 0);
    }

    /// Safepoint offsets widen with the code length: CeilOfLog2(300) = 9.
    #[test]
    fn safepoint_bit_width_follows_code_length() {
        let blob = encode(&input(300, &[300])).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(2), 0b10, "slim + sbr");
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 300);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 1);
        assert_eq!(r.read(9), 300, "the last byte is the safepoint");
    }

    /// One untracked stack slot, read back in the decoder's field order
    /// (gcinfodecoder.cpp `GcSlotDecoder::DecodeSlotTable`, untracked
    /// section). `delta_from` states the caller's expectation: `Some` =
    /// the delta form against that predecessor (same flags), `None` =
    /// the full varl_s + flags form. Returns (normalized offset, flags).
    fn read_untracked_slot(r: &mut Reader, delta_from: Option<(i32, u64)>) -> (i32, u64) {
        assert_eq!(r.read(2), GC_FRAMEREG_REL, "rbp-based frame slot");
        match delta_from {
            Some((last_norm, flags)) => {
                let delta = r.read_varl_u(STACK_SLOT_DELTA_ENCBASE) as i32;
                (last_norm + delta, flags)
            }
            None => {
                let norm = r.read_varl_s(STACK_SLOT_ENCBASE) as i32;
                let flags = r.read(2);
                (norm, flags)
            }
        }
    }

    fn root(offset: u32, is_byref: bool, pinned: bool) -> rokajit::pipeline::GcRootSlot {
        rokajit::pipeline::GcRootSlot {
            offset,
            is_byref,
            pinned,
        }
    }

    /// The step_10.3 acceptance shape: one live ref local in a leaf-shaped
    /// frame. Hand-computed bit stream (LSB-first; field order per the
    /// module docs):
    ///   [0] slim=0, [1] sbr=1
    ///   [2..10]  varl8(9): 1,0,0,1,0,0,0,0 + ext 0
    ///   [11..13] varl2(0 safepoints)
    ///   [14] no register slots, [15] stack slots follow
    ///   [16..18] varl2(0 tracked), [19..20] varl1(1 untracked): 1 + ext 0
    ///   [21..22] base=GC_FRAMEREG_REL: 0,1
    ///   [23..29] varl_s(-1, base 6): six 1 bits + ext 0
    ///   [30..31] flags 0 (plain ref, not pinned)
    #[test]
    fn one_live_ref_local_blob_is_byte_exact() {
        let mut i = input(9, &[]);
        i.gc_roots.push(root(8, false, false));
        let blob = encode(&i).expect("encodes");
        assert_eq!(blob, [0x26, 0x80, 0xC8, 0x1F]);
    }

    /// One string local live across a call: the safepoint list names the
    /// call's return address and the slot table names the local's frame
    /// slot as an always-live (untracked) root. Nothing follows the slot
    /// table — the decoder reads no live-state section when there are no
    /// tracked slots (gcinfodecoder.cpp:833).
    #[test]
    fn live_ref_local_across_a_call_decodes_back() {
        let mut i = input(73, &[37, 56]);
        i.gc_roots.push(root(16, false, false));
        let blob = encode(&i).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(1), 0, "slim header");
        assert_eq!(r.read(1), 1, "rbp stack base register");
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 73);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 2);
        assert_eq!(r.read(7), 37);
        assert_eq!(r.read(7), 56);
        assert_eq!(r.read(1), 0, "no register slots");
        assert_eq!(r.read(1), 1, "stack slots follow");
        assert_eq!(r.read_varl_u(NUM_STACK_SLOTS_ENCBASE), 0, "none tracked");
        assert_eq!(r.read_varl_u(NUM_UNTRACKED_SLOTS_ENCBASE), 1);
        assert_eq!(read_untracked_slot(&mut r, None), (-2, 0), "rbp - 16");
        assert!(blob.len() * 8 - r.bit < 8, "only padding remains");
    }

    /// Same-flag neighbors delta-encode: offsets 24 (norm -3) then 8
    /// (norm -1) after the ascending sort, delta 2.
    #[test]
    fn same_flag_slots_delta_encode_in_ascending_order() {
        let mut i = input(9, &[]);
        // Deliberately unsorted input: the encoder sorts.
        i.gc_roots.push(root(8, false, false));
        i.gc_roots.push(root(24, false, false));
        let blob = encode(&i).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(2), 0b10);
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 9);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 0);
        assert_eq!(r.read(1), 0);
        assert_eq!(r.read(1), 1);
        assert_eq!(r.read_varl_u(NUM_STACK_SLOTS_ENCBASE), 0);
        assert_eq!(r.read_varl_u(NUM_UNTRACKED_SLOTS_ENCBASE), 2);
        let first = read_untracked_slot(&mut r, None);
        assert_eq!(first, (-3, 0), "the deeper slot first (ascending norm)");
        let second = read_untracked_slot(&mut r, Some(first));
        assert_eq!(second, (-1, 0), "delta of 2 to rbp - 8");
    }

    /// A flags change (byref = interior) forces the full varl_s + flags
    /// form for the next slot, per the decoder's `if (flags)` branch.
    #[test]
    fn a_flag_change_forces_the_full_slot_form() {
        let mut i = input(9, &[]);
        i.gc_roots.push(root(16, false, false));
        i.gc_roots.push(root(8, true, false));
        let blob = encode(&i).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(2), 0b10);
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 9);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 0);
        assert_eq!(r.read(1), 0);
        assert_eq!(r.read(1), 1);
        assert_eq!(r.read_varl_u(NUM_STACK_SLOTS_ENCBASE), 0);
        assert_eq!(r.read_varl_u(NUM_UNTRACKED_SLOTS_ENCBASE), 2);
        // Flagged slots sort first (gcinfoencoder.cpp:785), so the byref
        // heads the table in full form...
        let first = read_untracked_slot(&mut r, None);
        assert_eq!(first, (-1, 1), "rbp - 8, interior");
        // ...and the plain ref follows it, also in full form, because the
        // delta rule keys on the PREDECESSOR's flags being nonzero
        // (gcinfoencoder.cpp:1586), not on flag equality.
        let second = read_untracked_slot(&mut r, None);
        assert_eq!(second, (-2, 0));
    }

    /// Two byref roots in a row: both take the full form (the latent
    /// pre-10.9 bug this regression guards — a delta between flagged
    /// slots misaligns the whole table, gcinfodecoder.cpp:1264).
    #[test]
    fn consecutive_byref_roots_both_take_the_full_form() {
        let mut i = input(9, &[]);
        i.gc_roots.push(root(16, true, false));
        i.gc_roots.push(root(8, true, false));
        i.gc_roots.push(root(24, false, false));
        let blob = encode(&i).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(2), 0b10);
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 9);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 0);
        assert_eq!(r.read(2), 0b10);
        assert_eq!(r.read_varl_u(NUM_STACK_SLOTS_ENCBASE), 0);
        assert_eq!(r.read_varl_u(NUM_UNTRACKED_SLOTS_ENCBASE), 3);
        // Flagged first (offset-descending within the group): both
        // byrefs in full form; then the plain ref in full form
        // (predecessor flagged); no delta entries at all.
        assert_eq!(read_untracked_slot(&mut r, None), (-2, 1));
        assert_eq!(read_untracked_slot(&mut r, None), (-1, 1));
        assert_eq!(read_untracked_slot(&mut r, None), (-3, 0));
    }

    /// A plain-ref run after the flagged slots delta-encodes.
    #[test]
    fn plain_roots_after_flagged_delta_encode() {
        let mut i = input(9, &[]);
        i.gc_roots.push(root(8, true, false));
        i.gc_roots.push(root(16, false, false));
        i.gc_roots.push(root(24, false, false));
        let blob = encode(&i).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(2), 0b10);
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 9);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 0);
        assert_eq!(r.read(2), 0b10);
        assert_eq!(r.read_varl_u(NUM_STACK_SLOTS_ENCBASE), 0);
        assert_eq!(r.read_varl_u(NUM_UNTRACKED_SLOTS_ENCBASE), 3);
        assert_eq!(
            read_untracked_slot(&mut r, None),
            (-1, 1),
            "the byref first"
        );
        assert_eq!(
            read_untracked_slot(&mut r, None),
            (-3, 0),
            "flag change: full"
        );
        // The second plain ref's predecessor has zero flags: the delta.
        assert_eq!(
            read_untracked_slot(&mut r, Some((-3, 0))),
            (-2, 0),
            "delta +1 from the previous plain ref"
        );
    }

    /// Pinned sets the second flag bit.
    #[test]
    fn pinned_sets_the_second_flag_bit() {
        let mut i = input(9, &[]);
        i.gc_roots.push(root(8, false, true));
        let blob = encode(&i).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(2), 0b10);
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 9);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 0);
        assert_eq!(r.read(2), 0b10, "no register slots, stack slots follow");
        assert_eq!(r.read_varl_u(NUM_STACK_SLOTS_ENCBASE), 0);
        assert_eq!(r.read_varl_u(NUM_UNTRACKED_SLOTS_ENCBASE), 1);
        assert_eq!(read_untracked_slot(&mut r, None), (-1, 2));
    }

    /// GC-root slots are 8-aligned by the frame contract; anything else
    /// is an upstream bug, not an encoding.
    #[test]
    fn misaligned_root_is_an_internal_error() {
        let mut i = input(9, &[]);
        i.gc_roots.push(root(4, false, false));
        assert!(matches!(encode(&i), Err(CompileError::Internal(_))));
    }

    #[test]
    fn ceil_log2_boundaries() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(64), 6);
        assert_eq!(ceil_log2(73), 7);
        assert_eq!(ceil_log2(128), 7);
        assert_eq!(ceil_log2(129), 8);
    }

    /// An EH-shaped input: total code length 100 (main + one funclet),
    /// interruptible ranges covering the main body [8, 73) and the
    /// funclet body [84, 96).
    fn eh_input() -> GcInfoInput {
        GcInfoInput {
            code_len: 100,
            interruptible_ranges: vec![(8, 73), (84, 96)],
            ..input(100, &[37, 56])
        }
    }

    /// Read one interruptible range in the decoder's order
    /// (gcinfodecoder.cpp:599-617): start delta from the previous range's
    /// stop, then the length minus one.
    fn read_range(r: &mut Reader, last_stop: u32) -> (u32, u32) {
        let start = last_stop + r.read_varl_u(INTERRUPTIBLE_RANGE_DELTA1_ENCBASE) as u32;
        let stop = start + r.read_varl_u(INTERRUPTIBLE_RANGE_DELTA2_ENCBASE) as u32 + 1;
        (start, stop)
    }

    /// The slim regression pin: empty ranges keep the exact pre-EH
    /// encoding (the other tests above pin the same bytes against
    /// hand-computed bit streams).
    #[test]
    fn empty_ranges_keep_the_slim_encoding() {
        let blob = encode(&input(73, &[37, 56])).expect("encodes");
        assert_eq!(blob, [0x26, 0x51, 0x09, 0x07], "the fib blob");
        assert_eq!(blob[0] & 1, 0, "slim bit");
    }

    /// Fat header, no roots: hand-computed bit stream (LSB-first).
    ///   [0] fat=1
    ///   [1..10]  flags 0xC0 (stack-base-register | report-only-leaf)
    ///   [11..19] varl8(100): 100 = 64+32+4 → 0,0,1,0,0,1,1,0 + ext 0
    ///   [20..23] varl3(0): the normalized stack base register (rbp → 0)
    ///   [24..27] varl3(0): outgoing area 0 >> 3
    ///   [28..30] varl2(0): no safepoints (fully interruptible)
    ///   [31..34] varl1(2 ranges): 0,1 | 1,0
    ///   [35..41] varl6(8):  start delta 8 - 0
    ///   [42..55] varl6(64): length-1 = 73-8-1 = 64 (two 7-bit chunks:
    ///            0+cont, then 1)
    ///   [56..62] varl6(11): start delta 84 - 73
    ///   [63..69] varl6(11): length-1 = 96-84-1
    ///   [70] no register slots, [71] no stack/untracked slots
    ///   (no chunk-pointer varl: the empty slot table early-exits,
    ///   gcinfoencoder.cpp:1448)
    #[test]
    fn fat_header_blob_is_byte_exact() {
        let blob = encode(&eh_input()).expect("encodes");
        assert_eq!(blob, [0x81, 0x21, 0x03, 0x00, 0x43, 0x00, 0x03, 0x8B, 0x05]);
    }

    /// Decode the fat blob back in the VM's read order
    /// (gcinfodecoder.cpp:294-411): fat flags word, TOTAL code length,
    /// normalized stack base register, outgoing area, zero safepoints,
    /// then the ranges — the slot table and chunk-pointer varl follow.
    #[test]
    fn fat_header_decodes_back() {
        let mut i = eh_input();
        i.outgoing_area_size = 32;
        i.gc_roots.push(root(16, false, false));
        let blob = encode(&i).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(1), 1, "fat header");
        assert_eq!(
            r.read(10),
            0xC0,
            "HAS_STACK_BASE_REGISTER | WANTS_REPORT_ONLY_LEAF"
        );
        assert_eq!(
            r.read_varl_u(CODE_LENGTH_ENCBASE),
            100,
            "TOTAL code length (main + funclet)"
        );
        assert_eq!(
            r.read_varl_u(STACK_BASE_REGISTER_ENCBASE),
            0,
            "rbp normalizes to 0"
        );
        assert_eq!(r.read_varl_u(SIZE_OF_STACK_AREA_ENCBASE), 4, "32 >> 3");
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 0, "no safepoints");
        assert_eq!(r.read_varl_u(NUM_INTERRUPTIBLE_RANGES_ENCBASE), 2);
        let first = read_range(&mut r, 0);
        assert_eq!(first, (8, 73), "the main body range");
        let second = read_range(&mut r, first.1);
        assert_eq!(second, (84, 96), "the funclet body range");
        // The untracked slot table is unchanged from the slim path.
        assert_eq!(r.read(1), 0, "no register slots");
        assert_eq!(r.read(1), 1, "stack slots follow");
        assert_eq!(r.read_varl_u(NUM_STACK_SLOTS_ENCBASE), 0, "none tracked");
        assert_eq!(r.read_varl_u(NUM_UNTRACKED_SLOTS_ENCBASE), 1);
        assert_eq!(read_untracked_slot(&mut r, None), (-2, 0), "rbp - 16");
        // No tracked slots ⇒ no transitions ⇒ the chunk-pointer section is
        // just the pointer-size varl encoding 0.
        assert_eq!(r.read_varl_u(POINTER_SIZE_ENCBASE), 0);
        assert!(blob.len() * 8 - r.bit < 8, "only padding remains");
    }

    /// A filter method (step_11.11): the GC blob needs nothing
    /// filter-specific — one more interruptible range for the filter
    /// funclet's body, in the same unified method-relative offsets (the
    /// VM derives filter-ness from the clause's FilterOffset matching
    /// the funclet's RUNTIME_FUNCTION, codeman.cpp:1372). This pins the
    /// three-range round-trip.
    #[test]
    fn filter_funclet_adds_an_interruptible_range() {
        let mut i = eh_input();
        i.code_len = 130;
        i.interruptible_ranges.push((105, 121)); // the filter funclet body
        let blob = encode(&i).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(1), 1, "fat header");
        assert_eq!(r.read(10), 0xC0, "SBR | WANTS_REPORT_ONLY_LEAF");
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 130, "TOTAL code length");
        assert_eq!(r.read_varl_u(STACK_BASE_REGISTER_ENCBASE), 0);
        assert_eq!(r.read_varl_u(SIZE_OF_STACK_AREA_ENCBASE), 0);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 0);
        assert_eq!(r.read_varl_u(NUM_INTERRUPTIBLE_RANGES_ENCBASE), 3);
        let first = read_range(&mut r, 0);
        assert_eq!(first, (8, 73), "the main body range");
        let second = read_range(&mut r, first.1);
        assert_eq!(second, (84, 96), "the handler funclet body");
        let third = read_range(&mut r, second.1);
        assert_eq!(third, (105, 121), "the filter funclet body");
    }

    /// Contract violations in the ranges are upstream bugs, not encodings.
    #[test]
    fn bad_ranges_are_internal_errors() {
        // Empty range.
        let mut i = eh_input();
        i.interruptible_ranges = vec![(8, 8)];
        assert!(matches!(encode(&i), Err(CompileError::Internal(_))));
        // Overlapping the previous range's stop.
        let mut i = eh_input();
        i.interruptible_ranges = vec![(8, 73), (72, 96)];
        assert!(matches!(encode(&i), Err(CompileError::Internal(_))));
        // The outgoing area must survive >> 3.
        let mut i = eh_input();
        i.outgoing_area_size = 12;
        assert!(matches!(encode(&i), Err(CompileError::Internal(_))));
    }

    /// A reported context (step_11.3B), hand-computed bit stream
    /// (LSB-first; gcinfoencoder.cpp:936-1046's Build order):
    ///   [0] fat=1
    ///   [1..10]  flags 0x60 (SBR | contextParamType=MD)
    ///   [11..19] varl8(73): the code length
    ///   [20..25] varl5(7): normPrologSize - 1 = prolog_end 8 - 1
    ///   [26..32] varl_s(-2, base 6): the context slot (rbp - 16)
    ///   [33..36] varl3(0): the stack base register
    ///   [37..40] varl3(0): the outgoing area
    ///   [41..43] varl2(2): two safepoints
    ///   [44..45] varl1(0): no interruptible ranges
    ///   [46..59] the two safepoint offsets (CeilOfLog2(73) = 7 bits)
    ///   [60..61] no register slots, no stack slots
    #[test]
    fn generics_context_forces_the_fat_header() {
        let mut i = input(73, &[37, 56]);
        i.generics_context = Some(rokajit::pipeline::GenericsContextGcInfo {
            slot_offset: -16,
            kind: rokajit::ir::GenericsContext::MethodDesc,
            prolog_end: 8,
        });
        let blob = encode(&i).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(1), 1, "fat header");
        assert_eq!(
            r.read(10),
            0x60,
            "HAS_STACK_BASE_REGISTER | contextParamType=MD (0x20)"
        );
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 73);
        assert_eq!(
            r.read_varl_u(NORM_PROLOG_SIZE_ENCBASE),
            7,
            "normPrologSize - 1"
        );
        assert_eq!(
            r.read_varl_s(GENERICS_INST_CONTEXT_STACK_SLOT_ENCBASE),
            -2,
            "the context slot at rbp - 16"
        );
        assert_eq!(r.read_varl_u(STACK_BASE_REGISTER_ENCBASE), 0);
        assert_eq!(r.read_varl_u(SIZE_OF_STACK_AREA_ENCBASE), 0);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 2);
        assert_eq!(
            r.read_varl_u(NUM_INTERRUPTIBLE_RANGES_ENCBASE),
            0,
            "partially interruptible: no ranges"
        );
        // The safepoint offsets follow the ranges count (Build order).
        assert_eq!(r.read(7), 37);
        assert_eq!(r.read(7), 56);
        assert_eq!(r.read(1), 0, "no register slots");
        assert_eq!(r.read(1), 0, "no stack/untracked slots");
        assert!(blob.len() * 8 - r.bit < 8, "only padding remains");
    }

    /// Context AND a call safepoint (step_11.3C's "GC-info context
    /// reporting under stub dispatch" shape): the fat header carries
    /// contextParamType=MT and the safepoint table rides behind the
    /// context fields.
    #[test]
    fn generics_context_and_a_safepoint_encode_together() {
        let mut i = input(40, &[22]);
        i.generics_context = Some(rokajit::pipeline::GenericsContextGcInfo {
            slot_offset: -16,
            kind: rokajit::ir::GenericsContext::MethodTable,
            prolog_end: 8,
        });
        let blob = encode(&i).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(1), 1, "fat header");
        assert_eq!(
            r.read(10),
            0x50,
            "HAS_STACK_BASE_REGISTER | contextParamType=MT (0x10)"
        );
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 40);
        assert_eq!(r.read_varl_u(NORM_PROLOG_SIZE_ENCBASE), 7);
        assert_eq!(
            r.read_varl_s(GENERICS_INST_CONTEXT_STACK_SLOT_ENCBASE),
            -2,
            "the context slot at rbp - 16"
        );
        assert_eq!(r.read_varl_u(STACK_BASE_REGISTER_ENCBASE), 0);
        assert_eq!(r.read_varl_u(SIZE_OF_STACK_AREA_ENCBASE), 0);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 1);
        assert_eq!(r.read_varl_u(NUM_INTERRUPTIBLE_RANGES_ENCBASE), 0);
        assert_eq!(r.read(6), 22, "the safepoint (CeilOfLog2(40) = 6 bits)");
        assert_eq!(r.read(1), 0, "no register slots");
        assert_eq!(r.read(1), 0, "no stack/untracked slots");
        assert!(blob.len() * 8 - r.bit < 8, "only padding remains");
    }

    /// Context AND EH (a shared generic method with funclets): both flag
    /// sets, the prolog varl and context slot before the stack base
    /// register, then the EH tail (zero safepoints, the ranges).
    #[test]
    fn generics_context_composes_with_the_eh_fat_path() {
        let mut i = eh_input();
        i.generics_context = Some(rokajit::pipeline::GenericsContextGcInfo {
            slot_offset: -8,
            kind: rokajit::ir::GenericsContext::This,
            prolog_end: 6,
        });
        let blob = encode(&i).expect("encodes");
        let mut r = Reader {
            bytes: &blob,
            bit: 0,
        };
        assert_eq!(r.read(1), 1, "fat header");
        assert_eq!(
            r.read(10),
            0xF0,
            "SBR | WANTS_REPORT_ONLY_LEAF | contextParamType=THIS (0x30)"
        );
        assert_eq!(r.read_varl_u(CODE_LENGTH_ENCBASE), 100);
        assert_eq!(r.read_varl_u(NORM_PROLOG_SIZE_ENCBASE), 5);
        assert_eq!(
            r.read_varl_s(GENERICS_INST_CONTEXT_STACK_SLOT_ENCBASE),
            -1,
            "the context slot at rbp - 8"
        );
        assert_eq!(r.read_varl_u(STACK_BASE_REGISTER_ENCBASE), 0);
        assert_eq!(r.read_varl_u(SIZE_OF_STACK_AREA_ENCBASE), 0);
        assert_eq!(r.read_varl_u(NUM_SAFE_POINTS_ENCBASE), 0);
        assert_eq!(r.read_varl_u(NUM_INTERRUPTIBLE_RANGES_ENCBASE), 2);
        let first = read_range(&mut r, 0);
        assert_eq!(first, (8, 73));
        let second = read_range(&mut r, first.1);
        assert_eq!(second, (84, 96));
        assert_eq!(r.read(1), 0, "no register slots");
        assert_eq!(r.read(1), 0, "no stack/untracked slots");
        assert!(blob.len() * 8 - r.bit < 8, "only padding remains");
    }
}
