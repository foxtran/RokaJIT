//! The x64 unwind encoder (step_07.7): renders [`UnwindInput`] into the
//! blob `allocUnwindInfo` consumes — a Windows AMD64 `UNWIND_INFO` struct
//! (references: `runtime/src/coreclr/jit/unwindamd64.cpp`,
//! `runtime/src/coreclr/vm/jitinterface.cpp` `CEEJitInfo::allocUnwindInfo`,
//! the unwinder in `runtime/src/coreclr/unwinder/amd64/unwinder.cpp`, and
//! `runtime/docs/design/coreclr/botr/stackwalking.md`).
//!
//! The JIT hands over only the `UNWIND_INFO`; the EE builds the
//! `RUNTIME_FUNCTION` itself, stamps `Flags = UNW_FLAG_EHANDLER |
//! UNW_FLAG_UHANDLER` over whatever we write (jitinterface.cpp:12301), and
//! appends the CLR personality-routine slot after our bytes (its
//! `reservePersonalityRoutineSpace` adds those 4 bytes to the reservation,
//! so our blob is exactly `4 + 2 * count` bytes — no padding).
//!
//! The blob describes the tier-0 frame contract's fixed prolog (`push rbp;
//! mov rbp, rsp; sub rsp, N` — see `codegen.rs`):
//!
//! - Unwind codes are stored in **reverse prolog order** (RyuJIT builds
//!   them by prepending, unwindamd64.cpp:268; the unwinder applies them in
//!   array order, so array order is undo order), each with the offset of
//!   the *end* of its prolog instruction (`unwindGetCurrentOffset` is
//!   called after emission; the unwinder applies a code when
//!   `prolog_offset >= code_offset`).
//! - `UWOP_SET_FPREG` with `FrameRegister = rbp`, `FrameOffset = 0`: after
//!   `mov rbp, rsp` the frame pointer IS the stack pointer, so unwinding
//!   is `rsp = rbp - 0` (unwinder.cpp:817-820).
//! - The stack allocation is `UWOP_ALLOC_SMALL` (≤ 128 bytes), else
//!   `UWOP_ALLOC_LARGE` (16-bit or 32-bit form), matching
//!   `unwindAllocStackWindows` (unwindamd64.cpp:321-360).
//!
//! Not yet encoded: cold fragments (no hot-cold split in tier 0), saved
//! non-volatiles (the tier-0 scratch pool is caller-saved only), epilog
//! codes (the x64 format has none).
//!
//! ## Funclets (10.6)
//!
//! Each `UnwindInput::funclets` entry becomes one standalone
//! `UNWIND_INFO` blob after the root blob — one `RUNTIME_FUNCTION` per
//! funclet, reported with `CorJitFuncKind::Handler` and hot-chunk-relative
//! offsets. A funclet
//! inherits the parent's rbp (the VM restores it from the walked context),
//! so its prolog is only `sub rsp, N` and its unwind info is a single
//! `UWOP_ALLOC_*` code at offset `prolog_len`: NO `UWOP_SET_FPREG`
//! (genFuncletProlog never calls unwindSetFrameReg) and no pushes (the VM
//! preserves non-volatiles). `FrameRegister`/`FrameOffset` stay 0. The
//! blob is NOT chained (`UNW_FLAG_CHAININFO` is R2R-only — verified:
//! unwindamd64.cpp only reads that flag in the disassembler, and
//! `CEEJitInfo::allocUnwindInfo` stamps `UNW_FLAG_EHANDLER|UNW_FLAG_UHANDLER`
//! plus the personality slot, jitinterface.cpp:12288-12301).

use rokajit::artifact::UnwindBlob;
use rokajit::error::{CompileError, CompileResult};
use rokajit::metadata::UnwindInput;
use rokajit_ee::enums::CorJitFuncKind;

// `UnwindOp` values (win64unwind.h:14-24).
const UWOP_PUSH_NONVOL: u8 = 0;
const UWOP_ALLOC_LARGE: u8 = 1;
const UWOP_ALLOC_SMALL: u8 = 2;
const UWOP_SET_FPREG: u8 = 3;

/// rbp's register number, used as `UWOP_PUSH_NONVOL`'s OpInfo and as the
/// `UNWIND_INFO::FrameRegister`.
const RBP: u8 = 5;

// End offsets of the frame contract's prolog instructions (the emitter
// always produces exactly `push rbp; mov rbp, rsp; sub rsp, N`:
// 55 / 48 89 E5 / 48 83 EC imm8 — or `48 81 EC imm32`, 7 bytes, when N
// doesn't fit the imm8 form, i.e. frames over 127 bytes; step_10.9).
const AFTER_PUSH_RBP: u8 = 1;
const AFTER_MOV_RBP_RSP: u8 = 4;

/// The largest `UWOP_ALLOC_SMALL` allocation (`opinfo * 8 + 8`, opinfo 4
/// bits) and the largest 16-bit-scaled `UWOP_ALLOC_LARGE`.
const ALLOC_SMALL_MAX: u32 = 128;
const ALLOC_LARGE_16_MAX: u32 = 0xFFFF * 8;

/// The largest frame allocation the encoder's imm8 `sub rsp` form covers
/// (`48 83 EC ib` — the assembler-canonical form for values that fit a
/// sign-extended byte; larger frames encode `48 81 EC id`, three bytes
/// longer).
const SUB_RSP_IMM8_MAX: u32 = 127;

/// The prolog length for a frame size: 1 (`push rbp`) + 3 (`mov rbp,
/// rsp`) + 4 or 7 (`sub rsp, N`).
fn prolog_len(frame_size: u32) -> u8 {
    if frame_size <= SUB_RSP_IMM8_MAX {
        8
    } else {
        11
    }
}

/// The `Target::encode_unwind_info` body for x64: the root-fragment blob
/// covering the main body (`[0, code_len)`), then one blob per funclet in
/// emission order.
pub fn encode(input: &UnwindInput) -> CompileResult<Vec<UnwindBlob>> {
    if !input.frame_size.is_multiple_of(8) {
        return Err(CompileError::Internal(
            "frame size outside the 8-aligned frame contract",
        ));
    }
    let prolog_len = prolog_len(input.frame_size);
    // Unwind codes, reverse prolog order (undo order): the allocation, the
    // frame-pointer establishment, the rbp push.
    let mut codes = Vec::new();
    if input.frame_size > 0 {
        push_alloc_code(&mut codes, prolog_len, input.frame_size)?;
    }
    push_code(&mut codes, AFTER_MOV_RBP_RSP, UWOP_SET_FPREG, 0);
    push_code(&mut codes, AFTER_PUSH_RBP, UWOP_PUSH_NONVOL, RBP);

    let count_of_unwind_codes = (codes.len() / 2) as u8;
    // UNWIND_INFO header: Version 1 (low 3 bits of byte 0); Flags 0 (the EE
    // overwrites them); FrameRegister rbp / FrameOffset 0 (byte 3 nibbles).
    let mut bytes = vec![1u8, prolog_len, count_of_unwind_codes, RBP];
    bytes.extend_from_slice(&codes);
    let mut blobs = vec![UnwindBlob {
        func_kind: CorJitFuncKind::Root,
        is_cold_code: false,
        start_offset: 0,
        end_offset: input.code_len,
        bytes,
    }];

    // Funclets (10.6): standalone UNWIND_INFO per funclet, one ALLOC code
    // only — no SET_FPREG (rbp is inherited from the parent frame) and no
    // pushes. See the module docs.
    for f in &input.funclets {
        let mut codes = Vec::new();
        push_alloc_code(&mut codes, f.prolog_len, f.sp_delta)?;
        let mut bytes = vec![
            1u8,
            f.prolog_len,
            (codes.len() / 2) as u8,
            0, // FrameRegister 0 / FrameOffset 0: unused without SET_FPREG
        ];
        bytes.extend_from_slice(&codes);
        blobs.push(UnwindBlob {
            func_kind: f.kind,
            is_cold_code: false,
            start_offset: f.start_offset,
            end_offset: f.end_offset,
            bytes,
        });
    }
    Ok(blobs)
}

/// One `UWOP_ALLOC_*` code (plus its trailing size word for the LARGE
/// forms) at `code_offset`, exactly `unwindAllocStackWindows`
/// (unwindamd64.cpp:321-359): ALLOC_SMALL for ≤ 128 bytes, the 16-bit
/// scaled ALLOC_LARGE to 0x7FFF8, the 32-bit form beyond. `size` is the
/// `sub rsp` amount: 8-aligned and ≥ 8 by the frame contracts.
fn push_alloc_code(codes: &mut Vec<u8>, code_offset: u8, size: u32) -> CompileResult<()> {
    if size < 8 || !size.is_multiple_of(8) {
        return Err(CompileError::Internal(
            "stack allocation outside the 8-aligned ≥ 8 contract",
        ));
    }
    if size <= ALLOC_SMALL_MAX {
        // ≤ 128 bytes, so the scaled size fits OpInfo's 4 bits.
        push_code(codes, code_offset, UWOP_ALLOC_SMALL, ((size - 8) / 8) as u8);
    } else if size <= ALLOC_LARGE_16_MAX {
        push_code(codes, code_offset, UWOP_ALLOC_LARGE, 0);
        codes.extend_from_slice(&((size / 8) as u16).to_le_bytes());
    } else {
        push_code(codes, code_offset, UWOP_ALLOC_LARGE, 1);
        codes.extend_from_slice(&size.to_le_bytes());
    }
    Ok(())
}

/// One `UNWIND_CODE`: offset byte, then UnwindOp in the low nibble and
/// OpInfo in the high nibble of the second byte (little-endian bitfield
/// layout).
fn push_code(codes: &mut Vec<u8>, code_offset: u8, op: u8, op_info: u8) {
    codes.push(code_offset);
    codes.push(op | (op_info << 4));
}

#[cfg(test)]
mod tests {
    //! Expected bytes are hand-derived from the UNWIND_INFO layout and the
    //! unwinder's per-op semantics (unwinder.cpp:762-830).

    use super::*;
    use rokajit::pipeline::FuncletInfo;

    fn input(frame_size: u32) -> UnwindInput {
        UnwindInput {
            frame_size,
            code_len: 73,
            funclets: Vec::new(),
        }
    }

    /// fib's real shape: 32-byte frame → ALLOC_SMALL (32-8)/8 = 3.
    #[test]
    fn fib_unwind_blob_is_byte_exact() {
        let blobs = encode(&input(32)).expect("encodes");
        assert_eq!(blobs.len(), 1);
        let blob = &blobs[0];
        assert_eq!(blob.func_kind, CorJitFuncKind::Root);
        assert!(!blob.is_cold_code);
        assert_eq!((blob.start_offset, blob.end_offset), (0, 73));
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x01, // Version 1, Flags 0
            0x08, // SizeOfProlog = 8
            0x03, // CountOfUnwindCodes = 3
            0x05, // FrameRegister = rbp, FrameOffset = 0
            0x08, 0x32, // @8:  UWOP_ALLOC_SMALL, opinfo 3 (32 bytes)
            0x04, 0x03, // @4:  UWOP_SET_FPREG
            0x01, 0x50, // @1:  UWOP_PUSH_NONVOL rbp
        ];
        assert_eq!(blob.bytes, expected);
    }

    /// Unwind simulation, mid-body (past the prolog): apply every code in
    /// array order over a synthetic context and check the frame chain.
    #[test]
    fn codes_actually_unwind_the_frame() {
        // Synthetic stack: return address at 1000, saved rbp at 992, then
        // the 32-byte frame below it; rbp = 992, rsp = 960.
        let mut stack = [0u64; 6];
        stack[5] = 0xDEAD; // return address
        stack[4] = 0xBEEF; // saved rbp
        let (mut rsp, mut rbp) = (960u64, 992u64);
        let read = |addr: u64| stack[((addr - 960) / 8) as usize];

        let blob = encode(&input(32)).expect("encodes").remove(0);
        let count = blob.bytes[2] as usize;
        for i in 0..count {
            let op = blob.bytes[4 + 2 * i + 1] & 0xF;
            let op_info = blob.bytes[4 + 2 * i + 1] >> 4;
            match op {
                UWOP_ALLOC_SMALL => rsp += u64::from(op_info) * 8 + 8,
                UWOP_SET_FPREG => rsp = rbp, // FrameOffset = 0
                UWOP_PUSH_NONVOL => {
                    assert_eq!(op_info, RBP);
                    rbp = read(rsp);
                    rsp += 8;
                }
                other => panic!("unexpected op {other}"),
            }
        }
        assert_eq!(rsp, 1000, "rsp now points at the return address");
        assert_eq!(rbp, 0xBEEF, "caller's rbp restored");
        assert_eq!(read(rsp), 0xDEAD);
    }

    #[test]
    fn zero_frame_omits_the_alloc_code() {
        let blob = encode(&input(0)).expect("encodes").remove(0);
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x01, 0x08, 0x02, 0x05,
            0x04, 0x03, // UWOP_SET_FPREG
            0x01, 0x50, // UWOP_PUSH_NONVOL rbp
        ];
        assert_eq!(blob.bytes, expected);
    }

    #[test]
    fn alloc_boundary_forms() {
        // 128: the largest ALLOC_SMALL — and the smallest frame whose
        // `sub rsp, N` needs the imm32 form (step_10.9: the prolog is 11
        // bytes, and the unwind code's offset tracks it).
        let blob = encode(&input(128)).expect("encodes").remove(0);
        assert_eq!(blob.bytes[1], 11, "SizeOfProlog follows the encoder");
        assert_eq!(blob.bytes[4..6], [0x0B, (15 << 4) | UWOP_ALLOC_SMALL]);
        assert_eq!(blob.bytes[2], 3);
        // 136: the smallest 16-bit ALLOC_LARGE (size/8 as a trailing u16).
        let blob = encode(&input(136)).expect("encodes").remove(0);
        assert_eq!(blob.bytes[2], 4, "the size word counts as a code slot");
        assert_eq!(&blob.bytes[4..8], &[0x0B, UWOP_ALLOC_LARGE, 17, 0]);
        // 0x80000: past the 16-bit form → opinfo 1, size as a u32.
        let blob = encode(&input(0x80000)).expect("encodes").remove(0);
        assert_eq!(blob.bytes[2], 5);
        assert_eq!(blob.bytes[5] >> 4, 1);
        assert_eq!(&blob.bytes[6..10], &0x80000u32.to_le_bytes());
        // 112: fits the imm8 form — the prolog stays 8 bytes.
        let blob = encode(&input(112)).expect("encodes").remove(0);
        assert_eq!(blob.bytes[1], 8);
    }

    #[test]
    fn unaligned_frame_is_an_internal_error() {
        assert!(matches!(encode(&input(12)), Err(CompileError::Internal(_))));
    }

    fn funclet(start: u32, end: u32, prolog_len: u8, sp_delta: u32) -> FuncletInfo {
        FuncletInfo {
            start_offset: start,
            end_offset: end,
            prolog_len,
            sp_delta,
            kind: CorJitFuncKind::Handler,
        }
    }

    /// A catch funclet: standalone UNWIND_INFO — Version 1, Flags 0 (the
    /// EE stamps them), one ALLOC_SMALL code at the end of the 4-byte
    /// `sub rsp, 16` prolog, FrameRegister/FrameOffset 0 (rbp is inherited
    /// from the parent frame — no SET_FPREG). The root blob keeps its
    /// current shape and covers the main body only.
    #[test]
    fn funclet_blob_is_byte_exact() {
        let mut i = input(32);
        i.funclets.push(funclet(73, 105, 4, 16));
        let blobs = encode(&i).expect("encodes");
        assert_eq!(blobs.len(), 2);

        let root = &blobs[0];
        assert_eq!(root.func_kind, CorJitFuncKind::Root);
        assert_eq!((root.start_offset, root.end_offset), (0, 73));
        assert_eq!(root.bytes.len(), 10, "the fib shape, unchanged");

        let f = &blobs[1];
        assert_eq!(f.func_kind, CorJitFuncKind::Handler);
        assert!(!f.is_cold_code);
        assert_eq!((f.start_offset, f.end_offset), (73, 105));
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x01, // Version 1, Flags 0
            0x04, // SizeOfProlog = 4 (`sub rsp, imm8`)
            0x01, // CountOfUnwindCodes = 1
            0x00, // FrameRegister 0, FrameOffset 0
            0x04, 0x12, // @4: UWOP_ALLOC_SMALL, opinfo 1 (16 bytes)
        ];
        assert_eq!(f.bytes, expected);
    }

    /// Unwind simulation of a funclet blob, mid-body: the single ALLOC
    /// code raises rsp by the funclet's allocation and leaves rbp (the
    /// PARENT's frame pointer) untouched.
    #[test]
    fn funclet_blob_unwinds_rsp_and_keeps_rbp() {
        let mut i = input(32);
        i.funclets.push(funclet(73, 105, 4, 16));
        let f = encode(&i).expect("encodes").remove(1);
        let (mut rsp, rbp) = (960u64, 992u64);
        let count = f.bytes[2] as usize;
        for i in 0..count {
            let op = f.bytes[4 + 2 * i + 1] & 0xF;
            let op_info = f.bytes[4 + 2 * i + 1] >> 4;
            match op {
                UWOP_ALLOC_SMALL => rsp += u64::from(op_info) * 8 + 8,
                other => panic!("unexpected op {other}"),
            }
        }
        assert_eq!(rsp, 976, "rsp += 16");
        assert_eq!(rbp, 992, "rbp inherited from the parent, untouched");
    }

    /// Funclet allocation boundary forms, exactly the root's: 128 is the
    /// largest ALLOC_SMALL; 136 takes the 16-bit ALLOC_LARGE (the size
    /// word counts as a code slot); past 0x7FFF8 the 32-bit form.
    #[test]
    fn funclet_alloc_boundary_forms() {
        let mut i = input(32);
        i.funclets.push(funclet(73, 100, 4, 128));
        let f = encode(&i).expect("encodes").remove(1);
        assert_eq!(f.bytes[2], 1);
        assert_eq!(f.bytes[4..6], [4, (15 << 4) | UWOP_ALLOC_SMALL]);

        let mut i = input(32);
        i.funclets.push(funclet(73, 100, 7, 136));
        let f = encode(&i).expect("encodes").remove(1);
        assert_eq!(f.bytes[1], 7, "SizeOfProlog follows the funclet prolog");
        assert_eq!(f.bytes[2], 2, "the size word counts as a code slot");
        assert_eq!(&f.bytes[4..8], &[7, UWOP_ALLOC_LARGE, 17, 0]);

        let mut i = input(32);
        i.funclets.push(funclet(73, 100, 7, 0x80000));
        let f = encode(&i).expect("encodes").remove(1);
        assert_eq!(f.bytes[2], 3);
        assert_eq!(f.bytes[5] >> 4, 1);
        assert_eq!(&f.bytes[6..10], &0x80000u32.to_le_bytes());
    }

    /// Two funclets: one blob each, in emission order after the root.
    #[test]
    fn funclets_emit_in_order_after_the_root() {
        let mut i = input(32);
        i.funclets.push(funclet(73, 105, 4, 16));
        i.funclets.push(funclet(105, 130, 4, 32));
        let blobs = encode(&i).expect("encodes");
        assert_eq!(blobs.len(), 3);
        assert_eq!(blobs[0].func_kind, CorJitFuncKind::Root);
        assert_eq!((blobs[1].start_offset, blobs[1].end_offset), (73, 105));
        assert_eq!((blobs[2].start_offset, blobs[2].end_offset), (105, 130));
        assert_eq!(blobs[2].bytes[4..6], [4, (3 << 4) | UWOP_ALLOC_SMALL]);
    }

    /// A filter funclet (step_11.11): the same standalone ALLOC-only
    /// blob — only the funcKind the EE sees differs (the EE uses it for
    /// debug overlap asserts; the VM learns filter-ness from the EH
    /// clause's FilterOffset).
    #[test]
    fn filter_funclet_blob_keeps_the_filter_kind() {
        let mut i = input(32);
        let mut f = funclet(73, 105, 4, 16);
        f.kind = CorJitFuncKind::Filter;
        i.funclets.push(f);
        let blobs = encode(&i).expect("encodes");
        assert_eq!(blobs.len(), 2);
        assert_eq!(blobs[1].func_kind, CorJitFuncKind::Filter);
        // The blob shape is byte-identical to a handler funclet's.
        let mut h = funclet(73, 105, 4, 16);
        h.kind = CorJitFuncKind::Handler;
        let mut j = input(32);
        j.funclets.push(h);
        let handler_blobs = encode(&j).expect("encodes");
        assert_eq!(blobs[1].bytes, handler_blobs[1].bytes);
    }

    /// The funclet contract: `sub rsp, N` with N 8-aligned and ≥ 8.
    #[test]
    fn bad_funclet_sp_delta_is_an_internal_error() {
        let mut i = input(32);
        i.funclets.push(funclet(73, 105, 4, 12));
        assert!(matches!(encode(&i), Err(CompileError::Internal(_))));
        let mut i = input(32);
        i.funclets.push(funclet(73, 105, 4, 0));
        assert!(matches!(encode(&i), Err(CompileError::Internal(_))));
    }
}
