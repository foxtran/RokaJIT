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
//!   two bits here.
//! - **Code length** as `varl_u(code_len, CODE_LENGTH_ENCBASE=8)`;
//!   `NORMALIZE_CODE_LENGTH` is the identity on AMD64.
//! - **Safepoints** (`NUM_SAFE_POINTS_ENCBASE=2`): the managed calls'
//!   return addresses, as recorded by codegen (the encoder's
//!   `callSite += m_pCallSiteSizes[...]`, gcinfoencoder.cpp:1100, already
//!   applied — [`GcInfoInput::safepoints`]). Each offset is written in
//!   `CeilOfLog2(code_len)` bits (`NORMALIZE_CODE_OFFSET` is the identity
//!   on AMD64).
//! - **Empty slot table**: one 0 bit for register slots, one 0 bit for
//!   stack/untracked slots (gcinfoencoder.cpp:1432-1447), then — zero used
//!   slots — `Build` jumps straight to its exit (gcinfoencoder.cpp:1449).
//!
//! Bit packing is LSB-first within each byte (`BitStreamWriter::Write`,
//! gcinfoencoder.h: the first bit written lands in bit 0 of the first
//! byte), matching `BitStreamReader` in gcinfodecoder.h.
//!
//! Not yet encoded (named causes, later steps): tracked GC-root slots
//! (register/stack slot tables, chunk live states), interruptible ranges,
//! and every fat-header feature.

use rokajit::error::{CompileError, CompileResult};
use rokajit::metadata::GcInfoInput;

/// `AMD64GcInfoEncoding::CODE_LENGTH_ENCBASE` (gcinfotypes.h:595).
const CODE_LENGTH_ENCBASE: u32 = 8;
/// `AMD64GcInfoEncoding::NUM_SAFE_POINTS_ENCBASE` (gcinfotypes.h:617).
const NUM_SAFE_POINTS_ENCBASE: u32 = 2;

/// The `Target::encode_gc_info` body for x64.
pub fn encode(input: &GcInfoInput) -> CompileResult<Vec<u8>> {
    if !input.gc_roots.is_empty() {
        return Err(CompileError::Unsupported(
            "GC-info slot tables for tracked references: a later step",
        ));
    }
    let mut w = BitWriter::new();
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
    // Slot table: no register slots, no stack or untracked slots.
    w.write(0, 1);
    w.write(0, 1);
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

    #[test]
    fn tracked_gc_roots_are_a_named_unsupported() {
        let mut i = input(9, &[]);
        i.gc_roots.push(rokajit::pipeline::GcRootSlot {
            offset: 16,
            is_byref: false,
            pinned: false,
        });
        assert!(matches!(encode(&i), Err(CompileError::Unsupported(_))));
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
}
