//! Byte-level x86-64 encoder (step_07.6): an Xbyak-shaped runtime
//! assembler (decisions/2026-09-11-x64-encoder-model.md) — typed
//! operands, one fluent call per instruction, label/fixup management
//! built in.
//!
//! The assembler speaks **physical operands only**: [`Gpr`] registers,
//! [`Mem`] addressing modes, and immediates. The
//! [`Inst`](crate::inst::Inst) descriptors lowering produces are
//! pre-allocation (`Val`s, symbolic frame slots, symbolic frame size);
//! binding those to physical operands is codegen's (07.5) duty, per the
//! `inst.rs` contract. The method set here covers exactly the forms the
//! fib-subset descriptors denote once bound:
//!
//! | `Inst` variant        | `Asm` entry points                          |
//! |-----------------------|---------------------------------------------|
//! | `Mov`                 | [`Asm::mov`]                                |
//! | `Lea`                 | [`Asm::lea`]                                |
//! | `Arith`               | [`Asm::add`] / [`Asm::sub`] / [`Asm::and`] / [`Asm::or`] / [`Asm::xor`] / [`Asm::imul`] (+ [`Asm::imul_imm`]) |
//! | `Cdq`                 | [`Asm::cdq`]                                |
//! | `Idiv` / `Div`        | [`Asm::idiv`] / [`Asm::div`]                |
//! | `Shift`               | [`Asm::shift_cl`] / [`Asm::shift_imm`]      |
//! | `Unary`               | [`Asm::neg`] / [`Asm::not`]                 |
//! | `Setcc`               | [`Asm::setcc`]                              |
//! | `MovExt`              | [`Asm::movsxd`] / a 32-bit [`Asm::mov`]     |
//! | `Cmp`                 | [`Asm::cmp`] (+ [`Asm::test`])              |
//! | `LoadMem` / `StoreMem` / `NullCheck` | [`Asm::mov`] (reg↔mem, through a scratch-GPR address) |
//! | `Jcc` / `Jmp`         | [`Asm::jcc`] / [`Asm::jmp`]                 |
//! | `CallDirect` / `CallHelper` | [`Asm::call`]                       |
//! | `Push` / `AllocFrame` | [`Asm::push`] / [`Asm::sub`] on `rsp`       |
//! | `Leave` / `Ret`       | [`Asm::leave`] / [`Asm::ret`]               |
//! | `ConstF`              | [`Asm::mov`] (GPR) + [`Asm::mov_gpr_to_xmm`] |
//! | `MovF`                | [`Asm::mov_f_load`] / [`Asm::mov_f_store`]  |
//! | `ArithF` / `NegF`     | [`Asm::arith_f`] / [`Asm::xor_f`]           |
//! | `CmpF`                | [`Asm::ucomis`]                             |
//! | `SetccF` / `JccF`     | [`Asm::setcc`] / [`Asm::jcc`] sequences     |
//! | `CvtIntToF` / `CvtFToInt` / `CvtFToF` | [`Asm::cvtsi2s`] / [`Asm::cvtts2si`] / [`Asm::cvts2s`] |
//!
//! REX/ModRM/SIB selection is table-driven: one generic ModRM/SIB/disp
//! encoder ([`encode_modrm`]) plus per-group rows ([`AluRow`],
//! [`jcc_tttn`]); no per-instruction bit-twiddling. Branches are rel32
//! everywhere (no shortening in tier 0): [`Asm::jcc`]/[`Asm::jmp`] record
//! fixups keyed on [`Label`], resolved by [`Asm::finalize`].
//! [`Asm::call`] emits `call rel32` with a placeholder and records a
//! [`CallReloc`]; 07.7 patches it once the EE supplies the target address
//! (the call site is also a GC safepoint, drained from codegen's records).

use crate::inst::{ArithFOp, CondCode, FWidth, Width};
use crate::regs::{Gpr, Xmm};
use rokajit::lower::Label;
use rokajit_ee::handles::MethodHandle;
use std::collections::HashMap;
use std::fmt;

/// Index scale factor in a SIB byte; the discriminant is the hardware
/// encoding (Intel SDM vol. 2A, table 2-3).
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Scale {
    X1 = 0,
    X2 = 1,
    X4 = 2,
    X8 = 3,
}

/// A memory operand, `[base + index*scale + disp]`. Every part but the
/// displacement is optional; the encoder picks the ModRM/SIB form (disp0
/// / disp8 / disp32, SIB when forced). Frame slots arrive from codegen
/// (07.5) as `Mem::base_disp(Gpr::Rbp, -off)`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Mem {
    base: Option<Gpr>,
    index: Option<(Gpr, Scale)>,
    disp: i32,
}

impl Mem {
    /// `[base]`.
    pub fn base(base: Gpr) -> Self {
        Self::base_disp(base, 0)
    }

    /// `[base + disp]`.
    pub fn base_disp(base: Gpr, disp: i32) -> Self {
        Mem {
            base: Some(base),
            index: None,
            disp,
        }
    }

    /// `[base + index*scale + disp]`.
    pub fn base_index_disp(base: Gpr, index: Gpr, scale: Scale, disp: i32) -> Self {
        Mem {
            base: Some(base),
            index: Some((index, scale)),
            disp,
        }
    }

    /// `[index*scale + disp]` — SIB with no base (mod=00, base=101,
    /// always disp32).
    pub fn index_disp(index: Gpr, scale: Scale, disp: i32) -> Self {
        Mem {
            base: None,
            index: Some((index, scale)),
            disp,
        }
    }

    /// Absolute `[disp32]` — SIB with no base and no index.
    pub fn abs(disp: i32) -> Self {
        Mem {
            base: None,
            index: None,
            disp,
        }
    }
}

/// A register-or-memory operand: the `r/m` side of a ModRM pair, and the
/// writable destination of every mutating instruction (the descriptor
/// contract's [`Place`](crate::inst::Place) after codegen binding).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Rm {
    Reg(Gpr),
    Mem(Mem),
}

/// A readable source: register, memory, or immediate (the descriptor
/// contract's [`Src`](crate::inst::Src) after codegen binding).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Rmi {
    Reg(Gpr),
    Mem(Mem),
    /// The payload follows the `inst.rs` rule: always `i64`; a 32-bit
    /// instruction truncates to imm32.
    Imm(i64),
}

impl From<Gpr> for Rm {
    fn from(reg: Gpr) -> Self {
        Rm::Reg(reg)
    }
}

impl From<Mem> for Rm {
    fn from(mem: Mem) -> Self {
        Rm::Mem(mem)
    }
}

impl From<Gpr> for Rmi {
    fn from(reg: Gpr) -> Self {
        Rmi::Reg(reg)
    }
}

impl From<Mem> for Rmi {
    fn from(mem: Mem) -> Self {
        Rmi::Mem(mem)
    }
}

impl From<i64> for Rmi {
    fn from(imm: i64) -> Self {
        Rmi::Imm(imm)
    }
}

/// A register-or-memory operand on the XMM side: the `r/m` of a scalar
/// SSE instruction ([`Rm`]'s floating-point twin).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RmX {
    Reg(Xmm),
    Mem(Mem),
}

impl From<Xmm> for RmX {
    fn from(reg: Xmm) -> Self {
        RmX::Reg(reg)
    }
}

impl From<Mem> for RmX {
    fn from(mem: Mem) -> Self {
        RmX::Mem(mem)
    }
}

/// A call-site relocation recorded by [`Asm::call`]: the rel32 field at
/// `offset` in the finalized buffer must be patched with the callee
/// address the EE supplies at emit time (07.7 drains these).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CallReloc {
    pub method: MethodHandle,
    /// Offset of the rel32 displacement field within the code buffer.
    pub offset: u32,
}

impl CallReloc {
    /// Patch the rel32 field once the buffer is at its final address:
    /// `rel32 = target - (code_base + offset + 4)`.
    pub fn patch(
        &self,
        code: &mut [u8],
        code_base: usize,
        target: usize,
    ) -> Result<(), EncodeError> {
        let rel = target as i64 - (code_base + self.offset as usize + 4) as i64;
        let rel = i32::try_from(rel).map_err(|_| EncodeError::Rel32OutOfRange {
            at: self.offset,
            displacement: rel,
        })?;
        let at = self.offset as usize;
        code[at..at + 4].copy_from_slice(&rel.to_le_bytes());
        Ok(())
    }
}

/// Why finalization failed.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum EncodeError {
    /// A branch references a label that was never bound.
    UnboundLabel(Label),
    /// A rel32 displacement does not fit in 32 bits (unreachable for one
    /// method's code; guarded anyway).
    Rel32OutOfRange { at: u32, displacement: i64 },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::UnboundLabel(label) => write!(f, "unbound label {label:?}"),
            EncodeError::Rel32OutOfRange { at, displacement } => {
                write!(f, "rel32 at {at} out of range: {displacement}")
            }
        }
    }
}

impl std::error::Error for EncodeError {}

/// The resolved code: bytes plus the call relocations 07.7 must patch
/// after the EE supplies target addresses.
pub struct FinalizedCode {
    pub bytes: Vec<u8>,
    pub call_relocs: Vec<CallReloc>,
}

/// A pending branch fixup: the rel32 field at `at` must be filled with
/// `target`'s bound offset minus the field's end.
struct Fixup {
    at: u32,
    target: Label,
}

/// The displacement form one ModRM encoding carries.
enum Disp {
    None,
    D8(i8),
    D32(i32),
}

/// One encoded ModRM/SIB/disp tail plus the REX bits it needs.
/// `rex` packs W/R/X/B in bits 3/2/1/0; 0 means no REX prefix at all.
struct ModRmEnc {
    rex: u8,
    modrm: u8,
    sib: Option<u8>,
    disp: Disp,
}

/// The XMM-side twin of [`encode_modrm`]: same ModRM/SIB/disp selection,
/// with an XMM register on the `r/m` side (Intel SDM vol. 2A — the SSE
/// scalar forms share the GPR ModRM encoding rules).
fn encode_modrm_x(wide: bool, reg: u8, rm: RmX) -> ModRmEnc {
    match rm {
        RmX::Reg(x) => {
            let mut rex = if wide { 0b1000 } else { 0 };
            rex |= (reg >> 3) << 2; // REX.R
            rex |= (x as u8) >> 3; // REX.B
            ModRmEnc {
                rex,
                modrm: 0xC0 | ((reg & 7) << 3) | (x as u8 & 7),
                sib: None,
                disp: Disp::None,
            }
        }
        RmX::Mem(mem) => {
            let rex = (if wide { 0b1000 } else { 0 }) | ((reg >> 3) << 2);
            encode_mem(rex, reg, mem)
        }
    }
}

/// The single ModRM/SIB/disp encoder every instruction form goes through
/// (table-driven core; Intel SDM vol. 2A tables 2-1..2-3). `reg` is the
/// full 4-bit reg field (bit 3 lands in REX.R).
fn encode_modrm(wide: bool, reg: u8, rm: Rm) -> ModRmEnc {
    let mut rex = if wide { 0b1000 } else { 0 };
    rex |= (reg >> 3) << 2; // REX.R
    match rm {
        Rm::Reg(r) => {
            rex |= (r as u8) >> 3; // REX.B
            ModRmEnc {
                rex,
                modrm: 0xC0 | ((reg & 7) << 3) | (r as u8 & 7),
                sib: None,
                disp: Disp::None,
            }
        }
        Rm::Mem(mem) => encode_mem(rex, reg, mem),
    }
}

fn encode_mem(mut rex: u8, reg: u8, mem: Mem) -> ModRmEnc {
    let reg_field = (reg & 7) << 3;
    let (index_enc, scale) = match mem.index {
        Some((index, scale)) => {
            rex |= ((index as u8) >> 3) << 1; // REX.X
            (index as u8 & 7, scale as u8)
        }
        // index=100 with no REX.X encodes "no index".
        None => (4, 0),
    };
    match mem.base {
        // No base: mod=00, rm=100, SIB base=101 → disp32 always.
        None => ModRmEnc {
            rex,
            modrm: reg_field | 4,
            sib: Some((scale << 6) | (index_enc << 3) | 5),
            disp: Disp::D32(mem.disp),
        },
        Some(base) => {
            let b = base as u8;
            rex |= b >> 3; // REX.B
                           // rsp/r12 (base & 7 == 4) require a SIB byte; so does any
                           // indexed form.
            let needs_sib = mem.index.is_some() || b & 7 == 4;
            let (mode, disp) = if mem.disp == 0 && b & 7 != 5 {
                (0, Disp::None)
            } else if i8::try_from(mem.disp).is_ok() {
                // rbp/r13 with disp 0 land here: mod=01, disp8 = 0.
                (1, Disp::D8(mem.disp as i8))
            } else {
                (2, Disp::D32(mem.disp))
            };
            let (rm_field, sib) = if needs_sib {
                (4, Some((scale << 6) | (index_enc << 3) | (b & 7)))
            } else {
                (b & 7, None)
            };
            ModRmEnc {
                rex,
                modrm: (mode << 6) | reg_field | rm_field,
                sib,
                disp,
            }
        }
    }
}

/// One row of the integer-ALU encoding table: the opcodes shared by
/// `add`/`sub`/`cmp`/`test`, which differ only in opcode number and
/// immediate-group extension (Intel SDM vol. 2A, opcode map).
struct AluRow {
    /// `op r/m, r` — dst is the r/m operand, src in the reg field.
    rm_r: u8,
    /// `op r, r/m` — dst in the reg field, src is memory. For `test`
    /// (symmetric operands) this repeats `rm_r`.
    r_rm: u8,
    /// Immediate form: `0x81` (imm32) / `0x83` (imm8) for the group-1
    /// instructions; `0xF7` for `test` (which has no imm8 form).
    imm_op: u8,
    /// The `/ext` reg field of the immediate form.
    imm_ext: u8,
}

const ADD_ROW: AluRow = AluRow {
    rm_r: 0x01,
    r_rm: 0x03,
    imm_op: 0x81,
    imm_ext: 0,
};
const SUB_ROW: AluRow = AluRow {
    rm_r: 0x29,
    r_rm: 0x2B,
    imm_op: 0x81,
    imm_ext: 5,
};
const CMP_ROW: AluRow = AluRow {
    rm_r: 0x39,
    r_rm: 0x3B,
    imm_op: 0x81,
    imm_ext: 7,
};
const TEST_ROW: AluRow = AluRow {
    rm_r: 0x85,
    r_rm: 0x85,
    imm_op: 0xF7,
    imm_ext: 0,
};
const AND_ROW: AluRow = AluRow {
    rm_r: 0x21,
    r_rm: 0x23,
    imm_op: 0x81,
    imm_ext: 4,
};
const OR_ROW: AluRow = AluRow {
    rm_r: 0x09,
    r_rm: 0x0B,
    imm_op: 0x81,
    imm_ext: 1,
};
const XOR_ROW: AluRow = AluRow {
    rm_r: 0x31,
    r_rm: 0x33,
    imm_op: 0x81,
    imm_ext: 6,
};

/// The `/ext` reg field of the shift group (`D1`/`C1`/`D3`) for a
/// [`ShiftOp`](crate::inst::ShiftOp) (Intel SDM vol. 2A).
fn shift_ext(op: crate::inst::ShiftOp) -> u8 {
    match op {
        crate::inst::ShiftOp::Shl => 4,
        crate::inst::ShiftOp::Shr => 5,
        crate::inst::ShiftOp::Sar => 7,
    }
}

/// The `tttn` low nibble of the `0F 8x`/`0F 9x` jcc/setcc opcode for a
/// [`CondCode`] (the descriptor contract's condition vocabulary →
/// hardware codes).
fn cc_tttn(cc: CondCode) -> u8 {
    match cc {
        CondCode::Eq => 0x4,
        CondCode::Ne => 0x5,
        CondCode::Lt => 0xC,
        CondCode::Le => 0xE,
        CondCode::Gt => 0xF,
        CondCode::Ge => 0xD,
        CondCode::ULt => 0x2,
        CondCode::ULe => 0x6,
        CondCode::UGt => 0x7,
        CondCode::UGe => 0x3,
        CondCode::Parity => 0xA,
        CondCode::NotParity => 0xB,
    }
}

/// The legacy opcode prefix for the scalar SSE forms that distinguish
/// `ss` from `sd` by prefix: F3 = single, F2 = double (Intel SDM vol. 2A).
fn sse_prefix(width: FWidth) -> u8 {
    match width {
        FWidth::S => 0xF3,
        FWidth::D => 0xF2,
    }
}

/// The assembler: a growable code buffer with label/fixup and call-
/// relocation management. One fluent call per emitted instruction.
#[derive(Default)]
pub struct Asm {
    buf: Vec<u8>,
    labels: HashMap<Label, u32>,
    fixups: Vec<Fixup>,
    call_relocs: Vec<CallReloc>,
}

impl Asm {
    pub fn new() -> Self {
        Asm::default()
    }

    /// The current write position (a bound label's value).
    pub fn offset(&self) -> u32 {
        self.buf.len() as u32
    }

    /// Bind a label at the current position. Rebinding is an internal
    /// bug (codegen binds each block label once).
    pub fn bind(&mut self, label: Label) {
        let prev = self.labels.insert(label, self.offset());
        assert!(prev.is_none(), "label {label:?} bound twice");
    }

    /// `mov dst, src` — reg/reg, reg/imm (imm32 with sign-extension, or
    /// imm64 via `movabs` when it doesn't fit i32), reg/mem in either
    /// direction, mem/imm (imm32).
    pub fn mov(&mut self, width: Width, dst: Rm, src: Rmi) {
        let wide = matches!(width, Width::W64);
        match (dst, src) {
            (dst, Rmi::Reg(s)) => self.emit_modrm_insn(wide, s as u8, dst, &[0x89]),
            (Rm::Reg(d), Rmi::Mem(m)) => self.emit_modrm_insn(wide, d as u8, Rm::Mem(m), &[0x8B]),
            (Rm::Reg(d), Rmi::Imm(v)) => self.mov_reg_imm(wide, d, v),
            (Rm::Mem(m), Rmi::Imm(v)) => {
                self.emit_modrm_insn(wide, 0, Rm::Mem(m), &[0xC7]);
                self.emit_u32(v as i32 as u32);
            }
            (Rm::Mem(_), Rmi::Mem(_)) => {
                unreachable!("x64 has no mem,mem mov; codegen must not produce one")
            }
        }
    }

    fn mov_reg_imm(&mut self, wide: bool, dst: Gpr, v: i64) {
        let d = dst as u8;
        if wide && i32::try_from(v).is_err() {
            // movabs: REX.W + B8+r, imm64.
            self.emit_u8(0x48 | (d >> 3));
            self.emit_u8(0xB8 | (d & 7));
            self.emit_u64(v as u64);
        } else if wide {
            // mov r/m64, imm32 (sign-extended) — the assembler-canonical
            // form for small constants.
            self.emit_modrm_insn(true, 0, Rm::Reg(dst), &[0xC7]);
            self.emit_u32(v as i32 as u32);
        } else {
            // B8+rd, imm32; REX.B only for r8d–r15d (never REX.W).
            if d >= 8 {
                self.emit_u8(0x41);
            }
            self.emit_u8(0xB8 | (d & 7));
            self.emit_u32(v as u32);
        }
    }

    /// `lea dst, [addr]` (always 64-bit).
    pub fn lea(&mut self, dst: Gpr, addr: Mem) {
        self.emit_modrm_insn(true, dst as u8, Rm::Mem(addr), &[0x8D]);
    }

    /// `add dst, src`.
    pub fn add(&mut self, width: Width, dst: Rm, src: Rmi) {
        self.alu(&ADD_ROW, width, dst, src);
    }

    /// `sub dst, src`.
    pub fn sub(&mut self, width: Width, dst: Rm, src: Rmi) {
        self.alu(&SUB_ROW, width, dst, src);
    }

    /// `cmp lhs, rhs` — sets the flags a following [`Asm::jcc`] consumes.
    pub fn cmp(&mut self, width: Width, lhs: Rm, rhs: Rmi) {
        self.alu(&CMP_ROW, width, lhs, rhs);
    }

    /// `test lhs, rhs` — reg/reg, mem/reg, reg/mem (symmetric), or
    /// reg-or-mem/imm (`F7 /0`, imm32; no imm8 form exists).
    pub fn test(&mut self, width: Width, lhs: Rm, rhs: Rmi) {
        self.alu(&TEST_ROW, width, lhs, rhs);
    }

    /// `and dst, src`.
    pub fn and(&mut self, width: Width, dst: Rm, src: Rmi) {
        self.alu(&AND_ROW, width, dst, src);
    }

    /// `or dst, src`.
    pub fn or(&mut self, width: Width, dst: Rm, src: Rmi) {
        self.alu(&OR_ROW, width, dst, src);
    }

    /// `xor dst, src`.
    pub fn xor(&mut self, width: Width, dst: Rm, src: Rmi) {
        self.alu(&XOR_ROW, width, dst, src);
    }

    /// `neg dst` (`F7 /3`).
    pub fn neg(&mut self, width: Width, dst: Rm) {
        let wide = matches!(width, Width::W64);
        self.emit_modrm_insn(wide, 3, dst, &[0xF7]);
    }

    /// `not dst` (`F7 /2`).
    pub fn not(&mut self, width: Width, dst: Rm) {
        let wide = matches!(width, Width::W64);
        self.emit_modrm_insn(wide, 2, dst, &[0xF7]);
    }

    /// `div divisor` — unsigned `rdx:rax / divisor` (`F7 /6`); no
    /// immediate form exists (the `inst.rs` contract makes that shape
    /// unreachable).
    pub fn div(&mut self, width: Width, divisor: Rm) {
        let wide = matches!(width, Width::W64);
        self.emit_modrm_insn(wide, 6, divisor, &[0xF7]);
    }

    /// `shl`/`shr`/`sar dst, cl` (`D3 /ext`) — the variable-count form;
    /// the count register is implicit (`cl`), as the descriptor's
    /// fixed-duty declaration records.
    pub fn shift_cl(&mut self, op: crate::inst::ShiftOp, width: Width, dst: Rm) {
        let wide = matches!(width, Width::W64);
        self.emit_modrm_insn(wide, shift_ext(op), dst, &[0xD3]);
    }

    /// `shl`/`shr`/`sar dst, imm8` — the constant-count form. A count of
    /// 1 uses the shorter `D1 /ext` encoding, as every assembler picks;
    /// other counts use `C1 /ext ib`. The hardware masks the count
    /// (5/6 bits by operand width) — ECMA-335's masking rule.
    pub fn shift_imm(&mut self, op: crate::inst::ShiftOp, width: Width, dst: Rm, count: i64) {
        let wide = matches!(width, Width::W64);
        if count == 1 {
            self.emit_modrm_insn(wide, shift_ext(op), dst, &[0xD1]);
        } else {
            self.emit_modrm_insn(wide, shift_ext(op), dst, &[0xC1]);
            self.emit_u8(count as u8);
        }
    }

    /// `setcc dst_low_byte` (`0F 9x /r`, 8-bit operand). A REX prefix is
    /// emitted when the register's low byte needs one (`sil`/`dil`/
    /// `r8b`+ — without REX, encodings 4–7 mean the legacy high bytes
    /// `ah`/`ch`/`dh`/`bh`).
    pub fn setcc(&mut self, cc: CondCode, dst: Gpr) {
        let d = dst as u8;
        if d >= 4 {
            self.emit_u8(0x40 | (d >> 3)); // REX (+ REX.B for r8b+)
        }
        self.emit_u8(0x0F);
        self.emit_u8(0x90 | cc_tttn(cc));
        self.emit_u8(0xC0 | (d & 7));
    }

    /// `movsxd dst, src` (`63 /r`, REX.W): sign-extend the 32-bit source
    /// into the 64-bit destination.
    pub fn movsxd(&mut self, dst: Gpr, src: Rm) {
        self.emit_modrm_insn(true, dst as u8, src, &[0x63]);
    }

    // ---- scalar SSE (x64 floating point; step_10.2) ----

    /// `movss`/`movsd dst, src` — the load/reg-reg form (`0F 10 /r`,
    /// F3 for `ss`, F2 for `sd`).
    pub fn mov_f_load(&mut self, width: FWidth, dst: Xmm, src: RmX) {
        self.emit_sse(sse_prefix(width), false, dst as u8, src, 0x10);
    }

    /// `movss`/`movsd [mem], src` — the store form (`0F 11 /r`).
    pub fn mov_f_store(&mut self, width: FWidth, dst: Mem, src: Xmm) {
        self.emit_sse(sse_prefix(width), false, src as u8, RmX::Mem(dst), 0x11);
    }

    /// `add`/`sub`/`mul`/`div`/`min`/`max ss|sd dst, src`
    /// (`0F 58/5C/59/5E/5D/5F /r`). Destructive two-operand: `dst` is both
    /// source and destination. `min`/`max` return the *second* operand on
    /// NaN — the property the float→unsigned conversions clamp with.
    pub fn arith_f(&mut self, op: ArithFOp, width: FWidth, dst: Xmm, src: RmX) {
        let opcode = match op {
            ArithFOp::Add => 0x58,
            ArithFOp::Sub => 0x5C,
            ArithFOp::Mul => 0x59,
            ArithFOp::Div => 0x5E,
            ArithFOp::Min => 0x5D,
            ArithFOp::Max => 0x5F,
        };
        self.emit_sse(sse_prefix(width), false, dst as u8, src, opcode);
    }

    /// `ucomiss`/`ucomisd lhs, rhs` (`0F 2E /r`; `ss` unprefixed, `sd`
    /// 66-prefixed). Sets ZF/PF/CF: unordered (NaN) reports all three set,
    /// which the parity-aware compare expansions in codegen key off.
    /// `ucomis*`, not `comis*`: quiet NaNs must not trap (#IA is only
    /// raised for SNaN, which IL never produces).
    pub fn ucomis(&mut self, width: FWidth, lhs: Xmm, rhs: RmX) {
        let prefix = match width {
            FWidth::S => 0,
            FWidth::D => 0x66,
        };
        self.emit_sse(prefix, false, lhs as u8, rhs, 0x2E);
    }

    /// `xorps`/`xorpd dst, src` (`0F 57 /r`; same prefix rule as
    /// [`Asm::ucomis`]). The float-`neg` mechanism: XOR with the sign mask.
    pub fn xor_f(&mut self, width: FWidth, dst: Xmm, src: RmX) {
        let prefix = match width {
            FWidth::S => 0,
            FWidth::D => 0x66,
        };
        self.emit_sse(prefix, false, dst as u8, src, 0x57);
    }

    /// `cvtsi2ss`/`cvtsi2sd dst, src` (`0F 2A /r`, F3/F2): signed integer
    /// to float. `src_w64` selects the 64-bit integer source (REX.W).
    pub fn cvtsi2s(&mut self, width: FWidth, dst: Xmm, src: Rm, src_w64: bool) {
        // The r/m side is a GPR or memory — the GPR-side encoder as-is.
        let enc = encode_modrm(src_w64, dst as u8, src);
        self.emit_sse_enc(sse_prefix(width), enc, 0x2A);
    }

    /// `cvttss2si`/`cvttsd2si dst, src` (`0F 2C /r`, F3/F2): float to
    /// integer, truncation toward zero; out-of-range/NaN yields the
    /// "integer indefinite" `0x8000…`. `dst_w64` selects the 64-bit
    /// destination (REX.W).
    pub fn cvtts2si(&mut self, width: FWidth, dst: Gpr, src: RmX, dst_w64: bool) {
        self.emit_sse(sse_prefix(width), dst_w64, dst as u8, src, 0x2C);
    }

    /// `cvtss2sd` (to `FWidth::D`, F3) / `cvtsd2ss` (to `FWidth::S`, F2)
    /// (`0F 5A /r`): the float↔double conversions.
    pub fn cvts2s(&mut self, to: FWidth, dst: Xmm, src: RmX) {
        // The prefix names the *source* width: F3 reads ss, F2 reads sd.
        let prefix = match to {
            FWidth::D => 0xF3,
            FWidth::S => 0xF2,
        };
        self.emit_sse(prefix, false, dst as u8, src, 0x5A);
    }

    /// `movq xmm, r64` (`FWidth::D`) / `movd xmm, r32` (`FWidth::S`) —
    /// `66 0F 6E /r`, REX.W for the 64-bit form. Bits cross from the GPR
    /// world unchanged; this is how float constants reach an XMM register.
    pub fn mov_gpr_to_xmm(&mut self, width: FWidth, dst: Xmm, src: Gpr) {
        let w64 = matches!(width, FWidth::D);
        let enc = encode_modrm(w64, dst as u8, Rm::Reg(src));
        self.emit_sse_enc(0x66, enc, 0x6E);
    }

    fn alu(&mut self, row: &AluRow, width: Width, dst: Rm, src: Rmi) {
        let wide = matches!(width, Width::W64);
        match (dst, src) {
            (dst, Rmi::Reg(s)) => self.emit_modrm_insn(wide, s as u8, dst, &[row.rm_r]),
            (Rm::Reg(d), Rmi::Mem(m)) => {
                self.emit_modrm_insn(wide, d as u8, Rm::Mem(m), &[row.r_rm]);
            }
            (dst, Rmi::Imm(v)) => {
                let imm32 = v as i32; // W32 truncation per the inst.rs rule
                if row.imm_op == 0x81 && i8::try_from(imm32).is_ok() {
                    // Group-1 imm8 (sign-extended) — the form every
                    // assembler picks when the value fits.
                    self.emit_modrm_insn(wide, row.imm_ext, dst, &[0x83]);
                    self.emit_u8(imm32 as u8);
                } else {
                    self.emit_modrm_insn(wide, row.imm_ext, dst, &[row.imm_op]);
                    self.emit_u32(imm32 as u32);
                }
            }
            (Rm::Mem(_), Rmi::Mem(_)) => {
                unreachable!("x64 has no mem,mem ALU form; codegen must not produce one")
            }
        }
    }

    /// `imul dst, src` — two-operand signed multiply (`0F AF`); dst is
    /// both source and destination (codegen emits the `mov`).
    pub fn imul(&mut self, width: Width, dst: Gpr, src: Rm) {
        let wide = matches!(width, Width::W64);
        self.emit_modrm_insn(wide, dst as u8, src, &[0x0F, 0xAF]);
    }

    /// `imul dst, lhs, imm` — three-operand signed multiply (`6B` imm8 /
    /// `69` imm32); dst := lhs * imm.
    pub fn imul_imm(&mut self, width: Width, dst: Gpr, lhs: Rm, imm: i64) {
        let wide = matches!(width, Width::W64);
        let imm32 = imm as i32;
        if i8::try_from(imm32).is_ok() {
            self.emit_modrm_insn(wide, dst as u8, lhs, &[0x6B]);
            self.emit_u8(imm32 as u8);
        } else {
            self.emit_modrm_insn(wide, dst as u8, lhs, &[0x69]);
            self.emit_u32(imm32 as u32);
        }
    }

    /// `idiv divisor` — `rdx:rax / divisor`; no immediate form exists
    /// (the `inst.rs` contract makes that shape unreachable).
    pub fn idiv(&mut self, width: Width, divisor: Rm) {
        let wide = matches!(width, Width::W64);
        self.emit_modrm_insn(wide, 7, divisor, &[0xF7]);
    }

    /// `cdq` (W32) / `cqo` (W64): sign-extend `rax` into `rdx:rax`.
    pub fn cdq(&mut self, width: Width) {
        if matches!(width, Width::W64) {
            self.emit_u8(0x48);
        }
        self.emit_u8(0x99);
    }

    /// `jcc target` — `0F 8x rel32`, fixup resolved at finalize.
    pub fn jcc(&mut self, cc: CondCode, target: Label) {
        self.emit_u8(0x0F);
        self.emit_u8(0x80 | cc_tttn(cc));
        self.emit_rel32_fixup(target);
    }

    /// `jmp target` — `E9 rel32`, fixup resolved at finalize.
    pub fn jmp(&mut self, target: Label) {
        self.emit_u8(0xE9);
        self.emit_rel32_fixup(target);
    }

    /// `call rel32` to a resolved method. The displacement is a
    /// placeholder; the recorded [`CallReloc`] is patched once the EE
    /// supplies the target address (see [`FinalizedCode`]).
    /// `call rel32` (E8): direct call to a method. The rel32 field stays
    /// zero here; it is patched by [`CallReloc::patch`] once the target
    /// address is known (07.7: by the EE, via the recorded relocation).
    pub fn call(&mut self, method: MethodHandle) {
        self.emit_u8(0xE8);
        let offset = self.offset();
        self.call_relocs.push(CallReloc { method, offset });
        self.emit_u32(0);
    }

    /// `call rel32` (E8) to an in-chunk label (a funclet entry,
    /// step_10.6): the displacement is a label fixup, resolved at
    /// finalize. No [`CallReloc`] — the target is not an EE address.
    pub fn call_label(&mut self, target: Label) {
        self.emit_u8(0xE8);
        self.emit_rel32_fixup(target);
    }

    /// `lea dst, [rip + rel32]` (8D /r with mod=00 r/m=101) — a code
    /// address materialized into a register (a catch funclet's resume
    /// address, step_10.6). The displacement is a label fixup; the
    /// finalize rule (`target − (field_end)`) is exactly rip-relative
    /// addressing, since rip reads as the next instruction's address.
    pub fn lea_rip(&mut self, dst: Gpr, target: Label) {
        let d = dst as u8;
        self.emit_u8(0x48 | ((d >> 3) << 2)); // REX.W (+ REX.R for r8+)
        self.emit_u8(0x8D);
        self.emit_u8(0x05 | ((d & 7) << 3)); // mod=00, r/m=101 → [rip + disp32]
        self.emit_rel32_fixup(target);
    }

    /// `call rel32` (E8) with a placeholder displacement and no
    /// [`CallReloc`] record: codegen records the relocation through its
    /// own drain, and helper calls have no method handle to key one on.
    pub fn call_unlinked(&mut self) {
        self.emit_u8(0xE8);
        self.emit_u32(0);
    }

    /// `call qword ptr [rip + rel32]` (FF /2, mod=00 r/m=101): the
    /// IAT_PVALUE form — the disp32 field points at the EE's entry-point
    /// *slot* (a precode target slot, whose contents the EE keeps current),
    /// patched by the EE through the recorded RELATIVE32 relocation exactly
    /// like the direct form. No [`CallReloc`] entry: the relocation is the
    /// drain's, keyed by offset, not by method handle.
    pub fn call_indirect(&mut self) {
        self.emit_u8(0xFF);
        self.emit_u8(0x15);
        self.emit_u32(0);
    }

    /// `push reg`.
    pub fn push(&mut self, reg: Gpr) {
        self.emit_op_plus_reg(0x50, reg);
    }

    /// `pop reg`.
    pub fn pop(&mut self, reg: Gpr) {
        self.emit_op_plus_reg(0x58, reg);
    }

    /// `leave` — the frame teardown matching `push rbp; mov rbp, rsp`.
    pub fn leave(&mut self) {
        self.emit_u8(0xC9);
    }

    /// `ret`.
    pub fn ret(&mut self) {
        self.emit_u8(0xC3);
    }

    /// `nop` — available for alignment.
    pub fn nop(&mut self) {
        self.emit_u8(0x90);
    }

    /// Resolve all branch fixups and return the code. Fails if a branch
    /// targets a label that was never bound.
    pub fn finalize(mut self) -> Result<FinalizedCode, EncodeError> {
        for fixup in &self.fixups {
            let target = *self
                .labels
                .get(&fixup.target)
                .ok_or(EncodeError::UnboundLabel(fixup.target))?;
            let rel = target as i64 - (fixup.at as i64 + 4);
            let rel = i32::try_from(rel).map_err(|_| EncodeError::Rel32OutOfRange {
                at: fixup.at,
                displacement: rel,
            })?;
            self.buf[fixup.at as usize..fixup.at as usize + 4].copy_from_slice(&rel.to_le_bytes());
        }
        Ok(FinalizedCode {
            bytes: self.buf,
            call_relocs: self.call_relocs,
        })
    }

    fn emit_rel32_fixup(&mut self, target: Label) {
        let at = self.offset();
        self.fixups.push(Fixup { at, target });
        self.emit_u32(0);
    }

    fn emit_op_plus_reg(&mut self, opcode: u8, reg: Gpr) {
        let r = reg as u8;
        if r >= 8 {
            self.emit_u8(0x41); // REX.B, never REX.W
        }
        self.emit_u8(opcode | (r & 7));
    }

    /// Emit `[rex?] opcode modrm [sib] [disp]` for one ModRM instruction.
    fn emit_modrm_insn(&mut self, wide: bool, reg: u8, rm: Rm, opcode: &[u8]) {
        let enc = encode_modrm(wide, reg, rm);
        if enc.rex != 0 {
            self.emit_u8(0x40 | enc.rex);
        }
        self.buf.extend_from_slice(opcode);
        self.emit_u8(enc.modrm);
        if let Some(sib) = enc.sib {
            self.emit_u8(sib);
        }
        match enc.disp {
            Disp::None => {}
            Disp::D8(d) => self.emit_u8(d as u8),
            Disp::D32(d) => self.emit_u32(d as u32),
        }
    }

    /// Emit `[prefix?] [rex?] 0F opcode modrm [sib] [disp]` — every scalar
    /// SSE form here is a two-byte `0F xx` opcode, optionally preceded by
    /// a legacy prefix (F2/F3/66; 0 = none). Prefixes precede REX (Intel
    /// SDM vol. 2A §2.2.1).
    fn emit_sse_enc(&mut self, prefix: u8, enc: ModRmEnc, opcode: u8) {
        if prefix != 0 {
            self.emit_u8(prefix);
        }
        if enc.rex != 0 {
            self.emit_u8(0x40 | enc.rex);
        }
        self.emit_u8(0x0F);
        self.emit_u8(opcode);
        self.emit_u8(enc.modrm);
        if let Some(sib) = enc.sib {
            self.emit_u8(sib);
        }
        match enc.disp {
            Disp::None => {}
            Disp::D8(d) => self.emit_u8(d as u8),
            Disp::D32(d) => self.emit_u32(d as u32),
        }
    }

    /// The XMM-operand half: encodes the `r/m` side from an [`RmX`].
    fn emit_sse(&mut self, prefix: u8, wide: bool, reg: u8, rm: RmX, opcode: u8) {
        let enc = encode_modrm_x(wide, reg, rm);
        self.emit_sse_enc(prefix, enc, opcode);
    }

    /// `movzx r32, r/m8` (`0F B6`) or `movzx r32, r/m16` (`0F B7`) — the
    /// narrow block-copy loads (step_10.9): a 1- or 2-byte chunk loads
    /// zero-extended into the full register.
    pub fn movzx_load(&mut self, size: u8, dst: Gpr, src: Mem) {
        let opcode = match size {
            1 => 0xB6,
            2 => 0xB7,
            _ => unreachable!("movzx_load covers 1- and 2-byte chunks"),
        };
        self.emit_modrm_insn(false, dst as u8, Rm::Mem(src), &[0x0F, opcode]);
    }

    /// `movsx r32, r/m8` (`0F BE`) or `movsx r32, r/m16` (`0F BF`) — the
    /// signed twin of [`Asm::movzx_load`]: sub-Int32 field loads of the
    /// I1/I2 metadata types sign-extend (ECMA-335 §III.1.1.1).
    pub fn movsx_load(&mut self, size: u8, dst: Gpr, src: Mem) {
        let opcode = match size {
            1 => 0xBE,
            2 => 0xBF,
            _ => unreachable!("movsx_load covers 1- and 2-byte chunks"),
        };
        self.emit_modrm_insn(false, dst as u8, Rm::Mem(src), &[0x0F, opcode]);
    }

    /// `mov r/m8, r8` (`88 /r`) or `mov r/m16, r16` (`66 89 /r`) — the
    /// narrow block-copy stores (step_10.9). The 8-bit form always carries
    /// a REX prefix so `sil`/`dil` and r8+ stay encodable.
    pub fn mov_store_narrow(&mut self, size: u8, dst: Mem, src: Gpr) {
        match size {
            1 => {
                let enc = encode_modrm(false, src as u8, Rm::Mem(dst));
                self.emit_u8(0x40 | enc.rex);
                self.emit_u8(0x88);
                self.emit_modrm_tail(enc);
            }
            2 => {
                self.emit_u8(0x66);
                let enc = encode_modrm(false, src as u8, Rm::Mem(dst));
                if enc.rex != 0 {
                    self.emit_u8(0x40 | enc.rex);
                }
                self.emit_u8(0x89);
                self.emit_modrm_tail(enc);
            }
            _ => unreachable!("mov_store_narrow covers 1- and 2-byte chunks"),
        }
    }

    /// The ModRM/SIB/displacement tail of an instruction whose opcode is
    /// already emitted.
    fn emit_modrm_tail(&mut self, enc: ModRmEnc) {
        self.emit_u8(enc.modrm);
        if let Some(sib) = enc.sib {
            self.emit_u8(sib);
        }
        match enc.disp {
            Disp::None => {}
            Disp::D8(d) => self.emit_u8(d as u8),
            Disp::D32(d) => self.emit_u32(d as u32),
        }
    }

    fn emit_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn emit_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn emit_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    //! Byte-exact tests per instruction form. Every expected byte string
    //! was produced by an independent assembler — `llvm-mc
    //! -triple=x86_64-unknown-linux-gnu -show-encoding` on the mnemonic in
    //! the test name/comment — and the reverse direction (our bytes
    //! disassembled with `objdump -D -b binary -m i386:x86-64`) was checked
    //! during development for every form here. REX edge cases (r8–r15,
    //! imm64 `mov`, SIB with no base) and negative rel32 branches are
    //! covered explicitly, per the step_07.6 acceptance list.

    use super::*;
    use crate::inst::Width::*;
    use rokajit::ir::BlockId;

    use Gpr::*;

    fn label(block: u32) -> Label {
        Label(BlockId(block))
    }

    /// Assemble `f` and finalize; fails the test on unbound labels.
    fn finish(f: impl FnOnce(&mut Asm)) -> Vec<u8> {
        let mut asm = Asm::new();
        f(&mut asm);
        asm.finalize().expect("all labels bound").bytes
    }

    // ---- narrow block-copy moves (step_10.9) ----

    #[test]
    fn movzx_load_forms() {
        // movzxb -4(%rbp), %eax
        assert_eq!(
            finish(|a| a.movzx_load(1, Rax, Mem::base_disp(Rbp, -4))),
            [0x0F, 0xB6, 0x45, 0xFC]
        );
        // movzxw -4(%rbp), %eax
        assert_eq!(
            finish(|a| a.movzx_load(2, Rax, Mem::base_disp(Rbp, -4))),
            [0x0F, 0xB7, 0x45, 0xFC]
        );
        // movzxb 8(%rsp), %r9d (REX.R, SIB)
        assert_eq!(
            finish(|a| a.movzx_load(1, R9, Mem::base_disp(Rsp, 8))),
            [0x44, 0x0F, 0xB6, 0x4C, 0x24, 0x08]
        );
    }

    #[test]
    fn mov_store_narrow_forms() {
        // movb %al, -4(%rbp) — a bare REX is always emitted so the
        // uniform register set (sil/dil/r8b+) stays encodable.
        assert_eq!(
            finish(|a| a.mov_store_narrow(1, Mem::base_disp(Rbp, -4), Rax)),
            [0x40, 0x88, 0x45, 0xFC]
        );
        // movb %sil, -4(%rbp)
        assert_eq!(
            finish(|a| a.mov_store_narrow(1, Mem::base_disp(Rbp, -4), Rsi)),
            [0x40, 0x88, 0x75, 0xFC]
        );
        // movb %r8b, -4(%rbp) (REX.R)
        assert_eq!(
            finish(|a| a.mov_store_narrow(1, Mem::base_disp(Rbp, -4), R8)),
            [0x44, 0x88, 0x45, 0xFC]
        );
        // movw %ax, 8(%rsp) (66 prefix, SIB)
        assert_eq!(
            finish(|a| a.mov_store_narrow(2, Mem::base_disp(Rsp, 8), Rax)),
            [0x66, 0x89, 0x44, 0x24, 0x08]
        );
    }

    // ---- mov reg/reg ----

    #[test]
    fn mov_rr() {
        // llvm-mc: movl %eax, %ecx
        assert_eq!(
            finish(|a| a.mov(W32, Rm::Reg(Rcx), Rmi::Reg(Rax))),
            [0x89, 0xC1]
        );
        // movq %rax, %rcx
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rcx), Rmi::Reg(Rax))),
            [0x48, 0x89, 0xC1]
        );
        // movq %rax, %r8 (REX.B)
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(R8), Rmi::Reg(Rax))),
            [0x49, 0x89, 0xC0]
        );
        // movq %r8, %rax (REX.R)
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Reg(R8))),
            [0x4C, 0x89, 0xC0]
        );
        // movq %r15, %r8 (REX.R+REX.B)
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(R8), Rmi::Reg(R15))),
            [0x4D, 0x89, 0xF8]
        );
        // movl %r8d, %r15d (REX.R+REX.B, no REX.W)
        assert_eq!(
            finish(|a| a.mov(W32, Rm::Reg(R15), Rmi::Reg(R8))),
            [0x45, 0x89, 0xC7]
        );
    }

    // ---- mov reg/imm ----

    #[test]
    fn mov_ri() {
        // movl $5, %eax
        assert_eq!(
            finish(|a| a.mov(W32, Rm::Reg(Rax), Rmi::Imm(5))),
            [0xB8, 5, 0, 0, 0]
        );
        // movl $-1, %r10d (REX.B, no REX.W)
        assert_eq!(
            finish(|a| a.mov(W32, Rm::Reg(R10), Rmi::Imm(-1))),
            [0x41, 0xBA, 0xFF, 0xFF, 0xFF, 0xFF]
        );
        // movq $5, %rax — sign-extended imm32 form (C7 /0)
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Imm(5))),
            [0x48, 0xC7, 0xC0, 5, 0, 0, 0]
        );
        // movq $-1, %rax
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Imm(-1))),
            [0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF]
        );
    }

    #[test]
    fn mov_ri_imm64_movabs() {
        // movabsq $0x123456789ABCDEF0, %rax (REX.W + B8+r, imm64)
        let bytes = finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Imm(0x1234_5678_9ABC_DEF0)));
        assert_eq!(
            bytes,
            [0x48, 0xB8, 0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12]
        );
        // movabsq $0x123456789ABCDEF0, %r11 (REX.W + REX.B)
        let bytes = finish(|a| a.mov(W64, Rm::Reg(R11), Rmi::Imm(0x1234_5678_9ABC_DEF0)));
        assert_eq!(
            bytes,
            [0x49, 0xBB, 0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12]
        );
        // The boundary: i32::MAX still uses the imm32 form…
        let bytes = finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Imm(i32::MAX as i64)));
        assert_eq!(bytes, [0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0x7F]);
        // …i32::MAX + 1 forces imm64.
        let bytes = finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Imm(i32::MAX as i64 + 1)));
        assert_eq!(bytes, [0x48, 0xB8, 0x00, 0x00, 0x00, 0x80, 0, 0, 0, 0]);
    }

    #[test]
    fn mov_ri_w32_truncates_the_i64_payload() {
        // inst.rs rule: Src::Imm is i64, W32 truncates to imm32.
        // Same bytes as movl $5, %eax above.
        let bytes = finish(|a| a.mov(W32, Rm::Reg(Rax), Rmi::Imm(0x1_0000_0005)));
        assert_eq!(bytes, [0xB8, 5, 0, 0, 0]);
    }

    // ---- mov mem forms (frame slots are [rbp - off]) ----

    #[test]
    fn mov_mem_forms() {
        let slot8 = Mem::base_disp(Rbp, -8);
        // movq -8(%rbp), %rax
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Mem(slot8))),
            [0x48, 0x8B, 0x45, 0xF8]
        );
        // movq %rax, -8(%rbp)
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Mem(slot8), Rmi::Reg(Rax))),
            [0x48, 0x89, 0x45, 0xF8]
        );
        // movl %eax, -8(%rbp)
        assert_eq!(
            finish(|a| a.mov(W32, Rm::Mem(slot8), Rmi::Reg(Rax))),
            [0x89, 0x45, 0xF8]
        );
        // movq %r9, -8(%rbp) (REX.R)
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Mem(slot8), Rmi::Reg(R9))),
            [0x4C, 0x89, 0x4D, 0xF8]
        );
        // movq -8(%rbp), %r9 (REX.R)
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(R9), Rmi::Mem(slot8))),
            [0x4C, 0x8B, 0x4D, 0xF8]
        );
        // movq $5, -8(%rbp) (C7 /0, imm32)
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Mem(slot8), Rmi::Imm(5))),
            [0x48, 0xC7, 0x45, 0xF8, 5, 0, 0, 0]
        );
    }

    #[test]
    fn mov_mem_displacement_widths() {
        // movq (%rbp), %rax — rbp/r13 with disp 0 force mod=01 disp8.
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Mem(Mem::base(Rbp)))),
            [0x48, 0x8B, 0x45, 0x00]
        );
        // movq (%r13), %rax (REX.B, same forced disp8)
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Mem(Mem::base(R13)))),
            [0x49, 0x8B, 0x45, 0x00]
        );
        // movq (%rbx), %rax — ordinary base, no displacement.
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Mem(Mem::base(Rbx)))),
            [0x48, 0x8B, 0x03]
        );
        // movq -200(%rbp), %rax — disp8 overflow → mod=10 disp32.
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Mem(Mem::base_disp(Rbp, -200)))),
            [0x48, 0x8B, 0x85, 0x38, 0xFF, 0xFF, 0xFF]
        );
        // movq -128(%rbp), %rax — the disp8 boundary: -128 still fits.
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Mem(Mem::base_disp(Rbp, -128)))),
            [0x48, 0x8B, 0x45, 0x80]
        );
        // movl 0x1234(%rbx), %eax — disp32 without REX.
        assert_eq!(
            finish(|a| a.mov(W32, Rm::Reg(Rax), Rmi::Mem(Mem::base_disp(Rbx, 0x1234)))),
            [0x8B, 0x83, 0x34, 0x12, 0x00, 0x00]
        );
    }

    #[test]
    fn mov_mem_sib_forms() {
        // movq 8(%rsp), %rax — rsp/r12 base forces a SIB byte.
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Mem(Mem::base_disp(Rsp, 8)))),
            [0x48, 0x8B, 0x44, 0x24, 0x08]
        );
        // movq 8(%r12), %rax (REX.B + SIB)
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Mem(Mem::base_disp(R12, 8)))),
            [0x49, 0x8B, 0x44, 0x24, 0x08]
        );
        // movq %rax, (%rsp) — disp0 with SIB.
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Mem(Mem::base(Rsp)), Rmi::Reg(Rax))),
            [0x48, 0x89, 0x04, 0x24]
        );
        // movq -8(%rbp,%rcx,2), %rax — base + index.
        assert_eq!(
            finish(|a| {
                a.mov(
                    W64,
                    Rm::Reg(Rax),
                    Rmi::Mem(Mem::base_index_disp(Rbp, Rcx, Scale::X2, -8)),
                );
            }),
            [0x48, 0x8B, 0x44, 0x4D, 0xF8]
        );
    }

    #[test]
    fn mov_mem_sib_no_base() {
        // movq 0x20(,%rcx,4), %rax — SIB no base: mod=00 rm=100, disp32.
        assert_eq!(
            finish(|a| a.mov(
                W64,
                Rm::Reg(Rax),
                Rmi::Mem(Mem::index_disp(Rcx, Scale::X4, 0x20))
            )),
            [0x48, 0x8B, 0x04, 0x8D, 0x20, 0x00, 0x00, 0x00]
        );
        // movq 0x20(,%r9,8), %rax — high index register (REX.X).
        assert_eq!(
            finish(|a| a.mov(
                W64,
                Rm::Reg(Rax),
                Rmi::Mem(Mem::index_disp(R9, Scale::X8, 0x20))
            )),
            [0x4A, 0x8B, 0x04, 0xCD, 0x20, 0x00, 0x00, 0x00]
        );
        // movabsq 0x1234, %rax — absolute [disp32]: SIB index=100 base=101.
        assert_eq!(
            finish(|a| a.mov(W64, Rm::Reg(Rax), Rmi::Mem(Mem::abs(0x1234)))),
            [0x48, 0x8B, 0x04, 0x25, 0x34, 0x12, 0x00, 0x00]
        );
    }

    // ---- lea ----

    #[test]
    fn lea_frame_slot_address() {
        // leaq -0x20(%rbp), %r11 (REX.W + REX.R)
        assert_eq!(
            finish(|a| a.lea(R11, Mem::base_disp(Rbp, -0x20))),
            [0x4C, 0x8D, 0x5D, 0xE0]
        );
        // leaq (%rbx), %rax
        assert_eq!(finish(|a| a.lea(Rax, Mem::base(Rbx))), [0x48, 0x8D, 0x03]);
    }

    // ---- add / sub (and the shared ALU machinery) ----

    #[test]
    fn add_forms() {
        // addl %ecx, %eax
        assert_eq!(
            finish(|a| a.add(W32, Rm::Reg(Rax), Rmi::Reg(Rcx))),
            [0x01, 0xC8]
        );
        // addq %rcx, %rax
        assert_eq!(
            finish(|a| a.add(W64, Rm::Reg(Rax), Rmi::Reg(Rcx))),
            [0x48, 0x01, 0xC8]
        );
        // addl %r10d, %r9d
        assert_eq!(
            finish(|a| a.add(W32, Rm::Reg(R9), Rmi::Reg(R10))),
            [0x45, 0x01, 0xD1]
        );
        // addq $5, %rax — group-1 imm8 (83 /0), as every assembler picks.
        assert_eq!(
            finish(|a| a.add(W64, Rm::Reg(Rax), Rmi::Imm(5))),
            [0x48, 0x83, 0xC0, 0x05]
        );
        // addq $0x1234, %rcx — imm8 overflow → 81 /0 imm32. (Not rax:
        // llvm-mc picks the accumulator-special 05 id there; the group-1
        // form is what the encoder emits, so test it on rcx.)
        assert_eq!(
            finish(|a| a.add(W64, Rm::Reg(Rcx), Rmi::Imm(0x1234))),
            [0x48, 0x81, 0xC1, 0x34, 0x12, 0x00, 0x00]
        );
        // addq $-1, %rax — negative imm8.
        assert_eq!(
            finish(|a| a.add(W64, Rm::Reg(Rax), Rmi::Imm(-1))),
            [0x48, 0x83, 0xC0, 0xFF]
        );
        // addl $5, %r9d (REX.B)
        assert_eq!(
            finish(|a| a.add(W32, Rm::Reg(R9), Rmi::Imm(5))),
            [0x41, 0x83, 0xC1, 0x05]
        );
        // addq $1, -8(%rbp) — immediate to a frame slot.
        assert_eq!(
            finish(|a| a.add(W64, Rm::Mem(Mem::base_disp(Rbp, -8)), Rmi::Imm(1))),
            [0x48, 0x83, 0x45, 0xF8, 0x01]
        );
        // addq -8(%rbp), %rax — the load form (03 /r).
        assert_eq!(
            finish(|a| a.add(W64, Rm::Reg(Rax), Rmi::Mem(Mem::base_disp(Rbp, -8)))),
            [0x48, 0x03, 0x45, 0xF8]
        );
        // W32 truncates the i64 immediate payload: addl $5, %eax.
        assert_eq!(
            finish(|a| a.add(W32, Rm::Reg(Rax), Rmi::Imm(0x1_0000_0005))),
            [0x83, 0xC0, 0x05]
        );
    }

    #[test]
    fn sub_forms() {
        // subq %rax, %rdx
        assert_eq!(
            finish(|a| a.sub(W64, Rm::Reg(Rdx), Rmi::Reg(Rax))),
            [0x48, 0x29, 0xC2]
        );
        // subq $0x28, %rsp — the AllocFrame shape codegen emits.
        assert_eq!(
            finish(|a| a.sub(W64, Rm::Reg(Rsp), Rmi::Imm(0x28))),
            [0x48, 0x83, 0xEC, 0x28]
        );
        // subq $0x100, %rsp — larger frame, imm32 form.
        assert_eq!(
            finish(|a| a.sub(W64, Rm::Reg(Rsp), Rmi::Imm(0x100))),
            [0x48, 0x81, 0xEC, 0x00, 0x01, 0x00, 0x00]
        );
        // subl $0x80, %ecx — 0x80 does not fit a *signed* imm8. (On rcx:
        // rax would take the accumulator-special 2D id form.)
        assert_eq!(
            finish(|a| a.sub(W32, Rm::Reg(Rcx), Rmi::Imm(0x80))),
            [0x81, 0xE9, 0x80, 0x00, 0x00, 0x00]
        );
        // subq -8(%rbp), %rax — load form (2B /r).
        assert_eq!(
            finish(|a| a.sub(W64, Rm::Reg(Rax), Rmi::Mem(Mem::base_disp(Rbp, -8)))),
            [0x48, 0x2B, 0x45, 0xF8]
        );
    }

    // ---- imul ----

    #[test]
    fn imul_forms() {
        // imulq %rcx, %rax (0F AF /r)
        assert_eq!(
            finish(|a| a.imul(W64, Rax, Rm::Reg(Rcx))),
            [0x48, 0x0F, 0xAF, 0xC1]
        );
        // imull %edx, %eax
        assert_eq!(
            finish(|a| a.imul(W32, Rax, Rm::Reg(Rdx))),
            [0x0F, 0xAF, 0xC2]
        );
        // imulq %r9, %r10 (REX.W+R+B)
        assert_eq!(
            finish(|a| a.imul(W64, R10, Rm::Reg(R9))),
            [0x4D, 0x0F, 0xAF, 0xD1]
        );
        // imulq -8(%rbp), %rax — memory source.
        assert_eq!(
            finish(|a| a.imul(W64, Rax, Rm::Mem(Mem::base_disp(Rbp, -8)))),
            [0x48, 0x0F, 0xAF, 0x45, 0xF8]
        );
        // imulq $5, %rcx, %rax — three-operand imm8 (6B /r).
        assert_eq!(
            finish(|a| a.imul_imm(W64, Rax, Rm::Reg(Rcx), 5)),
            [0x48, 0x6B, 0xC1, 0x05]
        );
        // imulq $0x1234, %rcx, %rax — imm32 (69 /r).
        assert_eq!(
            finish(|a| a.imul_imm(W64, Rax, Rm::Reg(Rcx), 0x1234)),
            [0x48, 0x69, 0xC1, 0x34, 0x12, 0x00, 0x00]
        );
        // imulq $-3, %r9, %r8 — negative imm8, REX.W+R+B.
        assert_eq!(
            finish(|a| a.imul_imm(W64, R8, Rm::Reg(R9), -3)),
            [0x4D, 0x6B, 0xC1, 0xFD]
        );
    }

    // ---- cmp / test ----

    #[test]
    fn cmp_forms() {
        // cmpl %ecx, %eax
        assert_eq!(
            finish(|a| a.cmp(W32, Rm::Reg(Rax), Rmi::Reg(Rcx))),
            [0x39, 0xC8]
        );
        // cmpq %r8, %r15 (REX.R+B)
        assert_eq!(
            finish(|a| a.cmp(W64, Rm::Reg(R15), Rmi::Reg(R8))),
            [0x4D, 0x39, 0xC7]
        );
        // cmpq $0, %rax
        assert_eq!(
            finish(|a| a.cmp(W64, Rm::Reg(Rax), Rmi::Imm(0))),
            [0x48, 0x83, 0xF8, 0x00]
        );
        // cmpq $0x1234, %rdi — the fib recursion guard shape, imm32 form.
        assert_eq!(
            finish(|a| a.cmp(W64, Rm::Reg(Rdi), Rmi::Imm(0x1234))),
            [0x48, 0x81, 0xFF, 0x34, 0x12, 0x00, 0x00]
        );
        // cmpl $0, -4(%rbp)
        assert_eq!(
            finish(|a| a.cmp(W32, Rm::Mem(Mem::base_disp(Rbp, -4)), Rmi::Imm(0))),
            [0x83, 0x7D, 0xFC, 0x00]
        );
        // cmpq -8(%rbp), %rax — load form (3B /r).
        assert_eq!(
            finish(|a| a.cmp(W64, Rm::Reg(Rax), Rmi::Mem(Mem::base_disp(Rbp, -8)))),
            [0x48, 0x3B, 0x45, 0xF8]
        );
    }

    #[test]
    fn test_forms() {
        // testl %eax, %eax
        assert_eq!(
            finish(|a| a.test(W32, Rm::Reg(Rax), Rmi::Reg(Rax))),
            [0x85, 0xC0]
        );
        // testq %rax, %rax
        assert_eq!(
            finish(|a| a.test(W64, Rm::Reg(Rax), Rmi::Reg(Rax))),
            [0x48, 0x85, 0xC0]
        );
        // testq $1, %rcx — F7 /0 imm32 (no imm8 form exists). (On rcx:
        // rax would take the accumulator-special A9 id form.)
        assert_eq!(
            finish(|a| a.test(W64, Rm::Reg(Rcx), Rmi::Imm(1))),
            [0x48, 0xF7, 0xC1, 0x01, 0x00, 0x00, 0x00]
        );
        // testq $1, -8(%rbp)
        assert_eq!(
            finish(|a| a.test(W64, Rm::Mem(Mem::base_disp(Rbp, -8)), Rmi::Imm(1))),
            [0x48, 0xF7, 0x45, 0xF8, 0x01, 0x00, 0x00, 0x00]
        );
        // testq %rax, -8(%rbp) — 85 is symmetric; reg source, mem dst.
        assert_eq!(
            finish(|a| a.test(W64, Rm::Mem(Mem::base_disp(Rbp, -8)), Rmi::Reg(Rax))),
            [0x48, 0x85, 0x45, 0xF8]
        );
    }

    // ---- idiv / cdq ----

    #[test]
    fn idiv_and_cdq_forms() {
        // cltd
        assert_eq!(finish(|a| a.cdq(W32)), [0x99]);
        // cqto
        assert_eq!(finish(|a| a.cdq(W64)), [0x48, 0x99]);
        // idivl %ecx
        assert_eq!(finish(|a| a.idiv(W32, Rm::Reg(Rcx))), [0xF7, 0xF9]);
        // idivq %rcx
        assert_eq!(finish(|a| a.idiv(W64, Rm::Reg(Rcx))), [0x48, 0xF7, 0xF9]);
        // idivl %r8d (REX.B)
        assert_eq!(finish(|a| a.idiv(W32, Rm::Reg(R8))), [0x41, 0xF7, 0xF8]);
        // idivl -4(%rbp)
        assert_eq!(
            finish(|a| a.idiv(W32, Rm::Mem(Mem::base_disp(Rbp, -4)))),
            [0xF7, 0x7D, 0xFC]
        );
    }

    // ---- prolog/epilog singletons ----

    #[test]
    fn push_pop_forms() {
        // pushq %rbp
        assert_eq!(finish(|a| a.push(Rbp)), [0x55]);
        // pushq %r12 (REX.B, no REX.W)
        assert_eq!(finish(|a| a.push(R12)), [0x41, 0x54]);
        // pushq %r15
        assert_eq!(finish(|a| a.push(R15)), [0x41, 0x57]);
        // popq %rbp
        assert_eq!(finish(|a| a.pop(Rbp)), [0x5D]);
        // popq %r12
        assert_eq!(finish(|a| a.pop(R12)), [0x41, 0x5C]);
    }

    #[test]
    fn leave_ret_nop() {
        assert_eq!(finish(|a| a.leave()), [0xC9]);
        assert_eq!(finish(|a| a.ret()), [0xC3]);
        assert_eq!(finish(|a| a.nop()), [0x90]);
    }

    // ---- branches: labels, fixups, negative rel32 ----

    #[test]
    fn jmp_forward_fixup() {
        // jmp l; nop; l: ret — disp = 1 (skips the nop).
        let bytes = finish(|a| {
            a.jmp(label(1));
            a.nop();
            a.bind(label(1));
            a.ret();
        });
        assert_eq!(bytes, [0xE9, 0x01, 0x00, 0x00, 0x00, 0x90, 0xC3]);
    }

    #[test]
    fn jmp_backward_negative_rel32() {
        // l: nop; jmp l — the branch spans itself; disp = -6.
        let bytes = finish(|a| {
            a.bind(label(0));
            a.nop();
            a.jmp(label(0));
        });
        assert_eq!(bytes, [0x90, 0xE9, 0xFA, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn jcc_opcode_table_and_negative_rel32() {
        // Every CondCode in one pass: l: jcc l — disp = -6 from the end
        // of each 6-byte instruction; pins the 0F 8x tttn nibble.
        let cases: [(CondCode, u8); 10] = [
            (CondCode::Eq, 0x84),  // je
            (CondCode::Ne, 0x85),  // jne
            (CondCode::Lt, 0x8C),  // jl
            (CondCode::Le, 0x8E),  // jle
            (CondCode::Gt, 0x8F),  // jg
            (CondCode::Ge, 0x8D),  // jge
            (CondCode::ULt, 0x82), // jb
            (CondCode::ULe, 0x86), // jbe
            (CondCode::UGt, 0x87), // ja
            (CondCode::UGe, 0x83), // jae
        ];
        for (cc, opcode) in cases {
            let bytes = finish(|a| {
                a.bind(label(0));
                a.jcc(cc, label(0));
            });
            assert_eq!(bytes, [0x0F, opcode, 0xFA, 0xFF, 0xFF, 0xFF], "{cc:?}");
        }
    }

    #[test]
    fn loop_skeleton_backward_branch() {
        // l: subl $1, %eax; testl %eax, %eax; jne l; ret
        let bytes = finish(|a| {
            a.bind(label(0));
            a.sub(W32, Rm::Reg(Rax), Rmi::Imm(1));
            a.test(W32, Rm::Reg(Rax), Rmi::Reg(Rax));
            a.jcc(CondCode::Ne, label(0));
            a.ret();
        });
        assert_eq!(
            bytes,
            [
                0x83, 0xE8, 0x01, // subl $1, %eax
                0x85, 0xC0, // testl %eax, %eax
                0x0F, 0x85, 0xF5, 0xFF, 0xFF, 0xFF, // jne l (disp = -11)
                0xC3,
            ]
        );
    }

    // ---- step_10.6: funclet call / resume-address forms ----

    #[test]
    fn call_label_fixup() {
        // call l; nop; l: ret — disp = 1 from the rel32 field's end.
        let bytes = finish(|a| {
            a.call_label(label(1));
            a.nop();
            a.bind(label(1));
            a.ret();
        });
        assert_eq!(bytes, [0xE8, 0x01, 0x00, 0x00, 0x00, 0x90, 0xC3]);
    }

    #[test]
    fn lea_rip_fixup() {
        // leaq [rip + l], %rax; nop; l: ret — the target is the address
        // of `ret`: disp = 1 (the field ends at the nop, llvm-mc:
        // `lea 1f(%rip), %rax`).
        let bytes = finish(|a| {
            a.lea_rip(Rax, label(1));
            a.nop();
            a.bind(label(1));
            a.ret();
        });
        assert_eq!(
            bytes,
            [0x48, 0x8D, 0x05, 0x01, 0x00, 0x00, 0x00, 0x90, 0xC3]
        );
        // A high destination register: REX.R (leaq 1f(%rip), %r9).
        let bytes = finish(|a| {
            a.lea_rip(R9, label(1));
            a.bind(label(1));
        });
        assert_eq!(bytes, [0x4C, 0x8D, 0x0D, 0x00, 0x00, 0x00, 0x00]);
        // Backward reference: l: nop; leaq [rip + l], %rax — disp = -8.
        let bytes = finish(|a| {
            a.bind(label(0));
            a.nop();
            a.lea_rip(Rax, label(0));
        });
        assert_eq!(bytes, [0x90, 0x48, 0x8D, 0x05, 0xF8, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn unbound_label_is_an_error_not_a_panic() {
        let mut asm = Asm::new();
        asm.jmp(label(7));
        assert_eq!(
            asm.finalize().map(|_| ()).unwrap_err(),
            EncodeError::UnboundLabel(label(7))
        );
    }

    #[test]
    #[should_panic(expected = "bound twice")]
    fn rebinding_a_label_is_rejected() {
        let mut asm = Asm::new();
        asm.bind(label(0));
        asm.nop();
        asm.bind(label(0));
    }

    // ---- call rel32 + relocation ----

    fn fake_method() -> MethodHandle {
        let mut cell = 0u8;
        MethodHandle::from_raw(&mut cell as *mut u8 as _).unwrap()
    }

    #[test]
    fn call_direct_emits_placeholder_and_reloc() {
        let mut asm = Asm::new();
        let method = fake_method();
        asm.call(method);
        asm.ret();
        let code = asm.finalize().unwrap();
        assert_eq!(code.bytes, [0xE8, 0, 0, 0, 0, 0xC3]);
        assert_eq!(code.call_relocs.len(), 1);
        assert_eq!(code.call_relocs[0].method, method);
        assert_eq!(code.call_relocs[0].offset, 1);
    }

    #[test]
    fn call_reloc_patches_rel32_against_runtime_addresses() {
        let mut asm = Asm::new();
        asm.call(fake_method());
        asm.ret();
        let mut code = asm.finalize().unwrap();
        // Buffer loaded at 0x1000, callee at 0x1010:
        // rel = 0x1010 - (0x1000 + 1 + 4) = 11.
        code.call_relocs[0]
            .patch(&mut code.bytes, 0x1000, 0x1010)
            .unwrap();
        assert_eq!(code.bytes, [0xE8, 0x0B, 0x00, 0x00, 0x00, 0xC3]);
        // A backward call: callee at 0x1000, buffer at 0x2000:
        // rel = 0x1000 - 0x2005 = -0x1005.
        code.call_relocs[0]
            .patch(&mut code.bytes, 0x2000, 0x1000)
            .unwrap();
        assert_eq!(code.bytes, [0xE8, 0xFB, 0xEF, 0xFF, 0xFF, 0xC3]);
    }

    // ---- and / or / xor (the shared ALU machinery, step_10.1) ----

    #[test]
    fn and_or_xor_forms() {
        // andl %ecx, %eax
        assert_eq!(
            finish(|a| a.and(W32, Rm::Reg(Rax), Rmi::Reg(Rcx))),
            [0x21, 0xC8]
        );
        // orq %rcx, %rax
        assert_eq!(
            finish(|a| a.or(W64, Rm::Reg(Rax), Rmi::Reg(Rcx))),
            [0x48, 0x09, 0xC8]
        );
        // xorl %ecx, %eax
        assert_eq!(
            finish(|a| a.xor(W32, Rm::Reg(Rax), Rmi::Reg(Rcx))),
            [0x31, 0xC8]
        );
        // xorl $5, %eax — group-1 imm8 (83 /6).
        assert_eq!(
            finish(|a| a.xor(W32, Rm::Reg(Rax), Rmi::Imm(5))),
            [0x83, 0xF0, 0x05]
        );
        // andq $0x1234, %rcx — imm32 form (81 /4). (Not rax: llvm-mc
        // picks the accumulator-special form there.)
        assert_eq!(
            finish(|a| a.and(W64, Rm::Reg(Rcx), Rmi::Imm(0x1234))),
            [0x48, 0x81, 0xE1, 0x34, 0x12, 0x00, 0x00]
        );
        // orl $-1, %r9d (REX.B, 83 /1).
        assert_eq!(
            finish(|a| a.or(W32, Rm::Reg(R9), Rmi::Imm(-1))),
            [0x41, 0x83, 0xC9, 0xFF]
        );
        // andl -8(%rbp), %eax — the load form (23 /r).
        assert_eq!(
            finish(|a| a.and(W32, Rm::Reg(Rax), Rmi::Mem(Mem::base_disp(Rbp, -8)))),
            [0x23, 0x45, 0xF8]
        );
        // xorq %rax, -8(%rbp) — memory destination.
        assert_eq!(
            finish(|a| a.xor(W64, Rm::Mem(Mem::base_disp(Rbp, -8)), Rmi::Reg(Rax))),
            [0x48, 0x31, 0x45, 0xF8]
        );
    }

    // ---- neg / not / div (F7 group) ----

    #[test]
    fn neg_not_div_forms() {
        // negl %eax
        assert_eq!(finish(|a| a.neg(W32, Rm::Reg(Rax))), [0xF7, 0xD8]);
        // negq %r8 (REX.W+B)
        assert_eq!(finish(|a| a.neg(W64, Rm::Reg(R8))), [0x49, 0xF7, 0xD8]);
        // notq %rax
        assert_eq!(finish(|a| a.not(W64, Rm::Reg(Rax))), [0x48, 0xF7, 0xD0]);
        // notl %r15d (REX.B)
        assert_eq!(finish(|a| a.not(W32, Rm::Reg(R15))), [0x41, 0xF7, 0xD7]);
        // divl %ecx
        assert_eq!(finish(|a| a.div(W32, Rm::Reg(Rcx))), [0xF7, 0xF1]);
        // divq %r8 (REX.W+B)
        assert_eq!(finish(|a| a.div(W64, Rm::Reg(R8))), [0x49, 0xF7, 0xF0]);
        // divl -4(%rbp) — memory divisor.
        assert_eq!(
            finish(|a| a.div(W32, Rm::Mem(Mem::base_disp(Rbp, -4)))),
            [0xF7, 0x75, 0xFC]
        );
    }

    // ---- shifts (D1 / C1 / D3 forms) ----

    #[test]
    fn shift_forms() {
        use crate::inst::ShiftOp;
        // shll %cl, %eax
        assert_eq!(
            finish(|a| a.shift_cl(ShiftOp::Shl, W32, Rm::Reg(Rax))),
            [0xD3, 0xE0]
        );
        // sarq %cl, %rax
        assert_eq!(
            finish(|a| a.shift_cl(ShiftOp::Sar, W64, Rm::Reg(Rax))),
            [0x48, 0xD3, 0xF8]
        );
        // shrl %cl, %r10d (REX.B)
        assert_eq!(
            finish(|a| a.shift_cl(ShiftOp::Shr, W32, Rm::Reg(R10))),
            [0x41, 0xD3, 0xEA]
        );
        // sarl %cl, %ecx
        assert_eq!(
            finish(|a| a.shift_cl(ShiftOp::Sar, W32, Rm::Reg(Rcx))),
            [0xD3, 0xF9]
        );
        // shrl $7, %eax (C1 /5 ib)
        assert_eq!(
            finish(|a| a.shift_imm(ShiftOp::Shr, W32, Rm::Reg(Rax), 7)),
            [0xC1, 0xE8, 0x07]
        );
        // shll $24, %eax — the conv.i1 expansion shape.
        assert_eq!(
            finish(|a| a.shift_imm(ShiftOp::Shl, W32, Rm::Reg(Rax), 24)),
            [0xC1, 0xE0, 0x18]
        );
        // sarl $24, %eax
        assert_eq!(
            finish(|a| a.shift_imm(ShiftOp::Sar, W32, Rm::Reg(Rax), 24)),
            [0xC1, 0xF8, 0x18]
        );
        // shll $1, %eax — the one-bit short form (D1 /4), as assemblers pick.
        assert_eq!(
            finish(|a| a.shift_imm(ShiftOp::Shl, W32, Rm::Reg(Rax), 1)),
            [0xD1, 0xE0]
        );
        // shrq $63, %r9 (REX.W+B, C1 /5 ib)
        assert_eq!(
            finish(|a| a.shift_imm(ShiftOp::Shr, W64, Rm::Reg(R9), 63)),
            [0x49, 0xC1, 0xE9, 0x3F]
        );
    }

    // ---- setcc (0F 9x, 8-bit destination) ----

    #[test]
    fn setcc_forms() {
        // setae %al
        assert_eq!(finish(|a| a.setcc(CondCode::UGe, Rax)), [0x0F, 0x93, 0xC0]);
        // sete %cl
        assert_eq!(finish(|a| a.setcc(CondCode::Eq, Rcx)), [0x0F, 0x94, 0xC1]);
        // setle %bl
        assert_eq!(finish(|a| a.setcc(CondCode::Le, Rbx)), [0x0F, 0x9E, 0xC3]);
        // setb %sil — REX required: without it rm=6 encodes %dh.
        assert_eq!(
            finish(|a| a.setcc(CondCode::ULt, Rsi)),
            [0x40, 0x0F, 0x92, 0xC6]
        );
        // setg %r8b (REX.B)
        assert_eq!(
            finish(|a| a.setcc(CondCode::Gt, R8)),
            [0x41, 0x0F, 0x9F, 0xC0]
        );
    }

    // ---- movsxd (63 /r) ----

    #[test]
    fn movsxd_forms() {
        // movslq %ecx, %rax
        assert_eq!(finish(|a| a.movsxd(Rax, Rm::Reg(Rcx))), [0x48, 0x63, 0xC1]);
        // movslq %r8d, %r9 (REX.W+R+B)
        assert_eq!(finish(|a| a.movsxd(R9, Rm::Reg(R8))), [0x4D, 0x63, 0xC8]);
        // movslq -8(%rbp), %r11 — memory source.
        assert_eq!(
            finish(|a| a.movsxd(R11, Rm::Mem(Mem::base_disp(Rbp, -8)))),
            [0x4C, 0x63, 0x5D, 0xF8]
        );
    }

    // ---- scalar SSE (step_10.2) ----

    use crate::inst::{ArithFOp, FWidth};
    use crate::regs::Xmm::*;
    use FWidth::{D, S};

    #[test]
    fn movs_forms() {
        // movsd %xmm1, %xmm0
        assert_eq!(
            finish(|a| a.mov_f_load(D, Xmm0, RmX::Reg(Xmm1))),
            [0xF2, 0x0F, 0x10, 0xC1]
        );
        // movss -8(%rbp), %xmm3
        assert_eq!(
            finish(|a| a.mov_f_load(S, Xmm3, RmX::Mem(Mem::base_disp(Rbp, -8)))),
            [0xF3, 0x0F, 0x10, 0x5D, 0xF8]
        );
        // movsd %xmm5, -16(%rbp) — the store form (0F 11).
        assert_eq!(
            finish(|a| a.mov_f_store(D, Mem::base_disp(Rbp, -16), Xmm5)),
            [0xF2, 0x0F, 0x11, 0x6D, 0xF0]
        );
        // movsd %xmm14, %xmm15 — REX.R+REX.B, prefix before REX.
        assert_eq!(
            finish(|a| a.mov_f_load(D, Xmm15, RmX::Reg(Xmm14))),
            [0xF2, 0x45, 0x0F, 0x10, 0xFE]
        );
    }

    #[test]
    fn arith_f_forms() {
        // addsd %xmm1, %xmm0
        assert_eq!(
            finish(|a| a.arith_f(ArithFOp::Add, D, Xmm0, RmX::Reg(Xmm1))),
            [0xF2, 0x0F, 0x58, 0xC1]
        );
        // addss -4(%rbp), %xmm2 — memory source.
        assert_eq!(
            finish(|a| a.arith_f(ArithFOp::Add, S, Xmm2, RmX::Mem(Mem::base_disp(Rbp, -4)))),
            [0xF3, 0x0F, 0x58, 0x55, 0xFC]
        );
        // subsd %xmm2, %xmm3
        assert_eq!(
            finish(|a| a.arith_f(ArithFOp::Sub, D, Xmm3, RmX::Reg(Xmm2))),
            [0xF2, 0x0F, 0x5C, 0xDA]
        );
        // mulsd -24(%rbp), %xmm4
        assert_eq!(
            finish(|a| a.arith_f(ArithFOp::Mul, D, Xmm4, RmX::Mem(Mem::base_disp(Rbp, -24)))),
            [0xF2, 0x0F, 0x59, 0x65, 0xE8]
        );
        // divsd %xmm6, %xmm7
        assert_eq!(
            finish(|a| a.arith_f(ArithFOp::Div, D, Xmm7, RmX::Reg(Xmm6))),
            [0xF2, 0x0F, 0x5E, 0xFE]
        );
        // minsd %xmm1, %xmm0 / maxsd %xmm2, %xmm3 — the float->unsigned
        // conversion clamp forms (step_10.11).
        assert_eq!(
            finish(|a| a.arith_f(ArithFOp::Min, D, Xmm0, RmX::Reg(Xmm1))),
            [0xF2, 0x0F, 0x5D, 0xC1]
        );
        assert_eq!(
            finish(|a| a.arith_f(ArithFOp::Max, D, Xmm3, RmX::Reg(Xmm2))),
            [0xF2, 0x0F, 0x5F, 0xDA]
        );
        // minss %xmm5, %xmm4 / maxss -8(%rbp), %xmm6
        assert_eq!(
            finish(|a| a.arith_f(ArithFOp::Min, S, Xmm4, RmX::Reg(Xmm5))),
            [0xF3, 0x0F, 0x5D, 0xE5]
        );
        assert_eq!(
            finish(|a| a.arith_f(ArithFOp::Max, S, Xmm6, RmX::Mem(Mem::base_disp(Rbp, -8)))),
            [0xF3, 0x0F, 0x5F, 0x75, 0xF8]
        );
    }

    #[test]
    fn ucomis_and_xor_forms() {
        // ucomiss %xmm1, %xmm0 — no prefix for the ss form.
        assert_eq!(
            finish(|a| a.ucomis(S, Xmm0, RmX::Reg(Xmm1))),
            [0x0F, 0x2E, 0xC1]
        );
        // ucomisd %xmm1, %xmm0 — 66 prefix.
        assert_eq!(
            finish(|a| a.ucomis(D, Xmm0, RmX::Reg(Xmm1))),
            [0x66, 0x0F, 0x2E, 0xC1]
        );
        // ucomisd -8(%rbp), %xmm2 — memory rhs.
        assert_eq!(
            finish(|a| a.ucomis(D, Xmm2, RmX::Mem(Mem::base_disp(Rbp, -8)))),
            [0x66, 0x0F, 0x2E, 0x55, 0xF8]
        );
        // xorps %xmm1, %xmm0
        assert_eq!(
            finish(|a| a.xor_f(S, Xmm0, RmX::Reg(Xmm1))),
            [0x0F, 0x57, 0xC1]
        );
        // xorpd %xmm9, %xmm8 — 66 + REX.R+B.
        assert_eq!(
            finish(|a| a.xor_f(D, Xmm8, RmX::Reg(Xmm9))),
            [0x66, 0x45, 0x0F, 0x57, 0xC1]
        );
    }

    #[test]
    fn cvt_forms() {
        // cvtsi2sdl %eax, %xmm0
        assert_eq!(
            finish(|a| a.cvtsi2s(D, Xmm0, Rm::Reg(Rax), false)),
            [0xF2, 0x0F, 0x2A, 0xC0]
        );
        // cvtsi2ssq %rax, %xmm1 — REX.W for the 64-bit integer source.
        assert_eq!(
            finish(|a| a.cvtsi2s(S, Xmm1, Rm::Reg(Rax), true)),
            [0xF3, 0x48, 0x0F, 0x2A, 0xC8]
        );
        // cvtsi2sdl -8(%rbp), %xmm2 — memory source.
        assert_eq!(
            finish(|a| a.cvtsi2s(D, Xmm2, Rm::Mem(Mem::base_disp(Rbp, -8)), false)),
            [0xF2, 0x0F, 0x2A, 0x55, 0xF8]
        );
        // cvttsd2si %xmm0, %eax
        assert_eq!(
            finish(|a| a.cvtts2si(D, Rax, RmX::Reg(Xmm0), false)),
            [0xF2, 0x0F, 0x2C, 0xC0]
        );
        // cvttsd2si %xmm0, %rax — REX.W for the 64-bit destination.
        assert_eq!(
            finish(|a| a.cvtts2si(D, Rax, RmX::Reg(Xmm0), true)),
            [0xF2, 0x48, 0x0F, 0x2C, 0xC0]
        );
        // cvttss2si %xmm2, %ecx
        assert_eq!(
            finish(|a| a.cvtts2si(S, Rcx, RmX::Reg(Xmm2), false)),
            [0xF3, 0x0F, 0x2C, 0xCA]
        );
        // cvtss2sd %xmm1, %xmm0 — F3 (the prefix names the source width).
        assert_eq!(
            finish(|a| a.cvts2s(D, Xmm0, RmX::Reg(Xmm1))),
            [0xF3, 0x0F, 0x5A, 0xC1]
        );
        // cvtsd2ss -8(%rbp), %xmm0 — F2, memory source.
        assert_eq!(
            finish(|a| a.cvts2s(S, Xmm0, RmX::Mem(Mem::base_disp(Rbp, -8)))),
            [0xF2, 0x0F, 0x5A, 0x45, 0xF8]
        );
    }

    #[test]
    fn mov_gpr_to_xmm_forms() {
        // movq %rax, %xmm0 — 66 REX.W 0F 6E.
        assert_eq!(
            finish(|a| a.mov_gpr_to_xmm(D, Xmm0, Rax)),
            [0x66, 0x48, 0x0F, 0x6E, 0xC0]
        );
        // movd %eax, %xmm1
        assert_eq!(
            finish(|a| a.mov_gpr_to_xmm(S, Xmm1, Rax)),
            [0x66, 0x0F, 0x6E, 0xC8]
        );
        // movq %r9, %xmm10 — REX.W+R+B.
        assert_eq!(
            finish(|a| a.mov_gpr_to_xmm(D, Xmm10, R9)),
            [0x66, 0x4D, 0x0F, 0x6E, 0xD1]
        );
    }

    #[test]
    fn parity_condition_codes() {
        // The float-compare expansions' jp/jnp: 0F 8A / 0F 8B rel32.
        let bytes = finish(|a| {
            a.bind(label(0));
            a.jcc(CondCode::Parity, label(0));
        });
        assert_eq!(bytes, [0x0F, 0x8A, 0xFA, 0xFF, 0xFF, 0xFF]);
        let bytes = finish(|a| {
            a.bind(label(0));
            a.jcc(CondCode::NotParity, label(0));
        });
        assert_eq!(bytes, [0x0F, 0x8B, 0xFA, 0xFF, 0xFF, 0xFF]);
    }

    // ---- the fib prolog/epilog shape, end to end ----

    #[test]
    fn fib_frame_prolog_epilog() {
        // pushq %rbp; movq %rsp, %rbp; subq $0x20, %rsp
        // movq %rdi, -8(%rbp)   (arg spill)
        // leaq -0x20(%rbp), %rax
        // leave; ret
        let bytes = finish(|a| {
            a.push(Rbp);
            a.mov(W64, Rm::Reg(Rbp), Rmi::Reg(Rsp));
            a.sub(W64, Rm::Reg(Rsp), Rmi::Imm(0x20));
            a.mov(W64, Rm::Mem(Mem::base_disp(Rbp, -8)), Rmi::Reg(Rdi));
            a.lea(Rax, Mem::base_disp(Rbp, -0x20));
            a.leave();
            a.ret();
        });
        assert_eq!(
            bytes,
            [
                0x55, // pushq %rbp
                0x48, 0x89, 0xE5, // movq %rsp, %rbp
                0x48, 0x83, 0xEC, 0x20, // subq $0x20, %rsp
                0x48, 0x89, 0x7D, 0xF8, // movq %rdi, -8(%rbp)
                0x48, 0x8D, 0x45, 0xE0, // leaq -0x20(%rbp), %rax
                0xC9, // leave
                0xC3, // ret
            ]
        );
    }
}
