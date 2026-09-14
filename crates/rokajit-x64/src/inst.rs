//! x64 machine-instruction descriptors — the contract between lowering
//! (step_07.4, which produces them) and the byte encoder (step_07.6,
//! which implements against them).
//!
//! Descriptors are **pre-allocation**: value operands are [`Val`]s (LIR
//! temps/locals) that codegen (step_07.5, Winch-style tier 0) binds to
//! registers or frame slots. Physical registers appear only where the
//! architecture or the SysV ABI pins them — argument registers, `rax` for
//! returns and `idiv`, `rsp`/`rbp` in the frame sequence. The operand
//! types are the typed terms (ISLE lesson, docs/JITs/cranelift.md §2):
//! [`Place`] (a writable destination) can never hold an immediate,
//! [`Src`] can never hold a destination-only entity, and [`Amode`] is
//! distinct from both, so invalid sequences are unrepresentable.
//!
//! Encoding notes for step_07.6:
//!
//! - `Width` selects the 32- vs 64-bit form (REX.W, imm32 vs imm64).
//!   `Src::Imm` payloads are always `i64`; a `W32` instruction truncates
//!   to imm32.
//! - Not every `Src` shape has an encoding: `idiv` has no immediate
//!   form, and `cmp`/`Arith` require a non-immediate lhs. Materializing
//!   an immediate into a scratch register when the encoding requires it
//!   is codegen's (07.5's) duty — the encoder may treat such shapes as
//!   unreachable.
//! - `AllocFrame` carries no operand: the frame size is codegen's frame
//!   layout result, not a lowering-time fact.

use rokajit::ir::LocalId;
use rokajit::lower::{Label, Val};
use rokajit_ee::enums::CorInfoHelpFunc;
use rokajit_ee::handles::MethodHandle;

use crate::regs::{Gpr, Xmm};

/// Operand width of an instruction: the 32- or 64-bit form.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Width {
    W32,
    W64,
}

/// A writable destination: an unallocated value or a fixed physical
/// register. Immediates and memory addressing modes are not
/// destinations.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Place {
    Val(Val),
    Reg(Gpr),
}

/// A readable source: an unallocated value, a fixed physical register,
/// or an immediate. Memory reads are descriptor-level ([`Inst::LoadMem`]);
/// frame-slot traffic is between codegen and the encoder.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Src {
    Val(Val),
    Reg(Gpr),
    Imm(i64),
}

/// An addressing mode. Frame slots are symbolic — codegen's frame layout
/// (07.5) assigns each local its `[rbp - off]` offset.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Amode {
    /// The frame slot of a local/arg/temp.
    FrameSlot(LocalId),
}

/// Scalar floating-point width: the `ss` (f32) or `sd` (f64) SSE form.
/// x64 floating point *is* SSE — there is no x87 anywhere in the backend.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum FWidth {
    /// 32-bit (`movss`/`addss`/…).
    S,
    /// 64-bit (`movsd`/`addsd`/…).
    D,
}

impl FWidth {
    /// The width matching an IR float type, or `None` for non-floats.
    pub fn of(ty: rokajit::ir::Type) -> Option<Self> {
        match ty {
            rokajit::ir::Type::Float => Some(FWidth::S),
            rokajit::ir::Type::Double => Some(FWidth::D),
            _ => None,
        }
    }
}

/// A writable XMM-side destination. Same separation as [`Place`]:
/// immediates and addressing modes are not destinations.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum XmmPlace {
    Val(Val),
    Reg(Xmm),
}

/// A readable XMM-side source. [`XmmSrc::Bits`] is a float constant as a
/// raw bit pattern (`f32::to_bits` zero-extended for `FWidth::S`) — SSE
/// has no immediate operand forms, so codegen materializes the bits
/// through a GPR scratch (`movabs` + `movq`/`movd`). [`Inst::ConstF`]
/// denotes the same materialization when the constant is the statement's
/// *result* (a `Copy`).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum XmmSrc {
    Val(Val),
    Reg(Xmm),
    Bits(u64),
}

/// The scalar SSE arithmetic instructions lowering emits (`add`/`sub`/
/// `mul`/`div`; `rem` on floats is an EE helper call, not an instruction
/// — see `decisions/` for step_10.2).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ArithFOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl ArithFOp {
    /// The IR operator this instruction lowers, or `None` when the
    /// operator has no scalar SSE form (`rem`, and every integer-only op).
    pub fn of(op: rokajit::ir::BinaryOp) -> Option<Self> {
        use rokajit::ir::BinaryOp as B;
        match op {
            B::Add => Some(ArithFOp::Add),
            B::Sub => Some(ArithFOp::Sub),
            B::Mul => Some(ArithFOp::Mul),
            B::Div => Some(ArithFOp::Div),
            _ => None,
        }
    }
}

/// The arithmetic instructions lowering emits (`imul` is the signed
/// multiply; tier 0 does not distinguish for `add`/`sub`/the bitwise
/// logic ops).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ArithOp {
    Add,
    Sub,
    Imul,
    And,
    Or,
    Xor,
}

impl ArithOp {
    /// The IR operator this instruction lowers, or `None` when the
    /// operator is outside the lowering subset.
    pub fn of(op: rokajit::ir::BinaryOp) -> Option<Self> {
        use rokajit::ir::BinaryOp as B;
        match op {
            B::Add => Some(ArithOp::Add),
            B::Sub => Some(ArithOp::Sub),
            B::Mul => Some(ArithOp::Imul),
            B::And => Some(ArithOp::And),
            B::Or => Some(ArithOp::Or),
            B::Xor => Some(ArithOp::Xor),
            _ => None,
        }
    }
}

/// The shift instructions lowering emits. `BinaryOp::Shr` (IL `shr`) is
/// the *arithmetic* shift right (`sar`); `BinaryOp::UShr` (IL `shr.un`)
/// is the logical one (`shr`) — the signedness flip is exactly here.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ShiftOp {
    Shl,
    /// Logical shift right (IL `shr.un`).
    Shr,
    /// Arithmetic shift right (IL `shr`).
    Sar,
}

impl ShiftOp {
    /// The IR shift operator this instruction lowers, or `None` for
    /// non-shift operators.
    pub fn of(op: rokajit::ir::BinaryOp) -> Option<Self> {
        use rokajit::ir::BinaryOp as B;
        match op {
            B::Shl => Some(ShiftOp::Shl),
            B::Shr => Some(ShiftOp::Sar),
            B::UShr => Some(ShiftOp::Shr),
            _ => None,
        }
    }
}

/// An x64 condition code for `jcc`/`setcc`, mapped from the IR's
/// comparison operators. The unsigned forms map onto the carry-based
/// codes (`b`/`be`/`a`/`ae`).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CondCode {
    /// `je`/`jz`.
    Eq,
    /// `jne`/`jnz`.
    Ne,
    /// `jl` (signed).
    Lt,
    /// `jle` (signed).
    Le,
    /// `jg` (signed).
    Gt,
    /// `jge` (signed).
    Ge,
    /// `jb` (unsigned).
    ULt,
    /// `jbe` (unsigned).
    ULe,
    /// `ja` (unsigned).
    UGt,
    /// `jae` (unsigned).
    UGe,
    /// `jp` — parity set. Only float-compare expansions emit this:
    /// `ucomis*` reports an unordered (NaN) operand pair as PF=1.
    Parity,
    /// `jnp` — parity clear (the ordered case of `ucomis*`).
    NotParity,
}

impl CondCode {
    /// The condition code implementing an IR comparison, or `None` for
    /// non-comparison operators.
    pub fn of(op: rokajit::ir::BinaryOp) -> Option<Self> {
        use rokajit::ir::BinaryOp as B;
        match op {
            B::Eq => Some(CondCode::Eq),
            B::Ne => Some(CondCode::Ne),
            B::Lt => Some(CondCode::Lt),
            B::Le => Some(CondCode::Le),
            B::Gt => Some(CondCode::Gt),
            B::Ge => Some(CondCode::Ge),
            B::ULt => Some(CondCode::ULt),
            B::ULe => Some(CondCode::ULe),
            B::UGt => Some(CondCode::UGt),
            B::UGe => Some(CondCode::UGe),
            _ => None,
        }
    }
}

/// One x64 machine instruction, as a typed descriptor. The variants are
/// the instruction set the lowering rules emit (the fib subset plus the
/// step_10.1 scalar-cheap pack); each variant's fields admit only the
/// operand shapes the instruction takes.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Inst {
    /// `mov dst, src` — reg/reg, reg/imm, or (after codegen binds
    /// values) frame-slot forms.
    Mov { width: Width, dst: Place, src: Src },
    /// A float constant: `dst` receives the value whose bit pattern is
    /// `bits` (`f32::to_bits` zero-extended for `FWidth::S`). SSE has no
    /// immediate forms; codegen materializes the bits through a GPR
    /// (`movabs` + `movq`/`movd`) — no rodata pool (step_10.2 decision).
    ConstF {
        width: FWidth,
        dst: XmmPlace,
        bits: u64,
    },
    /// `movss`/`movsd dst, src` — xmm/xmm or, after binding, frame-slot
    /// forms in either direction.
    MovF {
        width: FWidth,
        dst: XmmPlace,
        src: XmmSrc,
    },
    /// Scalar SSE arithmetic: `dst := lhs op rhs`. Destructive two-operand
    /// like [`Inst::Arith`]; codegen emits the `movs*` + op.
    ArithF {
        op: ArithFOp,
        width: FWidth,
        dst: XmmPlace,
        lhs: XmmSrc,
        rhs: XmmSrc,
    },
    /// Float `neg`: `dst := -src`, implemented as an XOR with the sign
    /// mask (flips the sign bit exactly — `0.0 − x` would get `-0.0` and
    /// NaN signs wrong). The mask is a [`FWidth`]-sized constant.
    NegF {
        width: FWidth,
        dst: XmmPlace,
        src: XmmSrc,
    },
    /// `ucomiss`/`ucomisd lhs, rhs` — the unordered-aware compare; sets
    /// ZF/PF/CF for the [`Inst::SetccF`]/[`Inst::JccF`] that immediately
    /// follows (same adjacency contract as [`Inst::Cmp`]). `ucomis*` (not
    /// `comis*`) so quiet NaNs don't trap.
    CmpF {
        width: FWidth,
        lhs: XmmSrc,
        rhs: XmmSrc,
    },
    /// Materialize a float compare's flags into an Int32 0/1. `op` is the
    /// IR comparison operator — ordered forms (`Eq`/`Gt`/`Lt`) are false
    /// on NaN, `.un` forms true on NaN (ECMA-335 table III.4); codegen
    /// owns the parity-aware expansion. Same adjacency contract as
    /// [`Inst::Setcc`].
    SetccF {
        op: rokajit::ir::BinaryOp,
        dst: Place,
    },
    /// Conditional branch on a float compare's flags (the branch-folded
    /// `SetccF`). Codegen expands the unordered forms to a `jp` + `jcc`
    /// pair, and the ordered equality/less forms to `jp`-guarded jumps.
    JccF {
        op: rokajit::ir::BinaryOp,
        target: Label,
    },
    /// `cvtsi2ss`/`cvtsi2sd dst, src` — signed integer to float
    /// (`conv.r4`/`conv.r8` from an integer operand). `src_w64` selects
    /// the 32- vs 64-bit integer source form.
    CvtIntToF {
        width: FWidth,
        src_w64: bool,
        dst: XmmPlace,
        src: Src,
    },
    /// `cvttss2si`/`cvttsd2si dst, src` — float to integer with truncation
    /// (`conv.i4`/`i8`/`u4` from a float operand). Out-of-range input
    /// yields the "integer indefinite" value (`0x8000…`), matching RyuJIT.
    /// `dst_w64` selects the 64-bit destination form; a 32-bit unsigned
    /// conversion also uses it (the low half is the result — values up to
    /// 2³²−1 convert exactly, everything beyond is unspecified by ECMA).
    CvtFToInt {
        src_width: FWidth,
        dst_w64: bool,
        dst: Place,
        src: XmmSrc,
    },
    /// `cvtss2sd`/`cvtsd2ss` — float↔double (`conv.r4`/`conv.r8` with a
    /// float operand). `to` is the destination width.
    CvtFToF {
        to: FWidth,
        dst: XmmPlace,
        src: XmmSrc,
    },
    /// `lea dst, [addr]` — address of a frame slot (always 64-bit).
    Lea { dst: Place, addr: Amode },
    /// Three-operand arithmetic: `dst := lhs op rhs`. x64 arithmetic is
    /// destructive two-operand; codegen (07.5) emits the `mov` + op.
    Arith {
        op: ArithOp,
        width: Width,
        dst: Place,
        lhs: Src,
        rhs: Src,
    },
    /// `cdq`/`cqo`: sign-extend `rax` into `rdx:rax` before `idiv`.
    Cdq { width: Width },
    /// `idiv divisor`: signed `rdx:rax / divisor`; quotient → `rax`,
    /// remainder → `rdx`. No immediate form exists.
    Idiv { width: Width, divisor: Src },
    /// `div divisor`: unsigned `rdx:rax / divisor` (IL `div.un`/`rem.un`;
    /// lowering zeroes `edx` instead of emitting `cdq`). No immediate
    /// form exists.
    Div { width: Width, divisor: Src },
    /// `shl`/`shr`/`sar dst, count`: the count is an immediate (the `C1`
    /// imm8 form) or a value that codegen places in `cl` (the `D3` form).
    /// Hardware masks the count to 5/6 bits by operand width — ECMA-335's
    /// masking rule exactly, so no mask instruction is emitted.
    Shift {
        op: ShiftOp,
        width: Width,
        dst: Place,
        lhs: Src,
        rhs: Src,
    },
    /// `neg`/`not dst` — one-operand complement forms.
    Unary {
        op: rokajit::ir::UnaryOp,
        width: Width,
        dst: Place,
        src: Src,
    },
    /// `setcc dst_low_byte` — the compare-as-value (`ceq`/`clt`/…)
    /// materialization. Reads the flags a preceding `Cmp` left; codegen
    /// zeroes the destination register first, so the result is a clean
    /// 0/1. Same adjacency contract as `Jcc`.
    Setcc { cc: CondCode, dst: Place },
    /// 32→64 extension (IL `conv.i8`/`conv.u8` from a 32-bit operand):
    /// `movsxd` for the signed form, a 32-bit `mov` (which zero-extends)
    /// for the unsigned form. Codegen always materializes into a scratch
    /// register — a widening copy may never alias the source's slot.
    MovExt { dst: Place, src: Src, signed: bool },
    /// `cmp lhs, rhs` — sets the flags a following `Jcc` consumes.
    Cmp { width: Width, lhs: Src, rhs: Src },
    /// `mov width, dst, [addr + disp]` — a load through a computed
    /// address (LIR `Load`: `ldfld`'s field read; step_10.4). `addr` is
    /// the address *value* (usually a frame-resident GC reference), so
    /// codegen materializes it into a scratch GPR before the load.
    LoadMem {
        width: Width,
        dst: Place,
        addr: Src,
        disp: i32,
    },
    /// `mov [addr + disp], src` — a store through a computed address
    /// (LIR `Store`: `stfld`'s non-reference field store; a reference
    /// store is the write-barrier helper call instead). A too-wide
    /// constant `src` materializes into a scratch register at codegen
    /// (the `movq [mem], imm32` form sign-extends — the same wide-imm
    /// rule as ALU ops).
    StoreMem {
        width: Width,
        addr: Src,
        disp: i32,
        src: Src,
    },
    /// `movzx`/`movsx dst, [addr + disp]` — a sub-Int32 field load
    /// (LIR `Load` with a narrow [`rokajit::ir::MemAccess`]): the cell is `size` bytes
    /// (1 or 2), the Int32 result extends zero or sign per `signed`
    /// (ECMA-335 §III.1.1.1: I1/I2 sign-extend, BOOLEAN/CHAR/U1/U2
    /// zero-extend).
    LoadMemNarrow {
        size: u8,
        signed: bool,
        dst: Place,
        addr: Src,
        disp: i32,
    },
    /// `mov [addr + disp], src_low` — a sub-Int32 field store (LIR
    /// `Store` with a narrow [`rokajit::ir::MemAccess`]): only the low `size` bytes
    /// (1 or 2) of the Int32 value write to memory.
    StoreMemNarrow {
        size: u8,
        addr: Src,
        disp: i32,
        src: Src,
    },
    /// `movss`/`movsd dst, [addr + disp]` — a float field load (LIR
    /// `Load` of a Float/Double type). Same address discipline as
    /// [`Inst::LoadMem`]; the destination binds to an XMM slot.
    LoadMemF {
        width: FWidth,
        dst: XmmPlace,
        addr: Src,
        disp: i32,
    },
    /// `movss`/`movsd [addr + disp], src` — a float field store (LIR
    /// `Store` of a float source). A float value never needs a GC write
    /// barrier.
    StoreMemF {
        width: FWidth,
        addr: Src,
        disp: i32,
        src: XmmSrc,
    },
    /// The explicit, trap-based null check (step_10.4): a 32-bit load
    /// through the reference, its result unused — on a null `addr` the
    /// hardware fault *is* the NullReferenceException, translated by the
    /// EE's signal handler via the same unwind path as `idiv`'s #DE.
    /// Always explicit: the offset-vs-page-size folding RyuJIT does is a
    /// later optimization.
    NullCheck { addr: Src },
    /// The array bounds check (step_10.8): the 32-bit length load at
    /// `[array + 8]` doubles as the null check (a null array faults — the
    /// trap model), then `index < length` unsigned decides between
    /// fallthrough and a call to the EE's RNGCHKFAIL helper
    /// (`IndexOutOfRangeException`). `index_wide` selects the 64-bit
    /// compare for a native-int index (the length load zero-extends, so
    /// the wide compare is exact).
    BoundsCheck {
        index: Src,
        index_wide: bool,
        array: Src,
    },
    /// `jcc target` — reads the flags `Cmp` (or a future flag-setting
    /// instruction) left. Flag materialization is by adjacency: lowering
    /// emits the producer immediately before the consumer, and codegen
    /// must not insert a flag-clobbering instruction between them.
    Jcc { cc: CondCode, target: Label },
    /// `jmp target`.
    Jmp { target: Label },
    /// `call rel32` to a resolved method. The EE supplies the target
    /// address at emit time; the call site is a relocation and a GC
    /// safepoint (07.7 drains it from codegen's records).
    CallDirect { method: MethodHandle },
    /// `call rel32` to an EE runtime helper (float `rem` lowers to
    /// `CORINFO_HELP_FLTREM`/`DBLREM`, RyuJIT's morph.cpp GT_MOD path).
    /// The EE supplies the entry point via `getHelperFtn`; the call site
    /// records carry no method handle (artifact.rs's CallSite contract).
    CallHelper { id: CorInfoHelpFunc },
    /// `push reg` (prolog: the frame pointer).
    Push { reg: Gpr },
    /// `sub rsp, <frame size>` — the size is codegen's frame-layout
    /// result; lowering only declares that a frame exists.
    AllocFrame,
    // --- step_10.9: the struct ABI and block operations ---
    /// Load `size` bytes (1..=8) from `[addr + disp]` into an ABI-pinned
    /// register, zero-extended (GPR) or as `movss`/`movsd` (XMM). Used at
    /// the call boundary for register-passed struct arguments and returns
    /// (SysV eightbyte classification): the address is a struct value's
    /// memory.
    LoadEightbyte {
        addr: BlockAddr,
        disp: u32,
        size: u8,
        dst: EbReg,
    },
    /// Store `size` bytes (1..=8) from an ABI-pinned register into a
    /// local's frame slot at byte `offset` within the slot — the callee
    /// half of the struct ABI (incoming register-passed arguments, and a
    /// register-passed struct call result landing in its destination
    /// slot). The size is respected exactly: a 3-byte eightbyte stores 3
    /// bytes, never 8 (frame slots are exactly `size` bytes).
    StoreEightbyte {
        local: LocalId,
        offset: u32,
        size: u8,
        src: EbReg,
    },
    /// `mov [rsp + offset], src` — an outgoing scalar stack argument
    /// (SysV register-pool overflow; step_10.9). `offset` is relative to
    /// `rsp` at the call instruction; the outgoing area is sized into the
    /// frame.
    StoreStackArg { width: Width, offset: u32, src: Src },
    /// The float form of [`Inst::StoreStackArg`] (`movss`/`movsd`).
    StoreStackArgF {
        width: FWidth,
        offset: u32,
        src: XmmSrc,
    },
    /// Copy `size` bytes from `[addr]` to `[rsp + offset]` — an outgoing
    /// stack-passed struct argument (a whole struct that didn't fit the
    /// register pools, or one the EE never classifies for registers).
    /// Always inline (no helper call mid-argument-setup).
    CopyStackArg {
        addr: BlockAddr,
        offset: u32,
        size: u32,
    },
    /// Copy `size` bytes from `[src]` to `[dst + dst_disp]` (`cpobj`/
    /// `stobj`, struct `stloc`/`starg`/`stfld`, the hidden-retbuf copy).
    /// Codegen picks inline unrolled moves for small sizes or the EE's
    /// `CORINFO_HELP_MEMCPY` for large ones; GC-barriered copies are
    /// decided at import (the bulk-write-barrier helper call) and never
    /// reach this descriptor.
    BlockCopy {
        dst: BlockAddr,
        dst_disp: u32,
        src: BlockAddr,
        size: u32,
    },
    /// Zero `size` bytes at `[dst + dst_disp]` (`initobj`; struct local
    /// zero-init is a prolog matter and doesn't use this descriptor).
    BlockZero {
        dst: BlockAddr,
        dst_disp: u32,
        size: u32,
    },
    /// `leave` (`mov rsp, rbp; pop rbp`) — the frame teardown matching
    /// the [`Inst::Push`] + `mov rbp, rsp` + [`Inst::AllocFrame`] prolog.
    /// Main-area returns only; a funclet never re-establishes rbp, so its
    /// exit is [`Inst::FuncletEpilog`] (step_10.6).
    Leave,
    /// `ret`.
    Ret,
    // --- step_10.6: the EH shapes ---
    /// `nop` (0x90) — padding after a call whose return address must stay
    /// inside its EH region / RUNTIME_FUNCTION (the `Throw` and
    /// `CallFinally` shapes; clr-abi.md's region-padding rule).
    Nop,
    /// `call rel32` to an in-chunk label (a `CallFinally` step block
    /// calling its finally funclet). Resolved by the assembler's label
    /// fixups — no EE lookup, no relocation, no managed-call-site record
    /// (an EH method is fully interruptible: the safepoint list is empty).
    /// The spill discipline is any call's (the scratch pool is
    /// caller-saved), which codegen applies.
    CallLabel { target: Label },
    /// `lea dst, [rip + rel32]` — a code address materialized into a
    /// register (a catch funclet's resume address into `rax` for the
    /// funclet-exit `ret`; the VM resumes the parent frame there). The
    /// displacement is a label fixup, resolved at finalize.
    LeaLabel { dst: Gpr, target: Label },
    /// Funclet epilog: `add rsp, N; ret`. Like [`Inst::AllocFrame`], the
    /// adjustment is codegen's per-funclet fact (the funclet's own
    /// outgoing-argument reservation), not a lowering-time one.
    FuncletEpilog,
}

/// A block operation's address operand (step_10.9): either a byref value
/// (loaded from its slot/register into a scratch GPR) or a local's own
/// frame-slot address (`ldloca`-shaped — `lea`, no memory read).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BlockAddr {
    /// A byref value's current location.
    Val(Val),
    /// A local's frame slot address.
    FrameSlot(LocalId),
}

/// An ABI-pinned register for one struct eightbyte: GPR for
/// integer-class eightbytes, XMM for SSE.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum EbReg {
    Gpr(Gpr),
    Xmm(Xmm),
}

/// The fixed (architecture- or ABI-pinned) registers an instruction
/// reads or writes *beyond its explicit operands*. Codegen consults this
/// when binding [`Val`]s so a fixed duty never collides with an
/// allocated value. `rsp`/`rbp` duties are omitted: both are reserved,
/// never allocatable, so nothing can collide with them.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct FixedRegs {
    pub uses: &'static [Gpr],
    pub defs: &'static [Gpr],
}

/// The GPRs a `call` destroys: the caller-saved set (SysV §3.2.1 — the
/// allocatable class minus the callee-saved subset). All XMM registers
/// are additionally caller-saved; float values are always frame-resident
/// in tier 0 (the step_10.2 codegen policy), so no XMM value can be
/// live across a call and that half stays documentary.
pub const CALL_DEFS: &[Gpr] = &[
    Gpr::Rax,
    Gpr::Rcx,
    Gpr::Rdx,
    Gpr::Rsi,
    Gpr::Rdi,
    Gpr::R8,
    Gpr::R9,
    Gpr::R10,
    Gpr::R11,
];

const NO_FIXED: FixedRegs = FixedRegs {
    uses: &[],
    defs: &[],
};

impl Inst {
    /// The instruction's implicit fixed-register duties (see
    /// [`FixedRegs`]). Explicit operands — including explicit
    /// [`Place::Reg`]/[`Src::Reg`] mentions like the `mov rax, …` a
    /// return sequence emits — are visible in the descriptor itself and
    /// not repeated here.
    pub fn fixed_regs(&self) -> FixedRegs {
        match self {
            Inst::Cdq { .. } => FixedRegs {
                uses: &[Gpr::Rax],
                defs: &[Gpr::Rdx],
            },
            Inst::Idiv { .. } | Inst::Div { .. } => FixedRegs {
                uses: &[Gpr::Rax, Gpr::Rdx],
                defs: &[Gpr::Rax, Gpr::Rdx],
            },
            // A variable-count shift moves the count into `cl`.
            Inst::Shift { rhs, .. } if !matches!(rhs, Src::Imm(_)) => FixedRegs {
                uses: &[],
                defs: &[Gpr::Rcx],
            },
            Inst::CallDirect { .. } | Inst::CallHelper { .. } | Inst::CallLabel { .. } => {
                FixedRegs {
                    uses: &[],
                    defs: CALL_DEFS,
                }
            }
            // A failed bounds check calls RNGCHKFAIL (conditionally).
            Inst::BoundsCheck { .. } => FixedRegs {
                uses: &[],
                defs: CALL_DEFS,
            },
            _ => NO_FIXED,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rokajit::ir::BinaryOp as B;

    #[test]
    fn arith_op_maps_the_lowering_subset() {
        assert_eq!(ArithOp::of(B::Add), Some(ArithOp::Add));
        assert_eq!(ArithOp::of(B::Sub), Some(ArithOp::Sub));
        assert_eq!(ArithOp::of(B::Mul), Some(ArithOp::Imul));
        assert_eq!(ArithOp::of(B::And), Some(ArithOp::And));
        assert_eq!(ArithOp::of(B::Or), Some(ArithOp::Or));
        assert_eq!(ArithOp::of(B::Xor), Some(ArithOp::Xor));
        assert_eq!(ArithOp::of(B::Rem), None);
        assert_eq!(ArithOp::of(B::Lt), None);
    }

    #[test]
    fn shift_op_maps_the_shift_subset() {
        // The signedness flip: IL `shr` is arithmetic (sar), IL `shr.un`
        // is logical (shr). This is where a wrong choice is a silent bug.
        assert_eq!(ShiftOp::of(B::Shl), Some(ShiftOp::Shl));
        assert_eq!(ShiftOp::of(B::Shr), Some(ShiftOp::Sar));
        assert_eq!(ShiftOp::of(B::UShr), Some(ShiftOp::Shr));
        assert_eq!(ShiftOp::of(B::Add), None);
    }

    #[test]
    fn cond_code_maps_every_comparison() {
        let signed = [
            (B::Eq, CondCode::Eq),
            (B::Ne, CondCode::Ne),
            (B::Lt, CondCode::Lt),
            (B::Le, CondCode::Le),
            (B::Gt, CondCode::Gt),
            (B::Ge, CondCode::Ge),
        ];
        let unsigned = [
            (B::ULt, CondCode::ULt),
            (B::ULe, CondCode::ULe),
            (B::UGt, CondCode::UGt),
            (B::UGe, CondCode::UGe),
        ];
        for (op, cc) in signed.into_iter().chain(unsigned) {
            assert_eq!(CondCode::of(op), Some(cc));
        }
        assert_eq!(CondCode::of(B::Add), None);
    }

    #[test]
    fn fixed_reg_duties_are_declared() {
        let idiv = Inst::Idiv {
            width: Width::W32,
            divisor: Src::Imm(3),
        }
        .fixed_regs();
        assert_eq!(idiv.uses, &[Gpr::Rax, Gpr::Rdx]);
        assert_eq!(idiv.defs, &[Gpr::Rax, Gpr::Rdx]);

        let mut cell = 0u8;
        let method = MethodHandle::from_raw(&mut cell as *mut u8 as _).unwrap();
        let call = Inst::CallDirect { method }.fixed_regs();
        assert!(call.defs.contains(&Gpr::Rax), "return register clobbered");
        // No callee-saved register is call-clobbered.
        for reg in crate::regs::GPR_CALLEE_SAVED {
            assert!(!call.defs.iter().any(|g| g.phys() == reg));
        }

        let mov = Inst::Mov {
            width: Width::W64,
            dst: Place::Reg(Gpr::Rax),
            src: Src::Imm(0),
        };
        assert_eq!(mov.fixed_regs(), NO_FIXED);
    }
}
