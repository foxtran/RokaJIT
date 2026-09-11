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
use rokajit_ee::handles::MethodHandle;

use crate::regs::Gpr;

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
/// or an immediate. Memory reads arrive with load support (a later
/// step); frame-slot traffic is between codegen and the encoder.
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

/// The arithmetic instructions lowering emits (`imul` is the signed
/// multiply; tier 0 does not distinguish for `add`/`sub`).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ArithOp {
    Add,
    Sub,
    Imul,
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
/// exactly the fib-subset instruction set (step_07.6's list); each
/// variant's fields admit only the operand shapes the instruction takes.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Inst {
    /// `mov dst, src` — reg/reg, reg/imm, or (after codegen binds
    /// values) frame-slot forms.
    Mov { width: Width, dst: Place, src: Src },
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
    /// `idiv divisor`: `rdx:rax / divisor`; quotient → `rax`, remainder
    /// → `rdx`. No immediate form exists.
    Idiv { width: Width, divisor: Src },
    /// `cmp lhs, rhs` — sets the flags a following `Jcc` consumes.
    Cmp { width: Width, lhs: Src, rhs: Src },
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
    /// `push reg` (prolog: the frame pointer).
    Push { reg: Gpr },
    /// `sub rsp, <frame size>` — the size is codegen's frame-layout
    /// result; lowering only declares that a frame exists.
    AllocFrame,
    /// `leave` (`mov rsp, rbp; pop rbp`) — the frame teardown matching
    /// the [`Inst::Push`] + `mov rbp, rsp` + [`Inst::AllocFrame`] prolog.
    Leave,
    /// `ret`.
    Ret,
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
/// are additionally caller-saved; tier 0 keeps no live float values, so
/// that half is documentary.
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
            Inst::Idiv { .. } => FixedRegs {
                uses: &[Gpr::Rax, Gpr::Rdx],
                defs: &[Gpr::Rax, Gpr::Rdx],
            },
            Inst::CallDirect { .. } => FixedRegs {
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
        assert_eq!(ArithOp::of(B::Rem), None);
        assert_eq!(ArithOp::of(B::Lt), None);
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
