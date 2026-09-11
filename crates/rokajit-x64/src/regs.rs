//! x86-64 register tables and SysV AMD64 ABI constants.
//!
//! Values, not logic: everything here is data the generic core consumes
//! through [`rokajit::target`]. The ABI classification algorithm that uses
//! these tables arrives with step_07.5.
//!
//! Sources (System V AMD64 ABI v1.0, `x86-64-abi`):
//! - §3.2.1 — callee-saved set: `rbx`, `rbp`, `r12`–`r15` (+ `rsp`).
//! - §3.2.2 — stack alignment: `rsp` is 16-byte aligned immediately before
//!   every `call` (`rsp + 8` is a multiple of 16 at function entry).
//! - §3.2.3 — argument/return register assignment for INTEGER/POINTER
//!   (aggregate-classification pseudocode) and SSE classes.

use rokajit::target::{PhysReg, RegClassId, RegClassKind, RegisterClass};

/// General-purpose registers. The discriminants are the hardware encoding
/// (ModRM/REX `reg` field, Intel SDM vol. 2A) — the step_07.6 encoder maps
/// [`Gpr::phys`] straight onto that encoding.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Gpr {
    Rax = 0,
    Rcx = 1,
    Rdx = 2,
    Rbx = 3,
    Rsp = 4,
    Rbp = 5,
    Rsi = 6,
    Rdi = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

/// SSE registers (x64 floating point is SSE-only; no x87).
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Xmm {
    Xmm0 = 0,
    Xmm1 = 1,
    Xmm2 = 2,
    Xmm3 = 3,
    Xmm4 = 4,
    Xmm5 = 5,
    Xmm6 = 6,
    Xmm7 = 7,
    Xmm8 = 8,
    Xmm9 = 9,
    Xmm10 = 10,
    Xmm11 = 11,
    Xmm12 = 12,
    Xmm13 = 13,
    Xmm14 = 14,
    Xmm15 = 15,
}

impl Gpr {
    /// The core-facing register handle. GPRs occupy [`PhysReg`] 0–15.
    pub const fn phys(self) -> PhysReg {
        PhysReg(self as u8)
    }
}

impl Xmm {
    /// The core-facing register handle. XMM registers occupy [`PhysReg`]
    /// 16–31 so indices are unique across classes (the `PhysReg` contract).
    pub const fn phys(self) -> PhysReg {
        PhysReg(16 + self as u8)
    }
}

/// [`RegClassId`] of the general-purpose class in [`REGISTER_CLASSES`].
pub const GPR_CLASS_ID: RegClassId = RegClassId(0);
/// [`RegClassId`] of the SSE class in [`REGISTER_CLASSES`].
pub const XMM_CLASS_ID: RegClassId = RegClassId(1);

/// Integer argument registers, in assignment order (SysV §3.2.3).
pub const INT_ARG_REGS: [Gpr; 6] = [Gpr::Rdi, Gpr::Rsi, Gpr::Rdx, Gpr::Rcx, Gpr::R8, Gpr::R9];

/// Float argument registers, in assignment order (SysV §3.2.3).
pub const FLOAT_ARG_REGS: [Xmm; 8] = [
    Xmm::Xmm0,
    Xmm::Xmm1,
    Xmm::Xmm2,
    Xmm::Xmm3,
    Xmm::Xmm4,
    Xmm::Xmm5,
    Xmm::Xmm6,
    Xmm::Xmm7,
];

/// Integer return register; `Rdx` joins it for 128-bit returns
/// (SysV §3.2.3). Pointers (`Ref`/`ByRef`/`NativeInt`) return in `Rax`.
pub const INT_RETURN_REGS: [Gpr; 2] = [Gpr::Rax, Gpr::Rdx];

/// Float return register (SysV §3.2.3).
pub const FLOAT_RETURN_REG: Xmm = Xmm::Xmm0;

/// The stack pointer. Never allocatable.
pub const STACK_POINTER: Gpr = Gpr::Rsp;

/// The frame pointer. Reserved — not allocatable — in tier 0, whose frame
/// contract is rbp-based so that GC root slot offsets
/// ([`rokajit::pipeline::GcRootSlot`]) are stable while `rsp` moves.
pub const FRAME_POINTER: Gpr = Gpr::Rbp;

/// Required `rsp` alignment in bytes at every `call` (SysV §3.2.2).
pub const CALL_SITE_STACK_ALIGNMENT: u32 = 16;

/// Allocatable GPRs, caller-saved first (allocation-preference order).
/// `Rsp` (stack pointer) and `Rbp` (reserved frame pointer) are excluded.
pub const GPR_ALLOCATABLE: [PhysReg; 14] = [
    Gpr::Rax.phys(),
    Gpr::Rcx.phys(),
    Gpr::Rdx.phys(),
    Gpr::Rsi.phys(),
    Gpr::Rdi.phys(),
    Gpr::R8.phys(),
    Gpr::R9.phys(),
    Gpr::R10.phys(),
    Gpr::R11.phys(),
    Gpr::Rbx.phys(),
    Gpr::R12.phys(),
    Gpr::R13.phys(),
    Gpr::R14.phys(),
    Gpr::R15.phys(),
];

/// The callee-saved subset of [`GPR_ALLOCATABLE`] (SysV §3.2.1; `Rbp` is
/// callee-saved by the ABI but reserved, so it does not appear here).
pub const GPR_CALLEE_SAVED: [PhysReg; 5] = [
    Gpr::Rbx.phys(),
    Gpr::R12.phys(),
    Gpr::R13.phys(),
    Gpr::R14.phys(),
    Gpr::R15.phys(),
];

/// Allocatable SSE registers. All XMM registers are caller-saved on SysV,
/// so the class's callee-saved list is empty.
pub const XMM_ALLOCATABLE: [PhysReg; 16] = [
    Xmm::Xmm0.phys(),
    Xmm::Xmm1.phys(),
    Xmm::Xmm2.phys(),
    Xmm::Xmm3.phys(),
    Xmm::Xmm4.phys(),
    Xmm::Xmm5.phys(),
    Xmm::Xmm6.phys(),
    Xmm::Xmm7.phys(),
    Xmm::Xmm8.phys(),
    Xmm::Xmm9.phys(),
    Xmm::Xmm10.phys(),
    Xmm::Xmm11.phys(),
    Xmm::Xmm12.phys(),
    Xmm::Xmm13.phys(),
    Xmm::Xmm14.phys(),
    Xmm::Xmm15.phys(),
];

/// The target's register classes; index = [`RegClassId`] (see
/// [`GPR_CLASS_ID`], [`XMM_CLASS_ID`]).
pub static REGISTER_CLASSES: [RegisterClass; 2] = [
    RegisterClass {
        name: "x64-gpr",
        kind: RegClassKind::Int,
        registers: &GPR_ALLOCATABLE,
        callee_saved: &GPR_CALLEE_SAVED,
    },
    RegisterClass {
        name: "x64-xmm",
        kind: RegClassKind::Float,
        registers: &XMM_ALLOCATABLE,
        callee_saved: &[],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn callee_saved_is_a_subset_of_allocatable() {
        for class in &REGISTER_CLASSES {
            for reg in class.callee_saved {
                assert!(
                    class.registers.contains(reg),
                    "{}: {:?} callee-saved but not allocatable",
                    class.name,
                    reg
                );
            }
        }
    }

    #[test]
    fn phys_reg_indices_are_unique_across_classes() {
        let mut seen = HashSet::new();
        for class in &REGISTER_CLASSES {
            for reg in class.registers {
                assert!(seen.insert(*reg), "duplicate PhysReg {:?}", reg);
            }
        }
    }

    #[test]
    fn fixed_duty_registers_are_not_allocatable() {
        assert!(!GPR_ALLOCATABLE.contains(&STACK_POINTER.phys()));
        assert!(!GPR_ALLOCATABLE.contains(&FRAME_POINTER.phys()));
    }

    #[test]
    fn gpr_discriminants_match_the_hardware_encoding() {
        // The step_07.6 encoder relies on this identity.
        assert_eq!(Gpr::Rax as u8, 0);
        assert_eq!(Gpr::Rsp as u8, 4);
        assert_eq!(Gpr::R15 as u8, 15);
    }
}
