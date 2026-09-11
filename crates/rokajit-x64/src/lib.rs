//! rokajit-x64 — the x86-64 (SysV AMD64) backend for RokaJIT.
//!
//! A thin backend crate, the first customer of the generic core's
//! interfaces (docs/porting-strategy.md, "Target independence"). It
//! contains target description data ([`regs`]: register tables and SysV
//! ABI constants), the instruction-descriptor contract ([`inst`], produced
//! by lowering, consumed by the step_07.6 encoder), the lowering rule set
//! ([`lower`], step_07.4), and implements [`rokajit::target::Target`];
//! later sub-steps add the byte-level encoder (07.6) and the
//! GC-info/unwind encoders (07.7).
//!
//! No CIL knowledge appears here: the crate speaks [`rokajit::ir`] types
//! and [`rokajit::target`] vocabulary only, and names not a single IL
//! opcode.

pub mod inst;
pub mod lower;
pub mod regs;

use rokajit::error::{CompileError, CompileResult};
use rokajit::ir::{CallSig, Type};
use rokajit::target::{CallAbi, RegClassId, RegisterClass, Target};

/// The x86-64 System V AMD64 target. A unit struct: everything it reports
/// lives in the [`regs`] tables.
pub struct X64Target;

impl Target for X64Target {
    fn pointer_size(&self) -> u8 {
        8
    }

    fn register_classes(&self) -> &'static [RegisterClass] {
        &regs::REGISTER_CLASSES
    }

    fn class_of(&self, ty: Type) -> Option<RegClassId> {
        match ty {
            // Integers, native ints, and GC pointers all live in GPRs.
            Type::Int32 | Type::Int64 | Type::NativeInt | Type::Ref | Type::ByRef => {
                Some(regs::GPR_CLASS_ID)
            }
            Type::Float | Type::Double => Some(regs::XMM_CLASS_ID),
            // Structs have no register class until struct support lands;
            // Void never types a value.
            Type::Struct(_) | Type::Void => None,
        }
    }

    fn classify_call(&self, sig: &CallSig) -> CompileResult<CallAbi> {
        let _ = sig;
        Err(CompileError::Unsupported(
            "classify_call: implemented in step_07.5",
        ))
    }

    fn call_site_stack_alignment(&self) -> u32 {
        regs::CALL_SITE_STACK_ALIGNMENT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rokajit::target::RegClassKind;

    #[test]
    fn reports_x64_machine_facts() {
        let target = X64Target;
        assert_eq!(target.pointer_size(), 8);
        assert_eq!(target.call_site_stack_alignment(), 16);
    }

    #[test]
    fn register_classes_match_their_ids() {
        let classes = X64Target.register_classes();
        assert_eq!(
            classes[regs::GPR_CLASS_ID.0 as usize].kind,
            RegClassKind::Int
        );
        assert_eq!(
            classes[regs::XMM_CLASS_ID.0 as usize].kind,
            RegClassKind::Float
        );
    }

    #[test]
    fn class_of_maps_the_ir_type_vocabulary() {
        let target = X64Target;
        for ty in [
            Type::Int32,
            Type::Int64,
            Type::NativeInt,
            Type::Ref,
            Type::ByRef,
        ] {
            assert_eq!(target.class_of(ty), Some(regs::GPR_CLASS_ID));
        }
        for ty in [Type::Float, Type::Double] {
            assert_eq!(target.class_of(ty), Some(regs::XMM_CLASS_ID));
        }
        assert_eq!(target.class_of(Type::Void), None);
    }
}
