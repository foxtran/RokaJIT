//! The `Target` trait — the single extension point through which all target
//! knowledge enters the compiler (frozen;
//! `decisions/2026-09-11-pipeline-and-target-contracts.md`).
//!
//! Following `docs/porting-strategy.md` ("Target independence"): the core is
//! generic code; everything machine-specific — pointer size, registers,
//! register classes, ABI rules — lives behind [`Target`], implemented once
//! per backend crate (`rokajit-x64` is the first). The core never names a
//! physical register: [`PhysReg`] is an opaque, target-local index whose
//! meaning only the target's own tables define.
//!
//! The trait is deliberately small. Capability queries (addressing-mode
//! folds, div-by-constant rules, …), the instruction-selection rule tables
//! (step_07.4), the byte-level encoder (step_07.6), and the GC/unwind
//! encoders (step_07.7) extend this trait when their consumers arrive, each
//! with its own `decisions/` entry.

use rokajit_ee::ee_info::EeInfo;

use crate::error::{CompileError, CompileResult};
use crate::ir::{lir, CallSig, Type};
use crate::pipeline::CodegenOutput;
use crate::structs::StructLayouts;

/// A physical register, as an opaque target-local index. The core compares,
/// copies, and stores these but never interprets the value; the mapping to
/// hardware registers is defined entirely by the target's register tables.
///
/// Invariant: indices are unique across **all** of a target's register
/// classes, so a bare `PhysReg` unambiguously names one hardware register
/// (e.g. rokajit-x64 numbers GPRs 0–15 and XMM registers 16–31).
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct PhysReg(pub u8);

/// The only distinction between register classes the core needs: integer
/// (pointers included) vs floating-point/vector.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RegClassKind {
    Int,
    Float,
}

/// A register class: a set of interchangeable, allocatable registers. This
/// is data, not code — the generic allocator and emitter consume it, the
/// target's backend crate supplies it as constants.
pub struct RegisterClass {
    /// Stable name for diagnostics and tests (e.g. `"x64-gpr"`).
    pub name: &'static str,
    pub kind: RegClassKind,
    /// The allocatable registers of this class, in allocation-preference
    /// order (caller-saved first, so a naive allocator avoids callee-saved
    /// traffic until forced). Registers with fixed duties (stack pointer,
    /// reserved frame pointer) are **not** listed here.
    pub registers: &'static [PhysReg],
    /// The subset of `registers` that survives calls (callee-saved).
    /// Invariant: every entry also appears in `registers`.
    pub callee_saved: &'static [PhysReg],
}

/// Identifies one of a target's register classes: an index into the slice
/// returned by [`Target::register_classes`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct RegClassId(pub u8);

/// Where one call argument or return value lives, in target terms.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ArgLocation {
    /// In a physical register.
    Reg(PhysReg),
    /// On the stack, at a byte offset from the stack pointer at the call
    /// instruction.
    Stack { offset: u32 },
    /// A register-passed struct (SysV eightbyte classification; step_10.9):
    /// one register per eightbyte, per the descriptor's classes — integer
    /// eightbytes take GPRs, SSE eightbytes take XMM registers. `sizes` and
    /// `offsets` are the descriptor's `eightByteSizes`/`eightByteOffsets`;
    /// moves must respect the sizes exactly.
    StructRegs {
        regs: [PhysReg; 2],
        count: u8,
        sizes: [u8; 2],
        offsets: [u8; 2],
    },
}

/// The ABI assignment for one call: where every argument and the return
/// value live. Produced by [`Target::classify_call`].
pub struct CallAbi {
    /// One entry per argument, in signature order. When
    /// [`CallSig::has_this`] is set, entry 0 is the implicit `this`, so this
    /// vector has `sig.args.len() + 1` entries.
    pub args: Vec<ArgLocation>,
    /// The return value's location; `None` for `Type::Void`.
    pub ret: Option<ArgLocation>,
    /// Total bytes of outgoing stack argument space the caller must reserve
    /// (0 when every argument fits in registers).
    pub stack_arg_bytes: u32,
}

/// A compilation target (an OS/arch pair's machine model). Object-safe by
/// construction: the pipeline holds `&dyn Target` and concrete targets are
/// expected to be unit structs (e.g. `rokajit_x64::X64Target`).
///
/// Everything here is a query over target *description*; compilation
/// algorithms stay in the core. Implementations return
/// `Err(CompileError::Unsupported(..))` for IR features the backend does not
/// cover yet.
pub trait Target {
    /// Native pointer width in bytes — the width of [`Type::NativeInt`] (and
    /// of `Ref`/`ByRef` slots).
    fn pointer_size(&self) -> u8;

    /// The target's register classes. An entry's position in this slice is
    /// its [`RegClassId`].
    fn register_classes(&self) -> &'static [RegisterClass];

    /// The register class values of `ty` live in, or `None` when the target
    /// does not keep the type on the frame/in registers. Structs are
    /// memory-backed but legal whenever the layout side table knows the
    /// class (step_10.9); `Void` never types a value.
    fn class_of(&self, ty: Type, layouts: &StructLayouts) -> Option<RegClassId>;

    /// Assign a call's arguments and return value to registers/stack per
    /// the target's ABI (SysV AMD64 on x64-Unix). `layouts` answers the
    /// struct-classification questions (step_10.9).
    fn classify_call(&self, sig: &CallSig, layouts: &StructLayouts) -> CompileResult<CallAbi>;

    /// Required stack-pointer alignment in bytes at every call instruction
    /// (SysV AMD64: 16).
    fn call_site_stack_alignment(&self) -> u32;

    /// Stage 4 emission (step_07.5; extension recorded in
    /// `decisions/2026-09-11-tier0-winch-codegen.md`): single-pass
    /// Winch-style codegen over LIR → machine code, driving the generic
    /// value-stack machinery in [`crate::codegen`]. Called by
    /// [`crate::pipeline::codegen`] after its tier gate. The EE parameter
    /// resolves call-target addresses (results land in
    /// [`CodegenOutput::relocations`]). The default body is the "no tier-0
    /// emitter on this target" answer.
    fn emit_tier0(&self, method: &lir::Method, ee: &dyn EeInfo) -> CompileResult<CodegenOutput> {
        let _ = (method, ee);
        Err(CompileError::Unsupported(
            "this target has no tier-0 emitter",
        ))
    }

    /// Stage 5 encoding (step_07.7; extension recorded in
    /// `decisions/2026-09-11-metadata-builder-api.md`): render the GC-info
    /// blob in the target's EE-facing encoding (x64: the GcInfoEncoder
    /// format, GCINFO_VERSION 5) from the generic facts. Called by
    /// [`crate::metadata::MetadataBuilder::finish`]. The blob must be valid
    /// even for a method with no GC roots: the EE requires a parseable
    /// minimal encoding for every method. The default body is the "no
    /// GC-info encoder on this target" answer.
    fn encode_gc_info(&self, input: &crate::metadata::GcInfoInput) -> CompileResult<Vec<u8>> {
        let _ = input;
        Err(CompileError::Unsupported(
            "this target has no GC-info encoder",
        ))
    }

    /// Stage 5 encoding (step_07.7, same decision as
    /// [`Target::encode_gc_info`]): render the unwind blobs — one per
    /// function fragment, root first — in the target's EE-facing encoding
    /// (x64: Windows AMD64 `UNWIND_INFO`, consumed by `allocUnwindInfo`).
    /// The default body is the "no unwind encoder on this target" answer.
    fn encode_unwind_info(
        &self,
        input: &crate::metadata::UnwindInput,
    ) -> CompileResult<Vec<crate::artifact::UnwindBlob>> {
        let _ = input;
        Err(CompileError::Unsupported(
            "this target has no unwind encoder",
        ))
    }
}
