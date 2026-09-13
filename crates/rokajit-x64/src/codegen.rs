//! x64 tier-0 emission (step_07.5): binds the pre-allocation [`Inst`]
//! descriptors to physical operands and emits bytes through the step_07.6
//! assembler, driving the generic Winch machinery
//! ([`rokajit::codegen::ValueState`]). The invariants (frame-resident
//! roots, join-point spill discipline, no liveness) live with the generic
//! half and in `decisions/2026-09-11-tier0-winch-codegen.md`; this module
//! owns the x64 emission choices:
//!
//! - **Frame layout** ([`FrameLayout`]): every local/arg/temp gets a slot
//!   at `[rbp - off]`, naturally aligned; the frame size rounds up to 16
//!   so that with the `push rbp; mov rbp, rsp` prolog every call site has
//!   `rsp ≡ 0 (mod 16)` (SysV §3.2.2 — entry leaves `rsp ≡ 8`, the push
//!   and the 16-aligned `sub` preserve alignment).
//! - **Prolog**: the [`lower_frame`] contract sequence, then incoming
//!   argument registers spilled to their slots (assignment per
//!   [`classify_call`]), IL locals zeroed (the init-locals flag does not
//!   reach LIR — zeroing unconditionally is always permitted), and
//!   GC-typed temp slots zeroed so the static root set holds a valid
//!   value at every safepoint.
//! - **Scratch pool**: the caller-saved GPRs, round-robin
//!   ([`SCRATCH_GPRS`]). No callee-saved register is ever touched, so the
//!   prolog/epilog stays the fixed three-instruction frame contract.
//! - **Floats** (step_10.2): float values are always frame-resident —
//!   every definition stores to the value's slot immediately (the GC-ref
//!   discipline), and computation uses two fixed scratch XMM registers
//!   ([`SCRATCH_XMM_A`]/[`SCRATCH_XMM_B`]) strictly within one statement,
//!   so the value machine never tracks XMM residence. Constants
//!   materialize through a GPR (`movabs` + `movq`); compares are
//!   `ucomis*` plus a parity-aware expansion.
//! - **Calls**: all register-resident temps spill before the `call`
//!   (the pool is caller-saved); the target address is resolved through
//!   the EE at emit time and recorded as a `RELATIVE32` relocation plus
//!   a managed call site (a GC safepoint) for 07.7 to drain. EE helper
//!   calls (float `rem`) share the path with no method handle recorded.

use rokajit::artifact::{CallSite, ChunkRef, CodeChunk, CodeChunks, Relocation};
use rokajit::codegen::{Loc, Move, ReadSrc, ValueState};
use rokajit::error::{CompileError, CompileResult};
use rokajit::ir::lir::StmtKind;
use rokajit::ir::{hir, lir, BinaryOp, CallSig, LocalId, Type, UnaryOp};
use rokajit::lower::{Cx, Label};
use rokajit::pipeline::{CodegenOutput, FrameInfo};
#[cfg(test)]
use rokajit::structs::{GcCell, SysVPass};
use rokajit::structs::{StructLayout, StructLayouts};
use rokajit::target::{ArgLocation, CallAbi, PhysReg};
use rokajit_ee::ee_info::{const_lookup_addr, const_lookup_slot, EeInfo};
use rokajit_ee::enums::{CorInfoHelpFunc, RelocType};
use rokajit_ee::handles::{ClassHandle, MethodHandle};

use crate::encode::{Asm, Mem, Rm, RmX, Rmi};
use crate::inst::{
    ArithFOp, ArithOp, BlockAddr, CondCode, EbReg, FWidth, Inst, Place, ShiftOp, Src, Width,
    XmmPlace, XmmSrc,
};
use crate::lower::{lower_frame, lower_stmt, width_of_ty, FrameReq};
use crate::regs::{self, Gpr, Xmm};

/// The largest block copy/zero emitted inline (unrolled 8/4/2/1 moves);
/// larger blocks go through `CORINFO_HELP_MEMCPY`/`MEMSET` (step_10.9
/// threshold decision).
const BLOCK_INLINE_MAX: u32 = 64;

/// The greedy 8/4/2/1 decomposition of a `size`-byte block: `(offset,
/// chunk size)` pairs in ascending offset order.
fn chunk_plan(size: u32) -> Vec<(u32, u8)> {
    let mut plan = Vec::new();
    let mut done = 0u32;
    for chunk in [8u8, 4, 2, 1] {
        while size - done >= u32::from(chunk) {
            plan.push((done, chunk));
            done += u32::from(chunk);
        }
    }
    plan
}

/// The GPRs an instruction sequence pins to ABI duties (step_10.9):
/// explicit fixed-register destinations (argument setup) and, when
/// `pre_call` is false, the fixed registers result moves read (the return
/// registers — live from the call until their store executes).
fn fixed_gprs(insts: &[Inst], pre_call: bool) -> Vec<PhysReg> {
    let mut v: Vec<PhysReg> = Vec::new();
    let mut push = |g: Gpr| {
        let p = g.phys();
        if !v.contains(&p) {
            v.push(p);
        }
    };
    for inst in insts {
        match *inst {
            Inst::Mov {
                dst: Place::Reg(g), ..
            }
            | Inst::Lea {
                dst: Place::Reg(g), ..
            }
            | Inst::MovExt {
                dst: Place::Reg(g), ..
            }
            | Inst::LoadEightbyte {
                dst: EbReg::Gpr(g), ..
            } if pre_call => push(g),
            Inst::StoreEightbyte {
                src: EbReg::Gpr(g), ..
            } if !pre_call => push(g),
            _ => {}
        }
    }
    v
}

/// Tier-0 scratch pool: the caller-saved GPRs (SysV §3.2.1), in
/// round-robin allocation order. Calls spill the whole pool, so no
/// callee-saved register ever appears in emitted code and the frame
/// contract stays fixed.
const SCRATCH_GPRS: [Gpr; 9] = [
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

/// Tier-0 float scratch registers. Float *values* are always
/// frame-resident — every definition stores to the value's frame slot
/// immediately (the same discipline as GC references) — so an XMM
/// register never holds a value across a statement boundary, and the two
/// scratch registers need no allocation or spill tracking. xmm14/15 are
/// chosen so they never collide with the SysV argument registers
/// (xmm0–7) during call setup.
const SCRATCH_XMM_A: Xmm = Xmm::Xmm15;
const SCRATCH_XMM_B: Xmm = Xmm::Xmm14;

/// First synthetic label id (for the float-compare branch expansions'
/// internal skip labels). Block labels are small integers; synthetics
/// count down from here so the two spaces never meet.
const FIRST_SYNTHETIC_LABEL: u32 = 0x8000_0000;

/// Hot-code chunk alignment requested from `allocMem` (≤ 32, corjit.h:81).
const HOT_CODE_ALIGNMENT: u32 = 16;

/// Call instruction lengths: `call rel32` (E8 + 4) and `call [rip+rel32]`
/// (FF 15 + 4). Recorded in [`CallSite::size`]; the GC safepoint of a call
/// is the byte after it.
const DIRECT_CALL_LEN: u32 = 5;
const INDIRECT_CALL_LEN: u32 = 6;

/// `PhysReg` → `Gpr` (the registers codegen handles are always GPRs;
/// the tables in [`regs`] number them by hardware encoding).
fn gpr_of(reg: PhysReg) -> CompileResult<Gpr> {
    Gpr::from_phys(reg).ok_or(CompileError::Internal("PhysReg outside the GPR range"))
}

/// The SysV AMD64 argument/return assignment for one call (the
/// [`rokajit::target::Target::classify_call`] body). Integer-class values
/// (pointers included) take [`regs::INT_ARG_REGS`] in order, floats take
/// [`regs::FLOAT_ARG_REGS`]; overflow goes to 8-byte stack slots.
/// Aggregates (step_10.9) follow the EE's eightbyte classification
/// (`layouts`): a register-passed struct takes one register per eightbyte
/// from its class's pool — but only when BOTH pools have room for the
/// whole struct (otherwise the entire struct goes on the stack, 8-byte
/// rounded); a struct the EE never classifies for registers is always a
/// full-size stack copy. Return values: SSE eightbytes take
/// [`regs::FLOAT_RETURN_REGS`], integer eightbytes
/// [`regs::INT_RETURN_REGS`]; a non-register-passed struct returns the
/// hidden-retbuf pointer in `rax` (the importer supplies the buffer as an
/// implicit argument).
pub fn classify_call(sig: &CallSig, layouts: &StructLayouts) -> CompileResult<CallAbi> {
    let mut int_n = 0usize;
    let mut float_n = 0usize;
    let mut stack_bytes = 0u32;
    let mut args = Vec::with_capacity(sig.args.len() + usize::from(sig.has_this));
    if sig.has_this {
        args.push(place_arg(
            Type::Ref,
            &mut int_n,
            &mut float_n,
            &mut stack_bytes,
            layouts,
        )?);
    }
    for &ty in &sig.args {
        args.push(place_arg(
            ty,
            &mut int_n,
            &mut float_n,
            &mut stack_bytes,
            layouts,
        )?);
    }
    let ret = match sig.ret {
        Type::Void => None,
        Type::Float | Type::Double => Some(ArgLocation::Reg(regs::FLOAT_RETURN_REG.phys())),
        Type::Struct(class) => {
            let sysv = &layout_of(layouts, class)?.sysv;
            if sysv.passed_in_registers {
                // Per-register-file assignment: integer eightbytes take
                // rax then rdx, SSE eightbytes xmm0 then xmm1.
                let mut regs = [regs::INT_RETURN_REGS[0].phys(); 2];
                let (mut i, mut f) = (0usize, 0usize);
                for (k, class) in sysv.classes.iter().enumerate().take(sysv.count as usize) {
                    regs[k] = if class.is_sse() {
                        let r = regs::FLOAT_RETURN_REGS[f].phys();
                        f += 1;
                        r
                    } else {
                        let r = regs::INT_RETURN_REGS[i].phys();
                        i += 1;
                        r
                    };
                }
                if sysv.count == 1 {
                    regs[1] = regs[0];
                }
                Some(ArgLocation::StructRegs {
                    regs,
                    count: sysv.count,
                    sizes: sysv.sizes,
                    offsets: sysv.offsets,
                })
            } else {
                // The hidden-return-buffer convention: the callee returns
                // the buffer address in rax (clr-abi.md). The caller
                // ignores it — the value is the retbuf temp it passed.
                Some(ArgLocation::Reg(regs::INT_RETURN_REGS[0].phys()))
            }
        }
        _ => Some(ArgLocation::Reg(regs::INT_RETURN_REGS[0].phys())),
    };
    Ok(CallAbi {
        args,
        ret,
        stack_arg_bytes: stack_bytes.div_ceil(16) * 16,
    })
}

/// The layout of a struct mentioned in a signature, or an upstream-bug
/// error (the importer populates the side table for every class it emits).
fn layout_of(layouts: &StructLayouts, class: ClassHandle) -> CompileResult<&StructLayout> {
    layouts.get(&class).ok_or(CompileError::Internal(
        "struct class missing from the layout side table",
    ))
}

fn place_arg(
    ty: Type,
    int_n: &mut usize,
    float_n: &mut usize,
    stack_bytes: &mut u32,
    layouts: &StructLayouts,
) -> CompileResult<ArgLocation> {
    let is_float = match ty {
        Type::Int32 | Type::Int64 | Type::NativeInt | Type::Ref | Type::ByRef => false,
        Type::Float | Type::Double => true,
        Type::Struct(class) => {
            let layout = layout_of(layouts, class)?;
            return place_struct_arg(layout, int_n, float_n, stack_bytes);
        }
        Type::Void => return Err(CompileError::Internal("void-typed argument")),
    };
    let (n, limit) = if is_float {
        (float_n, regs::FLOAT_ARG_REGS.len())
    } else {
        (int_n, regs::INT_ARG_REGS.len())
    };
    if *n < limit {
        let phys = if is_float {
            regs::FLOAT_ARG_REGS[*n].phys()
        } else {
            regs::INT_ARG_REGS[*n].phys()
        };
        *n += 1;
        Ok(ArgLocation::Reg(phys))
    } else {
        let offset = *stack_bytes;
        *stack_bytes += 8;
        Ok(ArgLocation::Stack { offset })
    }
}

/// The whole-struct rule for a struct argument (SysV + clr-abi.md): when
/// the EE classifies the struct for registers AND both register pools
/// have room for every eightbyte, each eightbyte takes the next register
/// of its file; otherwise (either pool short, or the EE never classifies
/// it) the ENTIRE struct goes on the stack as a full `size`-byte copy,
/// 8-byte rounded.
fn place_struct_arg(
    layout: &StructLayout,
    int_n: &mut usize,
    float_n: &mut usize,
    stack_bytes: &mut u32,
) -> CompileResult<ArgLocation> {
    let sysv = &layout.sysv;
    if sysv.passed_in_registers {
        let need_int = sysv.classes[..sysv.count as usize]
            .iter()
            .filter(|c| !c.is_sse())
            .count();
        let need_sse = sysv.count as usize - need_int;
        if *int_n + need_int <= regs::INT_ARG_REGS.len()
            && *float_n + need_sse <= regs::FLOAT_ARG_REGS.len()
        {
            let mut regs = [regs::INT_ARG_REGS[0].phys(); 2];
            for (k, reg) in regs.iter_mut().enumerate().take(sysv.count as usize) {
                *reg = if sysv.classes[k].is_sse() {
                    let r = regs::FLOAT_ARG_REGS[*float_n].phys();
                    *float_n += 1;
                    r
                } else {
                    let r = regs::INT_ARG_REGS[*int_n].phys();
                    *int_n += 1;
                    r
                };
            }
            // A one-eightbyte struct duplicates its register in the
            // unused slot (a defined value, never read).
            if sysv.count == 1 {
                regs[1] = regs[0];
            }
            return Ok(ArgLocation::StructRegs {
                regs,
                count: sysv.count,
                sizes: sysv.sizes,
                offsets: sysv.offsets,
            });
        }
    }
    let offset = *stack_bytes;
    *stack_bytes += layout.size.div_ceil(8) * 8;
    Ok(ArgLocation::Stack { offset })
}

/// The `jcc` condition implementing the ordered half of a float compare
/// branch (after any parity handling): equality maps to `je`, the
/// ordered greater forms ride `ja`/`jae` (CF excludes unordered), the
/// less forms ride `jb`/`jbe` (CF/ZF include it, per the `.un` split in
/// [`Emitter::emit_jcc_f`]).
fn float_jcc_tail(op: BinaryOp) -> CondCode {
    match op {
        BinaryOp::Eq => CondCode::Eq,
        BinaryOp::Ne => CondCode::Ne,
        BinaryOp::Gt | BinaryOp::UGt => CondCode::UGt,
        BinaryOp::Ge | BinaryOp::UGe => CondCode::UGe,
        BinaryOp::Lt | BinaryOp::ULt => CondCode::ULt,
        BinaryOp::Le | BinaryOp::ULe => CondCode::ULe,
        _ => CondCode::Eq, // unreachable: the caller matches compares only
    }
}

/// The frame layout: every local/arg/temp's slot offset below `rbp`, plus
/// the 16-aligned frame size (the `sub rsp, N` immediate). Slot `i`'s
/// bytes are `[rbp - slots[i], rbp - slots[i] + size)` — so `slots[i]` is
/// the offset [`rokajit::pipeline::GcRootSlot`] records. The outgoing
/// stack-argument area (step_10.9) sits below every slot, addressed
/// `[rsp + off]` at call sites; `frame_size` covers it, so `rsp ≡ 0
/// (mod 16)` holds at every call as before.
pub struct FrameLayout {
    pub slots: Vec<u32>,
    pub frame_size: u32,
    /// Bytes reserved below the slots for outgoing stack arguments (the
    /// maximum `CallAbi::stack_arg_bytes` over the method's call sites).
    pub outgoing_bytes: u32,
}

impl FrameLayout {
    pub fn compute(method: &lir::Method, outgoing_bytes: u32) -> CompileResult<FrameLayout> {
        let mut offset = 0u32;
        let mut slots = Vec::with_capacity(method.locals.len());
        for local in &method.locals {
            let (size, align) = match local.ty {
                Type::Int32 | Type::Float => (4, 4),
                Type::Int64 | Type::NativeInt | Type::Ref | Type::ByRef | Type::Double => (8, 8),
                Type::Struct(class) => {
                    let layout =
                        method
                            .struct_layouts
                            .get(&class)
                            .ok_or(CompileError::Internal(
                                "struct local missing from the layout side table",
                            ))?;
                    // Slots are exactly `size` bytes at `align` alignment —
                    // except structs embedding GC pointers: those round the
                    // size up to 8 and force 8-alignment so every reported
                    // root (slot offset − cell offset) stays 8-aligned for
                    // the GC slot-table encoder.
                    if layout.gc_cells.is_empty() {
                        (layout.size, layout.align)
                    } else {
                        (layout.size.div_ceil(8) * 8, layout.align.max(8))
                    }
                }
                Type::Void => return Err(CompileError::Internal("void-typed local")),
            };
            // The slot's base (its lowest byte, at `rbp - slots[i]`) must
            // meet the type's alignment, so the slot's *end* rounds up to
            // `align` too — the tail padding is dead space. (For the
            // scalar types size is a multiple of align and nothing
            // changes; a 100-byte struct at 8-alignment would otherwise
            // land on a 4-aligned base.)
            offset = (offset.div_ceil(align) * align + size).div_ceil(align) * align;
            slots.push(offset);
        }
        Ok(FrameLayout {
            slots,
            frame_size: (offset + outgoing_bytes).div_ceil(16) * 16,
            outgoing_bytes,
        })
    }

    /// The largest outgoing-argument area any call site in `method`
    /// needs (0 when every argument of every call fits in registers).
    pub fn max_outgoing_bytes(method: &lir::Method) -> CompileResult<u32> {
        let mut max = 0u32;
        for block in &method.blocks {
            for stmt in &block.stmts {
                if let StmtKind::Call { sig, .. } = &stmt.kind {
                    max = max.max(classify_call(sig, &method.struct_layouts)?.stack_arg_bytes);
                }
            }
        }
        Ok(max)
    }
}

/// Stage 4 emission for x64 (the `Target::emit_tier0` body): LIR →
/// machine-code bytes plus the relocation/call-site/frame facts 07.7
/// drains. Single pass over the blocks in layout order.
pub fn emit_tier0(method: &lir::Method, ee: &dyn EeInfo) -> CompileResult<CodegenOutput> {
    let outgoing = FrameLayout::max_outgoing_bytes(method)?;
    let layout = FrameLayout::compute(method, outgoing)?;
    let cx = Cx::new(&method.locals, &method.struct_layouts);
    let mut em = Emitter::new(method, layout, ee);

    let prolog = lower_frame(&FrameReq, &cx).ok_or(CompileError::Internal(
        "the catch-all frame rule must match",
    ))?;
    for inst in &prolog {
        em.emit_inst(inst)?;
    }
    em.spill_incoming_args()?;
    em.zero_init_slots()?;

    for block in &method.blocks {
        em.asm.bind(Label(block.id));
        em.vs.reset();
        for stmt in &block.stmts {
            let insts = lower_stmt(stmt, &cx).ok_or(CompileError::Unsupported(
                "no x64 lowering rule matched an LIR statement",
            ))?;
            // Join discipline: before an edge, every value becomes
            // frame-resident (before the compare, so the flag pair stays
            // adjacent — spills are `mov`s and don't clobber flags).
            if matches!(stmt.kind, StmtKind::Branch { .. } | StmtKind::Jump { .. }) {
                let moves = em.vs.spill_all();
                em.apply(moves)?;
            }
            if let StmtKind::Call { sig, .. } = &stmt.kind {
                em.call_sig = Some(sig.clone());
            }
            // ABI-pinned registers that will hold live values during this
            // statement (step_10.9): argument registers written by the
            // pre-call moves, then the return registers after the call.
            // Scratch allocation must exclude them — they are not
            // value-machine-tracked.
            em.fixed_dests = fixed_gprs(&insts, true);
            for (i, inst) in insts.iter().enumerate() {
                if matches!(inst, Inst::CallDirect { .. } | Inst::CallHelper { .. }) {
                    em.fixed_dests = fixed_gprs(&insts[i + 1..], false);
                }
                em.emit_inst(inst)?;
            }
            em.fixed_dests.clear();
            em.call_sig = None;
        }
        // A block whose terminator lowered to a fallthrough (jump-to-next
        // elision) still ends in an edge: the successor must see the same
        // frame-resident state. After a `Return` there is no edge.
        if !matches!(
            block.stmts.last().map(|s| &s.kind),
            Some(StmtKind::Return { .. })
        ) {
            let moves = em.vs.spill_all();
            em.apply(moves)?;
        }
    }

    let code = em
        .asm
        .finalize()
        .map_err(|_| CompileError::Internal("unbound branch label at finalize"))?;
    Ok(CodegenOutput {
        code: CodeChunks {
            hot: CodeChunk {
                bytes: code.bytes,
                alignment: HOT_CODE_ALIGNMENT,
            },
            cold: None,
        },
        ro_data: Vec::new(),
        relocations: em.relocations,
        call_sites: em.call_sites,
        frame: FrameInfo {
            frame_size: em.layout.frame_size,
            gc_roots: rokajit::codegen::gc_roots(
                &method.locals,
                &em.layout.slots,
                &method.struct_layouts,
            ),
        },
    })
}

/// The emission state: the assembler, the generic value machine, the
/// frame layout, and the facts recorded for 07.7.
struct Emitter<'a> {
    asm: Asm,
    vs: ValueState,
    layout: FrameLayout,
    locals: &'a [hir::Local],
    layouts: &'a StructLayouts,
    num_args: usize,
    /// `num_args + num_il_locals`: ids below this are IL args/locals
    /// (their slots are written on copy), above it temps (tag-tracked).
    num_frame_fixed: usize,
    ee: &'a dyn EeInfo,
    /// The signature of the call statement currently being emitted (for
    /// the call-site record).
    call_sig: Option<CallSig>,
    call_sites: Vec<CallSite>,
    relocations: Vec<Relocation>,
    /// Synthetic-label supply for the float-compare branch expansions
    /// (counts down from [`FIRST_SYNTHETIC_LABEL`]).
    next_synthetic: u32,
    /// ABI-pinned registers currently holding live values (step_10.9):
    /// argument registers already written during a call's setup, return
    /// registers after it, and incoming argument registers during the
    /// prolog spill. Scratch allocation must never clobber them (they are
    /// not value-machine-tracked, so `take_scratch` would silently reuse
    /// one).
    fixed_dests: Vec<PhysReg>,
}

impl<'a> Emitter<'a> {
    fn new(method: &'a lir::Method, layout: FrameLayout, ee: &'a dyn EeInfo) -> Self {
        let pool: Vec<PhysReg> = SCRATCH_GPRS.iter().map(|g| g.phys()).collect();
        let tys: Vec<Type> = method.locals.iter().map(|l| l.ty).collect();
        Emitter {
            asm: Asm::new(),
            vs: ValueState::new(&pool, &layout.slots, &tys),
            layout,
            locals: &method.locals,
            layouts: &method.struct_layouts,
            num_args: method.num_args as usize,
            num_frame_fixed: (method.num_args + method.num_il_locals) as usize,
            ee,
            call_sig: None,
            call_sites: Vec::new(),
            relocations: Vec::new(),
            next_synthetic: FIRST_SYNTHETIC_LABEL,
            fixed_dests: Vec::new(),
        }
    }

    /// `fixed_dests ∪ extra` — the scratch-allocation exclusion list while
    /// ABI registers hold live values.
    fn scratch_exclude(&self, extra: &[PhysReg]) -> Vec<PhysReg> {
        let mut v = self.fixed_dests.clone();
        for &e in extra {
            if !v.contains(&e) {
                v.push(e);
            }
        }
        v
    }

    fn ty_of(&self, id: LocalId) -> Type {
        self.locals[id.0 as usize].ty
    }

    fn width_of(&self, id: LocalId) -> CompileResult<Width> {
        width_of_ty(self.ty_of(id)).ok_or(CompileError::Internal(
            "value-state entry without a GPR width",
        ))
    }

    fn slot_mem(&self, offset: u32) -> Mem {
        Mem::base_disp(regs::FRAME_POINTER, -(offset as i32))
    }

    fn own_slot(&self, id: LocalId) -> Mem {
        self.slot_mem(self.vs.slot_of(id))
    }

    /// Emit the moves the value machine decided on.
    fn apply(&mut self, moves: Vec<Move>) -> CompileResult<()> {
        for m in moves {
            match m {
                Move::Spill { reg, slot, ty } => {
                    let w = width_of_ty(ty)
                        .ok_or(CompileError::Internal("spill of a non-GPR value"))?;
                    self.asm
                        .mov(w, Rm::Mem(self.slot_mem(slot)), Rmi::Reg(gpr_of(reg)?));
                }
                Move::Reload { slot, ty, reg } => {
                    let w = width_of_ty(ty)
                        .ok_or(CompileError::Internal("reload of a non-GPR value"))?;
                    self.asm
                        .mov(w, Rm::Reg(gpr_of(reg)?), Rmi::Mem(self.slot_mem(slot)));
                }
                Move::Remat { imm, reg, ty } => {
                    let w = width_of_ty(ty).ok_or(CompileError::Internal(
                        "rematerialization of a non-GPR value",
                    ))?;
                    self.asm.mov(w, Rm::Reg(gpr_of(reg)?), Rmi::Imm(imm));
                }
            }
        }
        Ok(())
    }

    /// A descriptor source resolved to a physical operand.
    fn rmi_of(&self, src: Src) -> CompileResult<Rmi> {
        Ok(match src {
            Src::Reg(g) => Rmi::Reg(g),
            Src::Imm(i) => Rmi::Imm(i),
            Src::Val(v) => match self.vs.read(v.0) {
                ReadSrc::Reg(p) => Rmi::Reg(gpr_of(p)?),
                ReadSrc::Slot(off) => Rmi::Mem(self.slot_mem(off)),
                ReadSrc::Imm(i) => Rmi::Imm(i),
            },
        })
    }

    /// Like [`Emitter::rmi_of`], but an immediate too wide for the target
    /// encoding (a 64-bit ALU/cmp/mem-store form carries a sign-extended
    /// imm32) materializes into a scratch register first — the `inst.rs`
    /// contract makes that codegen's duty. Allocation order matters for
    /// callers: take this operand *before* resolving any sibling operand,
    /// so the scratch allocation can't evict a register the sibling read
    /// already captured.
    fn wide_imm(&mut self, width: Width, src: Src, exclude: &[PhysReg]) -> CompileResult<Rmi> {
        if let Src::Imm(i) = src {
            if matches!(width, Width::W64) && i32::try_from(i).is_err() {
                let exclude = &self.scratch_exclude(exclude);
                let (p, moves) = self.vs.take_scratch(exclude);
                self.apply(moves)?;
                let g = gpr_of(p)?;
                // `mov r64, imm64` (movabs) covers every payload.
                self.asm.mov(Width::W64, Rm::Reg(g), Rmi::Imm(i));
                return Ok(Rmi::Reg(g));
            }
        }
        self.rmi_of(src)
    }

    /// A descriptor source as the `r/m` side: immediates have no such
    /// encoding, so they materialize into a scratch register.
    fn rm_of(&mut self, src: Src, width: Width, exclude: &[PhysReg]) -> CompileResult<Rm> {
        match self.rmi_of(src)? {
            Rmi::Reg(g) => Ok(Rm::Reg(g)),
            Rmi::Mem(m) => Ok(Rm::Mem(m)),
            Rmi::Imm(i) => {
                let exclude = &self.scratch_exclude(exclude);
                let (p, moves) = self.vs.take_scratch(exclude);
                self.apply(moves)?;
                let g = gpr_of(p)?;
                self.asm.mov(width, Rm::Reg(g), Rmi::Imm(i));
                Ok(Rm::Reg(g))
            }
        }
    }

    /// Record a temp's location, emitting any defensive moves the machine
    /// decided on.
    fn define_loc(&mut self, id: LocalId, loc: Loc) -> CompileResult<()> {
        let moves = self.vs.define(id, loc);
        self.apply(moves)
    }

    /// Record a temp's value living in `g`. GC-typed temps spill to their
    /// slot immediately — every GC reference is frame-resident (the root
    /// invariant); the tag is then its own slot.
    fn define_temp_reg(&mut self, t: LocalId, g: Gpr) -> CompileResult<()> {
        if matches!(self.ty_of(t), Type::Ref | Type::ByRef) {
            let w = self.width_of(t)?;
            self.asm.mov(w, Rm::Mem(self.own_slot(t)), Rmi::Reg(g));
            let slot = self.vs.slot_of(t);
            self.define_loc(t, Loc::Mem(slot))
        } else {
            self.define_loc(t, Loc::Reg(g.phys()))
        }
    }

    // ---- float emission (step_10.2): values are always frame-resident ----

    /// A fresh synthetic label (float-compare branch expansions only).
    fn synthetic_label(&mut self) -> Label {
        self.next_synthetic += 1;
        Label(rokajit::ir::BlockId(self.next_synthetic - 1))
    }

    /// Materialize a float constant's bit pattern into `dst` through a GPR
    /// scratch (`mov`/`movabs` + `movq`/`movd`). This is the constant-load
    /// mechanism: no rodata pool, no relocations — the step_10.2 decision.
    fn materialize_bits(&mut self, dst: Xmm, width: FWidth, bits: u64) -> CompileResult<()> {
        let exclude = &self.scratch_exclude(&[]);
        let (p, moves) = self.vs.take_scratch(exclude);
        self.apply(moves)?;
        let g = gpr_of(p)?;
        let w = match width {
            FWidth::S => Width::W32,
            FWidth::D => Width::W64,
        };
        self.asm.mov(w, Rm::Reg(g), Rmi::Imm(bits as i64));
        self.asm.mov_gpr_to_xmm(width, dst, g);
        Ok(())
    }

    /// An XMM-side descriptor source resolved to a physical operand. Float
    /// values are always frame-resident, so a `Val` reads its frame slot;
    /// `Bits` materializes through a GPR into the given scratch register.
    fn rmx_of(&mut self, src: XmmSrc, width: FWidth, scratch: Xmm) -> CompileResult<RmX> {
        Ok(match src {
            XmmSrc::Reg(x) => RmX::Reg(x),
            XmmSrc::Val(v) => match self.vs.read(v.0) {
                ReadSrc::Slot(off) => RmX::Mem(self.slot_mem(off)),
                // The slot-resident policy: a float value never carries a
                // Reg/Const tag, so any other read is an upstream bug.
                _ => return Err(CompileError::Internal("float value not slot-resident")),
            },
            XmmSrc::Bits(bits) => {
                self.materialize_bits(scratch, width, bits)?;
                RmX::Reg(scratch)
            }
        })
    }

    /// A float result in `src` lands at `dst`: an XMM register directly
    /// (ABI duties), or a value's frame slot — the slot-resident policy.
    /// A frame-fixed destination (an IL local/arg) follows the store
    /// discipline: aliases of its slot materialize first.
    fn define_xmm(&mut self, dst: XmmPlace, width: FWidth, src: Xmm) -> CompileResult<()> {
        match dst {
            XmmPlace::Reg(x) => {
                if x != src {
                    self.asm.mov_f_load(width, x, RmX::Reg(src));
                }
            }
            XmmPlace::Val(v) if (v.0 .0 as usize) < self.num_frame_fixed => {
                let moves = self.vs.before_local_write(v.0);
                self.apply(moves)?;
                self.asm.mov_f_store(width, self.own_slot(v.0), src);
            }
            XmmPlace::Val(v) => {
                self.asm.mov_f_store(width, self.own_slot(v.0), src);
                let slot = self.vs.slot_of(v.0);
                self.define_loc(v.0, Loc::Mem(slot))?;
            }
        }
        Ok(())
    }

    /// A float constant definition: for a value destination the bit
    /// pattern stores straight to the slot from a GPR (no XMM register
    /// needed); a register destination materializes through a GPR.
    fn emit_const_f(&mut self, width: FWidth, dst: XmmPlace, bits: u64) -> CompileResult<()> {
        match dst {
            XmmPlace::Reg(x) => self.materialize_bits(x, width, bits),
            XmmPlace::Val(v) => {
                if (v.0 .0 as usize) < self.num_frame_fixed {
                    let moves = self.vs.before_local_write(v.0);
                    self.apply(moves)?;
                }
                let slot = self.own_slot(v.0);
                match width {
                    // `mov dword [slot], imm32` covers every f32 pattern.
                    FWidth::S => self
                        .asm
                        .mov(Width::W32, Rm::Mem(slot), Rmi::Imm(bits as i64)),
                    // `mov qword [mem], imm32` sign-extends, so a general
                    // f64 pattern goes through a GPR scratch.
                    FWidth::D => {
                        let (p, moves) = self.vs.take_scratch(&[]);
                        self.apply(moves)?;
                        let g = gpr_of(p)?;
                        self.asm.mov(Width::W64, Rm::Reg(g), Rmi::Imm(bits as i64));
                        self.asm.mov(Width::W64, Rm::Mem(slot), Rmi::Reg(g));
                    }
                }
                if (v.0 .0 as usize) >= self.num_frame_fixed {
                    let slot = self.vs.slot_of(v.0);
                    self.define_loc(v.0, Loc::Mem(slot))?;
                }
                Ok(())
            }
        }
    }

    /// `movss`/`movsd`: a value-to-value or register move. Slot-to-slot
    /// copies go through the scratch register (no mem,mem SSE form).
    fn emit_mov_f(&mut self, width: FWidth, dst: XmmPlace, src: XmmSrc) -> CompileResult<()> {
        let src = self.rmx_of(src, width, SCRATCH_XMM_A)?;
        match dst {
            XmmPlace::Reg(x) => {
                self.asm.mov_f_load(width, x, src);
                Ok(())
            }
            XmmPlace::Val(v) => {
                let reg = match src {
                    RmX::Reg(x) => Some(x),
                    RmX::Mem(m) => {
                        if m == self.own_slot(v.0) {
                            None // self-copy: nothing to do
                        } else {
                            self.asm.mov_f_load(width, SCRATCH_XMM_A, RmX::Mem(m));
                            Some(SCRATCH_XMM_A)
                        }
                    }
                };
                match reg {
                    Some(x) => self.define_xmm(XmmPlace::Val(v), width, x),
                    None => Ok(()),
                }
            }
        }
    }

    /// Scalar SSE arithmetic: `lhs` into scratch A, the operation against
    /// `rhs` applied in place, the result stored to `dst`'s slot.
    fn emit_arith_f(
        &mut self,
        op: ArithFOp,
        width: FWidth,
        dst: XmmPlace,
        lhs: XmmSrc,
        rhs: XmmSrc,
    ) -> CompileResult<()> {
        let lhs = self.rmx_of(lhs, width, SCRATCH_XMM_A)?;
        let rhs = self.rmx_of(rhs, width, SCRATCH_XMM_B)?;
        if lhs != RmX::Reg(SCRATCH_XMM_A) {
            self.asm.mov_f_load(width, SCRATCH_XMM_A, lhs);
        }
        self.asm.arith_f(op, width, SCRATCH_XMM_A, rhs);
        self.define_xmm(dst, width, SCRATCH_XMM_A)
    }

    /// Float `neg`: XOR with the sign mask (the only form that gets
    /// `-0.0` and NaN signs exactly right — `0.0 - x` does neither).
    fn emit_neg_f(&mut self, width: FWidth, dst: XmmPlace, src: XmmSrc) -> CompileResult<()> {
        let mask = match width {
            FWidth::S => u64::from(0x8000_0000u32),
            FWidth::D => 0x8000_0000_0000_0000u64,
        };
        self.materialize_bits(SCRATCH_XMM_B, width, mask)?;
        let src = self.rmx_of(src, width, SCRATCH_XMM_A)?;
        if src != RmX::Reg(SCRATCH_XMM_A) {
            self.asm.mov_f_load(width, SCRATCH_XMM_A, src);
        }
        self.asm
            .xor_f(width, SCRATCH_XMM_A, RmX::Reg(SCRATCH_XMM_B));
        self.define_xmm(dst, width, SCRATCH_XMM_A)
    }

    /// `ucomis*`: the lhs must be in a register (there is no
    /// memory-first form); the rhs rides as register or memory.
    fn emit_cmp_f(&mut self, width: FWidth, lhs: XmmSrc, rhs: XmmSrc) -> CompileResult<()> {
        let lhs = self.rmx_of(lhs, width, SCRATCH_XMM_A)?;
        let rhs = self.rmx_of(rhs, width, SCRATCH_XMM_B)?;
        let lhs_reg = match lhs {
            RmX::Reg(x) => x,
            RmX::Mem(m) => {
                self.asm.mov_f_load(width, SCRATCH_XMM_A, RmX::Mem(m));
                SCRATCH_XMM_A
            }
        };
        self.asm.ucomis(width, lhs_reg, rhs);
        Ok(())
    }

    /// The flag-to-value recipe for a float compare (ECMA-335 table III.4
    /// over `ucomis*`'s ZF/PF/CF): one or two `setcc` bytes, combined by
    /// `and` (both must hold — the ordered forms exclude PF=1) or `or`
    /// (either suffices — the unordered forms include PF=1).
    ///
    /// | op | flags when true | recipe |
    /// |----|-----------------|--------|
    /// | `Eq`   | equal, ordered: ZF·¬PF | `sete & setnp` |
    /// | `Ne`   | unordered or ≠: PF+¬ZF | `setne \| setp` |
    /// | `Gt`   | a>b: ¬CF·¬ZF (unord. sets CF) | `seta` |
    /// | `Ge`   | a≥b: ¬CF (unord. sets CF) | `setae` |
    /// | `Lt`   | a<b ordered: CF·¬PF | `setb & setnp` |
    /// | `Le`   | a≤b ordered: (CF+ZF)·¬PF | `setbe & setnp` |
    /// | `UGt`  | unord. or a>b | `seta \| setp` |
    /// | `UGe`  | unord. or a≥b | `setae \| setp` |
    /// | `ULt`  | unord. or a<b: CF | `setb` |
    /// | `ULe`  | unord. or a≤b: CF+ZF | `setbe` |
    fn emit_setcc_f(&mut self, op: BinaryOp, dst: Place) -> CompileResult<()> {
        let (first, second) = match op {
            BinaryOp::Eq => (CondCode::Eq, Some((false, CondCode::NotParity))),
            BinaryOp::Ne => (CondCode::Ne, Some((true, CondCode::Parity))),
            BinaryOp::Gt => (CondCode::UGt, None),
            BinaryOp::Ge => (CondCode::UGe, None),
            BinaryOp::Lt => (CondCode::ULt, Some((false, CondCode::NotParity))),
            BinaryOp::Le => (CondCode::ULe, Some((false, CondCode::NotParity))),
            BinaryOp::UGt => (CondCode::UGt, Some((true, CondCode::Parity))),
            BinaryOp::UGe => (CondCode::UGe, Some((true, CondCode::Parity))),
            BinaryOp::ULt => (CondCode::ULt, None),
            BinaryOp::ULe => (CondCode::ULe, None),
            _ => {
                return Err(CompileError::Internal(
                    "non-comparison operator in a float setcc",
                ));
            }
        };
        let (pa, moves) = self.vs.take_scratch(&[]);
        self.apply(moves)?;
        let ga = gpr_of(pa)?;
        // `mov` and `setcc` leave the flags alone: zero first, then set.
        self.asm.mov(Width::W32, Rm::Reg(ga), Rmi::Imm(0));
        self.asm.setcc(first, ga);
        if let Some((is_or, cc2)) = second {
            let (pb, moves) = self.vs.take_scratch(&[pa]);
            self.apply(moves)?;
            let gb = gpr_of(pb)?;
            self.asm.mov(Width::W32, Rm::Reg(gb), Rmi::Imm(0));
            self.asm.setcc(cc2, gb);
            if is_or {
                self.asm.or(Width::W32, Rm::Reg(ga), Rmi::Reg(gb));
            } else {
                self.asm.and(Width::W32, Rm::Reg(ga), Rmi::Reg(gb));
            }
        }
        match dst {
            Place::Val(t) => self.define_temp_reg(t.0, ga),
            Place::Reg(g) => {
                let moves = self.vs.clobber(g.phys());
                self.apply(moves)?;
                if g != ga {
                    self.asm.mov(Width::W32, Rm::Reg(g), Rmi::Reg(ga));
                }
                Ok(())
            }
        }
    }

    /// A conditional branch on float-compare flags (ECMA-335 table III.4).
    /// The unordered-or forms jump on parity first (`jp target`); the
    /// ordered equality/less forms skip over the jump on parity
    /// (`jp skip`); the rest are a single `jcc` (see [`float_jcc_tail`]).
    fn emit_jcc_f(&mut self, op: BinaryOp, target: Label) {
        match op {
            // Unordered counts as taken: jump on parity, else the flag cc.
            BinaryOp::Ne | BinaryOp::UGt | BinaryOp::UGe => {
                self.asm.jcc(CondCode::Parity, target);
                self.asm.jcc(float_jcc_tail(op), target);
            }
            // Ordered forms false on NaN: parity jumps over the branch.
            BinaryOp::Eq | BinaryOp::Lt | BinaryOp::Le => {
                let skip = self.synthetic_label();
                self.asm.jcc(CondCode::Parity, skip);
                self.asm.jcc(float_jcc_tail(op), target);
                self.asm.bind(skip);
            }
            // Single-cc forms: `Gt`/`Ge` exclude unordered via CF, the
            // `.un` less-forms include it via CF/ZF.
            BinaryOp::Gt | BinaryOp::Ge | BinaryOp::ULt | BinaryOp::ULe => {
                self.asm.jcc(float_jcc_tail(op), target);
            }
            _ => {}
        }
    }

    /// `cvtsi2ss`/`cvtsi2sd` (`conv.r4`/`conv.r8` from an integer): the
    /// source is a GPR/mem operand (immediates materialize first).
    fn emit_cvt_int_to_f(
        &mut self,
        width: FWidth,
        src_w64: bool,
        dst: XmmPlace,
        src: Src,
    ) -> CompileResult<()> {
        let src_w = if src_w64 { Width::W64 } else { Width::W32 };
        let rm = self.rm_of(src, src_w, &[])?;
        self.asm.cvtsi2s(width, SCRATCH_XMM_A, rm, src_w64);
        self.define_xmm(dst, width, SCRATCH_XMM_A)
    }

    /// `cvttss2si`/`cvttsd2si` (`conv.i4`/`i8`/`u4` from a float):
    /// truncation toward zero; out-of-range/NaN → the "integer indefinite"
    /// value, matching RyuJIT's cast lowering.
    fn emit_cvt_f_to_int(
        &mut self,
        src_width: FWidth,
        dst_w64: bool,
        dst: Place,
        src: XmmSrc,
    ) -> CompileResult<()> {
        let src = self.rmx_of(src, src_width, SCRATCH_XMM_A)?;
        let (p, moves) = self.vs.take_scratch(&[]);
        self.apply(moves)?;
        let g = gpr_of(p)?;
        self.asm.cvtts2si(src_width, g, src, dst_w64);
        match dst {
            Place::Val(t) => self.define_temp_reg(t.0, g),
            Place::Reg(g2) => {
                let moves = self.vs.clobber(g2.phys());
                self.apply(moves)?;
                if g2 != g {
                    self.asm.mov(Width::W64, Rm::Reg(g2), Rmi::Reg(g));
                }
                Ok(())
            }
        }
    }

    /// `cvtss2sd`/`cvtsd2ss` — the float↔double conversions.
    fn emit_cvt_f_to_f(&mut self, to: FWidth, dst: XmmPlace, src: XmmSrc) -> CompileResult<()> {
        let from = match to {
            FWidth::S => FWidth::D,
            FWidth::D => FWidth::S,
        };
        let src = self.rmx_of(src, from, SCRATCH_XMM_A)?;
        self.asm.cvts2s(to, SCRATCH_XMM_A, src);
        self.define_xmm(dst, to, SCRATCH_XMM_A)
    }

    /// Incoming argument registers → their frame slots, per the SysV
    /// classification of the method's own argument list (args include the
    /// receiver already, so the synthesized signature is `has_this: false`).
    /// Float arguments arrive in XMM registers and store with
    /// `movss`/`movsd`; integer-class arguments with a GPR `mov`.
    /// Struct arguments (step_10.9): register-passed eightbytes store with
    /// their exact descriptor sizes; stack-passed scalars load from the
    /// caller's outgoing area at `[rbp + 16 + offset]` (return address +
    /// saved rbp above the locals); stack-passed structs block-copy from
    /// there into the slot.
    fn spill_incoming_args(&mut self) -> CompileResult<()> {
        if self.num_args == 0 {
            return Ok(());
        }
        let sig = CallSig {
            ret: Type::Void,
            args: self.locals[..self.num_args].iter().map(|l| l.ty).collect(),
            has_this: false,
        };
        let abi = classify_call(&sig, self.layouts)?;
        // Incoming argument registers hold live values until their spill:
        // scratch allocation (struct eightbyte shifts, stack-struct
        // copies) must stay clear of the ones not yet spilled.
        self.fixed_dests = abi
            .args
            .iter()
            .flat_map(|loc| match loc {
                ArgLocation::Reg(p) => vec![*p],
                ArgLocation::StructRegs { regs, count, .. } => regs[..*count as usize]
                    .iter()
                    .copied()
                    .filter(|p| Gpr::from_phys(*p).is_some())
                    .collect(),
                ArgLocation::Stack { .. } => vec![],
            })
            .collect();
        for (i, loc) in abi.args.iter().enumerate() {
            let id = LocalId(i as u32);
            match *loc {
                ArgLocation::Reg(phys) => {
                    if let Some(g) = Gpr::from_phys(phys) {
                        let width = self.width_of(id)?;
                        self.asm.mov(width, Rm::Mem(self.own_slot(id)), Rmi::Reg(g));
                    } else if let Some(x) = Xmm::from_phys(phys) {
                        let fw = FWidth::of(self.ty_of(id)).ok_or(CompileError::Internal(
                            "XMM-classified argument without a float type",
                        ))?;
                        self.asm.mov_f_store(fw, self.own_slot(id), x);
                    } else {
                        return Err(CompileError::Internal("PhysReg outside both classes"));
                    }
                }
                ArgLocation::StructRegs {
                    regs,
                    count,
                    sizes,
                    offsets,
                } => {
                    for k in 0..count as usize {
                        let src =
                            match Gpr::from_phys(regs[k]) {
                                Some(g) => EbReg::Gpr(g),
                                None => EbReg::Xmm(Xmm::from_phys(regs[k]).ok_or(
                                    CompileError::Internal("PhysReg outside both classes"),
                                )?),
                            };
                        self.store_eightbyte_to_slot(id, u32::from(offsets[k]), sizes[k], src)?;
                    }
                }
                ArgLocation::Stack { offset } => {
                    let home = Mem::base_disp(regs::FRAME_POINTER, 16 + offset as i32);
                    match self.ty_of(id) {
                        Type::Float | Type::Double => {
                            let fw = FWidth::of(self.ty_of(id)).unwrap();
                            self.asm.mov_f_load(fw, SCRATCH_XMM_A, RmX::Mem(home));
                            self.asm.mov_f_store(fw, self.own_slot(id), SCRATCH_XMM_A);
                        }
                        Type::Struct(class) => {
                            let size = self.layouts[&class].size;
                            self.block_copy_mem_to_slot(id, offset, size)?;
                        }
                        _ => {
                            let width = self.width_of(id)?;
                            let exclude = &self.scratch_exclude(&[]);
                            let (p, moves) = self.vs.take_scratch(exclude);
                            self.apply(moves)?;
                            let g = gpr_of(p)?;
                            self.asm.mov(width, Rm::Reg(g), Rmi::Mem(home));
                            self.asm.mov(width, Rm::Mem(self.own_slot(id)), Rmi::Reg(g));
                        }
                    }
                }
            }
        }
        self.fixed_dests.clear();
        Ok(())
    }

    /// Zero the slots whose contents must be defined from the first
    /// safepoint on: IL locals (the init-locals flag does not reach LIR;
    /// zeroing unconditionally is always permitted), GC-typed temps
    /// (reported roots, so their slots must hold a valid value even before
    /// the temp's first definition), and struct temps embedding GC
    /// pointers (same reason — the cells are reported roots). Struct
    /// slots zero with inline decomposed stores (never a helper call in
    /// the prolog).
    fn zero_init_slots(&mut self) -> CompileResult<()> {
        for (i, local) in self.locals.iter().enumerate() {
            let is_il_local = i >= self.num_args && i < self.num_frame_fixed;
            let is_gc_temp = i >= self.num_frame_fixed
                && match local.ty {
                    Type::Ref | Type::ByRef => true,
                    Type::Struct(class) => !self.layouts[&class].gc_cells.is_empty(),
                    _ => false,
                };
            if !(is_il_local || is_gc_temp) {
                continue;
            }
            // The store width is the slot width: 4 bytes for Int32/Float,
            // 8 for the 64-bit types (zero bits are zero in every
            // interpretation, so floats zero with a GPR move).
            let width = match local.ty {
                Type::Int32 | Type::Float => Width::W32,
                Type::Int64 | Type::NativeInt | Type::Ref | Type::ByRef | Type::Double => {
                    Width::W64
                }
                Type::Struct(class) => {
                    let size = self.layouts[&class].size;
                    self.block_zero_slot(LocalId(i as u32), size)?;
                    continue;
                }
                // Void never types a local.
                Type::Void => continue,
            };
            self.asm.mov(
                width,
                Rm::Mem(self.slot_mem(self.layout.slots[i])),
                Rmi::Imm(0),
            );
        }
        Ok(())
    }

    // ---- step_10.9: struct ABI moves and block operations ----

    /// The memory of the byte at struct-relative `offset` within a
    /// local's frame slot (the slot's bytes are
    /// `[rbp - slot, rbp - slot + size)`).
    fn slot_mem_at(&self, id: LocalId, offset: u32) -> Mem {
        Mem::base_disp(
            regs::FRAME_POINTER,
            -((self.vs.slot_of(id) - offset) as i32),
        )
    }

    /// A block operand's address materialized into a scratch GPR: a
    /// byref value reads from wherever it currently lives (register
    /// reused directly when allowed), a frame-slot address is a `lea`.
    fn block_addr_gpr(&mut self, addr: BlockAddr, exclude: &[PhysReg]) -> CompileResult<Gpr> {
        if let BlockAddr::Val(v) = addr {
            if let ReadSrc::Reg(p) = self.vs.read(v.0) {
                if !exclude.contains(&p) && !self.fixed_dests.contains(&p) {
                    return gpr_of(p);
                }
            }
        }
        let exclude = &self.scratch_exclude(exclude);
        let (p, moves) = self.vs.take_scratch(exclude);
        self.apply(moves)?;
        let g = gpr_of(p)?;
        match addr {
            BlockAddr::FrameSlot(l) => {
                self.asm
                    .lea(g, self.slot_mem(self.layout.slots[l.0 as usize]));
            }
            BlockAddr::Val(v) => match self.vs.read(v.0) {
                ReadSrc::Reg(src) => {
                    let src = gpr_of(src)?;
                    if src != g {
                        self.asm.mov(Width::W64, Rm::Reg(g), Rmi::Reg(src));
                    }
                }
                ReadSrc::Slot(off) => {
                    self.asm
                        .mov(Width::W64, Rm::Reg(g), Rmi::Mem(self.slot_mem(off)));
                }
                ReadSrc::Imm(i) => {
                    self.asm.mov(Width::W64, Rm::Reg(g), Rmi::Imm(i));
                }
            },
        }
        Ok(g)
    }

    /// `dst := mem[size]` — an exact-width block-copy load (1- and
    /// 2-byte chunks zero-extend).
    fn load_chunk(&mut self, mem: Mem, size: u8, dst: Gpr) {
        match size {
            8 => self.asm.mov(Width::W64, Rm::Reg(dst), Rmi::Mem(mem)),
            4 => self.asm.mov(Width::W32, Rm::Reg(dst), Rmi::Mem(mem)),
            1 | 2 => self.asm.movzx_load(size, dst, mem),
            _ => unreachable!("chunk sizes are 8/4/2/1"),
        }
    }

    /// `mem[size] := src` — an exact-width block-copy store.
    fn store_chunk(&mut self, mem: Mem, size: u8, src: Gpr) {
        match size {
            8 => self.asm.mov(Width::W64, Rm::Mem(mem), Rmi::Reg(src)),
            4 => self.asm.mov(Width::W32, Rm::Mem(mem), Rmi::Reg(src)),
            1 | 2 => self.asm.mov_store_narrow(size, mem, src),
            _ => unreachable!("chunk sizes are 8/4/2/1"),
        }
    }

    /// One struct eightbyte from an ABI register into a frame slot at
    /// `offset`, respecting the descriptor's exact byte size (the callee
    /// spill of a register-passed argument; a register-passed call
    /// result landing in its destination slot). GPR eightbytes decompose
    /// into 8/4/2/1 chunks (later chunks shift the register through a
    /// scratch); SSE eightbyte sizes are always 4 or 8 (`movss`/`movsd`
    /// are exact).
    fn store_eightbyte_to_slot(
        &mut self,
        id: LocalId,
        offset: u32,
        size: u8,
        src: EbReg,
    ) -> CompileResult<()> {
        match src {
            EbReg::Xmm(x) => {
                let fw = if size <= 4 { FWidth::S } else { FWidth::D };
                self.asm.mov_f_store(fw, self.slot_mem_at(id, offset), x);
            }
            EbReg::Gpr(g) => {
                let mut done = 0u32;
                for (off, chunk) in chunk_plan(u32::from(size)) {
                    let mem = self.slot_mem_at(id, offset + off);
                    if off == 0 {
                        self.store_chunk(mem, chunk, g);
                    } else {
                        let exclude = &self.scratch_exclude(&[g.phys()]);
                        let (p, moves) = self.vs.take_scratch(exclude);
                        self.apply(moves)?;
                        let t = gpr_of(p)?;
                        self.asm.mov(Width::W64, Rm::Reg(t), Rmi::Reg(g));
                        self.asm.shift_imm(
                            ShiftOp::Shr,
                            Width::W64,
                            Rm::Reg(t),
                            i64::from(done * 8),
                        );
                        self.store_chunk(mem, chunk, t);
                    }
                    done += u32::from(chunk);
                }
            }
        }
        Ok(())
    }

    /// One struct eightbyte from memory into an ABI register (the caller
    /// side of a register-passed struct argument; a register-passed
    /// struct return). Exact-size loads decompose into 8/4/2/1 chunks,
    /// later chunks composed with shift+or — never a wider read, since a
    /// heap-resident struct can sit at the end of a committed page.
    fn emit_load_eightbyte(
        &mut self,
        addr: BlockAddr,
        disp: u32,
        size: u8,
        dst: EbReg,
    ) -> CompileResult<()> {
        match dst {
            EbReg::Xmm(x) => {
                let fw = if size <= 4 { FWidth::S } else { FWidth::D };
                let g = self.block_addr_gpr(addr, &[])?;
                self.asm
                    .mov_f_load(fw, x, RmX::Mem(Mem::base_disp(g, disp as i32)));
            }
            EbReg::Gpr(dst_g) => {
                let moves = self.vs.clobber(dst_g.phys());
                self.apply(moves)?;
                let g = self.block_addr_gpr(addr, &[dst_g.phys()])?;
                let mut done = 0u32;
                for (off, chunk) in chunk_plan(u32::from(size)) {
                    let mem = Mem::base_disp(g, (disp + off) as i32);
                    if off == 0 {
                        self.load_chunk(mem, chunk, dst_g);
                    } else {
                        let exclude = &self.scratch_exclude(&[g.phys(), dst_g.phys()]);
                        let (p, moves) = self.vs.take_scratch(exclude);
                        self.apply(moves)?;
                        let t = gpr_of(p)?;
                        self.load_chunk(mem, chunk, t);
                        self.asm.shift_imm(
                            ShiftOp::Shl,
                            Width::W64,
                            Rm::Reg(t),
                            i64::from(done * 8),
                        );
                        self.asm.or(Width::W64, Rm::Reg(dst_g), Rmi::Reg(t));
                    }
                    done += u32::from(chunk);
                }
            }
        }
        Ok(())
    }

    /// An incoming stack-passed struct argument: `size` bytes from the
    /// caller's outgoing area (`[rbp + 16 + offset]`) into the local's
    /// slot, chunk by chunk through one scratch.
    fn block_copy_mem_to_slot(&mut self, id: LocalId, offset: u32, size: u32) -> CompileResult<()> {
        let exclude = &self.scratch_exclude(&[]);
        let (p, moves) = self.vs.take_scratch(exclude);
        self.apply(moves)?;
        let t = gpr_of(p)?;
        for (done, chunk) in chunk_plan(size) {
            let from = Mem::base_disp(regs::FRAME_POINTER, (16 + offset + done) as i32);
            self.load_chunk(from, chunk, t);
            self.store_chunk(self.slot_mem_at(id, done), chunk, t);
        }
        Ok(())
    }

    /// Zero a struct slot in the prolog (IL locals zero-init; GC-cell
    /// struct temps). Always inline: no call in the prolog region.
    fn block_zero_slot(&mut self, id: LocalId, size: u32) -> CompileResult<()> {
        let (p, moves) = self.vs.take_scratch(&[]);
        self.apply(moves)?;
        let z = gpr_of(p)?;
        self.asm.xor(Width::W32, Rm::Reg(z), Rmi::Reg(z));
        for (done, chunk) in chunk_plan(size) {
            self.store_chunk(self.slot_mem_at(id, done), chunk, z);
        }
        Ok(())
    }

    /// A block copy between two memory addresses (`Inst::BlockCopy`):
    /// inline unrolled chunks at or below [`BLOCK_INLINE_MAX`] bytes, the
    /// EE's `CORINFO_HELP_MEMCPY` above.
    fn emit_block_copy(
        &mut self,
        dst: BlockAddr,
        dst_disp: u32,
        src: BlockAddr,
        size: u32,
    ) -> CompileResult<()> {
        if size > BLOCK_INLINE_MAX {
            return self.emit_block_helper(CorInfoHelpFunc::MEMCPY, dst, dst_disp, Some(src), size);
        }
        let dst_g = self.block_addr_gpr(dst, &[])?;
        let src_g = self.block_addr_gpr(src, &[dst_g.phys()])?;
        for (done, chunk) in chunk_plan(size) {
            let exclude = &self.scratch_exclude(&[dst_g.phys(), src_g.phys()]);
            let (p, moves) = self.vs.take_scratch(exclude);
            self.apply(moves)?;
            let t = gpr_of(p)?;
            self.load_chunk(Mem::base_disp(src_g, done as i32), chunk, t);
            self.store_chunk(Mem::base_disp(dst_g, (dst_disp + done) as i32), chunk, t);
        }
        Ok(())
    }

    /// A block zero (`Inst::BlockZero`, `initobj`): inline at or below
    /// [`BLOCK_INLINE_MAX`], `CORINFO_HELP_MEMSET` above. Zeroing needs
    /// no write barrier even on the heap (storing null creates no
    /// old→young edge).
    fn emit_block_zero(&mut self, dst: BlockAddr, dst_disp: u32, size: u32) -> CompileResult<()> {
        if size > BLOCK_INLINE_MAX {
            return self.emit_block_helper(CorInfoHelpFunc::MEMSET, dst, dst_disp, None, size);
        }
        let dst_g = self.block_addr_gpr(dst, &[])?;
        let exclude = &self.scratch_exclude(&[dst_g.phys()]);
        let (p, moves) = self.vs.take_scratch(exclude);
        self.apply(moves)?;
        let z = gpr_of(p)?;
        self.asm.xor(Width::W32, Rm::Reg(z), Rmi::Reg(z));
        for (done, chunk) in chunk_plan(size) {
            self.store_chunk(Mem::base_disp(dst_g, (dst_disp + done) as i32), chunk, z);
        }
        Ok(())
    }

    /// The helper-call form of a large block operation: `MEMCPY(dst,
    /// src, size)` / `MEMSET(dst, 0, size)` through the EE's
    /// `getHelperFtn` — the same emission path as any helper call (the
    /// call site is a recorded GC safepoint; the frame-resident roots
    /// stay authoritative).
    fn emit_block_helper(
        &mut self,
        id: CorInfoHelpFunc,
        dst: BlockAddr,
        dst_disp: u32,
        src: Option<BlockAddr>,
        size: u32,
    ) -> CompileResult<()> {
        for g in [Gpr::Rdi, Gpr::Rsi, Gpr::Rdx] {
            let moves = self.vs.clobber(g.phys());
            self.apply(moves)?;
        }
        self.block_addr_into_reg(Gpr::Rdi, dst, dst_disp)?;
        match src {
            Some(src) => self.block_addr_into_reg(Gpr::Rsi, src, 0)?,
            None => self.asm.mov(Width::W32, Rm::Reg(Gpr::Rsi), Rmi::Imm(0)),
        }
        self.asm
            .mov(Width::W64, Rm::Reg(Gpr::Rdx), Rmi::Imm(i64::from(size)));
        let lookup = self.ee.get_helper_ftn(id).entrypoint;
        let addr = const_lookup_addr(&lookup);
        let slot = const_lookup_slot(&lookup);
        self.emit_call_lookup(None, addr, slot)
    }

    /// A block address into a specific ABI register (the block-helper
    /// argument setup): value reads and frame-slot `lea`s, no scratch.
    fn block_addr_into_reg(&mut self, reg: Gpr, addr: BlockAddr, disp: u32) -> CompileResult<()> {
        match addr {
            BlockAddr::FrameSlot(l) => {
                self.asm
                    .lea(reg, self.slot_mem(self.layout.slots[l.0 as usize]));
            }
            BlockAddr::Val(v) => match self.vs.read(v.0) {
                ReadSrc::Reg(p) => {
                    let g = gpr_of(p)?;
                    if g != reg {
                        self.asm.mov(Width::W64, Rm::Reg(reg), Rmi::Reg(g));
                    }
                }
                ReadSrc::Slot(off) => {
                    self.asm
                        .mov(Width::W64, Rm::Reg(reg), Rmi::Mem(self.slot_mem(off)));
                }
                ReadSrc::Imm(i) => {
                    self.asm.mov(Width::W64, Rm::Reg(reg), Rmi::Imm(i));
                }
            },
        }
        if disp != 0 {
            self.asm
                .add(Width::W64, Rm::Reg(reg), Rmi::Imm(i64::from(disp)));
        }
        Ok(())
    }

    /// `mov [rsp + offset], src` — an outgoing scalar stack argument.
    /// A slot-resident source reloads through a scratch (no mem,mem
    /// form); a too-wide constant materializes (the wide-imm rule).
    fn emit_store_stack_arg(&mut self, width: Width, offset: u32, src: Src) -> CompileResult<()> {
        let mem = Mem::base_disp(regs::STACK_POINTER, offset as i32);
        match self.wide_imm(width, src, &[])? {
            src @ (Rmi::Reg(_) | Rmi::Imm(_)) => self.asm.mov(width, Rm::Mem(mem), src),
            Rmi::Mem(m) => {
                let exclude = &self.scratch_exclude(&[]);
                let (p, moves) = self.vs.take_scratch(exclude);
                self.apply(moves)?;
                let g = gpr_of(p)?;
                self.asm.mov(width, Rm::Reg(g), Rmi::Mem(m));
                self.asm.mov(width, Rm::Mem(mem), Rmi::Reg(g));
            }
        }
        Ok(())
    }

    /// The float form of [`Emitter::emit_store_stack_arg`].
    fn emit_store_stack_arg_f(
        &mut self,
        width: FWidth,
        offset: u32,
        src: XmmSrc,
    ) -> CompileResult<()> {
        let src = self.rmx_of(src, width, SCRATCH_XMM_A)?;
        match src {
            RmX::Reg(x) => {
                self.asm
                    .mov_f_store(width, Mem::base_disp(regs::STACK_POINTER, offset as i32), x)
            }
            RmX::Mem(m) => {
                self.asm.mov_f_load(width, SCRATCH_XMM_A, RmX::Mem(m));
                self.asm.mov_f_store(
                    width,
                    Mem::base_disp(regs::STACK_POINTER, offset as i32),
                    SCRATCH_XMM_A,
                );
            }
        }
        Ok(())
    }

    /// An outgoing stack-passed struct argument: `size` bytes from
    /// `[addr]` to `[rsp + offset]`, always inline (no helper call in
    /// the middle of argument setup).
    fn emit_copy_stack_arg(
        &mut self,
        addr: BlockAddr,
        offset: u32,
        size: u32,
    ) -> CompileResult<()> {
        let src_g = self.block_addr_gpr(addr, &[])?;
        for (done, chunk) in chunk_plan(size) {
            let exclude = &self.scratch_exclude(&[src_g.phys()]);
            let (p, moves) = self.vs.take_scratch(exclude);
            self.apply(moves)?;
            let t = gpr_of(p)?;
            self.load_chunk(Mem::base_disp(src_g, done as i32), chunk, t);
            self.store_chunk(
                Mem::base_disp(regs::STACK_POINTER, (offset + done) as i32),
                chunk,
                t,
            );
        }
        Ok(())
    }

    fn emit_inst(&mut self, inst: &Inst) -> CompileResult<()> {
        match *inst {
            Inst::Mov { width, dst, src } => self.emit_mov(width, dst, src),
            Inst::Lea { dst, addr } => self.emit_lea(dst, addr),
            Inst::ConstF { width, dst, bits } => self.emit_const_f(width, dst, bits),
            Inst::MovF { width, dst, src } => self.emit_mov_f(width, dst, src),
            Inst::ArithF {
                op,
                width,
                dst,
                lhs,
                rhs,
            } => self.emit_arith_f(op, width, dst, lhs, rhs),
            Inst::NegF { width, dst, src } => self.emit_neg_f(width, dst, src),
            Inst::CmpF { width, lhs, rhs } => self.emit_cmp_f(width, lhs, rhs),
            Inst::SetccF { op, dst } => self.emit_setcc_f(op, dst),
            Inst::JccF { op, target } => {
                self.emit_jcc_f(op, target);
                Ok(())
            }
            Inst::CvtIntToF {
                width,
                src_w64,
                dst,
                src,
            } => self.emit_cvt_int_to_f(width, src_w64, dst, src),
            Inst::CvtFToInt {
                src_width,
                dst_w64,
                dst,
                src,
            } => self.emit_cvt_f_to_int(src_width, dst_w64, dst, src),
            Inst::CvtFToF { to, dst, src } => self.emit_cvt_f_to_f(to, dst, src),
            Inst::Arith {
                op,
                width,
                dst,
                lhs,
                rhs,
            } => self.emit_arith(op, width, dst, lhs, rhs),
            Inst::Cdq { width } => {
                // `cdq` writes rdx; a temp living there spills first.
                let moves = self.vs.clobber(Gpr::Rdx.phys());
                self.apply(moves)?;
                self.asm.cdq(width);
                Ok(())
            }
            Inst::Idiv { width, divisor } => {
                // `idiv` reads and writes rdx:rax; both spill first, and
                // the divisor (materialized if immediate) must avoid them.
                let moves = self.vs.clobber(Gpr::Rax.phys());
                self.apply(moves)?;
                let moves = self.vs.clobber(Gpr::Rdx.phys());
                self.apply(moves)?;
                let rm = self.rm_of(divisor, width, &[Gpr::Rax.phys(), Gpr::Rdx.phys()])?;
                self.asm.idiv(width, rm);
                Ok(())
            }
            Inst::Div { width, divisor } => {
                // Same fixed-register discipline as `idiv` (unsigned form).
                let moves = self.vs.clobber(Gpr::Rax.phys());
                self.apply(moves)?;
                let moves = self.vs.clobber(Gpr::Rdx.phys());
                self.apply(moves)?;
                let rm = self.rm_of(divisor, width, &[Gpr::Rax.phys(), Gpr::Rdx.phys()])?;
                self.asm.div(width, rm);
                Ok(())
            }
            Inst::Shift {
                op,
                width,
                dst,
                lhs,
                rhs,
            } => self.emit_shift(op, width, dst, lhs, rhs),
            Inst::Unary {
                op,
                width,
                dst,
                src,
            } => self.emit_unary(op, width, dst, src),
            Inst::Setcc { cc, dst } => self.emit_setcc(cc, dst),
            Inst::MovExt { dst, src, signed } => self.emit_movext(dst, src, signed),
            Inst::Cmp { width, lhs, rhs } => self.emit_cmp(width, lhs, rhs),
            Inst::LoadMem {
                width,
                dst,
                addr,
                disp,
            } => self.emit_load_mem(width, dst, addr, disp),
            Inst::StoreMem {
                width,
                addr,
                disp,
                src,
            } => self.emit_store_mem(width, addr, disp, src),
            Inst::NullCheck { addr } => self.emit_null_check(addr),
            Inst::LoadEightbyte {
                addr,
                disp,
                size,
                dst,
            } => self.emit_load_eightbyte(addr, disp, size, dst),
            Inst::StoreEightbyte {
                local,
                offset,
                size,
                src,
            } => self.store_eightbyte_to_slot(local, offset, size, src),
            Inst::StoreStackArg { width, offset, src } => {
                self.emit_store_stack_arg(width, offset, src)
            }
            Inst::StoreStackArgF { width, offset, src } => {
                self.emit_store_stack_arg_f(width, offset, src)
            }
            Inst::CopyStackArg { addr, offset, size } => {
                self.emit_copy_stack_arg(addr, offset, size)
            }
            Inst::BlockCopy {
                dst,
                dst_disp,
                src,
                size,
            } => self.emit_block_copy(dst, dst_disp, src, size),
            Inst::BlockZero {
                dst,
                dst_disp,
                size,
            } => self.emit_block_zero(dst, dst_disp, size),
            Inst::Jcc { cc, target } => {
                self.asm.jcc(cc, target);
                Ok(())
            }
            Inst::Jmp { target } => {
                self.asm.jmp(target);
                Ok(())
            }
            Inst::CallDirect { method } => self.emit_call(method),
            Inst::CallHelper { id } => self.emit_helper_call(id),
            Inst::Push { reg } => {
                self.asm.push(reg);
                Ok(())
            }
            Inst::AllocFrame => {
                self.asm.sub(
                    Width::W64,
                    Rm::Reg(regs::STACK_POINTER),
                    Rmi::Imm(i64::from(self.layout.frame_size)),
                );
                Ok(())
            }
            Inst::Leave => {
                self.asm.leave();
                Ok(())
            }
            Inst::Ret => {
                self.asm.ret();
                Ok(())
            }
        }
    }

    fn emit_mov(&mut self, width: Width, dst: Place, src: Src) -> CompileResult<()> {
        match dst {
            // A fixed-register destination (arg setup, `idiv`, return):
            // spill whatever temp the register holds, then move.
            Place::Reg(g) => {
                let moves = self.vs.clobber(g.phys());
                self.apply(moves)?;
                let rmi = self.rmi_of(src)?;
                if rmi != Rmi::Reg(g) {
                    self.asm.mov(width, Rm::Reg(g), rmi);
                }
                Ok(())
            }
            Place::Val(v) if (v.0 .0 as usize) < self.num_frame_fixed => {
                self.emit_store_to_local(v.0, width, src)
            }
            Place::Val(v) => self.emit_define_temp(v.0, width, src),
        }
    }

    /// `stloc`: the IL local's slot must be written (locals are
    /// frame-resident, not tag-tracked); temps aliasing the slot
    /// ([`Loc::Local`]) materialize first so they keep the old value.
    fn emit_store_to_local(&mut self, id: LocalId, width: Width, src: Src) -> CompileResult<()> {
        let moves = self.vs.before_local_write(id);
        self.apply(moves)?;
        let slot = self.own_slot(id);
        match self.wide_imm(width, src, &[])? {
            Rmi::Reg(g) => self.asm.mov(width, Rm::Mem(slot), Rmi::Reg(g)),
            Rmi::Imm(i) => self.asm.mov(width, Rm::Mem(slot), Rmi::Imm(i)),
            Rmi::Mem(m) => {
                if m != slot {
                    let (p, moves) = self.vs.take_scratch(&[]);
                    self.apply(moves)?;
                    let g = gpr_of(p)?;
                    self.asm.mov(width, Rm::Reg(g), Rmi::Mem(m));
                    self.asm.mov(width, Rm::Mem(slot), Rmi::Reg(g));
                }
            }
        }
        Ok(())
    }

    /// A temp definition: tag the result instead of materializing it —
    /// constants stay constants, aliases stay aliases, register values
    /// keep their register. GC-typed temps are the exception: they
    /// materialize into their own slot at once (the root invariant).
    fn emit_define_temp(&mut self, id: LocalId, width: Width, src: Src) -> CompileResult<()> {
        if matches!(self.ty_of(id), Type::Ref | Type::ByRef) {
            match self.rmi_of(src)? {
                Rmi::Reg(g) => self.asm.mov(width, Rm::Mem(self.own_slot(id)), Rmi::Reg(g)),
                Rmi::Imm(i) => self.asm.mov(width, Rm::Mem(self.own_slot(id)), Rmi::Imm(i)),
                Rmi::Mem(m) => {
                    if m != self.own_slot(id) {
                        let (p, moves) = self.vs.take_scratch(&[]);
                        self.apply(moves)?;
                        let g = gpr_of(p)?;
                        self.asm.mov(width, Rm::Reg(g), Rmi::Mem(m));
                        self.asm.mov(width, Rm::Mem(self.own_slot(id)), Rmi::Reg(g));
                    }
                }
            }
            let slot = self.vs.slot_of(id);
            return self.define_loc(id, Loc::Mem(slot));
        }
        match src {
            Src::Imm(i) => self.define_loc(id, Loc::Const(i)),
            // A fixed-register source (call/`idiv` result): adopt the
            // register — zero instructions.
            Src::Reg(g) => self.define_temp_reg(id, g),
            Src::Val(v) => {
                // Copy of an IL local/arg: alias its slot (invalidated on
                // writes to it). Zero instructions.
                if (v.0 .0 as usize) < self.num_frame_fixed {
                    return self.define_loc(id, Loc::Local(v.0));
                }
                match self.vs.read(v.0) {
                    // Copy of a spilled temp: alias the (immutable) slot.
                    ReadSrc::Slot(off) => self.define_loc(id, Loc::Mem(off)),
                    ReadSrc::Imm(i) => self.define_loc(id, Loc::Const(i)),
                    ReadSrc::Reg(p) => {
                        // Allocate first (excluding the source register, so
                        // the eviction can't spill it), then copy.
                        let (d, moves) = self.vs.take_scratch(&[p]);
                        self.apply(moves)?;
                        let dg = gpr_of(d)?;
                        self.asm.mov(width, Rm::Reg(dg), Rmi::Reg(gpr_of(p)?));
                        self.define_temp_reg(id, dg)
                    }
                }
            }
        }
    }

    fn emit_lea(&mut self, dst: Place, addr: crate::inst::Amode) -> CompileResult<()> {
        let crate::inst::Amode::FrameSlot(l) = addr;
        let mem = self.slot_mem(self.layout.slots[l.0 as usize]);
        match dst {
            // A `lea` into an IL local/arg slot follows the store
            // discipline: aliases of the slot materialize first.
            Place::Val(t) if (t.0 .0 as usize) < self.num_frame_fixed => {
                let (p, moves) = self.vs.take_scratch(&[]);
                self.apply(moves)?;
                let g = gpr_of(p)?;
                self.asm.lea(g, mem);
                self.emit_store_to_local(t.0, Width::W64, Src::Reg(g))
            }
            Place::Val(t) => {
                let (p, moves) = self.vs.take_scratch(&[]);
                self.apply(moves)?;
                let g = gpr_of(p)?;
                self.asm.lea(g, mem);
                self.define_temp_reg(t.0, g)
            }
            Place::Reg(g) => {
                let moves = self.vs.clobber(g.phys());
                self.apply(moves)?;
                self.asm.lea(g, mem);
                Ok(())
            }
        }
    }

    /// A shift: the value moves into a scratch register and the shift
    /// applies in place. A constant count uses the imm8 form; a variable
    /// count must be in `cl` — `rcx` is clobbered (its temp spills), the
    /// count moves in with a 32-bit `mov` (only `cl` is ever read), and
    /// the `D3` form applies. The destination scratch excludes `rcx` so
    /// the count move can't evict the value. Hardware masks the count to
    /// 5/6 bits by operand width — ECMA-335's masking rule exactly.
    fn emit_shift(
        &mut self,
        op: ShiftOp,
        width: Width,
        dst: Place,
        lhs: Src,
        rhs: Src,
    ) -> CompileResult<()> {
        let Place::Val(t) = dst else {
            return Err(CompileError::Internal(
                "shift destination is always a value",
            ));
        };
        let (d, moves) = self.vs.take_scratch(&[Gpr::Rcx.phys()]);
        self.apply(moves)?;
        let dg = gpr_of(d)?;
        match self.rmi_of(lhs)? {
            Rmi::Reg(g) => self.asm.mov(width, Rm::Reg(dg), Rmi::Reg(g)),
            Rmi::Mem(m) => self.asm.mov(width, Rm::Reg(dg), Rmi::Mem(m)),
            Rmi::Imm(i) => self.asm.mov(width, Rm::Reg(dg), Rmi::Imm(i)),
        }
        match rhs {
            Src::Imm(count) => self.asm.shift_imm(op, width, Rm::Reg(dg), count),
            rhs => {
                let moves = self.vs.clobber(Gpr::Rcx.phys());
                self.apply(moves)?;
                let count = self.rmi_of(rhs)?;
                self.asm.mov(Width::W32, Rm::Reg(Gpr::Rcx), count);
                self.asm.shift_cl(op, width, Rm::Reg(dg));
            }
        }
        self.define_temp_reg(t.0, dg)
    }

    /// `neg`/`not`: the source moves into a scratch register and the
    /// operation applies in place.
    fn emit_unary(&mut self, op: UnaryOp, width: Width, dst: Place, src: Src) -> CompileResult<()> {
        let Place::Val(t) = dst else {
            return Err(CompileError::Internal(
                "unary destination is always a value",
            ));
        };
        let (d, moves) = self.vs.take_scratch(&[]);
        self.apply(moves)?;
        let dg = gpr_of(d)?;
        let rmi = self.rmi_of(src)?;
        self.asm.mov(width, Rm::Reg(dg), rmi);
        match op {
            UnaryOp::Neg => self.asm.neg(width, Rm::Reg(dg)),
            UnaryOp::Not => self.asm.not(width, Rm::Reg(dg)),
        }
        self.define_temp_reg(t.0, dg)
    }

    /// `setcc` materializes a compare's flags into an Int32 0/1. The flags
    /// come from the immediately preceding `Cmp` (the lowering rules emit
    /// the pair adjacently); every move emitted here — scratch spills, the
    /// zeroing — is flag-preserving.
    fn emit_setcc(&mut self, cc: crate::inst::CondCode, dst: Place) -> CompileResult<()> {
        match dst {
            Place::Val(t) => {
                let (p, moves) = self.vs.take_scratch(&[]);
                self.apply(moves)?;
                let g = gpr_of(p)?;
                // `mov` doesn't touch the flags: zero, then set the low byte.
                self.asm.mov(Width::W32, Rm::Reg(g), Rmi::Imm(0));
                self.asm.setcc(cc, g);
                self.define_temp_reg(t.0, g)
            }
            Place::Reg(g) => {
                let moves = self.vs.clobber(g.phys());
                self.apply(moves)?;
                self.asm.mov(Width::W32, Rm::Reg(g), Rmi::Imm(0));
                self.asm.setcc(cc, g);
                Ok(())
            }
        }
    }

    /// A 32→64 extension (`conv.i8`/`conv.u8`): `movsxd` when signed, a
    /// 32-bit `mov` (which zeroes the upper half) when unsigned. Always
    /// materializes into a register — a widening copy must never alias
    /// the source's slot (a 64-bit read of a 32-bit slot would pick up
    /// the neighboring 4 bytes).
    fn emit_movext(&mut self, dst: Place, src: Src, signed: bool) -> CompileResult<()> {
        match dst {
            Place::Val(t) => {
                let (p, moves) = self.vs.take_scratch(&[]);
                self.apply(moves)?;
                let g = gpr_of(p)?;
                self.ext_into(g, src, signed)?;
                self.define_temp_reg(t.0, g)
            }
            Place::Reg(g) => {
                let moves = self.vs.clobber(g.phys());
                self.apply(moves)?;
                self.ext_into(g, src, signed)
            }
        }
    }

    /// The extension proper: `movsxd g, src32` or `mov g32, src32`.
    fn ext_into(&mut self, g: Gpr, src: Src, signed: bool) -> CompileResult<()> {
        if signed {
            let rm = self.rm_of(src, Width::W32, &[g.phys()])?;
            self.asm.movsxd(g, rm);
        } else {
            let rmi = self.rmi_of(src)?;
            self.asm.mov(Width::W32, Rm::Reg(g), rmi);
        }
        Ok(())
    }

    fn emit_arith(
        &mut self,
        op: ArithOp,
        width: Width,
        dst: Place,
        lhs: Src,
        rhs: Src,
    ) -> CompileResult<()> {
        let Place::Val(t) = dst else {
            return Err(CompileError::Internal(
                "arithmetic destination is always a value",
            ));
        };
        let (d, moves) = self.vs.take_scratch(&[]);
        self.apply(moves)?;
        let dg = gpr_of(d)?;
        // `imul` by a constant: the three-operand form folds the move —
        // but only while the constant fits the imm32 field.
        if op == ArithOp::Imul {
            if let Src::Imm(i) = rhs {
                if i32::try_from(i).is_ok() {
                    let lhs_rm = self.rm_of(lhs, width, &[d])?;
                    self.asm.imul_imm(width, dg, lhs_rm, i);
                    return self.define_temp_reg(t.0, dg);
                }
            }
        }
        // Destructive two-operand form: lhs into the scratch, then op.
        match self.rmi_of(lhs)? {
            Rmi::Reg(g) => self.asm.mov(width, Rm::Reg(dg), Rmi::Reg(g)),
            Rmi::Mem(m) => self.asm.mov(width, Rm::Reg(dg), Rmi::Mem(m)),
            Rmi::Imm(i) => self.asm.mov(width, Rm::Reg(dg), Rmi::Imm(i)),
        }
        match op {
            ArithOp::Add => {
                let rhs = self.wide_imm(width, rhs, &[d])?;
                self.asm.add(width, Rm::Reg(dg), rhs);
            }
            ArithOp::Sub => {
                let rhs = self.wide_imm(width, rhs, &[d])?;
                self.asm.sub(width, Rm::Reg(dg), rhs);
            }
            ArithOp::And => {
                let rhs = self.wide_imm(width, rhs, &[d])?;
                self.asm.and(width, Rm::Reg(dg), rhs);
            }
            ArithOp::Or => {
                let rhs = self.wide_imm(width, rhs, &[d])?;
                self.asm.or(width, Rm::Reg(dg), rhs);
            }
            ArithOp::Xor => {
                let rhs = self.wide_imm(width, rhs, &[d])?;
                self.asm.xor(width, Rm::Reg(dg), rhs);
            }
            ArithOp::Imul => {
                let rhs = self.rm_of(rhs, width, &[d])?;
                self.asm.imul(width, dg, rhs);
            }
        }
        self.define_temp_reg(t.0, dg)
    }

    fn emit_cmp(&mut self, width: Width, lhs: Src, rhs: Src) -> CompileResult<()> {
        // The rhs resolves first: a too-wide immediate materializes into a
        // scratch register, and that allocation must not evict a register
        // an already-resolved lhs sits in.
        let rhs_rmi = self.wide_imm(width, rhs, &[])?;
        let lhs_rmi = self.rmi_of(lhs)?;
        // x64 `cmp` needs a non-immediate lhs and no mem,mem pair;
        // materialize the lhs into a scratch register in either case —
        // excluding any register an operand already sits in, so the
        // allocation's eviction can't spill it.
        let needs_reg = matches!(lhs_rmi, Rmi::Imm(_))
            || matches!((lhs_rmi, rhs_rmi), (Rmi::Mem(_), Rmi::Mem(_)));
        if needs_reg {
            let mut exclude = Vec::new();
            for rmi in [lhs_rmi, rhs_rmi] {
                if let Rmi::Reg(g) = rmi {
                    exclude.push(g.phys());
                }
            }
            let (p, moves) = self.vs.take_scratch(&exclude);
            self.apply(moves)?;
            let g = gpr_of(p)?;
            self.asm.mov(width, Rm::Reg(g), lhs_rmi);
            self.asm.cmp(width, Rm::Reg(g), rhs_rmi);
        } else {
            let lhs_rm = match lhs_rmi {
                Rmi::Reg(g) => Rm::Reg(g),
                Rmi::Mem(m) => Rm::Mem(m),
                Rmi::Imm(_) => unreachable!("immediate lhs handled above"),
            };
            self.asm.cmp(width, lhs_rm, rhs_rmi);
        }
        Ok(())
    }

    /// A computed address value materialized into a scratch GPR: a
    /// frame-resident reference (the common case — GC refs never stay in
    /// registers) reloads from its slot; a constant goes through the
    /// wide-imm rule (a null or frozen-ref constant is just an address).
    fn addr_into(&mut self, src: Src) -> CompileResult<Gpr> {
        let (p, moves) = self.vs.take_scratch(&[]);
        self.apply(moves)?;
        let g = gpr_of(p)?;
        match self.wide_imm(Width::W64, src, &[p])? {
            Rmi::Reg(r) => {
                if r != g {
                    self.asm.mov(Width::W64, Rm::Reg(g), Rmi::Reg(r));
                }
            }
            Rmi::Mem(m) => self.asm.mov(Width::W64, Rm::Reg(g), Rmi::Mem(m)),
            Rmi::Imm(i) => self.asm.mov(Width::W64, Rm::Reg(g), Rmi::Imm(i)),
        }
        Ok(g)
    }

    /// `mov width, dst, [addr + disp]` (LIR `Load`, `ldfld`): address into
    /// one scratch, the load into another, the destination defined per the
    /// value discipline (a Ref result goes frame-resident at once — the
    /// root invariant).
    fn emit_load_mem(
        &mut self,
        width: Width,
        dst: Place,
        addr: Src,
        disp: i32,
    ) -> CompileResult<()> {
        let g = self.addr_into(addr)?;
        let (p, moves) = self.vs.take_scratch(&[g.phys()]);
        self.apply(moves)?;
        let gd = gpr_of(p)?;
        self.asm
            .mov(width, Rm::Reg(gd), Rmi::Mem(Mem::base_disp(g, disp)));
        match dst {
            Place::Val(t) => self.define_temp_reg(t.0, gd),
            Place::Reg(g2) => {
                let moves = self.vs.clobber(g2.phys());
                self.apply(moves)?;
                if g2 != gd {
                    self.asm.mov(width, Rm::Reg(g2), Rmi::Reg(gd));
                }
                Ok(())
            }
        }
    }

    /// `mov [addr + disp], src` (LIR `Store`, `stfld` of a non-reference
    /// field). A too-wide constant source materializes through a scratch
    /// register (the wide-imm rule); a slot-resident source reloads
    /// through one (no mem,mem form).
    fn emit_store_mem(
        &mut self,
        width: Width,
        addr: Src,
        disp: i32,
        src: Src,
    ) -> CompileResult<()> {
        let g = self.addr_into(addr)?;
        match self.wide_imm(width, src, &[g.phys()])? {
            src @ (Rmi::Reg(_) | Rmi::Imm(_)) => {
                self.asm.mov(width, Rm::Mem(Mem::base_disp(g, disp)), src)
            }
            Rmi::Mem(m) => {
                let (p, moves) = self.vs.take_scratch(&[g.phys()]);
                self.apply(moves)?;
                let gs = gpr_of(p)?;
                self.asm.mov(width, Rm::Reg(gs), Rmi::Mem(m));
                self.asm
                    .mov(width, Rm::Mem(Mem::base_disp(g, disp)), Rmi::Reg(gs));
            }
        }
        Ok(())
    }

    /// The explicit null check (step_10.4): the reference into one scratch
    /// GPR, then a 32-bit load through it into a second scratch, result
    /// unused — on null the hardware fault is the NullReferenceException
    /// (the EE's signal translation, the same path `idiv`'s #DE rides).
    /// Always explicit; folding the check into a small-offset access is a
    /// later optimization.
    fn emit_null_check(&mut self, addr: Src) -> CompileResult<()> {
        let g = self.addr_into(addr)?;
        let (p, moves) = self.vs.take_scratch(&[g.phys()]);
        self.apply(moves)?;
        let gd = gpr_of(p)?;
        self.asm
            .mov(Width::W32, Rm::Reg(gd), Rmi::Mem(Mem::base(g)));
        Ok(())
    }

    /// A direct IL call: spill every register-resident temp (the pool is
    /// caller-saved), resolve the target through the EE, emit the call, and
    /// record the safepoint + relocation for 07.7. Two machine forms:
    ///
    /// - **IAT_VALUE** (already-compiled target): `call rel32` (E8), the
    ///   relocation's rel32 field right after the opcode, targeting the
    ///   entry point.
    /// - **IAT_PVALUE** (target not yet compiled — e.g. fib's recursive
    ///   call when `Main` is jitted first): `call [rip+rel32]` (FF /2), the
    ///   relocation targeting the EE's entry-point *slot* (a fixup
    ///   precode's target slot the EE keeps current). This is RyuJIT's
    ///   `EC_FUNC_TOKEN_INDIR` form (codegenxarch.cpp:10785,
    ///   jitinterface.cpp `getFunctionEntryPoint`).
    fn emit_call(&mut self, method: MethodHandle) -> CompileResult<()> {
        let lookup = self.ee.get_function_entry_point(method);
        let addr = const_lookup_addr(&lookup);
        let slot = const_lookup_slot(&lookup);
        self.emit_call_lookup(method.into(), addr, slot)
    }

    /// An EE helper call (float `rem` → `fmod`/`fmodf` via
    /// `CORINFO_HELP_FLTREM`/`DBLREM`): the target comes from
    /// `getHelperFtn`; the call site records no method handle (the
    /// artifact.rs CallSite contract for helper calls).
    fn emit_helper_call(&mut self, id: CorInfoHelpFunc) -> CompileResult<()> {
        let lookup = self.ee.get_helper_ftn(id).entrypoint;
        let addr = const_lookup_addr(&lookup);
        let slot = const_lookup_slot(&lookup);
        self.emit_call_lookup(None, addr, slot)
    }

    /// The shared call-emission tail behind [`Emitter::emit_call`] and
    /// [`Emitter::emit_helper_call`]: the lookup is already split into the
    /// direct address (IAT_VALUE) or the indirection slot (IAT_PVALUE).
    fn emit_call_lookup(
        &mut self,
        method: Option<MethodHandle>,
        addr: Option<usize>,
        slot: Option<usize>,
    ) -> CompileResult<()> {
        let moves = self.vs.spill_registers();
        self.apply(moves)?;
        let instr_offset = self.asm.offset();
        let (size, reloc_offset, target) = if let Some(target) = addr {
            self.asm.call_unlinked();
            (DIRECT_CALL_LEN, instr_offset + 1, target)
        } else if let Some(slot) = slot {
            self.asm.call_indirect();
            (INDIRECT_CALL_LEN, instr_offset + 2, slot)
        } else {
            return Err(CompileError::Unsupported(
                "call target with more than one indirection (IAT_PPVALUE/IAT_RELPVALUE)",
            ));
        };
        self.call_sites.push(CallSite {
            chunk: ChunkRef::HotCode,
            offset: instr_offset,
            size,
            // Helper calls record neither signature nor method (the
            // CallSite contract); managed calls carry both.
            sig: if method.is_some() {
                self.call_sig.clone()
            } else {
                None
            },
            method,
        });
        self.relocations.push(Relocation {
            chunk: ChunkRef::HotCode,
            offset: reloc_offset,
            target,
            reloc_type: RelocType::RELATIVE32,
            addl_delta: 0,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Whole-method byte tests: hand-built LIR in → emitted machine-code
    //! bytes out. Every expected byte string was hand-traced through the
    //! emitter and cross-checked by disassembling with `objdump -D
    //! -b binary -m i386:x86-64` (the fib test comments carry the
    //! disassembly). Methods: a trivial return, a call (stack-alignment
    //! check), a branch join (deterministic spill-state check), a `rem`
    //! (idiv fixed-register sequence), GC-slot residency, and fib's exact
    //! IL bytes end to end through the pipeline.

    use super::*;
    use rokajit::ir::hir::{Local, LocalKind};
    use rokajit::ir::lir::{Block, BranchCond, Operand, Stmt, StmtKind};
    use rokajit::ir::{BinaryOp, BlockId, Const, IL_OFFSET_NONE};
    use rokajit::pipeline::Tier;
    use rokajit_ee::enums::CorJitFuncKind;
    use rokajit_ee::mock::MockEe;

    fn local(ty: Type, kind: LocalKind) -> Local {
        Local {
            ty,
            kind,
            pinned: false,
        }
    }

    fn int_arg(i: u32) -> Local {
        local(Type::Int32, LocalKind::IlArg(i))
    }

    fn int_temp() -> Local {
        local(Type::Int32, LocalKind::Temp)
    }

    fn stmt(kind: StmtKind) -> Stmt {
        Stmt {
            il_offset: IL_OFFSET_NONE,
            kind,
        }
    }

    fn block(id: u32, stmts: Vec<Stmt>) -> Block {
        Block {
            id: BlockId(id),
            stmts,
        }
    }

    fn method(
        locals: Vec<Local>,
        num_args: u32,
        num_il_locals: u32,
        blocks: Vec<Block>,
    ) -> lir::Method {
        lir::Method {
            blocks,
            locals,
            eh_regions: Vec::new(),
            num_args,
            num_il_locals,
            struct_layouts: StructLayouts::new(),
        }
    }

    fn handle(raw: usize) -> MethodHandle {
        MethodHandle::from_raw(raw as *mut u8 as _).unwrap()
    }

    fn int_sig() -> CallSig {
        CallSig {
            ret: Type::Int32,
            args: vec![Type::Int32],
            has_this: false,
        }
    }

    fn emit(m: &lir::Method, ee: &MockEe) -> CodegenOutput {
        emit_tier0(m, ee).expect("emits")
    }

    // ---- SysV classification ----

    #[test]
    fn classify_assigns_int_arg_registers_in_order() {
        let sig = CallSig {
            ret: Type::Int32,
            args: vec![Type::Int32, Type::Ref, Type::Int64],
            has_this: false,
        };
        let abi = classify_call(&sig, &StructLayouts::new()).expect("classifies");
        assert_eq!(
            abi.args,
            vec![
                ArgLocation::Reg(Gpr::Rdi.phys()),
                ArgLocation::Reg(Gpr::Rsi.phys()),
                ArgLocation::Reg(Gpr::Rdx.phys()),
            ]
        );
        assert_eq!(abi.ret, Some(ArgLocation::Reg(Gpr::Rax.phys())));
        assert_eq!(abi.stack_arg_bytes, 0);
    }

    #[test]
    fn classify_places_this_first_and_floats_in_xmm() {
        let sig = CallSig {
            ret: Type::Float,
            args: vec![Type::Int32, Type::Double],
            has_this: true,
        };
        let abi = classify_call(&sig, &StructLayouts::new()).expect("classifies");
        assert_eq!(
            abi.args,
            vec![
                ArgLocation::Reg(Gpr::Rdi.phys()), // this
                ArgLocation::Reg(Gpr::Rsi.phys()),
                ArgLocation::Reg(regs::Xmm::Xmm0.phys()),
            ]
        );
        assert_eq!(abi.ret, Some(ArgLocation::Reg(regs::Xmm::Xmm0.phys())));
    }

    #[test]
    fn classify_overflows_to_the_stack_and_void_has_no_return() {
        let sig = CallSig {
            ret: Type::Void,
            args: vec![Type::Int32; 8],
            has_this: false,
        };
        let abi = classify_call(&sig, &StructLayouts::new()).expect("classifies");
        assert_eq!(abi.args[5], ArgLocation::Reg(Gpr::R9.phys()));
        assert_eq!(abi.args[6], ArgLocation::Stack { offset: 0 });
        assert_eq!(abi.args[7], ArgLocation::Stack { offset: 8 });
        assert_eq!(abi.stack_arg_bytes, 16, "16-aligned outgoing area");
        assert_eq!(abi.ret, None);
    }

    // ---- SysV struct classification (step_10.9): descriptor → CallAbi ----
    //
    // The cases mirror runtime/src/tests/JIT/Directed/StructABI/StructABI.cs;
    // the descriptors are canned (the EE computes them — classification is
    // NOT reimplemented here; what we test is our mapping of a descriptor
    // to registers/stack). sN = eightbyte size.

    use rokajit::structs::SysVClass;

    fn class(raw: usize) -> rokajit_ee::handles::ClassHandle {
        rokajit_ee::handles::ClassHandle::from_raw(raw as *mut u8 as _).unwrap()
    }

    /// A canned descriptor: `(class, size)` per eightbyte, offsets 0/8.
    fn sysv(eightbytes: &[(SysVClass, u8)]) -> SysVPass {
        let mut pass = SysVPass::memory();
        pass.passed_in_registers = !eightbytes.is_empty();
        pass.count = eightbytes.len() as u8;
        for (i, &(class, size)) in eightbytes.iter().enumerate() {
            pass.classes[i] = class;
            pass.sizes[i] = size;
            pass.offsets[i] = (i * 8) as u8;
        }
        pass
    }

    /// One canned class in a fresh side table.
    fn one_layout(
        size: u32,
        eightbytes: &[(SysVClass, u8)],
    ) -> (StructLayouts, rokajit_ee::handles::ClassHandle) {
        let c = class(0x9000 + size as usize);
        let mut layouts = StructLayouts::new();
        layouts.insert(
            c,
            StructLayout {
                size,
                align: 8,
                gc_cells: vec![],
                sysv: sysv(eightbytes),
            },
        );
        (layouts, c)
    }

    fn struct_arg_abi(
        layouts: &StructLayouts,
        c: rokajit_ee::handles::ClassHandle,
        pre: &[Type],
    ) -> CallAbi {
        let mut args = pre.to_vec();
        args.push(Type::Struct(c));
        classify_call(
            &CallSig {
                ret: Type::Void,
                args,
                has_this: false,
            },
            layouts,
        )
        .expect("classifies")
    }

    use SysVClass::{Integer as I, Sse as F};

    #[test]
    fn classify_single_eightbyte_structs() {
        // SingleByte{byte} → [Integer s1]; SingleLong → [Integer s8];
        // SingleFloat → [SSE s4]; SingleDouble → [SSE s8].
        let (layouts, c) = one_layout(1, &[(I, 1)]);
        let abi = struct_arg_abi(&layouts, c, &[]);
        assert_eq!(
            abi.args[0],
            ArgLocation::StructRegs {
                regs: [Gpr::Rdi.phys(), Gpr::Rdi.phys()],
                count: 1,
                sizes: [1, 0],
                offsets: [0, 0],
            }
        );
        let (layouts, c) = one_layout(8, &[(I, 8)]);
        assert_eq!(
            struct_arg_abi(&layouts, c, &[]).args[0],
            ArgLocation::StructRegs {
                regs: [Gpr::Rdi.phys(), Gpr::Rdi.phys()],
                count: 1,
                sizes: [8, 0],
                offsets: [0, 0],
            }
        );
        let (layouts, c) = one_layout(4, &[(F, 4)]);
        assert_eq!(
            struct_arg_abi(&layouts, c, &[]).args[0],
            ArgLocation::StructRegs {
                regs: [regs::Xmm::Xmm0.phys(); 2],
                count: 1,
                sizes: [4, 0],
                offsets: [0, 0],
            }
        );
        let (layouts, c) = one_layout(8, &[(F, 8)]);
        assert_eq!(
            struct_arg_abi(&layouts, c, &[]).args[0],
            ArgLocation::StructRegs {
                regs: [regs::Xmm::Xmm0.phys(); 2],
                count: 1,
                sizes: [8, 0],
                offsets: [0, 0],
            }
        );
    }

    #[test]
    fn classify_merged_and_mixed_eightbytes() {
        // ByteAndFloat{byte;float} → one [Integer s8] (merge: Integer wins).
        let (layouts, c) = one_layout(8, &[(I, 8)]);
        assert!(matches!(
            struct_arg_abi(&layouts, c, &[]).args[0],
            ArgLocation::StructRegs {
                regs: [r, _],
                count: 1,
                ..
            } if r == Gpr::Rdi.phys()
        ));
        // LongAndFloat{ulong;float} → [Integer s8, SSE s4] → rdi + xmm0.
        let (layouts, c) = one_layout(16, &[(I, 8), (F, 4)]);
        assert_eq!(
            struct_arg_abi(&layouts, c, &[]).args[0],
            ArgLocation::StructRegs {
                regs: [Gpr::Rdi.phys(), regs::Xmm::Xmm0.phys()],
                count: 2,
                sizes: [8, 4],
                offsets: [0, 8],
            }
        );
        // DoubleAndByte{double;byte} → [SSE s8, Integer s1] → xmm0 + rdi
        // (per-register-file assignment, NOT declaration order).
        let (layouts, c) = one_layout(16, &[(F, 8), (I, 1)]);
        assert_eq!(
            struct_arg_abi(&layouts, c, &[]).args[0],
            ArgLocation::StructRegs {
                regs: [regs::Xmm::Xmm0.phys(), Gpr::Rdi.phys()],
                count: 2,
                sizes: [8, 1],
                offsets: [0, 8],
            }
        );
        // TwoLongs → rdi + rsi; TwoFloats → one [SSE s8] → xmm0;
        // TwoDoubles → xmm0 + xmm1.
        let (layouts, c) = one_layout(16, &[(I, 8), (I, 8)]);
        assert_eq!(
            struct_arg_abi(&layouts, c, &[]).args[0],
            ArgLocation::StructRegs {
                regs: [Gpr::Rdi.phys(), Gpr::Rsi.phys()],
                count: 2,
                sizes: [8, 8],
                offsets: [0, 8],
            }
        );
        let (layouts, c) = one_layout(8, &[(F, 8)]);
        assert!(matches!(
            struct_arg_abi(&layouts, c, &[]).args[0],
            ArgLocation::StructRegs { count: 1, .. }
        ));
        let (layouts, c) = one_layout(16, &[(F, 8), (F, 8)]);
        assert_eq!(
            struct_arg_abi(&layouts, c, &[]).args[0],
            ArgLocation::StructRegs {
                regs: [regs::Xmm::Xmm0.phys(), regs::Xmm::Xmm1.phys()],
                count: 2,
                sizes: [8, 8],
                offsets: [0, 8],
            }
        );
    }

    #[test]
    fn classify_memory_structs_go_on_the_stack() {
        // A 17-byte struct and FourLongs (32 bytes): passedInRegisters =
        // false → full-size stack copy, 8-rounded.
        for (size, rounded) in [(17u32, 24u32), (32, 32)] {
            let (layouts, c) = one_layout(size, &[]);
            let abi = struct_arg_abi(&layouts, c, &[]);
            assert_eq!(abi.args[0], ArgLocation::Stack { offset: 0 });
            assert_eq!(abi.stack_arg_bytes, rounded.div_ceil(16) * 16);
        }
    }

    #[test]
    fn classify_register_exhaustion_sends_the_whole_struct_to_the_stack() {
        // StructABI.cs's NotEnoughRegisters: 6 GPRs occupied, then
        // TwoLongs — the ENTIRE struct goes on the stack (no splitting).
        let (layouts, c) = one_layout(16, &[(I, 8), (I, 8)]);
        let abi = struct_arg_abi(&layouts, c, &[Type::Int64; 6]);
        assert_eq!(abi.args[6], ArgLocation::Stack { offset: 0 });
        // ...but the float pool is untouched, so TwoDoubles still goes
        // xmm0:xmm1 (EnoughRegisters).
        let (layouts, c) = one_layout(16, &[(F, 8), (F, 8)]);
        assert_eq!(
            struct_arg_abi(&layouts, c, &[Type::Int64; 6]).args[6],
            ArgLocation::StructRegs {
                regs: [regs::Xmm::Xmm0.phys(), regs::Xmm::Xmm1.phys()],
                count: 2,
                sizes: [8, 8],
                offsets: [0, 8],
            }
        );
        // 5 GPRs occupied: TwoLongs still doesn't fit (needs 2) → stack.
        let (layouts, c) = one_layout(16, &[(I, 8), (I, 8)]);
        assert_eq!(
            struct_arg_abi(&layouts, c, &[Type::Int64; 5]).args[5],
            ArgLocation::Stack { offset: 0 }
        );
        // 5 GPRs + 1 XMM occupied, then LongAndFloat: one eightbyte per
        // file → r9 + xmm1.
        let (layouts, c) = one_layout(16, &[(I, 8), (F, 4)]);
        let mut pre = vec![Type::Int64; 5];
        pre.push(Type::Double);
        assert_eq!(
            struct_arg_abi(&layouts, c, &pre).args[6],
            ArgLocation::StructRegs {
                regs: [Gpr::R9.phys(), regs::Xmm::Xmm1.phys()],
                count: 2,
                sizes: [8, 4],
                offsets: [0, 8],
            }
        );
    }

    #[test]
    fn classify_hidden_retbuf_argument_order() {
        // The importer models the hidden return buffer as a leading ByRef
        // argument (immediately after `this`); classification places it in
        // declaration order. Static method: retbuf in rdi.
        let (layouts, c) = one_layout(17, &[]);
        let abi = classify_call(
            &CallSig {
                ret: Type::Struct(c),
                args: vec![Type::ByRef, Type::Int32],
                has_this: false,
            },
            &layouts,
        )
        .expect("classifies");
        assert_eq!(abi.args[0], ArgLocation::Reg(Gpr::Rdi.phys()), "retbuf");
        assert_eq!(abi.args[1], ArgLocation::Reg(Gpr::Rsi.phys()));
        // rax carries the buffer address back.
        assert_eq!(abi.ret, Some(ArgLocation::Reg(Gpr::Rax.phys())));
        // Instance method: this in rdi, retbuf in rsi, user args after.
        let abi = classify_call(
            &CallSig {
                ret: Type::Struct(c),
                args: vec![Type::ByRef, Type::Int32],
                has_this: true,
            },
            &layouts,
        )
        .expect("classifies");
        assert_eq!(abi.args[0], ArgLocation::Reg(Gpr::Rdi.phys()), "this");
        assert_eq!(abi.args[1], ArgLocation::Reg(Gpr::Rsi.phys()), "retbuf");
        assert_eq!(abi.args[2], ArgLocation::Reg(Gpr::Rdx.phys()));
    }

    #[test]
    fn classify_struct_returns_per_register_file() {
        // 1-EB Integer s3 → rax with a 3-byte size.
        let (layouts, c) = one_layout(3, &[(I, 3)]);
        let abi = classify_call(
            &CallSig {
                ret: Type::Struct(c),
                args: vec![],
                has_this: false,
            },
            &layouts,
        )
        .expect("classifies");
        assert_eq!(
            abi.ret,
            Some(ArgLocation::StructRegs {
                regs: [Gpr::Rax.phys(); 2],
                count: 1,
                sizes: [3, 0],
                offsets: [0, 0],
            })
        );
        // 2-EB mixed {long; float} → rax + xmm0.
        let (layouts, c) = one_layout(16, &[(I, 8), (F, 4)]);
        let abi = classify_call(
            &CallSig {
                ret: Type::Struct(c),
                args: vec![],
                has_this: false,
            },
            &layouts,
        )
        .expect("classifies");
        assert_eq!(
            abi.ret,
            Some(ArgLocation::StructRegs {
                regs: [Gpr::Rax.phys(), regs::Xmm::Xmm0.phys()],
                count: 2,
                sizes: [8, 4],
                offsets: [0, 8],
            })
        );
        // TwoDoubles → xmm0:xmm1.
        let (layouts, c) = one_layout(16, &[(F, 8), (F, 8)]);
        let abi = classify_call(
            &CallSig {
                ret: Type::Struct(c),
                args: vec![],
                has_this: false,
            },
            &layouts,
        )
        .expect("classifies");
        assert_eq!(
            abi.ret,
            Some(ArgLocation::StructRegs {
                regs: [regs::Xmm::Xmm0.phys(), regs::Xmm::Xmm1.phys()],
                count: 2,
                sizes: [8, 8],
                offsets: [0, 8],
            })
        );
        // Non-register-passed → the retbuf address in rax.
        let (layouts, c) = one_layout(32, &[]);
        let abi = classify_call(
            &CallSig {
                ret: Type::Struct(c),
                args: vec![Type::ByRef],
                has_this: false,
            },
            &layouts,
        )
        .expect("classifies");
        assert_eq!(abi.ret, Some(ArgLocation::Reg(Gpr::Rax.phys())));
    }

    // ---- frame layout ----

    #[test]
    fn frame_slots_are_naturally_aligned_and_the_frame_is_16_aligned() {
        let m = method(
            vec![
                local(Type::Int32, LocalKind::IlArg(0)),
                local(Type::Int64, LocalKind::IlLocal(0)),
                int_temp(),
            ],
            1,
            1,
            vec![],
        );
        let layout = FrameLayout::compute(&m, 0).expect("layout");
        assert_eq!(layout.slots, vec![4, 16, 20]);
        assert_eq!(layout.frame_size, 32);
    }

    /// Struct locals (step_10.9): a 12-byte struct at 8-alignment takes a
    /// 12-byte slot; a struct embedding GC pointers rounds to 8 and
    /// 8-aligns so the reported roots stay 8-aligned; a struct missing
    /// from the side table is an upstream bug (Internal).
    #[test]
    fn struct_locals_size_their_slots_from_the_layout() {
        let small = rokajit_ee::handles::ClassHandle::from_raw(0x111usize as _).unwrap();
        let with_gc = rokajit_ee::handles::ClassHandle::from_raw(0x222usize as _).unwrap();
        let missing = rokajit_ee::handles::ClassHandle::from_raw(0x333usize as _).unwrap();
        let mut layouts = StructLayouts::new();
        layouts.insert(
            small,
            StructLayout {
                size: 12,
                align: 4,
                gc_cells: vec![],
                sysv: SysVPass::memory(),
            },
        );
        layouts.insert(
            with_gc,
            StructLayout {
                size: 12,
                align: 4,
                gc_cells: vec![GcCell {
                    offset: 0,
                    is_byref: false,
                }],
                sysv: SysVPass::memory(),
            },
        );
        let mut m = method(
            vec![
                local(Type::Struct(small), LocalKind::IlLocal(0)),
                local(Type::Struct(with_gc), LocalKind::IlLocal(1)),
            ],
            0,
            2,
            vec![],
        );
        m.struct_layouts = layouts;
        let layout = FrameLayout::compute(&m, 0).expect("layout");
        // small: 12 bytes at align 4; with_gc: rounded to 16 at align 8.
        assert_eq!(layout.slots, vec![12, 32]);
        assert_eq!(layout.frame_size, 32);
        m.locals
            .push(local(Type::Struct(missing), LocalKind::IlLocal(2)));
        assert!(matches!(
            FrameLayout::compute(&m, 0),
            Err(CompileError::Internal(_))
        ));
    }

    // ---- whole-method bytes ----

    /// `int id(int x) { return x; }` — the minimal frame.
    #[test]
    fn identity_method_bytes() {
        let m = method(
            vec![int_arg(0)],
            1,
            0,
            vec![block(
                0,
                vec![stmt(StmtKind::Return {
                    value: Some(Operand::Local(LocalId(0))),
                })],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)   — arg spill
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax  — return value
            0xC9, // leave
            0xC3, // ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
        assert_eq!(out.code.hot.alignment, 16);
        assert!(out.code.cold.is_none());
        assert_eq!(out.frame.frame_size, 16);
        assert!(out.frame.gc_roots.is_empty());
        assert!(out.relocations.is_empty() && out.call_sites.is_empty());
    }

    /// `int g(int n) { return f(n - 1); }` — the call test: 16-byte
    /// stack alignment at the call site, plus the relocation and
    /// call-site records 07.7 drains.
    #[test]
    fn call_method_bytes_and_alignment() {
        let f = handle(0xF00);
        let mut ee = MockEe::default();
        ee.entry_points.insert(0xF00, 0x5000);
        let m = method(
            vec![int_arg(0), int_temp(), int_temp()],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Binary {
                        dst: LocalId(1),
                        op: BinaryOp::Sub,
                        lhs: Operand::Local(LocalId(0)),
                        rhs: Operand::Const(Const::Int32(1)),
                    }),
                    stmt(StmtKind::Call {
                        dst: Some(LocalId(2)),
                        target: rokajit::ir::CallTarget::Direct(f),
                        sig: int_sig(),
                        args: vec![Operand::Temp(LocalId(1))],
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(2))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &ee);
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)  — arg n
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax
            0x83, 0xE8, 0x01, // subl $1, %eax        — t1 = n-1
            0x89, 0xC7, // movl %eax, %edi           — arg setup
            0x89, 0x45, 0xF8, // movl %eax, -8(%rbp) — call spill of t1
            0xE8, 0, 0, 0, 0, // call rel32 (patched by 07.7)
            0x89, 0x45, 0xF4, // movl %eax, -12(%rbp) — spill for the
            0x8B, 0x45, 0xF4, // movl -12(%rbp), %eax — return-value move
            0xC9, // leave
            0xC3, // ret
        ];
        assert_eq!(out.code.hot.bytes, expected);

        // The SysV alignment check: at entry rsp ≡ 8 (mod 16); the push
        // and the 16-aligned frame keep every call site at rsp ≡ 0.
        assert_eq!(out.frame.frame_size % 16, 0);
        assert_eq!((8 + 8 + out.frame.frame_size) % 16, 0);

        // The call is a GC safepoint and a relocation, in native order.
        assert_eq!(out.call_sites.len(), 1);
        assert_eq!(out.call_sites[0].chunk, ChunkRef::HotCode);
        assert_eq!(out.call_sites[0].offset, 22, "the E8 opcode offset");
        assert_eq!(out.call_sites[0].method, Some(f));
        assert_eq!(out.call_sites[0].sig, Some(int_sig()));
        assert_eq!(out.relocations.len(), 1);
        assert_eq!(out.relocations[0].chunk, ChunkRef::HotCode);
        assert_eq!(out.relocations[0].offset, 23, "the rel32 field");
        assert_eq!(out.relocations[0].target, 0x5000);
        assert_eq!(out.relocations[0].reloc_type, RelocType::RELATIVE32);
    }

    /// The IAT_PVALUE form of the call test: the callee is not yet
    /// compiled, so the EE answers with an entry-point *slot* — the emitter
    /// produces `call [rip+rel32]` (one byte longer; the tail shifts) and
    /// the relocation targets the slot, which the EE keeps current.
    #[test]
    fn indirect_call_method_bytes() {
        let f = handle(0xF00);
        let mut ee = MockEe::default();
        ee.entry_point_slots.insert(0xF00, 0x9000);
        let m = method(
            vec![int_arg(0), int_temp(), int_temp()],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Binary {
                        dst: LocalId(1),
                        op: BinaryOp::Sub,
                        lhs: Operand::Local(LocalId(0)),
                        rhs: Operand::Const(Const::Int32(1)),
                    }),
                    stmt(StmtKind::Call {
                        dst: Some(LocalId(2)),
                        target: rokajit::ir::CallTarget::Direct(f),
                        sig: int_sig(),
                        args: vec![Operand::Temp(LocalId(1))],
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(2))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &ee);
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)  — arg n
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax
            0x83, 0xE8, 0x01, // subl $1, %eax        — t1 = n-1
            0x89, 0xC7, // movl %eax, %edi           — arg setup
            0x89, 0x45, 0xF8, // movl %eax, -8(%rbp) — call spill of t1
            0xFF, 0x15, 0, 0, 0, 0, // call [rip+rel32] (patched by 07.7)
            0x89, 0x45, 0xF4, // movl %eax, -12(%rbp) — spill for the
            0x8B, 0x45, 0xF4, // movl -12(%rbp), %eax — return-value move
            0xC9, // leave
            0xC3, // ret
        ];
        assert_eq!(out.code.hot.bytes, expected);

        assert_eq!(out.call_sites.len(), 1);
        assert_eq!(out.call_sites[0].offset, 22, "the FF opcode offset");
        assert_eq!(out.call_sites[0].size, 6);
        assert_eq!(out.relocations.len(), 1);
        assert_eq!(out.relocations[0].offset, 24, "the disp32 field");
        assert_eq!(out.relocations[0].target, 0x9000, "the slot, not the entry");
        assert_eq!(out.relocations[0].reloc_type, RelocType::RELATIVE32);
    }

    /// `t = a + 1; if (a < b) goto B2; B1: return t; B2: return t;` —
    /// the branch-join test: `t` is defined in B0 and read in both
    /// successors, so the join spill makes merge state deterministic and
    /// both successors emit the identical reload from t's slot.
    #[test]
    fn branch_join_has_deterministic_spill_state() {
        let m = method(
            vec![int_arg(0), int_arg(1), int_temp()],
            2,
            0,
            vec![
                block(
                    0,
                    vec![
                        stmt(StmtKind::Binary {
                            dst: LocalId(2),
                            op: BinaryOp::Add,
                            lhs: Operand::Local(LocalId(0)),
                            rhs: Operand::Const(Const::Int32(1)),
                        }),
                        stmt(StmtKind::Branch {
                            cond: BranchCond::Cmp {
                                op: BinaryOp::Lt,
                                lhs: Operand::Local(LocalId(0)),
                                rhs: Operand::Local(LocalId(1)),
                            },
                            target: BlockId(2),
                        }),
                    ],
                ),
                block(
                    1,
                    vec![stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(2))),
                    })],
                ),
                block(
                    2,
                    vec![stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(2))),
                    })],
                ),
            ],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)  — arg a
            0x89, 0x75, 0xF8, // movl %esi, -8(%rbp)  — arg b
            // B0:
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax
            0x83, 0xC0, 0x01, // addl $1, %eax        — t = a+1
            0x89, 0x45, 0xF4, // movl %eax, -12(%rbp) — join spill of t
            0x8B, 0x4D, 0xFC, // movl -4(%rbp), %ecx  (mem,mem cmp avoided)
            0x3B, 0x4D, 0xF8, // cmpl -8(%rbp), %ecx
            0x0F, 0x8C, 0x05, 0, 0, 0, // jl B2 (rel = +5)
            // B1 (offset 35):
            0x8B, 0x45, 0xF4, // movl -12(%rbp), %eax
            0xC9, 0xC3,
            // B2 (offset 40): byte-identical — the deterministic merge state.
            0x8B, 0x45, 0xF4, // movl -12(%rbp), %eax
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);
        assert_eq!(out.code.hot.bytes.len(), 45);
        assert_eq!(out.code.hot.bytes[35..40], out.code.hot.bytes[40..45]);
    }

    /// `int m(int a, int b) => a < b` — compare-as-a-value: the `cmp` /
    /// `setcc` pair, with the zeroing `mov` in between (flag-preserving).
    #[test]
    fn compare_value_method_bytes() {
        let m = method(
            vec![int_arg(0), int_arg(1), int_temp()],
            2,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Binary {
                        dst: LocalId(2),
                        op: BinaryOp::Lt,
                        lhs: Operand::Local(LocalId(0)),
                        rhs: Operand::Local(LocalId(1)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(2))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)  — arg a
            0x89, 0x75, 0xF8, // movl %esi, -8(%rbp)  — arg b
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax  (mem,mem cmp avoided)
            0x3B, 0x45, 0xF8, // cmpl -8(%rbp), %eax
            0xB9, 0, 0, 0, 0, // movl $0, %ecx       — flag-preserving zeroing
            0x0F, 0x9C, 0xC1, // setl %cl
            0x89, 0xC8, // movl %ecx, %eax           — return value
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `uint m(uint a) => a / 3` — the unsigned-divide sequence: `edx`
    /// zeroed (never `cdq`), the `div` form, quotient out of `rax`.
    #[test]
    fn udiv_method_bytes() {
        let m = method(
            vec![int_arg(0), int_temp()],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Binary {
                        dst: LocalId(1),
                        op: BinaryOp::UDiv,
                        lhs: Operand::Local(LocalId(0)),
                        rhs: Operand::Const(Const::Int32(3)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(1))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax  — dividend to rax
            0xBA, 0, 0, 0, 0, // movl $0, %edx       — zero, not sign-extend
            0xB9, 0x03, 0, 0, 0, // movl $3, %ecx    — imm divisor materialized
            0xF7, 0xF1, // divl %ecx                  — the unsigned form
            0x89, 0x45, 0xF8, // movl %eax, -8(%rbp) — rax clobber spill at ret
            0x8B, 0x45, 0xF8, // movl -8(%rbp), %eax
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `int m(int a, int b) => a << b` — the variable-count shift: the
    /// count moves into `cl` with a 32-bit `mov`, the `D3` form applies.
    #[test]
    fn shift_by_cl_method_bytes() {
        let m = method(
            vec![int_arg(0), int_arg(1), int_temp()],
            2,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Binary {
                        dst: LocalId(2),
                        op: BinaryOp::Shl,
                        lhs: Operand::Local(LocalId(0)),
                        rhs: Operand::Local(LocalId(1)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(2))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)  — arg a
            0x89, 0x75, 0xF8, // movl %esi, -8(%rbp)  — arg b
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax
            0x8B, 0x4D, 0xF8, // movl -8(%rbp), %ecx — count into cl
            0xD3, 0xE0, // shll %cl, %eax
            0x89, 0x45, 0xF4, // movl %eax, -12(%rbp) — rax clobber spill at ret
            0x8B, 0x45, 0xF4, // movl -12(%rbp), %eax
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `ulong m(uint a) => a` — `conv.u8`: a 32-bit `mov` materializes the
    /// zero-extension (never an alias of the 32-bit slot, which a 64-bit
    /// read would overrun into the neighboring bytes).
    #[test]
    fn conv_u8_method_bytes() {
        let m = method(
            vec![int_arg(0), local(Type::Int64, LocalKind::Temp)],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Conv {
                        dst: LocalId(1),
                        to: Type::Int64,
                        overflow: false,
                        unsigned: true,
                        src: Operand::Local(LocalId(0)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(1))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax  — zero-extends into rax
            0x48, 0x89, 0x45, 0xF0, // movq %rax, -16(%rbp) — rax clobber spill
            0x48, 0x8B, 0x45, 0xF0, // movq -16(%rbp), %rax
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `long m(int a) => a` — `conv.i8`: `movsxd` sign-extends.
    #[test]
    fn conv_i8_method_bytes() {
        let m = method(
            vec![int_arg(0), local(Type::Int64, LocalKind::Temp)],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Conv {
                        dst: LocalId(1),
                        to: Type::Int64,
                        overflow: false,
                        unsigned: false,
                        src: Operand::Local(LocalId(0)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(1))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)
            0x48, 0x63, 0x45, 0xFC, // movslq -4(%rbp), %rax
            0x48, 0x89, 0x45, 0xF0, // movq %rax, -16(%rbp) — rax clobber spill
            0x48, 0x8B, 0x45, 0xF0, // movq -16(%rbp), %rax
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `int m(int n) { return n % 3; }` — the idiv fixed-register
    /// sequence, with the immediate divisor materialized clear of rax/rdx.
    #[test]
    fn rem_method_bytes() {
        let m = method(
            vec![int_arg(0), int_temp()],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Binary {
                        dst: LocalId(1),
                        op: BinaryOp::Rem,
                        lhs: Operand::Local(LocalId(0)),
                        rhs: Operand::Const(Const::Int32(3)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(1))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax  — dividend to rax
            0x99, // cltd
            0xB9, 0x03, 0, 0, 0, // movl $3, %ecx    — imm divisor materialized
            0xF7, 0xF9, // idivl %ecx
            0x89, 0xD0, // movl %edx, %eax           — remainder out of rdx
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// A `Ref` IL local and a `ByRef` temp (from `ldloca`): both slots are
    /// reported roots, the IL local is zero-initialized in the prolog, and
    /// the temp spills to its own slot at definition — frame-resident GC
    /// references, the invariant 07.7's GC info keys off.
    #[test]
    fn gc_references_are_frame_resident() {
        let m = method(
            vec![
                local(Type::Int32, LocalKind::IlLocal(0)),
                local(Type::Ref, LocalKind::IlLocal(1)),
                local(Type::ByRef, LocalKind::Temp),
            ],
            0,
            2,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Copy {
                        dst: LocalId(2),
                        src: Operand::AddrOf(LocalId(0)),
                    }),
                    stmt(StmtKind::Return { value: None }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        // Slots: Int32 local at 4, Ref local at 16 (8-aligned), ByRef temp
        // at 24; frame rounds to 32.
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x20, // subq $32, %rsp
            0xC7, 0x45, 0xFC, 0, 0, 0, 0, // movl $0, -4(%rbp)   — zero-init
            0x48, 0xC7, 0x45, 0xF0, 0, 0, 0, 0, // movq $0, -16(%rbp) — Ref local
            0x48, 0xC7, 0x45, 0xE8, 0, 0, 0, 0, // movq $0, -24(%rbp) — ByRef temp
            0x48, 0x8D, 0x45, 0xFC, // leaq -4(%rbp), %rax
            0x48, 0x89, 0x45, 0xE8, // movq %rax, -24(%rbp) — spilled at def
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);
        assert_eq!(
            out.frame.gc_roots,
            vec![
                rokajit::pipeline::GcRootSlot {
                    offset: 16,
                    is_byref: false,
                    pinned: false,
                },
                rokajit::pipeline::GcRootSlot {
                    offset: 24,
                    is_byref: true,
                    pinned: false,
                },
            ]
        );
    }

    /// More than six integer arguments: the seventh arrives on the stack
    /// (step_10.9) — the prolog spill reloads it from the caller's
    /// outgoing area at `[rbp + 16]` into the slot.
    #[test]
    fn incoming_stack_args_spill_from_the_caller_frame() {
        let m = method(
            (0..7).map(int_arg).collect(),
            7,
            0,
            vec![block(
                0,
                vec![stmt(StmtKind::Return {
                    value: Some(Operand::Local(LocalId(0))),
                })],
            )],
        );
        let out = emit_tier0(&m, &MockEe::default()).expect("emits");
        // The prolog spills args 0..5 from rdi..r9 (slots 4,8,...,24),
        // then arg 6: `mov eax, [rbp+16]`; `mov [rbp-28], eax`.
        let bytes = &out.code.hot.bytes;
        let tail = [
            0x8B, 0x45, 0x10, // mov eax, [rbp+16]
            0x89, 0x45, 0xE4, // mov [rbp-28], eax
        ];
        let at = bytes
            .windows(tail.len())
            .position(|w| w == tail)
            .expect("the stack-arg spill is emitted");
        assert!(at > 8, "after the frame prolog");
        // And the returned arg 0 reads its own slot: mov eax, [rbp-4].
        assert_eq!(&bytes[at + tail.len()..], &[0x8B, 0x45, 0xFC, 0xC9, 0xC3]);
    }

    // ---- step_10.9: struct ABI + block op byte tests ----

    /// A fresh side table with one canned class.
    fn layouts_for(
        size: u32,
        align: u32,
        gc_cells: Vec<GcCell>,
        eightbytes: &[(SysVClass, u8)],
    ) -> (StructLayouts, rokajit_ee::handles::ClassHandle) {
        let c = class(0x5000 + size as usize);
        let mut layouts = StructLayouts::new();
        layouts.insert(
            c,
            StructLayout {
                size,
                align,
                gc_cells,
                sysv: sysv(eightbytes),
            },
        );
        (layouts, c)
    }

    fn struct_method(
        locals: Vec<Local>,
        num_args: u32,
        num_il_locals: u32,
        blocks: Vec<Block>,
        layouts: StructLayouts,
    ) -> lir::Method {
        let mut m = method(locals, num_args, num_il_locals, blocks);
        m.struct_layouts = layouts;
        m
    }

    /// `stloc` of an 8-byte struct: a block copy slot-to-slot. Struct IL
    /// locals zero-init in the prolog with decomposed stores.
    #[test]
    fn block_copy_small_struct_bytes() {
        let (layouts, c) = layouts_for(8, 8, vec![], &[]);
        let m = struct_method(
            vec![
                local(Type::Struct(c), LocalKind::IlLocal(0)),
                local(Type::Struct(c), LocalKind::IlLocal(1)),
            ],
            0,
            2,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::BlockCopy {
                        dst_addr: Operand::AddrOf(LocalId(1)),
                        dst_offset: 0,
                        src_addr: Operand::AddrOf(LocalId(0)),
                        class: c,
                    }),
                    stmt(StmtKind::Return { value: None }),
                ],
            )],
            layouts,
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x31, 0xC0, // xorl %eax, %eax          — zero-init local 0
            0x48, 0x89, 0x45, 0xF8, // movq %rax, -8(%rbp)
            0x31, 0xC9, // xorl %ecx, %ecx          — zero-init local 1
            0x48, 0x89, 0x4D, 0xF0, // movq %rcx, -16(%rbp)
            0x48, 0x8D, 0x45, 0xF0, // leaq -16(%rbp), %rax  — dst
            0x48, 0x8D, 0x4D, 0xF8, // leaq -8(%rbp), %rcx   — src
            0x48, 0x8B, 0x11, // movq (%rcx), %rdx
            0x48, 0x89, 0x10, // movq %rdx, (%rax)
            0xC9, // leave
            0xC3, // ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `initobj` of a 12-byte struct: inline decomposed zeroing (8+4).
    #[test]
    fn block_zero_initobj_bytes() {
        let (layouts, c) = layouts_for(12, 4, vec![], &[]);
        let m = struct_method(
            vec![local(Type::Struct(c), LocalKind::IlLocal(0))],
            0,
            1,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::BlockZero {
                        dst_addr: Operand::AddrOf(LocalId(0)),
                        class: c,
                    }),
                    stmt(StmtKind::Return { value: None }),
                ],
            )],
            layouts,
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x31, 0xC0, // xorl %eax, %eax          — prolog zero-init
            0x48, 0x89, 0x45, 0xF4, // movq %rax, -12(%rbp)
            0x89, 0x45, 0xFC, // movl %eax, -4(%rbp)
            0x48, 0x8D, 0x45, 0xF4, // leaq -12(%rbp), %rax — the initobj
            0x31, 0xC9, // xorl %ecx, %ecx
            0x48, 0x89, 0x08, // movq %rcx, (%rax)
            0x89, 0x48, 0x08, // movl %ecx, 8(%rax)
            0xC9, // leave
            0xC3, // ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// A 100-byte block copy: above the inline threshold → the
    /// `CORINFO_HELP_MEMCPY` helper call. The 208-byte frame also
    /// exercises the imm32 `sub rsp` prolog form.
    #[test]
    fn large_block_copy_uses_memcpy_helper() {
        let (layouts, c) = layouts_for(100, 8, vec![], &[]);
        let m = struct_method(
            vec![
                local(Type::Struct(c), LocalKind::Temp),
                local(Type::Struct(c), LocalKind::Temp),
            ],
            0,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::BlockCopy {
                        dst_addr: Operand::AddrOf(LocalId(1)),
                        dst_offset: 0,
                        src_addr: Operand::AddrOf(LocalId(0)),
                        class: c,
                    }),
                    stmt(StmtKind::Return { value: None }),
                ],
            )],
            layouts,
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x81, 0xEC, 0xD0, 0x00, 0x00, 0x00, // subq $208, %rsp
            0x48, 0x8D, 0xBD, 0x30, 0xFF, 0xFF, 0xFF, // leaq -208(%rbp), %rdi
            0x48, 0x8D, 0x75, 0x98, // leaq -104(%rbp), %rsi
            0x48, 0xC7, 0xC2, 0x64, 0x00, 0x00, 0x00, // movq $100, %rdx
            0xE8, 0, 0, 0, 0, // call rel32 (MEMCPY)
            0xC9, // leave
            0xC3, // ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
        assert_eq!(out.call_sites.len(), 1);
        assert_eq!(out.call_sites[0].method, None, "a helper call site");
        assert_eq!(out.relocations.len(), 1);
    }

    /// Seven integer arguments: the seventh stores into the outgoing
    /// stack area at [rsp], sized into the frame.
    #[test]
    fn outgoing_stack_args_store_into_the_outgoing_area() {
        let f = handle(0xF00);
        let mut ee = MockEe::default();
        ee.entry_points.insert(0xF00, 0x5000);
        let sig = CallSig {
            ret: Type::Void,
            args: vec![Type::Int32; 7],
            has_this: false,
        };
        let m = method(
            vec![int_arg(0)],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Call {
                        dst: None,
                        target: rokajit::ir::CallTarget::Direct(f),
                        sig,
                        args: vec![Operand::Local(LocalId(0)); 7],
                    }),
                    stmt(StmtKind::Return { value: None }),
                ],
            )],
        );
        let out = emit(&m, &ee);
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x20, // subq $32, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)   — arg spill
            0x8B, 0x7D, 0xFC, // movl -4(%rbp), %edi
            0x8B, 0x75, 0xFC, // movl -4(%rbp), %esi
            0x8B, 0x55, 0xFC, // movl -4(%rbp), %edx
            0x8B, 0x4D, 0xFC, // movl -4(%rbp), %ecx
            0x44, 0x8B, 0x45, 0xFC, // movl -4(%rbp), %r8d
            0x44, 0x8B, 0x4D, 0xFC, // movl -4(%rbp), %r9d
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax  — the stack arg
            0x89, 0x04, 0x24, // movl %eax, (%rsp)
            0xE8, 0, 0, 0, 0, // call rel32
            0xC9, // leave
            0xC3, // ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
        assert_eq!(out.frame.frame_size, 32);
    }

    /// A register-passed struct argument and result (TwoLongs): the
    /// prolog spills rdi/rsi into the argument's slot, the call setup
    /// reloads them from the address, and the result lands back in the
    /// destination slot from rax/rdx.
    #[test]
    fn struct_arg_and_result_cross_the_call_in_registers() {
        let f = handle(0xF00);
        let mut ee = MockEe::default();
        ee.entry_points.insert(0xF00, 0x5000);
        let (layouts, c) = layouts_for(16, 8, vec![], &[(I, 8), (I, 8)]);
        let sig = CallSig {
            ret: Type::Struct(c),
            args: vec![Type::Struct(c)],
            has_this: false,
        };
        let m = struct_method(
            vec![
                local(Type::Struct(c), LocalKind::IlArg(0)),
                local(Type::Struct(c), LocalKind::Temp),
            ],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Call {
                        dst: Some(LocalId(1)),
                        target: rokajit::ir::CallTarget::Direct(f),
                        sig,
                        args: vec![Operand::AddrOf(LocalId(0))],
                    }),
                    stmt(StmtKind::Return { value: None }),
                ],
            )],
            layouts,
        );
        let out = emit(&m, &ee);
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x20, // subq $32, %rsp
            0x48, 0x89, 0x7D, 0xF0, // movq %rdi, -16(%rbp) — arg spill
            0x48, 0x89, 0x75, 0xF8, // movq %rsi, -8(%rbp)
            0x48, 0x8D, 0x45, 0xF0, // leaq -16(%rbp), %rax
            0x48, 0x8B, 0x38, // movq (%rax), %rdi     — eightbyte 0
            0x48, 0x8D, 0x4D, 0xF0, // leaq -16(%rbp), %rcx
            0x48, 0x8B, 0x71, 0x08, // movq 8(%rcx), %rsi  — eightbyte 1
            0xE8, 0, 0, 0, 0, // call rel32
            0x48, 0x89, 0x45, 0xE0, // movq %rax, -32(%rbp) — the result
            0x48, 0x89, 0x55, 0xE8, // movq %rdx, -24(%rbp)
            0xC9, // leave
            0xC3, // ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// A stack-passed struct argument (FourLongs, 32 bytes, after six
    /// int64 arguments): the prolog copies it from the caller's outgoing
    /// area at [rbp+16] into the slot, four 8-byte chunks.
    #[test]
    fn incoming_stack_struct_arg_copies_into_its_slot() {
        let (layouts, c) = layouts_for(32, 8, vec![], &[]);
        let mut locals: Vec<Local> = (0..6)
            .map(|i| local(Type::Int64, LocalKind::IlArg(i)))
            .collect();
        locals.push(local(Type::Struct(c), LocalKind::IlArg(6)));
        let m = struct_method(
            locals,
            7,
            0,
            vec![block(0, vec![stmt(StmtKind::Return { value: None })])],
            layouts,
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x50, // subq $80, %rsp
            0x48, 0x89, 0x7D, 0xF8, // movq %rdi, -8(%rbp)
            0x48, 0x89, 0x75, 0xF0, // movq %rsi, -16(%rbp)
            0x48, 0x89, 0x55, 0xE8, // movq %rdx, -24(%rbp)
            0x48, 0x89, 0x4D, 0xE0, // movq %rcx, -32(%rbp)
            0x4C, 0x89, 0x45, 0xD8, // movq %r8, -40(%rbp)
            0x4C, 0x89, 0x4D, 0xD0, // movq %r9, -48(%rbp)
            0x48, 0x8B, 0x45, 0x10, // movq 16(%rbp), %rax — the struct
            0x48, 0x89, 0x45, 0xB0, // movq %rax, -80(%rbp)
            0x48, 0x8B, 0x45, 0x18, // movq 24(%rbp), %rax
            0x48, 0x89, 0x45, 0xB8, // movq %rax, -72(%rbp)
            0x48, 0x8B, 0x45, 0x20, // movq 32(%rbp), %rax
            0x48, 0x89, 0x45, 0xC0, // movq %rax, -64(%rbp)
            0x48, 0x8B, 0x45, 0x28, // movq 40(%rbp), %rax
            0x48, 0x89, 0x45, 0xC8, // movq %rax, -56(%rbp)
            0xC9, // leave
            0xC3, // ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// A register-passed struct return (TwoDoubles): one `movsd` per
    /// eightbyte into xmm0/xmm1.
    #[test]
    fn register_passed_struct_return_loads_eightbytes() {
        let (layouts, c) = layouts_for(16, 8, vec![], &[(F, 8), (F, 8)]);
        let m = struct_method(
            vec![local(Type::Struct(c), LocalKind::IlLocal(0))],
            0,
            1,
            vec![block(
                0,
                vec![stmt(StmtKind::ReturnStruct {
                    addr: Operand::AddrOf(LocalId(0)),
                    class: c,
                })],
            )],
            layouts,
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x31, 0xC0, // xorl %eax, %eax           — prolog zero-init
            0x48, 0x89, 0x45, 0xF0, // movq %rax, -16(%rbp)
            0x48, 0x89, 0x45, 0xF8, // movq %rax, -8(%rbp)
            0x48, 0x8D, 0x45, 0xF0, // leaq -16(%rbp), %rax
            0xF2, 0x0F, 0x10, 0x00, // movsd (%rax), %xmm0
            0x48, 0x8D, 0x4D, 0xF0, // leaq -16(%rbp), %rcx
            0xF2, 0x0F, 0x10, 0x49, 0x08, // movsd 8(%rcx), %xmm1
            0xC9, // leave
            0xC3, // ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// A 3-byte struct (SingleByte-3, one [Integer s3] eightbyte): every
    /// move respects the exact size — the prolog spill stores 2+1 bytes,
    /// the call setup loads 2+1 bytes, the result stores 2+1 bytes.
    #[test]
    fn three_byte_eightbyte_moves_are_exact() {
        let f = handle(0xF00);
        let mut ee = MockEe::default();
        ee.entry_points.insert(0xF00, 0x5000);
        let (layouts, c) = layouts_for(3, 1, vec![], &[(I, 3)]);
        let sig = CallSig {
            ret: Type::Struct(c),
            args: vec![Type::Struct(c)],
            has_this: false,
        };
        let m = struct_method(
            vec![
                local(Type::Struct(c), LocalKind::IlArg(0)),
                local(Type::Struct(c), LocalKind::Temp),
            ],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Call {
                        dst: Some(LocalId(1)),
                        target: rokajit::ir::CallTarget::Direct(f),
                        sig,
                        args: vec![Operand::AddrOf(LocalId(0))],
                    }),
                    stmt(StmtKind::Return { value: None }),
                ],
            )],
            layouts,
        );
        let out = emit(&m, &ee);
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x66, 0x89, 0x7D, 0xFD, // movw %di, -3(%rbp)  — spill, 2 bytes
            0x48, 0x89, 0xF8, // movq %rdi, %rax
            0x48, 0xC1, 0xE8, 0x10, // shrq $16, %rax
            0x40, 0x88, 0x45, 0xFF, // movb %al, -1(%rbp)  — +1 byte
            0x48, 0x8D, 0x45, 0xFD, // leaq -3(%rbp), %rax
            0x0F, 0xB7, 0x38, // movzxw (%rax), %edi  — load, 2 bytes
            0x0F, 0xB6, 0x48, 0x02, // movzxb 2(%rax), %ecx
            0x48, 0xC1, 0xE1, 0x10, // shlq $16, %rcx
            0x48, 0x09, 0xCF, // orq %rcx, %rdi      — +1 byte
            0xE8, 0, 0, 0, 0, // call rel32
            0x66, 0x89, 0x45, 0xFA, // movw %ax, -6(%rbp)  — result, 2 bytes
            0x48, 0x89, 0xC2, // movq %rax, %rdx
            0x48, 0xC1, 0xEA, 0x10, // shrq $16, %rdx
            0x40, 0x88, 0x55, 0xFC, // movb %dl, -4(%rbp)  — +1 byte
            0xC9, // leave
            0xC3, // ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    // ---- fib, end to end: exact IL bytes → machine-code bytes ----

    #[test]
    fn fib_compiles_end_to_end() {
        use rokajit::pipeline::MethodInfo;
        use rokajit_ee::enums::CorInfoType;
        use rokajit_ee::mock::MockSig;

        const FIB_TOKEN: u32 = 0x0600_0001;
        let mut ee = MockEe::default();
        let fib_sig = MockSig {
            ret: CorInfoType::Int,
            args: vec![CorInfoType::Int],
            has_this: false,
            ret_class: None,
            arg_classes: Vec::new(),
        };
        ee.add_method(FIB_TOKEN, fib_sig.clone());
        let info = MethodInfo {
            ftn: handle(1),
            il: vec![
                0x02, 0x18, 0x32, 0x12, 0x02, 0x17, 0x59, 0x28, 0x01, 0x00, 0x00, 0x06, 0x02, 0x18,
                0x59, 0x28, 0x01, 0x00, 0x00, 0x06, 0x58, 0x2A, 0x02, 0x2A,
            ],
            max_stack: 8,
            eh_count: 0,
            init_locals: false,
            args: ee.make_method_sig(&fib_sig),
            locals: ee.make_locals_sig(&[]),
        };
        let hir = rokajit::pipeline::import(&info, &ee).expect("imports");
        let hir = rokajit::pipeline::morph(hir).expect("morphs");
        let lir = rokajit::pipeline::lower(hir, &crate::X64Target).expect("lowers");
        // The recursive call target, as the importer resolved it; give the
        // mock EE an entry point for it.
        let fib_handle = lir
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .find_map(|s| match &s.kind {
                StmtKind::Call {
                    target: rokajit::ir::CallTarget::Direct(m),
                    ..
                } => Some(*m),
                _ => None,
            })
            .expect("fib calls itself");
        ee.entry_points.insert(fib_handle.as_raw() as usize, 0x1000);

        let out = rokajit::pipeline::codegen(&lir, &ee, &crate::X64Target, Tier::Tier0)
            .expect("tier-0 codegen");

        // objdump of the expected bytes:
        //   push   %rbp
        //   mov    %rsp,%rbp
        //   sub    $0x20,%rsp
        //   mov    %edi,-0x4(%rbp)
        //   cmpl   $0x2,-0x4(%rbp)
        //   jl     <B2>                (rel +47)
        //   mov    -0x4(%rbp),%eax
        //   sub    $0x1,%eax
        //   mov    %eax,%edi
        //   mov    %eax,-0x8(%rbp)
        //   call   <fib>               (reloc @33)
        //   mov    -0x4(%rbp),%ecx
        //   sub    $0x2,%ecx
        //   mov    %ecx,%edi
        //   mov    %eax,-0xc(%rbp)
        //   mov    %ecx,-0x10(%rbp)
        //   call   <fib>               (reloc @52)
        //   mov    -0xc(%rbp),%edx
        //   add    %eax,%edx
        //   mov    %eax,-0x14(%rbp)
        //   mov    %edx,%eax
        //   leave; ret
        // B2:
        //   mov    -0x4(%rbp),%eax
        //   leave; ret
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x20, // subq $32, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)
            // B0: the `n < 2` guard
            0x83, 0x7D, 0xFC, 0x02, // cmpl $2, -4(%rbp)
            0x0F, 0x8C, 0x2F, 0, 0, 0, // jl B2
            // B1: fib(n-1) + fib(n-2)
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax
            0x83, 0xE8, 0x01, // subl $1, %eax
            0x89, 0xC7, // movl %eax, %edi
            0x89, 0x45, 0xF8, // movl %eax, -8(%rbp)  — call spill
            0xE8, 0, 0, 0, 0, // call fib
            0x8B, 0x4D, 0xFC, // movl -4(%rbp), %ecx
            0x83, 0xE9, 0x02, // subl $2, %ecx
            0x89, 0xCF, // movl %ecx, %edi
            0x89, 0x45, 0xF4, // movl %eax, -12(%rbp) — call spills,
            0x89, 0x4D, 0xF0, // movl %ecx, -16(%rbp) — pool order
            0xE8, 0, 0, 0, 0, // call fib
            0x8B, 0x55, 0xF4, // movl -12(%rbp), %edx
            0x01, 0xC2, // addl %eax, %edx
            0x89, 0x45, 0xEC, // movl %eax, -20(%rbp) — rax clobber spill
            0x89, 0xD0, // movl %edx, %eax
            0xC9, 0xC3, // leave; ret
            // B2: return n
            0x8B, 0x45, 0xFC, // movl -4(%rbp), %eax
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);

        // Frame: six 4-byte slots in 32 bytes, 16-aligned at the calls.
        assert_eq!(out.frame.frame_size, 32);
        assert!(out.frame.gc_roots.is_empty(), "fib has no GC refs");
        // Two calls: safepoints at the E8 opcodes, rel32 fields one past.
        let site_offsets: Vec<u32> = out.call_sites.iter().map(|c| c.offset).collect();
        assert_eq!(site_offsets, vec![32, 51]);
        let reloc_offsets: Vec<u32> = out.relocations.iter().map(|r| r.offset).collect();
        assert_eq!(reloc_offsets, vec![33, 52]);
        assert!(out.relocations.iter().all(|r| r.target == 0x1000));
        assert!(out
            .call_sites
            .iter()
            .all(|c| c.method == Some(fib_handle) && c.sig.is_some()));

        // Stage 5 (07.7): the metadata channel renders the EE-facing
        // encodings. Byte-exact expectations are derived bit-by-bit in
        // gcinfo.rs/unwind.rs's tests.
        let meta =
            rokajit::pipeline::build_metadata(&out, &lir, &crate::X64Target).expect("metadata");
        assert_eq!(meta.gc_info, [0x26, 0x51, 0x09, 0x07]);
        assert_eq!(meta.unwind.len(), 1);
        assert_eq!(meta.unwind[0].func_kind, CorJitFuncKind::Root);
        assert_eq!(
            (meta.unwind[0].start_offset, meta.unwind[0].end_offset),
            (0, 73)
        );
        assert_eq!(
            meta.unwind[0].bytes,
            [0x01, 0x08, 0x03, 0x05, 0x08, 0x32, 0x04, 0x03, 0x01, 0x50]
        );
        assert!(meta.eh_clauses.is_empty() && meta.il_map.is_empty());

        // The whole driver, one call, as the FFI edge runs it.
        let artifact = rokajit::pipeline::compile(&info, &ee, &crate::X64Target, Tier::Tier0)
            .expect("compile()");
        assert_eq!(artifact.code.hot.bytes, out.code.hot.bytes);
        assert_eq!(artifact.gc_info, meta.gc_info);
        assert_eq!(artifact.unwind.len(), 1);
    }

    // ---- step_10.2: float whole-method bytes ----

    fn dbl_arg(i: u32) -> Local {
        local(Type::Double, LocalKind::IlArg(i))
    }

    fn dbl_temp() -> Local {
        local(Type::Double, LocalKind::Temp)
    }

    /// `double f(double a, double b) => a + b * 2.0` — scalar SSE
    /// arithmetic: the constant materializes through a GPR (movabs +
    /// movq), operands load/store with movsd, results are slot-resident.
    #[test]
    fn float_arith_method_bytes() {
        let m = method(
            vec![dbl_arg(0), dbl_arg(1), dbl_temp(), dbl_temp()],
            2,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Binary {
                        dst: LocalId(2),
                        op: BinaryOp::Mul,
                        lhs: Operand::Local(LocalId(1)),
                        rhs: Operand::Const(Const::Double(2.0)),
                    }),
                    stmt(StmtKind::Binary {
                        dst: LocalId(3),
                        op: BinaryOp::Add,
                        lhs: Operand::Local(LocalId(0)),
                        rhs: Operand::Temp(LocalId(2)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(3))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x20, // subq $32, %rsp
            0xF2, 0x0F, 0x11, 0x45, 0xF8, // movsd %xmm0, -8(%rbp)  — arg a
            0xF2, 0x0F, 0x11, 0x4D, 0xF0, // movsd %xmm1, -16(%rbp) — arg b
            // t2 = b * 2.0: the constant's bits through rax into xmm14
            0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // movabsq $2.0, %rax
            0x66, 0x4C, 0x0F, 0x6E, 0xF0, // movq %rax, %xmm14
            0xF2, 0x44, 0x0F, 0x10, 0x7D, 0xF0, // movsd -16(%rbp), %xmm15
            0xF2, 0x45, 0x0F, 0x59, 0xFE, // mulsd %xmm14, %xmm15
            0xF2, 0x44, 0x0F, 0x11, 0x7D, 0xE8, // movsd %xmm15, -24(%rbp)
            // t3 = a + t2
            0xF2, 0x44, 0x0F, 0x10, 0x7D, 0xF8, // movsd -8(%rbp), %xmm15
            0xF2, 0x44, 0x0F, 0x58, 0x7D, 0xE8, // addsd -24(%rbp), %xmm15
            0xF2, 0x44, 0x0F, 0x11, 0x7D, 0xE0, // movsd %xmm15, -32(%rbp)
            0xF2, 0x0F, 0x10, 0x45, 0xE0, // movsd -32(%rbp), %xmm0 — return
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `bool f(double a, double b) => a < b` — the ordered float compare:
    /// `ucomisd` then `setb & setnp` (a NaN operand must yield false).
    #[test]
    fn float_compare_value_method_bytes() {
        let m = method(
            vec![dbl_arg(0), dbl_arg(1), int_temp()],
            2,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Binary {
                        dst: LocalId(2),
                        op: BinaryOp::Lt,
                        lhs: Operand::Local(LocalId(0)),
                        rhs: Operand::Local(LocalId(1)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(2))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x20, // subq $32, %rsp
            0xF2, 0x0F, 0x11, 0x45, 0xF8, // movsd %xmm0, -8(%rbp)
            0xF2, 0x0F, 0x11, 0x4D, 0xF0, // movsd %xmm1, -16(%rbp)
            0xF2, 0x44, 0x0F, 0x10, 0x7D, 0xF8, // movsd -8(%rbp), %xmm15
            0x66, 0x44, 0x0F, 0x2E, 0x7D, 0xF0, // ucomisd -16(%rbp), %xmm15
            0xB8, 0, 0, 0, 0, // movl $0, %eax — flag-preserving zeroing
            0x0F, 0x92, 0xC0, // setb %al   — CF: a < b (or unordered)
            0xB9, 0, 0, 0, 0, // movl $0, %ecx
            0x0F, 0x9B, 0xC1, // setnp %cl  — ¬PF: not unordered
            0x21, 0xC8, // andl %ecx, %eax — ordered-and-less
            0x89, 0x45, 0xEC, // movl %eax, -20(%rbp) — rax clobber spill at ret
            0x8B, 0x45, 0xEC, // movl -20(%rbp), %eax
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `int f(double a, double b) { if (a == b) return 1; return 0; }` —
    /// the float branch expansion: `jp` over the `je` (NaN must not take
    /// the ordered-equal edge).
    #[test]
    fn float_branch_method_bytes() {
        let m = method(
            vec![dbl_arg(0), dbl_arg(1)],
            2,
            0,
            vec![
                block(
                    0,
                    vec![stmt(StmtKind::Branch {
                        cond: BranchCond::Cmp {
                            op: BinaryOp::Eq,
                            lhs: Operand::Local(LocalId(0)),
                            rhs: Operand::Local(LocalId(1)),
                        },
                        target: BlockId(2),
                    })],
                ),
                block(
                    1,
                    vec![stmt(StmtKind::Return {
                        value: Some(Operand::Const(Const::Int32(0))),
                    })],
                ),
                block(
                    2,
                    vec![stmt(StmtKind::Return {
                        value: Some(Operand::Const(Const::Int32(1))),
                    })],
                ),
            ],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0xF2, 0x0F, 0x11, 0x45, 0xF8, // movsd %xmm0, -8(%rbp)
            0xF2, 0x0F, 0x11, 0x4D, 0xF0, // movsd %xmm1, -16(%rbp)
            0xF2, 0x44, 0x0F, 0x10, 0x7D, 0xF8, // movsd -8(%rbp), %xmm15
            0x66, 0x44, 0x0F, 0x2E, 0x7D, 0xF0, // ucomisd -16(%rbp), %xmm15
            0x0F, 0x8A, 0x06, 0, 0, 0, // jp skip (rel +6) — NaN is not equal
            0x0F, 0x84, 0x07, 0, 0, 0, // je B2 (rel +7)
            // skip / B1 (offset 42): return 0
            0xB8, 0, 0, 0, 0, // movl $0, %eax
            0xC9, 0xC3,
            // B2 (offset 49): return 1
            0xB8, 0x01, 0, 0, 0, // movl $1, %eax
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `long f(double a) => (long)a` — `cvttsd2si` (conv.i8 from float).
    #[test]
    fn float_to_int_conv_method_bytes() {
        let m = method(
            vec![dbl_arg(0), local(Type::Int64, LocalKind::Temp)],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Conv {
                        dst: LocalId(1),
                        to: Type::Int64,
                        overflow: false,
                        unsigned: false,
                        src: Operand::Local(LocalId(0)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(1))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0xF2, 0x0F, 0x11, 0x45, 0xF8, // movsd %xmm0, -8(%rbp)
            0xF2, 0x48, 0x0F, 0x2C, 0x45, 0xF8, // cvttsd2si -8(%rbp), %rax
            0x48, 0x89, 0x45, 0xF0, // movq %rax, -16(%rbp) — rax clobber spill at ret
            0x48, 0x8B, 0x45, 0xF0, // movq -16(%rbp), %rax
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `double f(double a) => -a` — the sign-mask XOR: `0.0 - x` would
    /// mishandle `-0.0` and NaN signs.
    #[test]
    fn float_neg_method_bytes() {
        let m = method(
            vec![dbl_arg(0), dbl_temp()],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Unary {
                        dst: LocalId(1),
                        op: UnaryOp::Neg,
                        src: Operand::Local(LocalId(0)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(1))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0xF2, 0x0F, 0x11, 0x45, 0xF8, // movsd %xmm0, -8(%rbp)
            // the sign mask 0x8000000000000000 through rax into xmm14
            0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // movabsq
            0x66, 0x4C, 0x0F, 0x6E, 0xF0, // movq %rax, %xmm14
            0xF2, 0x44, 0x0F, 0x10, 0x7D, 0xF8, // movsd -8(%rbp), %xmm15
            0x66, 0x45, 0x0F, 0x57, 0xFE, // xorpd %xmm14, %xmm15
            0xF2, 0x44, 0x0F, 0x11, 0x7D, 0xF0, // movsd %xmm15, -16(%rbp)
            0xF2, 0x0F, 0x10, 0x45, 0xF0, // movsd -16(%rbp), %xmm0
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `int g(int i, double d) => h(i, d)` — the mixed-signature call:
    /// int args take the GPR sequence, float args the XMM one, each in
    /// its own order (SysV §3.2.3).
    #[test]
    fn mixed_signature_call_method_bytes() {
        let h = handle(0xF00);
        let mut ee = MockEe::default();
        ee.entry_points.insert(0xF00, 0x5000);
        let sig = CallSig {
            ret: Type::Int32,
            args: vec![Type::Int32, Type::Double],
            has_this: false,
        };
        let m = method(
            vec![int_arg(0), dbl_arg(1), int_temp()],
            2,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Call {
                        dst: Some(LocalId(2)),
                        target: rokajit::ir::CallTarget::Direct(h),
                        sig: sig.clone(),
                        args: vec![Operand::Local(LocalId(0)), Operand::Local(LocalId(1))],
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(2))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &ee);
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x20, // subq $32, %rsp
            0x89, 0x7D, 0xFC, // movl %edi, -4(%rbp)   — arg i
            0xF2, 0x0F, 0x11, 0x45, 0xF0, // movsd %xmm0, -16(%rbp) — arg d
            0x8B, 0x7D, 0xFC, // movl -4(%rbp), %edi   — arg setup (GPR lane)
            0xF2, 0x0F, 0x10, 0x45, 0xF0, // movsd -16(%rbp), %xmm0 — (XMM lane)
            0xE8, 0, 0, 0, 0, // call rel32
            0x89, 0x45, 0xEC, // movl %eax, -20(%rbp) — rax clobber spill at ret
            0x8B, 0x45, 0xEC, // movl -20(%rbp), %eax
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
        assert_eq!(out.call_sites.len(), 1);
        assert_eq!(out.call_sites[0].offset, 24);
        assert_eq!(out.relocations[0].target, 0x5000);
    }

    /// `double f(double a, double b) => a % b` — float `rem` is the EE
    /// helper call (CORINFO_HELP_DBLREM → fmod), args in xmm0/xmm1, the
    /// result out of xmm0; the call site records no method handle.
    #[test]
    fn float_rem_helper_call_method_bytes() {
        let sig = CallSig {
            ret: Type::Double,
            args: vec![Type::Double, Type::Double],
            has_this: false,
        };
        let m = method(
            vec![dbl_arg(0), dbl_arg(1), dbl_temp()],
            2,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Call {
                        dst: Some(LocalId(2)),
                        target: rokajit::ir::CallTarget::Helper(
                            rokajit_ee::enums::CorInfoHelpFunc::DBLREM,
                        ),
                        sig: sig.clone(),
                        args: vec![Operand::Local(LocalId(0)), Operand::Local(LocalId(1))],
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(2))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x20, // subq $32, %rsp
            0xF2, 0x0F, 0x11, 0x45, 0xF8, // movsd %xmm0, -8(%rbp)
            0xF2, 0x0F, 0x11, 0x4D, 0xF0, // movsd %xmm1, -16(%rbp)
            0xF2, 0x0F, 0x10, 0x45, 0xF8, // movsd -8(%rbp), %xmm0
            0xF2, 0x0F, 0x10, 0x4D, 0xF0, // movsd -16(%rbp), %xmm1
            0xE8, 0, 0, 0, 0, // call rel32 (helper; patched by 07.7)
            0xF2, 0x0F, 0x11, 0x45, 0xE8, // movsd %xmm0, -24(%rbp) — slot-resident
            0xF2, 0x0F, 0x10, 0x45, 0xE8, // movsd -24(%rbp), %xmm0 — return
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
        // The helper call site carries neither signature nor method.
        assert_eq!(out.call_sites.len(), 1);
        assert_eq!(out.call_sites[0].offset, 28);
        assert_eq!(out.call_sites[0].method, None);
        assert_eq!(out.call_sites[0].sig, None);
        assert_eq!(out.relocations.len(), 1);
    }

    /// Incoming float arguments spill from XMM registers; a float IL
    /// local zero-initializes with a GPR store of zero bits.
    #[test]
    fn float_locals_zero_init_and_arg_spill() {
        let m = method(
            vec![
                local(Type::Float, LocalKind::IlArg(0)),
                local(Type::Double, LocalKind::IlLocal(0)),
            ],
            1,
            1,
            vec![block(
                0,
                vec![stmt(StmtKind::Return {
                    value: Some(Operand::Local(LocalId(1))),
                })],
            )],
        );
        let out = emit(&m, &MockEe::default());
        // Slots: Float arg at 4, Double local at 16; frame 16.
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0xF3, 0x0F, 0x11, 0x45, 0xFC, // movss %xmm0, -4(%rbp)  — arg spill
            0x48, 0xC7, 0x45, 0xF0, 0, 0, 0, 0, // movq $0, -16(%rbp) — zero bits
            0xF2, 0x0F, 0x10, 0x45, 0xF0, // movsd -16(%rbp), %xmm0 — return 0.0
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    // ---- wide immediates (the inst.rs contract: codegen materializes) ----

    /// `long f(long a) => a + 499999500000` — a W64 ALU op with an
    /// immediate beyond the sign-extended imm32 field: the constant goes
    /// through a scratch register (`movabs`), never a truncated `81 /0`.
    #[test]
    fn wide_imm_arith_materializes_the_constant() {
        const BIG: i64 = 499_999_500_000; // 0x746A4AE6E0 — past i32
        let m = method(
            vec![
                local(Type::Int64, LocalKind::IlArg(0)),
                local(Type::Int64, LocalKind::Temp),
            ],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Binary {
                        dst: LocalId(1),
                        op: BinaryOp::Add,
                        lhs: Operand::Local(LocalId(0)),
                        rhs: Operand::Const(Const::Int64(BIG)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(1))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x48, 0x89, 0x7D, 0xF8, // movq %rdi, -8(%rbp)
            0x48, 0x8B, 0x45, 0xF8, // movq -8(%rbp), %rax
            0x48, 0xB9, 0xE0, 0xE6, 0x4A, 0x6A, 0x74, 0, 0, 0, // movabsq $BIG, %rcx
            0x48, 0x01, 0xC8, // addq %rcx, %rax
            0x48, 0x89, 0x45, 0xF0, // movq %rax, -16(%rbp) — rax clobber spill at ret
            0x48, 0x8B, 0x45, 0xF0, // movq -16(%rbp), %rax
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `long f(long a) { if (a == 499999500000) return 1; return 0; }` —
    /// the OSR-mainloop shape: a W64 compare against a wide constant
    /// (silently truncating it was the MISMATCH this test pins).
    #[test]
    fn wide_imm_cmp_materializes_the_constant() {
        const BIG: i64 = 499_999_500_000;
        let m = method(
            vec![local(Type::Int64, LocalKind::IlArg(0))],
            1,
            0,
            vec![
                block(
                    0,
                    vec![stmt(StmtKind::Branch {
                        cond: BranchCond::Cmp {
                            op: BinaryOp::Eq,
                            lhs: Operand::Local(LocalId(0)),
                            rhs: Operand::Const(Const::Int64(BIG)),
                        },
                        target: BlockId(2),
                    })],
                ),
                block(
                    1,
                    vec![stmt(StmtKind::Return {
                        value: Some(Operand::Const(Const::Int32(0))),
                    })],
                ),
                block(
                    2,
                    vec![stmt(StmtKind::Return {
                        value: Some(Operand::Const(Const::Int32(1))),
                    })],
                ),
            ],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x48, 0x89, 0x7D, 0xF8, // movq %rdi, -8(%rbp)
            0x48, 0xB8, 0xE0, 0xE6, 0x4A, 0x6A, 0x74, 0, 0, 0, // movabsq $BIG, %rax
            0x48, 0x39, 0x45, 0xF8, // cmpq %rax, -8(%rbp)
            0x0F, 0x84, 0x07, 0, 0, 0, // je B2 (rel +7)
            // B1: return 0
            0xB8, 0, 0, 0, 0,
            0xC9, 0xC3,
            // B2: return 1
            0xB8, 0x01, 0, 0, 0,
            0xC9, 0xC3,
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `long f() { long x = 499999500000; return x; }` — a wide constant
    /// stored to a frame slot (`mov qword [mem], imm32` would
    /// sign-extend; the constant must go through a register).
    #[test]
    fn wide_imm_store_to_local_materializes_the_constant() {
        const BIG: i64 = 499_999_500_000;
        let m = method(
            vec![local(Type::Int64, LocalKind::IlLocal(0))],
            0,
            1,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::Copy {
                        dst: LocalId(0),
                        src: Operand::Const(Const::Int64(BIG)),
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Local(LocalId(0))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x48, 0xC7, 0x45, 0xF8, 0, 0, 0, 0, // movq $0, -8(%rbp) — zero-init
            0x48, 0xB8, 0xE0, 0xE6, 0x4A, 0x6A, 0x74, 0, 0, 0, // movabsq $BIG, %rax
            0x48, 0x89, 0x45, 0xF8, // movq %rax, -8(%rbp)
            0x48, 0x8B, 0x45, 0xF8, // movq -8(%rbp), %rax
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    // ---- step_10.4: the object pack (loads/stores through refs, null checks) ----

    fn ref_arg(i: u32) -> Local {
        local(Type::Ref, LocalKind::IlArg(i))
    }

    /// `int f(C this) => this.x` (x: Int32 at offset 16) — the trap-based
    /// null check (a throwaway 32-bit load through the reference) followed
    /// by the field load.
    #[test]
    fn ldfld_int_method_bytes() {
        let m = method(
            vec![ref_arg(0), int_temp()],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::NullCheck {
                        arg: Operand::Local(LocalId(0)),
                    }),
                    stmt(StmtKind::Load {
                        dst: LocalId(1),
                        addr: Operand::Local(LocalId(0)),
                        offset: 16,
                        ty: Type::Int32,
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(1))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x48, 0x89, 0x7D, 0xF8, // movq %rdi, -8(%rbp)  — arg this
            0x48, 0x8B, 0x45, 0xF8, // movq -8(%rbp), %rax — the null check:
            0x8B, 0x08, // movl (%rax), %ecx              — faults iff null
            0x48, 0x8B, 0x55, 0xF8, // movq -8(%rbp), %rdx — the field load:
            0x8B, 0x72, 0x10, // movl 16(%rdx), %esi
            0x89, 0xF0, // movl %esi, %eax              — return value
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
        assert_eq!(
            out.frame.gc_roots,
            vec![rokajit::pipeline::GcRootSlot {
                offset: 8,
                is_byref: false,
                pinned: false,
            }]
        );
    }

    /// `C f(C this) => this.r` (r: a Ref field at offset 24) — a Ref load
    /// result is frame-resident at definition (the root invariant).
    #[test]
    fn ldfld_ref_result_is_frame_resident() {
        let m = method(
            vec![ref_arg(0), local(Type::Ref, LocalKind::Temp)],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::NullCheck {
                        arg: Operand::Local(LocalId(0)),
                    }),
                    stmt(StmtKind::Load {
                        dst: LocalId(1),
                        addr: Operand::Local(LocalId(0)),
                        offset: 24,
                        ty: Type::Ref,
                    }),
                    stmt(StmtKind::Return {
                        value: Some(Operand::Temp(LocalId(1))),
                    }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x48, 0x89, 0x7D, 0xF8, // movq %rdi, -8(%rbp)
            0x48, 0xC7, 0x45, 0xF0, 0, 0, 0, 0, // movq $0, -16(%rbp) — root zero-init
            0x48, 0x8B, 0x45, 0xF8, // movq -8(%rbp), %rax
            0x8B, 0x08, // movl (%rax), %ecx           — the null check
            0x48, 0x8B, 0x55, 0xF8, // movq -8(%rbp), %rdx
            0x48, 0x8B, 0x72, 0x18, // movq 24(%rdx), %rsi
            0x48, 0x89, 0x75, 0xF0, // movq %rsi, -16(%rbp) — spilled at def
            0x48, 0x8B, 0x45, 0xF0, // movq -16(%rbp), %rax — return value
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
        assert_eq!(
            out.frame.gc_roots,
            vec![
                rokajit::pipeline::GcRootSlot {
                    offset: 8,
                    is_byref: false,
                    pinned: false,
                },
                rokajit::pipeline::GcRootSlot {
                    offset: 16,
                    is_byref: false,
                    pinned: false,
                },
            ]
        );
    }

    /// `void f(C this, int v) { this.x = v; }` (x: Int32 at offset 16) —
    /// the source reloads through a scratch (no mem,mem form).
    #[test]
    fn stfld_int_method_bytes() {
        let m = method(
            vec![ref_arg(0), int_arg(1)],
            2,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::NullCheck {
                        arg: Operand::Local(LocalId(0)),
                    }),
                    stmt(StmtKind::Store {
                        addr: Operand::Local(LocalId(0)),
                        offset: 16,
                        src: Operand::Local(LocalId(1)),
                    }),
                    stmt(StmtKind::Return { value: None }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x48, 0x89, 0x7D, 0xF8, // movq %rdi, -8(%rbp)  — arg this
            0x89, 0x75, 0xF4, // movl %esi, -12(%rbp)      — arg v
            0x48, 0x8B, 0x45, 0xF8, // movq -8(%rbp), %rax
            0x8B, 0x08, // movl (%rax), %ecx           — the null check
            0x48, 0x8B, 0x55, 0xF8, // movq -8(%rbp), %rdx — address scratch
            0x8B, 0x75, 0xF4, // movl -12(%rbp), %esi   — source reload
            0x89, 0x72, 0x10, // movl %esi, 16(%rdx)   — the store
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// A W64 store of a small constant (`null`) uses the imm32 form; one
    /// past i32's range (a frozen-ref-style address) materializes through
    /// `movabs` first — the same wide-imm rule as the ALU ops.
    #[test]
    fn stfld_constant_store_bytes() {
        // this.r = null (r: Ref at offset 8).
        let m = method(
            vec![ref_arg(0)],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::NullCheck {
                        arg: Operand::Local(LocalId(0)),
                    }),
                    stmt(StmtKind::Store {
                        addr: Operand::Local(LocalId(0)),
                        offset: 8,
                        src: Operand::Const(Const::NullRef),
                    }),
                    stmt(StmtKind::Return { value: None }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x48, 0x89, 0x7D, 0xF8, // movq %rdi, -8(%rbp)
            0x48, 0x8B, 0x45, 0xF8, // movq -8(%rbp), %rax
            0x8B, 0x08, // movl (%rax), %ecx           — the null check
            0x48, 0x8B, 0x55, 0xF8, // movq -8(%rbp), %rdx
            0x48, 0xC7, 0x42, 0x08, 0, 0, 0, 0, // movq $0, 8(%rdx)
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);

        // this.r = <frozen object at 0x1_2345_6789> — the constant does
        // not fit the sign-extended imm32 field, so it goes through a
        // scratch (`movabs`).
        let m = method(
            vec![ref_arg(0)],
            1,
            0,
            vec![block(
                0,
                vec![
                    stmt(StmtKind::NullCheck {
                        arg: Operand::Local(LocalId(0)),
                    }),
                    stmt(StmtKind::Store {
                        addr: Operand::Local(LocalId(0)),
                        offset: 8,
                        src: Operand::Const(Const::FrozenRef(0x1_2345_6789)),
                    }),
                    stmt(StmtKind::Return { value: None }),
                ],
            )],
        );
        let out = emit(&m, &MockEe::default());
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x10, // subq $16, %rsp
            0x48, 0x89, 0x7D, 0xF8, // movq %rdi, -8(%rbp)
            0x48, 0x8B, 0x45, 0xF8, // movq -8(%rbp), %rax
            0x8B, 0x08, // movl (%rax), %ecx           — the null check
            0x48, 0x8B, 0x55, 0xF8, // movq -8(%rbp), %rdx
            0x48, 0xBE, 0x89, 0x67, 0x45, 0x23, 0x01, 0, 0, 0, // movabsq $0x123456789, %rsi
            0x48, 0x89, 0x72, 0x08, // movq %rsi, 8(%rdx)
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);
    }

    /// `void f(C this, C v) { this.r = v; }` (r: Ref at offset 24), from
    /// IL bytes through the whole pipeline: the importer routes the store
    /// through `CORINFO_HELP_CHECKED_ASSIGN_REF(&this.r, v)` — the field
    /// address materializes as a ByRef temp (a GC interior-pointer root)
    /// computed as `this + 24`, and the helper's dst/src land in rdi/rsi.
    #[test]
    fn stfld_ref_write_barrier_end_to_end() {
        use rokajit::pipeline::MethodInfo;
        use rokajit_ee::enums::CorInfoType;
        use rokajit_ee::mock::MockSig;

        const REF_FIELD: u32 = 0x0400_0002;
        let mut ee = MockEe::default();
        ee.add_field(REF_FIELD, CorInfoType::Class, 24);
        let entry = MockSig {
            ret: CorInfoType::Void,
            args: vec![CorInfoType::Class],
            has_this: true,
            ret_class: None,
            arg_classes: Vec::new(),
        };
        // ldarg.0; ldarg.1; stfld 0x04000002; ret
        let il = [0x02, 0x03, 0x7D, 0x02, 0x00, 0x00, 0x04, 0x2A];
        let info = MethodInfo {
            ftn: handle(1),
            il: il.to_vec(),
            max_stack: 8,
            eh_count: 0,
            init_locals: false,
            args: ee.make_method_sig(&entry),
            locals: ee.make_locals_sig(&[]),
        };
        let hir = rokajit::pipeline::import(&info, &ee).expect("imports");
        let hir = rokajit::pipeline::morph(hir).expect("morphs");
        let lir = rokajit::pipeline::lower(hir, &crate::X64Target).expect("lowers");
        let out = rokajit::pipeline::codegen(&lir, &ee, &crate::X64Target, Tier::Tier0)
            .expect("tier-0 codegen");

        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x55, // pushq %rbp
            0x48, 0x89, 0xE5, // movq %rsp, %rbp
            0x48, 0x83, 0xEC, 0x20, // subq $32, %rsp
            0x48, 0x89, 0x7D, 0xF8, // movq %rdi, -8(%rbp)  — arg this
            0x48, 0x89, 0x75, 0xF0, // movq %rsi, -16(%rbp) — arg v
            0x48, 0xC7, 0x45, 0xE8, 0, 0, 0, 0, // movq $0, -24(%rbp) — ByRef temp root
            0x48, 0x8B, 0x45, 0xF8, // movq -8(%rbp), %rax
            0x8B, 0x08, // movl (%rax), %ecx           — the null check
            0x48, 0x8B, 0x55, 0xF8, // movq -8(%rbp), %rdx — &this.r:
            0x48, 0x83, 0xC2, 0x18, // addq $24, %rdx
            0x48, 0x89, 0x55, 0xE8, // movq %rdx, -24(%rbp) — ByRef root at def
            0x48, 0x8B, 0x7D, 0xE8, // movq -24(%rbp), %rdi — barrier arg 0 (dst)
            0x48, 0x8B, 0x75, 0xF0, // movq -16(%rbp), %rsi — barrier arg 1 (src)
            0xE8, 0, 0, 0, 0, // call CHECKED_ASSIGN_REF (patched by 07.7)
            0xC9, 0xC3, // leave; ret
        ];
        assert_eq!(out.code.hot.bytes, expected);

        // The barrier call is a helper safepoint (no method/sig recorded);
        // all three GC slots are roots, the ByRef temp flagged as such.
        assert_eq!(out.call_sites.len(), 1);
        assert_eq!(out.call_sites[0].offset, 50);
        assert_eq!(out.call_sites[0].method, None);
        assert_eq!(out.call_sites[0].sig, None);
        assert_eq!(out.relocations.len(), 1);
        assert_eq!(out.relocations[0].offset, 51);
        assert_eq!(
            out.frame.gc_roots,
            vec![
                rokajit::pipeline::GcRootSlot {
                    offset: 8,
                    is_byref: false,
                    pinned: false,
                },
                rokajit::pipeline::GcRootSlot {
                    offset: 16,
                    is_byref: false,
                    pinned: false,
                },
                rokajit::pipeline::GcRootSlot {
                    offset: 24,
                    is_byref: true,
                    pinned: false,
                },
            ]
        );
    }
}
