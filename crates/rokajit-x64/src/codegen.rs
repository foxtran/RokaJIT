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
//! - **Calls**: all register-resident temps spill before the `call`
//!   (the pool is caller-saved); the target address is resolved through
//!   the EE at emit time and recorded as a `RELATIVE32` relocation plus
//!   a managed call site (a GC safepoint) for 07.7 to drain.

use rokajit::artifact::{CallSite, ChunkRef, CodeChunk, CodeChunks, Relocation};
use rokajit::codegen::{Loc, Move, ReadSrc, ValueState};
use rokajit::error::{CompileError, CompileResult};
use rokajit::ir::lir::StmtKind;
use rokajit::ir::{hir, lir, CallSig, LocalId, Type, UnaryOp};
use rokajit::lower::{Cx, Label};
use rokajit::pipeline::{CodegenOutput, FrameInfo};
use rokajit::target::{ArgLocation, CallAbi, PhysReg};
use rokajit_ee::ee_info::{const_lookup_addr, const_lookup_slot, EeInfo};
use rokajit_ee::enums::RelocType;
use rokajit_ee::handles::MethodHandle;

use crate::encode::{Asm, Mem, Rm, Rmi};
use crate::inst::{ArithOp, Inst, Place, ShiftOp, Src, Width};
use crate::lower::{lower_frame, lower_stmt, width_of_ty, FrameReq};
use crate::regs::{self, Gpr};

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
    const BY_INDEX: [Gpr; 16] = [
        Gpr::Rax,
        Gpr::Rcx,
        Gpr::Rdx,
        Gpr::Rbx,
        Gpr::Rsp,
        Gpr::Rbp,
        Gpr::Rsi,
        Gpr::Rdi,
        Gpr::R8,
        Gpr::R9,
        Gpr::R10,
        Gpr::R11,
        Gpr::R12,
        Gpr::R13,
        Gpr::R14,
        Gpr::R15,
    ];
    BY_INDEX
        .get(reg.0 as usize)
        .copied()
        .ok_or(CompileError::Internal("PhysReg outside the GPR range"))
}

/// The SysV AMD64 argument/return assignment for one call (the
/// [`rokajit::target::Target::classify_call`] body). Integer-class values
/// (pointers included) take [`regs::INT_ARG_REGS`] in order, floats take
/// [`regs::FLOAT_ARG_REGS`]; overflow goes to 8-byte stack slots.
/// Aggregates are outside the tier-0 subset.
pub fn classify_call(sig: &CallSig) -> CompileResult<CallAbi> {
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
        )?);
    }
    for &ty in &sig.args {
        args.push(place_arg(ty, &mut int_n, &mut float_n, &mut stack_bytes)?);
    }
    let ret = match sig.ret {
        Type::Void => None,
        Type::Float | Type::Double => Some(ArgLocation::Reg(regs::FLOAT_RETURN_REG.phys())),
        Type::Struct(_) => {
            return Err(CompileError::Unsupported(
                "struct return values: outside the tier-0 subset",
            ));
        }
        _ => Some(ArgLocation::Reg(regs::INT_RETURN_REGS[0].phys())),
    };
    Ok(CallAbi {
        args,
        ret,
        stack_arg_bytes: stack_bytes.div_ceil(16) * 16,
    })
}

fn place_arg(
    ty: Type,
    int_n: &mut usize,
    float_n: &mut usize,
    stack_bytes: &mut u32,
) -> CompileResult<ArgLocation> {
    let is_float = match ty {
        Type::Int32 | Type::Int64 | Type::NativeInt | Type::Ref | Type::ByRef => false,
        Type::Float | Type::Double => true,
        Type::Struct(_) => {
            return Err(CompileError::Unsupported(
                "struct arguments: outside the tier-0 subset",
            ));
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

/// The frame layout: every local/arg/temp's slot offset below `rbp`, plus
/// the 16-aligned frame size (the `sub rsp, N` immediate). Slot `i`'s
/// bytes are `[rbp - slots[i], rbp - slots[i] + size)` — so `slots[i]` is
/// the offset [`rokajit::pipeline::GcRootSlot`] records.
pub struct FrameLayout {
    pub slots: Vec<u32>,
    pub frame_size: u32,
}

impl FrameLayout {
    pub fn compute(method: &lir::Method) -> CompileResult<FrameLayout> {
        let mut offset = 0u32;
        let mut slots = Vec::with_capacity(method.locals.len());
        for local in &method.locals {
            let (size, align) = match local.ty {
                Type::Int32 | Type::Float => (4, 4),
                Type::Int64 | Type::NativeInt | Type::Ref | Type::ByRef | Type::Double => (8, 8),
                Type::Struct(_) => {
                    return Err(CompileError::Unsupported(
                        "struct local in tier-0 frame layout",
                    ));
                }
                Type::Void => return Err(CompileError::Internal("void-typed local")),
            };
            offset = offset.div_ceil(align) * align + size;
            slots.push(offset);
        }
        Ok(FrameLayout {
            slots,
            frame_size: offset.div_ceil(16) * 16,
        })
    }
}

/// Stage 4 emission for x64 (the `Target::emit_tier0` body): LIR →
/// machine-code bytes plus the relocation/call-site/frame facts 07.7
/// drains. Single pass over the blocks in layout order.
pub fn emit_tier0(method: &lir::Method, ee: &dyn EeInfo) -> CompileResult<CodegenOutput> {
    let layout = FrameLayout::compute(method)?;
    let cx = Cx::new(&method.locals);
    let mut em = Emitter::new(method, layout, ee);

    let prolog = lower_frame(&FrameReq, &cx).ok_or(CompileError::Internal(
        "the catch-all frame rule must match",
    ))?;
    for inst in &prolog {
        em.emit_inst(inst)?;
    }
    em.spill_incoming_args()?;
    em.zero_init_slots();

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
            for inst in &insts {
                em.emit_inst(inst)?;
            }
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
            gc_roots: rokajit::codegen::gc_roots(&method.locals, &em.layout.slots),
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
            num_args: method.num_args as usize,
            num_frame_fixed: (method.num_args + method.num_il_locals) as usize,
            ee,
            call_sig: None,
            call_sites: Vec::new(),
            relocations: Vec::new(),
        }
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

    /// A descriptor source as the `r/m` side: immediates have no such
    /// encoding, so they materialize into a scratch register.
    fn rm_of(&mut self, src: Src, width: Width, exclude: &[PhysReg]) -> CompileResult<Rm> {
        match self.rmi_of(src)? {
            Rmi::Reg(g) => Ok(Rm::Reg(g)),
            Rmi::Mem(m) => Ok(Rm::Mem(m)),
            Rmi::Imm(i) => {
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

    /// Incoming argument registers → their frame slots, per the SysV
    /// classification of the method's own argument list (args include the
    /// receiver already, so the synthesized signature is `has_this: false`).
    fn spill_incoming_args(&mut self) -> CompileResult<()> {
        if self.num_args == 0 {
            return Ok(());
        }
        let sig = CallSig {
            ret: Type::Void,
            args: self.locals[..self.num_args].iter().map(|l| l.ty).collect(),
            has_this: false,
        };
        let abi = classify_call(&sig)?;
        for (i, loc) in abi.args.iter().enumerate() {
            let id = LocalId(i as u32);
            let width = self.width_of(id)?;
            match *loc {
                ArgLocation::Reg(phys) => {
                    let reg = gpr_of(phys).map_err(|_| {
                        CompileError::Unsupported("float argument in tier-0 codegen")
                    })?;
                    self.asm
                        .mov(width, Rm::Mem(self.own_slot(id)), Rmi::Reg(reg));
                }
                ArgLocation::Stack { .. } => {
                    return Err(CompileError::Unsupported(
                        "incoming stack arguments (more than six integer arguments)",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Zero the slots whose contents must be defined from the first
    /// safepoint on: IL locals (the init-locals flag does not reach LIR;
    /// zeroing unconditionally is always permitted) and GC-typed temps
    /// (reported roots, so their slots must hold a valid value even before
    /// the temp's first definition).
    fn zero_init_slots(&mut self) {
        for (i, local) in self.locals.iter().enumerate() {
            let is_il_local = i >= self.num_args && i < self.num_frame_fixed;
            let is_gc_temp =
                i >= self.num_frame_fixed && matches!(local.ty, Type::Ref | Type::ByRef);
            if !(is_il_local || is_gc_temp) {
                continue;
            }
            // A float IL local has no GPR move form; it is never consumed
            // (lowering rejects float operations), so its slot stays cold.
            let Some(width) = width_of_ty(local.ty) else {
                continue;
            };
            self.asm.mov(
                width,
                Rm::Mem(self.slot_mem(self.layout.slots[i])),
                Rmi::Imm(0),
            );
        }
    }

    fn emit_inst(&mut self, inst: &Inst) -> CompileResult<()> {
        match *inst {
            Inst::Mov { width, dst, src } => self.emit_mov(width, dst, src),
            Inst::Lea { dst, addr } => self.emit_lea(dst, addr),
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
            Inst::Jcc { cc, target } => {
                self.asm.jcc(cc, target);
                Ok(())
            }
            Inst::Jmp { target } => {
                self.asm.jmp(target);
                Ok(())
            }
            Inst::CallDirect { method } => self.emit_call(method),
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
        match self.rmi_of(src)? {
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
        // `imul` by a constant: the three-operand form folds the move.
        if op == ArithOp::Imul {
            if let Src::Imm(i) = rhs {
                let lhs_rm = self.rm_of(lhs, width, &[d])?;
                self.asm.imul_imm(width, dg, lhs_rm, i);
                return self.define_temp_reg(t.0, dg);
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
                let rhs = self.rmi_of(rhs)?;
                self.asm.add(width, Rm::Reg(dg), rhs);
            }
            ArithOp::Sub => {
                let rhs = self.rmi_of(rhs)?;
                self.asm.sub(width, Rm::Reg(dg), rhs);
            }
            ArithOp::And => {
                let rhs = self.rmi_of(rhs)?;
                self.asm.and(width, Rm::Reg(dg), rhs);
            }
            ArithOp::Or => {
                let rhs = self.rmi_of(rhs)?;
                self.asm.or(width, Rm::Reg(dg), rhs);
            }
            ArithOp::Xor => {
                let rhs = self.rmi_of(rhs)?;
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
        let lhs_rmi = self.rmi_of(lhs)?;
        let rhs_rmi = self.rmi_of(rhs)?;
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
        let moves = self.vs.spill_registers();
        self.apply(moves)?;
        let lookup = self.ee.get_function_entry_point(method);
        let instr_offset = self.asm.offset();
        let (size, reloc_offset, target) = if let Some(target) = const_lookup_addr(&lookup) {
            self.asm.call(method);
            (DIRECT_CALL_LEN, instr_offset + 1, target)
        } else if let Some(slot) = const_lookup_slot(&lookup) {
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
            sig: self.call_sig.clone(),
            method: Some(method),
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
        let abi = classify_call(&sig).expect("classifies");
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
        let abi = classify_call(&sig).expect("classifies");
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
        let abi = classify_call(&sig).expect("classifies");
        assert_eq!(abi.args[5], ArgLocation::Reg(Gpr::R9.phys()));
        assert_eq!(abi.args[6], ArgLocation::Stack { offset: 0 });
        assert_eq!(abi.args[7], ArgLocation::Stack { offset: 8 });
        assert_eq!(abi.stack_arg_bytes, 16, "16-aligned outgoing area");
        assert_eq!(abi.ret, None);
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
        let layout = FrameLayout::compute(&m).expect("layout");
        assert_eq!(layout.slots, vec![4, 16, 20]);
        assert_eq!(layout.frame_size, 32);
    }

    #[test]
    fn struct_locals_are_rejected_at_frame_layout() {
        let class = rokajit_ee::handles::ClassHandle::from_raw(std::ptr::dangling_mut()).unwrap();
        let m = method(
            vec![local(Type::Struct(class), LocalKind::IlLocal(0))],
            0,
            1,
            vec![],
        );
        assert!(matches!(
            FrameLayout::compute(&m),
            Err(CompileError::Unsupported(_))
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

    /// More than six integer arguments: the seventh arrives on the stack,
    /// which tier-0 emission does not cover — a clean `Unsupported`.
    #[test]
    fn incoming_stack_args_are_unsupported() {
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
        assert!(matches!(
            emit_tier0(&m, &MockEe::default()),
            Err(CompileError::Unsupported(_))
        ));
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
}
