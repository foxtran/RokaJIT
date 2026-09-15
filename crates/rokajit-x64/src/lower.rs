//! x64 lowering rules: LIR statements → [`Inst`] descriptors, written in
//! the `lower_rules!` DSL (docs/JITs/README.md verdict 4 — rules as
//! data, one declarative pattern per rule, no hand-written `match` over
//! opcodes with inline emission logic).
//!
//! Entry points:
//!
//! - [`lower_stmt`] — the statement ruleset; codegen (step_07.5) drives
//!   it per LIR statement. `None` means "no rule matched", which
//!   [`lower_method`] maps to `CompileError::Unsupported`.
//! - [`lower_frame`] — the frame ruleset: the prolog descriptor
//!   sequence. Matching on [`FrameReq`] (rather than a bare constructor)
//!   is the declared extension point for frame-shape decisions (callee-
//!   saved traffic, stack probes) when they arrive.
//! - [`lower_method`] — convenience whole-method driver for tests and
//!   codegen: prolog + per-block descriptor sequences.
//!
//! The rules produce pre-allocation descriptors: value operands stay
//! [`Val`]s; physical registers appear only where the SysV ABI or the
//! architecture pins them (arg moves per the [`crate::codegen::classify_call`]
//! assignment, returns through `rax`/`xmm0`, the `idiv` fixed-register
//! sequence, the `rbp`-based frame contract from `regs.rs`).

use rokajit::error::{CompileError, CompileResult};
use rokajit::ir::lir::{BranchCond, Operand, StmtKind::*};
use rokajit::ir::{lir, BinaryOp, BlockId, CallSig, Const, LocalId, MemAccess, Type};
use rokajit::lower::{Cx, Label, Val};
use rokajit::target::ArgLocation;

use crate::inst::{
    Amode, ArithFOp, ArithOp, BlockAddr, CondCode, EbReg, FWidth, Inst, Place, ShiftOp, Src, Width,
    XmmPlace, XmmSrc,
};
use crate::regs::{self, Gpr, Xmm};

/// The whole lowered method: the prolog sequence plus one descriptor
/// sequence per block, in layout order. Epilogs are per-block, emitted
/// by the return rules (the morph-certified multi-exit shape).
#[derive(Debug)]
pub struct LoweredMethod {
    pub prolog: Vec<Inst>,
    pub blocks: Vec<LoweredBlock>,
}

#[derive(Debug)]
pub struct LoweredBlock {
    pub id: BlockId,
    pub insts: Vec<Inst>,
}

// --- external extractors (plain functions; the DSL owns no IR) ---

/// The machine width a value of IR type `ty` occupies in a GPR, or
/// `None` when the type does not live in GPRs (floats: XMM, out of the
/// lowering subset; structs/void: no class at all).
pub(crate) fn width_of_ty(ty: Type) -> Option<Width> {
    match ty {
        Type::Int32 => Some(Width::W32),
        Type::Int64 | Type::NativeInt | Type::Ref | Type::ByRef => Some(Width::W64),
        Type::Float | Type::Double | Type::Struct(_) | Type::Void => None,
    }
}

/// `Some(())` when the memory access is the type's natural width — a
/// guard helper so the natural-width rules and the narrow/float field
/// rules stay disjoint.
fn natural_only(access: MemAccess) -> Option<()> {
    access.is_natural().then_some(())
}

/// The width of a local/arg/temp slot.
fn width_of(cx: &Cx, id: LocalId) -> Option<Width> {
    width_of_ty(cx.ty_of(id)?)
}

/// An integer constant as an immediate payload plus its width.
fn const_imm(k: Const) -> Option<(i64, Width)> {
    match k {
        Const::Int32(v) => Some((i64::from(v), Width::W32)),
        Const::Int64(v) => Some((v, Width::W64)),
        Const::NativeInt(v) => Some((v as i64, Width::W64)),
        Const::NullRef => Some((0, Width::W64)),
        Const::FrozenRef(v) => Some((v as i64, Width::W64)),
        Const::Float(_) | Const::Double(_) => None,
    }
}

/// Any LIR operand as an instruction source. `AddrOf` is not a `Src`:
/// an address materializes through `Lea` (the `copy_addr_of` rule),
/// never as an operand.
fn operand_src(op: Operand) -> Option<Src> {
    match op {
        Operand::Temp(id) | Operand::Local(id) => Some(Src::Val(Val(id))),
        Operand::Const(k) => Some(Src::Imm(const_imm(k)?.0)),
        Operand::AddrOf(_) => None,
    }
}

/// An LIR operand that is a plain value (temp/local) — constants and
/// addresses are claimed by the earlier, more specific rules.
fn value_src(op: Operand) -> Option<Src> {
    match op {
        Operand::Temp(id) | Operand::Local(id) => Some(Src::Val(Val(id))),
        Operand::Const(_) | Operand::AddrOf(_) => None,
    }
}

/// The width an operand carries: from the locals table for slots, from
/// the constant itself for immediates.
fn operand_width(cx: &Cx, op: Operand) -> Option<Width> {
    match op {
        Operand::Temp(id) | Operand::Local(id) => width_of(cx, id),
        Operand::Const(k) => Some(const_imm(k)?.1),
        Operand::AddrOf(_) => Some(Width::W64),
    }
}

// --- float extractors (step_10.2) ---

/// The SSE width of a local/arg/temp slot's float type.
fn fwidth_of(cx: &Cx, id: LocalId) -> Option<FWidth> {
    FWidth::of(cx.ty_of(id)?)
}

/// A float constant as its bit pattern plus its SSE width.
fn const_float(k: Const) -> Option<(u64, FWidth)> {
    match k {
        Const::Float(v) => Some((u64::from(v.to_bits()), FWidth::S)),
        Const::Double(v) => Some((v.to_bits(), FWidth::D)),
        _ => None,
    }
}

/// Any LIR operand as an XMM-side source: slots stay [`XmmSrc::Val`]
/// (float values are always frame-resident, so codegen reads a slot),
/// constants carry their bit pattern.
fn xmm_opnd(cx: &Cx, op: Operand) -> Option<XmmSrc> {
    match op {
        Operand::Temp(id) | Operand::Local(id) => {
            fwidth_of(cx, id)?;
            Some(XmmSrc::Val(Val(id)))
        }
        Operand::Const(k) => Some(XmmSrc::Bits(const_float(k)?.0)),
        Operand::AddrOf(_) => None,
    }
}

/// The SSE width a float operand carries (the operand must be
/// float-typed; used as a rule guard).
fn operand_fwidth(cx: &Cx, op: Operand) -> Option<FWidth> {
    match op {
        Operand::Temp(id) | Operand::Local(id) => fwidth_of(cx, id),
        Operand::Const(k) => Some(const_float(k)?.1),
        Operand::AddrOf(_) => None,
    }
}

/// The address operand of a struct value (step_10.9): `AddrOf` names a
/// frame slot directly; a byref temp/local is a pointer value.
fn block_addr(op: Operand) -> Option<BlockAddr> {
    match op {
        Operand::AddrOf(l) => Some(BlockAddr::FrameSlot(l)),
        Operand::Temp(id) | Operand::Local(id) => Some(BlockAddr::Val(Val(id))),
        Operand::Const(_) => None,
    }
}

/// The fixed register of one struct eightbyte, as an [`EbReg`].
fn eb_reg(phys: rokajit::target::PhysReg) -> Option<EbReg> {
    if let Some(g) = Gpr::from_phys(phys) {
        Some(EbReg::Gpr(g))
    } else {
        Xmm::from_phys(phys).map(EbReg::Xmm)
    }
}

/// One argument's setup moves for a classified ABI location (step_10.9:
/// several instructions for structs and stack arguments).
fn arg_move(cx: &Cx, arg: &Operand, loc: &ArgLocation) -> Option<Vec<Inst>> {
    match *loc {
        ArgLocation::Reg(phys) => {
            if let Some(g) = Gpr::from_phys(phys) {
                Some(vec![match arg {
                    Operand::AddrOf(l) => Inst::Lea {
                        dst: Place::Reg(g),
                        addr: Amode::FrameSlot(*l),
                    },
                    _ => Inst::Mov {
                        width: operand_width(cx, *arg)?,
                        dst: Place::Reg(g),
                        src: operand_src(*arg)?,
                    },
                }])
            } else {
                let x = Xmm::from_phys(phys)?;
                Some(vec![match arg {
                    Operand::Const(k) => {
                        let (bits, width) = const_float(*k)?;
                        Inst::ConstF {
                            width,
                            dst: XmmPlace::Reg(x),
                            bits,
                        }
                    }
                    _ => Inst::MovF {
                        width: operand_fwidth(cx, *arg)?,
                        dst: XmmPlace::Reg(x),
                        src: xmm_opnd(cx, *arg)?,
                    },
                }])
            }
        }
        ArgLocation::Stack { offset } => {
            // Outgoing stack arguments (SysV register-pool overflow; the
            // operand is a scalar here — structs arrive as StructRegs or
            // a struct-classified Stack, which `arg_moves` splits out).
            if let Some(width) = operand_width(cx, *arg) {
                let mut insts = Vec::new();
                let src = match arg {
                    Operand::AddrOf(l) => {
                        // An address constant has no `Src` form:
                        // materialize through the fixed scratch r11 (never
                        // an argument register).
                        insts.push(Inst::Lea {
                            dst: Place::Reg(Gpr::R11),
                            addr: Amode::FrameSlot(*l),
                        });
                        Src::Reg(Gpr::R11)
                    }
                    _ => operand_src(*arg)?,
                };
                insts.push(Inst::StoreStackArg { width, offset, src });
                Some(insts)
            } else {
                let width = operand_fwidth(cx, *arg)?;
                Some(vec![Inst::StoreStackArgF {
                    width,
                    offset,
                    src: xmm_opnd(cx, *arg)?,
                }])
            }
        }
        ArgLocation::StructRegs {
            regs,
            count,
            sizes,
            offsets,
        } => {
            // A register-passed struct argument: one exact-size load per
            // eightbyte from the value's memory.
            let addr = block_addr(*arg)?;
            let mut insts = Vec::with_capacity(count as usize);
            for k in 0..count as usize {
                insts.push(Inst::LoadEightbyte {
                    addr,
                    disp: u32::from(offsets[k]),
                    size: sizes[k],
                    dst: eb_reg(regs[k])?,
                });
            }
            Some(insts)
        }
    }
}

/// The per-ABI argument setup for a call: moves per argument into its
/// SysV location, per the [`crate::codegen::classify_call`] assignment
/// (entry 0 is the implicit `this` when present). A struct argument the
/// ABI placed on the stack (whole-struct rule or never-classified) is a
/// block copy into the outgoing area; its size comes from the signature
/// and the layout side table.
fn arg_moves(cx: &Cx, sig: &CallSig, args: &[Operand]) -> Option<Vec<Inst>> {
    let abi = crate::codegen::classify_call(sig, cx.layouts()).ok()?;
    let mut insts = Vec::new();
    for (i, (arg, loc)) in args.iter().zip(&abi.args).enumerate() {
        // The argument's IR type (entry 0 is `this` when present): a
        // struct type with a Stack location is the stack-passed struct
        // case.
        let ty = if sig.has_this {
            if i == 0 {
                None
            } else {
                Some(sig.args[i - 1])
            }
        } else {
            Some(sig.args[i])
        };
        match (ty, loc) {
            // A struct argument the ABI placed on the stack (whole-struct
            // rule or never-classified): a block copy into the outgoing
            // area, sized from the layout side table.
            (Some(Type::Struct(class)), ArgLocation::Stack { offset }) => {
                let size = cx.layout_of(class)?.size;
                insts.push(Inst::CopyStackArg {
                    addr: block_addr(*arg)?,
                    offset: *offset,
                    size,
                });
            }
            (Some(Type::Struct(_)), ArgLocation::StructRegs { .. }) => {
                insts.extend(arg_move(cx, arg, loc)?);
            }
            (Some(Type::Struct(_)), _) => return None,
            _ => insts.extend(arg_move(cx, arg, loc)?),
        }
    }
    Some(insts)
}

/// The result moves after a call, per the ABI return location (`rax` for
/// integers, `xmm0` for floats, the eightbyte registers for a
/// register-passed struct — stored into the destination slot with their
/// exact sizes; step_10.9). A non-register-passed struct call has no
/// destination (the retbuf temp received the value), so it never reaches
/// here.
fn call_result_move(cx: &Cx, sig: &CallSig, dst: LocalId) -> Option<Vec<Inst>> {
    let abi = crate::codegen::classify_call(sig, cx.layouts()).ok()?;
    match abi.ret {
        Some(ArgLocation::Reg(phys)) => {
            if let Some(g) = Gpr::from_phys(phys) {
                Some(vec![Inst::Mov {
                    width: width_of(cx, dst)?,
                    dst: Place::Val(Val(dst)),
                    src: Src::Reg(g),
                }])
            } else {
                let x = Xmm::from_phys(phys)?;
                Some(vec![Inst::MovF {
                    width: fwidth_of(cx, dst)?,
                    dst: XmmPlace::Val(Val(dst)),
                    src: XmmSrc::Reg(x),
                }])
            }
        }
        Some(ArgLocation::StructRegs {
            regs,
            count,
            sizes,
            offsets,
        }) => {
            let mut insts = Vec::with_capacity(count as usize);
            for k in 0..count as usize {
                insts.push(Inst::StoreEightbyte {
                    local: dst,
                    offset: u32::from(offsets[k]),
                    size: sizes[k],
                    src: eb_reg(regs[k])?,
                });
            }
            Some(insts)
        }
        _ => None,
    }
}

/// The per-block epilog, shared by both return rules. Matches the
/// prolog the `standard_frame` rule emits: `leave` undoes
/// `push rbp; mov rbp, rsp; sub rsp, frame`.
fn epilog() -> Vec<Inst> {
    vec![Inst::Leave, Inst::Ret]
}

rokajit::lower_rules! {
    /// LIR statement → x64 instruction descriptors, for the fib subset,
    /// the step_10.1 scalar-cheap pack, the step_10.2 float pack, and the
    /// step_10.4 object pack (loads/stores through computed addresses,
    /// the trap-based null check). Rules are tried in
    /// declaration order; the more specific operand shapes (constants,
    /// addresses) precede the general value rules.
    pub fn lower_stmt(stmt: &lir::Stmt, cx: &Cx<'_>) -> Option<Vec<Inst>>
    matching &stmt.kind;

    /// `t := C` — one `mov` of an immediate, width from the constant's
    /// own type.
    rule copy_const: Copy { dst, src: Operand::Const(k) }
        if let Some((imm, w)) = const_imm(*k)
        => |_| vec![Inst::Mov {
            width: w,
            dst: Place::Val(Val(*dst)),
            src: Src::Imm(imm),
        }];

    /// `t := f` — a float constant: the bit pattern materializes through
    /// a GPR at codegen (SSE has no immediate forms).
    rule copy_const_f: Copy { dst, src: Operand::Const(k) }
        if let (Some(w), Some((bits, _))) = (fwidth_of(cx, *dst), const_float(*k))
        => |_| vec![Inst::ConstF {
            width: w,
            dst: XmmPlace::Val(Val(*dst)),
            bits,
        }];

    /// `t := &local` — `lea` against the local's (symbolic) frame slot.
    rule copy_addr_of: Copy { dst, src: Operand::AddrOf(l) }
        => |_| vec![Inst::Lea {
            dst: Place::Val(Val(*dst)),
            addr: Amode::FrameSlot(*l),
        }];

    /// `t := v` — a value-to-value `mov`, width from the destination
    /// slot's type.
    rule copy: Copy { dst, src }
        if let (Some(w), Some(s)) = (width_of(cx, *dst), value_src(*src))
        => |_| vec![Inst::Mov {
            width: w,
            dst: Place::Val(Val(*dst)),
            src: s,
        }];

    /// `t := v` for float values — `movss`/`movsd`.
    rule copy_f: Copy { dst, src }
        if let (Some(w), Some(s)) = (fwidth_of(cx, *dst), xmm_opnd(cx, *src))
        => |_| vec![Inst::MovF {
            width: w,
            dst: XmmPlace::Val(Val(*dst)),
            src: s,
        }];

    /// `t := [addr + disp]` — `ldfld`'s load through the (already
    /// null-checked) object reference (step_10.4). The address operand is
    /// usually a frame-resident GC ref; codegen loads it into a scratch
    /// GPR first. Natural-width GPR types only: floats match
    /// `load_mem_f`, sub-Int32 fields match `load_mem_narrow`.
    rule load_mem: Load { dst, addr, offset, ty, access }
        if let (Some(w), Some(a), Some(())) = (
            width_of_ty(*ty),
            operand_src(*addr),
            natural_only(*access),
        )
        => |_| vec![Inst::LoadMem {
            width: w,
            dst: Place::Val(Val(*dst)),
            addr: a,
            disp: *offset as i32,
        }];

    /// `t := [addr + disp]` — a float field load (`movss`/`movsd`).
    rule load_mem_f: Load { dst, addr, offset, ty, access }
        if let (Some(w), Some(a), Some(())) = (
            FWidth::of(*ty),
            operand_src(*addr),
            natural_only(*access),
        )
        => |_| vec![Inst::LoadMemF {
            width: w,
            dst: XmmPlace::Val(Val(*dst)),
            addr: a,
            disp: *offset as i32,
        }];

    /// `t := [addr + disp]` — a sub-Int32 field load: `size` bytes,
    /// zero- or sign-extended per the field's metadata type.
    rule load_mem_narrow: Load { dst, addr, offset, access, .. }
        if let (Some(size), Some(a)) = (access.narrow_bytes(), operand_src(*addr))
        => |_| vec![Inst::LoadMemNarrow {
            size,
            signed: access.sign_extends(),
            dst: Place::Val(Val(*dst)),
            addr: a,
            disp: *offset as i32,
        }];

    /// `[addr + disp] := src` — `stfld`'s store through the (already
    /// null-checked) object reference. A reference-typed store never
    /// reaches here: the importer routes it through the write-barrier
    /// helper call. The width is the source operand's. Natural-width GPR
    /// sources only: floats match `store_mem_f`, sub-Int32 fields match
    /// `store_mem_narrow`.
    rule store_mem: Store { addr, offset, src, access }
        if let (Some(a), Some(s), Some(w), Some(())) = (
            operand_src(*addr),
            operand_src(*src),
            operand_width(cx, *src),
            natural_only(*access),
        )
        => |_| vec![Inst::StoreMem {
            width: w,
            addr: a,
            disp: *offset as i32,
            src: s,
        }];

    /// `[addr + disp] := src` — a float field store (`movss`/`movsd`).
    rule store_mem_f: Store { addr, offset, src, access }
        if let (Some(a), Some(s), Some(w), Some(())) = (
            operand_src(*addr),
            xmm_opnd(cx, *src),
            operand_fwidth(cx, *src),
            natural_only(*access),
        )
        => |_| vec![Inst::StoreMemF {
            width: w,
            addr: a,
            disp: *offset as i32,
            src: s,
        }];

    /// `[addr + disp] := src_low` — a sub-Int32 field store: only the
    /// low `size` bytes of the Int32 source write to memory.
    rule store_mem_narrow: Store { addr, offset, src, access }
        if let (Some(size), Some(a), Some(s)) = (
            access.narrow_bytes(),
            operand_src(*addr),
            operand_src(*src),
        )
        => |_| vec![Inst::StoreMemNarrow {
            size,
            addr: a,
            disp: *offset as i32,
            src: s,
        }];

    /// The trap-based null check (step_10.4): a throwaway 32-bit load
    /// through the reference — on null the hardware fault is the
    /// NullReferenceException, via the EE's signal translation (the same
    /// path `idiv`'s #DE rides, covered by the 07.7 unwind info).
    rule null_check: NullCheck { arg }
        if let Some(a) = operand_src(*arg)
        => |_| vec![Inst::NullCheck { addr: a }];

    /// The array bounds check (step_10.8): the only array-aware
    /// descriptor — the length load, the unsigned compare, and the
    /// conditional RNGCHKFAIL helper call. Element addressing never
    /// reaches here (the flattener already expanded it to plain
    /// address arithmetic). The compare widens to 64 bits for a
    /// native-int index (RyuJIT widens to TYP_I_IMPL, morph.cpp:3022).
    rule bounds_check: BoundsCheck { array, index }
        if let (Some(a), Some(i), Some(w)) = (
            operand_src(*array),
            operand_src(*index),
            operand_width(cx, *index),
        )
        => |_| vec![Inst::BoundsCheck {
            index: i,
            index_wide: matches!(w, Width::W64),
            array: a,
        }];

    /// `t := a + b`, `a - b`, `a * b` — one three-operand descriptor;
    /// codegen emits the destructive two-operand pair. Constants ride
    /// along as immediates (`Src::Imm`); the encoder picks the imm form.
    rule arith: Binary { dst, op, lhs, rhs }
        if let (Some(aop), Some(w), Some(l), Some(r)) = (
            ArithOp::of(*op),
            width_of(cx, *dst),
            operand_src(*lhs),
            operand_src(*rhs),
        )
        => |_| vec![Inst::Arith {
            op: aop,
            width: w,
            dst: Place::Val(Val(*dst)),
            lhs: l,
            rhs: r,
        }];

    /// `t := a + b` … `a / b` on floats — the scalar SSE forms
    /// (`addss`/`addsd`/…). (`rem` on floats never reaches here: the
    /// importer expands it to the `CORINFO_HELP_FLTREM`/`DBLREM` call,
    /// RyuJIT's morph.cpp GT_MOD lowering.)
    rule arith_f: Binary { dst, op, lhs, rhs }
        if let (Some(aop), Some(w), Some(l), Some(r)) = (
            ArithFOp::of(*op),
            fwidth_of(cx, *dst),
            xmm_opnd(cx, *lhs),
            xmm_opnd(cx, *rhs),
        )
        => |_| vec![Inst::ArithF {
            op: aop,
            width: w,
            dst: XmmPlace::Val(Val(*dst)),
            lhs: l,
            rhs: r,
        }];

    /// `t := a % b` — the signed-remainder sequence through the
    /// architecture's fixed registers: `rax := a; cdq; idiv b;
    /// t := rdx`. The fixed duties are declared on the descriptors
    /// (`Inst::fixed_regs`) so codegen can keep allocated values clear.
    rule rem: Binary { dst, op: BinaryOp::Rem, lhs, rhs }
        if let (Some(w), Some(l), Some(r)) = (
            width_of(cx, *dst),
            operand_src(*lhs),
            operand_src(*rhs),
        )
        => |_| vec![
            Inst::Mov { width: w, dst: Place::Reg(Gpr::Rax), src: l },
            Inst::Cdq { width: w },
            Inst::Idiv { width: w, divisor: r },
            Inst::Mov { width: w, dst: Place::Val(Val(*dst)), src: Src::Reg(Gpr::Rdx) },
        ];

    /// `t := a / b` (signed `div`) — the same fixed-register sequence,
    /// quotient out of `rax`. `DivideByZeroException` and the
    /// `int.MinValue / -1` overflow come from the `idiv` hardware trap
    /// (#DE → the EE's signal translation via our unwind info); there is
    /// deliberately no explicit divisor check.
    rule div_s: Binary { dst, op: BinaryOp::Div, lhs, rhs }
        if let (Some(w), Some(l), Some(r)) = (
            width_of(cx, *dst),
            operand_src(*lhs),
            operand_src(*rhs),
        )
        => |_| vec![
            Inst::Mov { width: w, dst: Place::Reg(Gpr::Rax), src: l },
            Inst::Cdq { width: w },
            Inst::Idiv { width: w, divisor: r },
            Inst::Mov { width: w, dst: Place::Val(Val(*dst)), src: Src::Reg(Gpr::Rax) },
        ];

    /// `t := a / b`, `a % b` (unsigned `div.un`/`rem.un`) — `edx` is
    /// zeroed rather than sign-extended (a 32-bit `mov` clears all of
    /// `rdx`), then the unsigned `div` form. The quotient is `rax`, the
    /// remainder `rdx`.
    rule div_un: Binary { dst, op, lhs, rhs }
        if let (Some(w), Some(l), Some(r), true) = (
            width_of(cx, *dst),
            operand_src(*lhs),
            operand_src(*rhs),
            matches!(op, BinaryOp::UDiv | BinaryOp::URem),
        )
        => |_| {
            let result = if matches!(op, BinaryOp::UDiv) {
                Gpr::Rax
            } else {
                Gpr::Rdx
            };
            vec![
                Inst::Mov { width: w, dst: Place::Reg(Gpr::Rax), src: l },
                Inst::Mov { width: Width::W32, dst: Place::Reg(Gpr::Rdx), src: Src::Imm(0) },
                Inst::Div { width: w, divisor: r },
                Inst::Mov { width: w, dst: Place::Val(Val(*dst)), src: Src::Reg(result) },
            ]
        };

    /// `t := a + b` … `a * b`, checked (`add.ovf`/`sub.ovf`/`mul.ovf` and
    /// the `.un` forms) — one descriptor carrying the operator, width,
    /// and signedness; codegen owns the conditional OVERFLOW helper call
    /// (the `Inst::BoundsCheck` conditional-throw shape) and, for `Mul`,
    /// the one-operand `imul`/`mul` fixed-register sequence.
    rule arith_ovf: BinaryOvf { dst, op, unsigned, lhs, rhs }
        if let (Some(w), Some(l), Some(r), true) = (
            width_of(cx, *dst),
            operand_src(*lhs),
            operand_src(*rhs),
            matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul),
        )
        => |_| vec![Inst::ArithOvf {
            op: *op,
            unsigned: *unsigned,
            width: w,
            dst: Place::Val(Val(*dst)),
            lhs: l,
            rhs: r,
        }];

    /// `t := (checked) a` — `conv.ovf.*` from an integer source (LIR
    /// `ConvOvf`): one descriptor carrying the source width/signedness and
    /// the target range; codegen owns the range check and conditional
    /// OVERFLOW helper call. Same-width cells the importer proved
    /// no-check never reach here.
    rule conv_ovf: ConvOvf { dst, dst_bits, signed_dst, unsigned_src, src }
        if let (Some(w), Some(s)) = (operand_width(cx, *src), operand_src(*src))
        => |_| vec![Inst::ConvOvf {
            width_src: w,
            width_dst: *dst_bits,
            signed_dst: *signed_dst,
            unsigned_src: *unsigned_src,
            dst: Place::Val(Val(*dst)),
            src: s,
        }];

    /// `t := ckfinite a` — one descriptor; codegen emits the exponent-mask
    /// check with the conditional OVERFLOW helper call and copies the
    /// value through on the no-throw edge.
    rule ckfinite: CkFinite { dst, src }
        if let (Some(w), Some(s)) = (operand_fwidth(cx, *src), xmm_opnd(cx, *src))
        => |_| vec![Inst::CkFinite {
            width: w,
            dst: XmmPlace::Val(Val(*dst)),
            src: s,
        }];

    /// `t := a << b`, `a >> b` — one descriptor; codegen emits the
    /// destructive pair, with the count in `cl` when it isn't constant.
    /// The signedness flip (`shr` → `sar`, `shr.un` → `shr`) happens in
    /// `ShiftOp::of`.
    rule shift: Binary { dst, op, lhs, rhs }
        if let (Some(sop), Some(w), Some(l), Some(r)) = (
            ShiftOp::of(*op),
            width_of(cx, *dst),
            operand_src(*lhs),
            operand_src(*rhs),
        )
        => |_| vec![Inst::Shift {
            op: sop,
            width: w,
            dst: Place::Val(Val(*dst)),
            lhs: l,
            rhs: r,
        }];

    /// `t := -a`, `~a` — the one-operand complement forms.
    rule unary: Unary { dst, op, src }
        if let (Some(w), Some(s)) = (width_of(cx, *dst), operand_src(*src))
        => |_| vec![Inst::Unary {
            op: *op,
            width: w,
            dst: Place::Val(Val(*dst)),
            src: s,
        }];

    /// `t := -a` on floats — the sign-mask XOR (`not` has no float form;
    /// the importer rejects it).
    rule neg_f: Unary { dst, op: rokajit::ir::UnaryOp::Neg, src }
        if let (Some(w), Some(s)) = (fwidth_of(cx, *dst), xmm_opnd(cx, *src))
        => |_| vec![Inst::NegF {
            width: w,
            dst: XmmPlace::Val(Val(*dst)),
            src: s,
        }];

    /// `t := (a cmp b)` — compare-as-a-value (`ceq`/`cgt`/`clt`/…): the
    /// flags materialization of `branch_cmp`, with `setcc` consuming them
    /// into an Int32 0/1. Producer and consumer are adjacent by
    /// construction.
    rule cmp_value: Binary { dst, op, lhs, rhs }
        if let (Some(cc), Some(w), Some(l), Some(r)) = (
            CondCode::of(*op),
            operand_width(cx, *lhs),
            operand_src(*lhs),
            operand_src(*rhs),
        )
        => |_| vec![
            Inst::Cmp { width: w, lhs: l, rhs: r },
            Inst::Setcc { cc, dst: Place::Val(Val(*dst)) },
        ];

    /// `t := (a cmp b)` on floats — `ucomis*` plus the parity-aware
    /// materialization: the ordered forms (`ceq`/`cgt`/`clt`) are false
    /// when either operand is NaN, the `.un` forms true (ECMA-335 table
    /// III.4). The descriptor pair carries the *IR operator*; codegen owns
    /// the flag-sequence expansion.
    rule cmp_value_f: Binary { dst, op, lhs, rhs }
        if let (true, Some(w), Some(l), Some(r)) = (
            CondCode::of(*op).is_some(),
            operand_fwidth(cx, *lhs),
            xmm_opnd(cx, *lhs),
            xmm_opnd(cx, *rhs),
        )
        => |_| vec![
            Inst::CmpF { width: w, lhs: l, rhs: r },
            Inst::SetccF {
                op: *op,
                dst: Place::Val(Val(*dst)),
            },
        ];

    /// `conv.i4`/`conv.u4` from a 64-bit operand: a 32-bit `mov` keeps the
    /// low half. (Signedness is unobservable in a truncation, and on the
    /// evaluation stack both forms normalize to Int32.)
    rule conv_trunc: Conv { dst, to: Type::Int32, src, .. }
        if let (Some(Width::W64), Some(s)) = (operand_width(cx, *src), operand_src(*src))
        => |_| vec![Inst::Mov {
            width: Width::W32,
            dst: Place::Val(Val(*dst)),
            src: s,
        }];

    /// `conv.i8`/`conv.u8`/`conv.u` from a 32-bit operand: sign-
    /// (`movsxd`) vs zero-extension (a 32-bit `mov`, which clears the
    /// upper half) — another spot where the `unsigned` flag is the whole
    /// semantics. `conv.u` targets `NativeInt` (step_10.7's rider), a
    /// W64 slot exactly like Int64.
    rule conv_ext: Conv { dst, to, unsigned, src, .. }
        if let (true, Some(Width::W32), Some(s)) = (
            matches!(*to, Type::Int64 | Type::NativeInt),
            operand_width(cx, *src),
            operand_src(*src),
        )
        => |_| vec![Inst::MovExt {
            dst: Place::Val(Val(*dst)),
            src: s,
            signed: !unsigned,
        }];

    /// `conv.r.un` from a 64-bit operand (u64 → f32/f64): no SSE2
    /// unsigned conversion exists; codegen emits the branchy fixup
    /// expansion (`Inst::CvtU64ToF`). A 32-bit unsigned source never
    /// reaches here — the importer pre-zero-extends it to Int64, and the
    /// signed 64-bit conversion below converts it exactly.
    rule conv_u64_to_f: Conv { dst, to, unsigned, src, .. }
        if let (true, Some(w), Some(Width::W64), Some(s)) = (
            *unsigned,
            FWidth::of(*to),
            operand_width(cx, *src),
            operand_src(*src),
        )
        => |_| vec![Inst::CvtU64ToF {
            width: w,
            dst: XmmPlace::Val(Val(*dst)),
            src: s,
        }];

    /// `conv.r4`/`conv.r8` from an integer operand: `cvtsi2ss`/`cvtsi2sd`,
    /// the 32- or 64-bit form from the source's width. (Signed sources
    /// only — an unsigned 64-bit source is `conv_u64_to_f` above.)
    rule conv_i_to_f: Conv { dst, to, unsigned, src, .. }
        if let (false, Some(w), Some(src_w), Some(s)) = (
            *unsigned,
            FWidth::of(*to),
            operand_width(cx, *src),
            operand_src(*src),
        )
        => |_| vec![Inst::CvtIntToF {
            width: w,
            src_w64: matches!(src_w, Width::W64),
            dst: XmmPlace::Val(Val(*dst)),
            src: s,
        }];

    /// `conv.i4`/`i8`/`u4` from a float operand: `cvttss2si`/`cvttsd2si`,
    /// truncating toward zero; out-of-range/NaN yields the hardware's
    /// "integer indefinite" value. (RyuJIT saturates since .NET 9 —
    /// closing that gap for the *signed* forms is the `convfloat`
    /// follow-up; the IL-level *unsigned* float conversions never reach
    /// here, the HIR→LIR lowering expands them — step_10.11.) A 32-bit
    /// *unsigned* target still converts through the 64-bit form (values
    /// up to 2³²−1 exact; beyond is ECMA-unspecified), keeping the low
    /// half. `NativeInt` (conv.i, step_11.9) is the signed 64-bit form.
    rule conv_f_to_i: Conv { dst, to, unsigned, src, .. }
        if let (Some(w64), Some(w), Some(s)) = (
            match to {
                Type::Int32 => Some(*unsigned),
                Type::Int64 | Type::NativeInt => Some(true),
                _ => None,
            },
            operand_fwidth(cx, *src),
            xmm_opnd(cx, *src),
        )
        => |_| vec![Inst::CvtFToInt {
            src_width: w,
            dst_w64: w64,
            dst: Place::Val(Val(*dst)),
            src: s,
        }];

    /// `conv.r4`/`conv.r8` from a float operand of the other width:
    /// `cvtsd2ss`/`cvtss2sd`. (Same-width conversions are the identity;
    /// the importer drops them.)
    rule conv_f_to_f: Conv { dst, to, src, .. }
        if let (Some(w), Some(_), Some(s), true) = (
            FWidth::of(*to),
            operand_fwidth(cx, *src),
            xmm_opnd(cx, *src),
            // The other float width only (same-width is the identity).
            matches!((FWidth::of(*to), operand_fwidth(cx, *src)), (Some(a), Some(b)) if a != b),
        )
        => |_| vec![Inst::CvtFToF {
            to: w,
            dst: XmmPlace::Val(Val(*dst)),
            src: s,
        }];

    /// `if (a cmp b) goto L` — flag materialization: the compare and the
    /// conditional jump are one rule's output, so the flag producer and
    /// consumer are adjacent by construction.
    rule branch_cmp: Branch { cond: BranchCond::Cmp { op, lhs, rhs }, target }
        if let (Some(cc), Some(w), Some(l), Some(r)) = (
            CondCode::of(*op),
            operand_width(cx, *lhs),
            operand_src(*lhs),
            operand_src(*rhs),
        )
        => |_| vec![
            Inst::Cmp { width: w, lhs: l, rhs: r },
            Inst::Jcc { cc, target: Label(*target) },
        ];

    /// `if (a cmp b) goto L` on floats — `ucomis*` + the parity-aware
    /// branch expansion (`beq`..`blt.un` on float operands; ECMA-335 table
    /// III.4 unordered semantics).
    rule branch_cmp_f: Branch { cond: BranchCond::Cmp { op, lhs, rhs }, target }
        if let (true, Some(w), Some(l), Some(r)) = (
            CondCode::of(*op).is_some(),
            operand_fwidth(cx, *lhs),
            xmm_opnd(cx, *lhs),
            xmm_opnd(cx, *rhs),
        )
        => |_| vec![
            Inst::CmpF { width: w, lhs: l, rhs: r },
            Inst::JccF {
                op: *op,
                target: Label(*target),
            },
        ];

    /// `if v goto L` — compare against zero.
    rule branch_true: Branch { cond: BranchCond::True(v), target }        if let (Some(w), Some(s)) = (operand_width(cx, *v), operand_src(*v))
        => |_| vec![
            Inst::Cmp { width: w, lhs: s, rhs: Src::Imm(0) },
            Inst::Jcc { cc: CondCode::Ne, target: Label(*target) },
        ];

    /// `if !v goto L`.
    rule branch_false: Branch { cond: BranchCond::False(v), target }
        if let (Some(w), Some(s)) = (operand_width(cx, *v), operand_src(*v))
        => |_| vec![
            Inst::Cmp { width: w, lhs: s, rhs: Src::Imm(0) },
            Inst::Jcc { cc: CondCode::Eq, target: Label(*target) },
        ];

    /// `goto L`.
    rule jump: Jump { target }
        => |_| vec![Inst::Jmp { target: Label(*target) }];

    /// `switch (v)` — the tier-0 compare chain: one `cmp`/`je` pair per
    /// case, then the default jump. A value outside [0, N) matches no
    /// `je` and reaches the default — no explicit range check. (The
    /// operand is Int32 by the importer's validation, so W32 compares.)
    rule switch_chain: Switch { value, targets, default }
        if let Some(s) = operand_src(*value)
        => |_| {
            let mut insts = Vec::with_capacity(2 * targets.len() + 1);
            for (i, target) in targets.iter().enumerate() {
                insts.push(Inst::Cmp {
                    width: Width::W32,
                    lhs: s,
                    rhs: Src::Imm(i as i64),
                });
                insts.push(Inst::Jcc {
                    cc: CondCode::Eq,
                    target: Label(*target),
                });
            }
            insts.push(Inst::Jmp {
                target: Label(*default),
            });
            insts
        };

    /// `call m(args)` — direct: argument moves per the ABI classification
    /// (mixed int/float signatures interleave the GPR and XMM sequences;
    /// struct arguments load per eightbyte or block-copy to the outgoing
    /// stack area; step_10.9), then the call, then the result out of
    /// `rax`/`xmm0`/the eightbyte registers. `?` on the
    /// result move aborts the match: a matched call whose destination
    /// can't receive the ABI return is an upstream bug, not a "try the
    /// next rule".
    rule call_direct: Call { dst, target: rokajit::ir::CallTarget::Direct(method), sig, args }
        if let Some(moves) = arg_moves(cx, sig, args)
        => |cx| {
            let mut insts = moves;
            insts.push(Inst::CallDirect { method: *method });
            if let Some(d) = dst {
                insts.extend(call_result_move(cx, sig, *d)?);
            }
            insts
        };

    /// `call helper(args)` — an EE runtime helper (float `rem`); same ABI
    /// treatment as a direct call, target resolved via `getHelperFtn` at
    /// emit time.
    rule call_helper: Call { dst, target: rokajit::ir::CallTarget::Helper(id), sig, args }
        if let Some(moves) = arg_moves(cx, sig, args)
        => |cx| {
            let mut insts = moves;
            insts.push(Inst::CallHelper { id: *id });
            if let Some(d) = dst {
                insts.extend(call_result_move(cx, sig, *d)?);
            }
            insts
        };

    /// `call fnptr(args)` — an indirect call through a computed target
    /// (step_10.12: `calli`, and the vtable slot the importer's
    /// `CORINFO_VIRTUALCALL_VTABLE` emission loads). The ABI argument
    /// moves are a direct call's; the target operand is read LAST — after
    /// the argument registers are filled — so the argument setup can never
    /// clobber it (a pool-resident target the moves evicted reads back
    /// from its frame slot at emit).
    rule call_indirect: Call { dst, target: rokajit::ir::CallTarget::Indirect(addr), sig, args }
        if let (Some(moves), Some(t)) = (arg_moves(cx, sig, args), operand_src(**addr))
        => |cx| {
            let mut insts = moves;
            insts.push(Inst::CallReg { target: t });
            if let Some(d) = dst {
                insts.extend(call_result_move(cx, sig, *d)?);
            }
            insts
        };

    // --- step_10.6: the EH shapes ---

    /// `throw` — the exception object into rdi (the normal first SysV
    /// argument register; CORINFO_HELP_THROW's only parameter), the
    /// never-returning helper call, then a NOP so the call's return
    /// address stays inside its EH region / RUNTIME_FUNCTION
    /// (clr-abi.md's padding rule).
    rule throw_helper: Throw { exception }
        if let Some(s) = operand_src(*exception)
        => |_| vec![
            Inst::Mov {
                width: Width::W64,
                dst: Place::Reg(Gpr::Rdi),
                src: s,
            },
            Inst::CallHelper {
                id: rokajit_ee::enums::CorInfoHelpFunc::THROW,
            },
            Inst::Nop,
        ];

    /// `rethrow` — the never-returning `CORINFO_HELP_RETHROW` call (no
    /// argument: the VM finds the in-flight exception via the stack
    /// walk), with the same padding NOP as `throw` so the call's return
    /// address stays a valid in-region byte (clr-abi.md's rule).
    rule rethrow_helper: Rethrow
        => |_| vec![
            Inst::CallHelper {
                id: rokajit_ee::enums::CorInfoHelpFunc::RETHROW,
            },
            Inst::Nop,
        ];

    /// `leave` in the main body or a finally funclet — a plain jump.
    /// (Inside a CATCH region a `Leave` is the funclet return instead;
    /// that choice needs the region table, which the statement ruleset
    /// context does not carry — codegen substitutes [`catch_leave`].)
    rule leave_jump: Leave { target }
        => |_| vec![Inst::Jmp { target: Label(*target) }];

    /// A `leave` chain hop (a statement-less step block): call the
    /// finally funclet (an in-chunk label; the call's return address is
    /// the NOP, so it stays inside the enclosing region), then jump to
    /// the continuation. Codegen elides the jump when the continuation
    /// is the next block in layout.
    rule call_finally: CallFinally { funclet, continuation }
        => |_| vec![
            Inst::CallLabel { target: Label(*funclet) },
            Inst::Nop,
            Inst::Jmp { target: Label(*continuation) },
        ];

    /// Catch-handler entry: the throwable arrives in rdi
    /// (targetamd64.h REG_EXCEPTION_OBJECT — the VM's CallEHFunclet
    /// contract) and lands in the destination's frame slot. `dst` is a
    /// Ref temp, so the GC-root discipline materializes it immediately.
    rule catch_arg: CatchArg { dst }
        => |_| vec![Inst::Mov {
            width: Width::W64,
            dst: Place::Val(Val(*dst)),
            src: Src::Reg(Gpr::Rdi),
        }];

    /// `endfinally` — the funclet epilog (`add rsp, N; ret`; N is
    /// codegen's per-funclet stack adjustment). No rax result.
    rule end_finally: EndFinally
        => |_| vec![Inst::FuncletEpilog];

    /// A struct block copy (`cpobj`/`stobj`, struct `stloc`/`starg`/
    /// `stfld`, the hidden-retbuf copy; step_10.9): one descriptor; the
    /// size comes from the layout side table. GC-barriered copies are
    /// import-level helper calls and never reach here.
    rule block_copy: BlockCopy { dst_addr, dst_offset, src_addr, class }
        if let (Some(d), Some(s), Some(layout)) = (
            block_addr(*dst_addr),
            block_addr(*src_addr),
            cx.layout_of(*class),
        )
        => |_| vec![Inst::BlockCopy {
            dst: d,
            dst_disp: *dst_offset,
            src: s,
            size: layout.size,
        }];

    /// `initobj`'s block zero.
    rule block_zero: BlockZero { dst_addr, class }
        if let (Some(d), Some(layout)) = (block_addr(*dst_addr), cx.layout_of(*class))
        => |_| vec![Inst::BlockZero {
            dst: d,
            dst_disp: 0,
            size: layout.size,
        }];

    /// `cpblk` — a runtime-sized block copy: one descriptor; codegen
    /// always emits the MEMCPY helper (the size is dynamic, so there is
    /// no inline-threshold decision to take at lowering).
    rule block_copy_dyn: BlockCopyDyn { dst_addr, src_addr, size }
        if let (Some(d), Some(s), Some(n)) = (
            block_addr(*dst_addr),
            block_addr(*src_addr),
            operand_src(*size),
        )
        => |_| vec![Inst::BlockCopyDyn {
            dst: d,
            src: s,
            size: n,
        }];

    /// `initblk` — a runtime-sized block fill: one descriptor; codegen
    /// always emits the MEMSET helper, masking the fill to its low byte.
    rule block_fill_dyn: BlockFillDyn { dst_addr, fill, size }
        if let (Some(d), Some(f), Some(n)) = (
            block_addr(*dst_addr),
            operand_src(*fill),
            operand_src(*size),
        )
        => |_| vec![Inst::BlockFillDyn {
            dst: d,
            fill: f,
            size: n,
        }];

    /// `localloc` — dynamic stack allocation: one descriptor; the
    /// rounding, the `sub rsp`, and the zero-init MEMSET call are
    /// codegen's emission (the frame layout's facts live there). The
    /// size is a native-width integer value or constant.
    rule loc_alloc: LocAlloc { dst, size }
        if let Some(s) = operand_src(*size)
        => |_| vec![Inst::LocAlloc {
            dst: Place::Val(Val(*dst)),
            size: s,
        }];

    /// `return <struct>` — a register-passed struct return (step_10.9):
    /// one exact-size load per eightbyte into its return register
    /// (integer eightbytes → rax then rdx, SSE → xmm0 then xmm1, per the
    /// EE's classification), then the per-block epilog. The
    /// non-register-passed form never reaches LIR (the importer rewrote
    /// it through the hidden retbuf pointer).
    rule return_struct: ReturnStruct { addr, class }
        if let (Some(a), Some(layout)) = (block_addr(*addr), cx.layout_of(*class))
        => |_| {
            // The non-register-passed form never reaches LIR (the
            // importer rewrote it through the hidden retbuf pointer).
            layout.sysv.passed_in_registers.then_some(())?;
            let mut insts = Vec::with_capacity(layout.sysv.count as usize + 2);
            let (mut i, mut f) = (0usize, 0usize);
            for k in 0..layout.sysv.count as usize {
                let dst = if layout.sysv.classes[k].is_sse() {
                    let r = EbReg::Xmm(regs::FLOAT_RETURN_REGS[f]);
                    f += 1;
                    r
                } else {
                    let r = EbReg::Gpr(regs::INT_RETURN_REGS[i]);
                    i += 1;
                    r
                };
                insts.push(Inst::LoadEightbyte {
                    addr: a,
                    disp: u32::from(layout.sysv.offsets[k]),
                    size: layout.sysv.sizes[k],
                    dst,
                });
            }
            insts.extend(epilog());
            insts
        };

    /// `return v` — result to `rax`, then the per-block epilog.
    rule return_value: Return { value: Some(v) }
        if let (Some(w), Some(s)) = (operand_width(cx, *v), operand_src(*v))
        => |_| {
            let mut insts = vec![Inst::Mov {
                width: w,
                dst: Place::Reg(Gpr::Rax),
                src: s,
            }];
            insts.extend(epilog());
            insts
        };

    /// `return f` — float result to `xmm0` (SysV §3.2.3), then the epilog.
    rule return_f: Return { value: Some(v) }
        if let (Some(w), Some(s)) = (operand_fwidth(cx, *v), xmm_opnd(cx, *v))
        => |_| {
            let mut insts = vec![Inst::MovF {
                width: w,
                dst: XmmPlace::Reg(regs::FLOAT_RETURN_REG),
                src: s,
            }];
            insts.extend(epilog());
            insts
        };

    /// `return` — just the epilog.
    rule return_void: Return { value: None }
        => |_| epilog();
}

/// Facts about a method's frame the prolog rules match on. Currently
/// empty: the tier-0 frame contract (rbp-based, so GC root slot offsets
/// are stable while rsp moves — regs.rs) is one fixed shape. Shape
/// choices (callee-saved pushes, large-frame probes) arrive as fields
/// here plus additional rules when a consumer needs them.
pub struct FrameReq;

rokajit::lower_rules! {
    /// The prolog descriptor sequence. One rule today; the frame
    /// contract's future alternatives are added here, not inlined into
    /// codegen.
    pub fn lower_frame(req: &FrameReq, cx: &Cx<'_>) -> Option<Vec<Inst>>
    matching req;

    /// `push rbp; mov rbp, rsp; sub rsp, <frame size>` — the size is
    /// symbolic (`Inst::AllocFrame`); codegen's frame layout fills it in
    /// and keeps it 16-byte aligned so every call site meets the SysV
    /// alignment contract.
    rule standard_frame: _req
        => |_| vec![
            Inst::Push { reg: regs::FRAME_POINTER },
            Inst::Mov {
                width: Width::W64,
                dst: Place::Reg(regs::FRAME_POINTER),
                src: Src::Reg(regs::STACK_POINTER),
            },
            Inst::AllocFrame,
        ];
}

/// The catch-handler form of `leave` (step_10.6): the funclet return.
/// The resume address goes in rax (`lea rax, [rip+target]`) — the VM
/// resumes the parent frame there after the funclet's `ret` — then the
/// funclet epilog. Codegen substitutes this for the `leave_jump` rule
/// when the `Leave`'s block sits inside a CATCH region (a fact the
/// statement ruleset's context does not carry).
pub fn catch_leave(target: BlockId) -> Vec<Inst> {
    vec![
        Inst::LeaLabel {
            dst: Gpr::Rax,
            target: Label(target),
        },
        Inst::FuncletEpilog,
    ]
}

/// Whole-method driver: prolog + every block's statements through the
/// rulesets. A statement no rule matches is an `Unsupported` feature,
/// not a panic (error-model decision).
pub fn lower_method(method: &lir::Method) -> CompileResult<LoweredMethod> {
    let cx = Cx::new(&method.locals, &method.struct_layouts);
    let prolog = lower_frame(&FrameReq, &cx).ok_or(CompileError::Internal(
        "the catch-all frame rule must match",
    ))?;
    let mut blocks = Vec::with_capacity(method.blocks.len());
    for block in &method.blocks {
        let mut insts = Vec::new();
        for stmt in &block.stmts {
            match lower_stmt(stmt, &cx) {
                Some(lowered) => insts.extend(lowered),
                None => {
                    // Audit tooling (the ROKAJIT_DUMP_IL precedent): the
                    // missed statement's kind, on request.
                    if std::env::var_os("ROKAJIT_DEBUG_LOWER").is_some() {
                        eprintln!("rokajit-x64: no rule for LIR {}", stmt.kind.kind_name());
                    }
                    return Err(CompileError::Unsupported(
                        "no x64 lowering rule matched an LIR statement",
                    ));
                }
            }
        }
        blocks.push(LoweredBlock {
            id: block.id,
            insts,
        });
    }
    Ok(LoweredMethod { prolog, blocks })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rokajit::ir::lir::StmtKind;
    use rokajit::ir::{hir, CallTarget, IlOffset, UnaryOp};
    use rokajit::structs::StructLayouts;
    use rokajit_ee::handles::MethodHandle;

    /// Locals: three Int32 slots (0, 1, 2) and one Float slot (3), so
    /// both width selection and float rejection are exercised.
    fn locals() -> Vec<hir::Local> {
        let int = |i: u32| hir::Local {
            ty: Type::Int32,
            kind: hir::LocalKind::IlLocal(i),
            pinned: false,
        };
        let float = || hir::Local {
            ty: Type::Float,
            kind: hir::LocalKind::Temp,
            pinned: false,
        };
        vec![int(0), int(1), int(2), float()]
    }

    fn stmt(kind: StmtKind) -> lir::Stmt {
        lir::Stmt {
            il_offset: IlOffset(0),
            kind,
        }
    }

    fn handle(raw: usize) -> MethodHandle {
        MethodHandle::from_raw(raw as *mut u8 as _).unwrap()
    }

    fn lower_one(s: &lir::Stmt) -> Option<Vec<Inst>> {
        lower_stmt(s, &Cx::new(&locals(), &StructLayouts::new()))
    }

    fn val(i: u32) -> Place {
        Place::Val(Val(LocalId(i)))
    }

    fn vsrc(i: u32) -> Src {
        Src::Val(Val(LocalId(i)))
    }

    #[test]
    fn copy_const_lowers_to_mov_imm() {
        let s = stmt(StmtKind::Copy {
            dst: LocalId(0),
            src: Operand::Const(Const::Int32(7)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::Mov {
                width: Width::W32,
                dst: val(0),
                src: Src::Imm(7),
            }])
        );
        // A null ref is a 64-bit zero immediate.
        let s = stmt(StmtKind::Copy {
            dst: LocalId(0),
            src: Operand::Const(Const::NullRef),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::Mov {
                width: Width::W64,
                dst: val(0),
                src: Src::Imm(0),
            }])
        );
        // A frozen ref (ldstr) is a 64-bit immediate holding the object
        // address; codegen's wide-imm machinery covers >imm32 payloads.
        let s = stmt(StmtKind::Copy {
            dst: LocalId(0),
            src: Operand::Const(Const::FrozenRef(0x1_2345_6789)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::Mov {
                width: Width::W64,
                dst: val(0),
                src: Src::Imm(0x1_2345_6789),
            }])
        );
        // A float constant materializes its bit pattern (ConstF).
        let s = stmt(StmtKind::Copy {
            dst: LocalId(3),
            src: Operand::Const(Const::Float(1.0)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::ConstF {
                width: crate::inst::FWidth::S,
                dst: crate::inst::XmmPlace::Val(Val(LocalId(3))),
                bits: u64::from(1.0f32.to_bits()),
            }])
        );
    }

    #[test]
    fn copy_addr_of_lowers_to_lea_frame_slot() {
        let s = stmt(StmtKind::Copy {
            dst: LocalId(1),
            src: Operand::AddrOf(LocalId(0)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::Lea {
                dst: val(1),
                addr: Amode::FrameSlot(LocalId(0)),
            }])
        );
    }

    #[test]
    fn copy_value_lowers_to_mov() {
        let s = stmt(StmtKind::Copy {
            dst: LocalId(1),
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::Mov {
                width: Width::W32,
                dst: val(1),
                src: vsrc(0),
            }])
        );
    }

    #[test]
    fn arith_lowers_to_three_operand_descriptor() {
        let s = stmt(StmtKind::Binary {
            dst: LocalId(2),
            op: BinaryOp::Add,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Temp(LocalId(1)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::Arith {
                op: ArithOp::Add,
                width: Width::W32,
                dst: val(2),
                lhs: vsrc(0),
                rhs: vsrc(1),
            }])
        );
        // Constants fold into the descriptor as immediates.
        let s = stmt(StmtKind::Binary {
            dst: LocalId(2),
            op: BinaryOp::Sub,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Const(Const::Int32(1)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::Arith {
                op: ArithOp::Sub,
                width: Width::W32,
                dst: val(2),
                lhs: vsrc(0),
                rhs: Src::Imm(1),
            }])
        );
        // `mul` is `imul`.
        let s = stmt(StmtKind::Binary {
            dst: LocalId(2),
            op: BinaryOp::Mul,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Local(LocalId(1)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::Arith {
                op: ArithOp::Imul,
                width: Width::W32,
                dst: val(2),
                lhs: vsrc(0),
                rhs: vsrc(1),
            }])
        );
        // Float arithmetic takes the scalar SSE descriptor.
        let s = stmt(StmtKind::Binary {
            dst: LocalId(3),
            op: BinaryOp::Add,
            lhs: Operand::Local(LocalId(3)),
            rhs: Operand::Local(LocalId(3)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::ArithF {
                op: crate::inst::ArithFOp::Add,
                width: crate::inst::FWidth::S,
                dst: crate::inst::XmmPlace::Val(Val(LocalId(3))),
                lhs: crate::inst::XmmSrc::Val(Val(LocalId(3))),
                rhs: crate::inst::XmmSrc::Val(Val(LocalId(3))),
            }])
        );
    }

    #[test]
    fn rem_lowers_to_the_idiv_fixed_register_sequence() {
        let s = stmt(StmtKind::Binary {
            dst: LocalId(2),
            op: BinaryOp::Rem,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Local(LocalId(1)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Mov {
                    width: Width::W32,
                    dst: Place::Reg(Gpr::Rax),
                    src: vsrc(0),
                },
                Inst::Cdq { width: Width::W32 },
                Inst::Idiv {
                    width: Width::W32,
                    divisor: vsrc(1),
                },
                Inst::Mov {
                    width: Width::W32,
                    dst: val(2),
                    src: Src::Reg(Gpr::Rdx),
                },
            ])
        );
    }

    // --- step_10.1: the scalar-cheap pack rules ---

    /// Locals: Int64 slots 0 and 1, an Int32 slot 2 (dst for 32-bit
    /// results), an Int32 slot 3 (a 32-bit source), an Int64 slot 4
    /// (dst for 64-bit results).
    fn locals_mixed() -> Vec<hir::Local> {
        let l = |ty: Type, i: u32| hir::Local {
            ty,
            kind: hir::LocalKind::IlLocal(i),
            pinned: false,
        };
        vec![
            l(Type::Int64, 0),
            l(Type::Int64, 1),
            l(Type::Int32, 2),
            l(Type::Int32, 3),
            l(Type::Int64, 4),
        ]
    }

    fn lower_with(locals: &[hir::Local], s: &lir::Stmt) -> Option<Vec<Inst>> {
        lower_stmt(s, &Cx::new(locals, &StructLayouts::new()))
    }

    #[test]
    fn div_lowers_like_rem_with_quotient_from_rax() {
        let s = stmt(StmtKind::Binary {
            dst: LocalId(2),
            op: BinaryOp::Div,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Local(LocalId(1)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Mov {
                    width: Width::W32,
                    dst: Place::Reg(Gpr::Rax),
                    src: vsrc(0),
                },
                Inst::Cdq { width: Width::W32 },
                Inst::Idiv {
                    width: Width::W32,
                    divisor: vsrc(1),
                },
                Inst::Mov {
                    width: Width::W32,
                    dst: val(2),
                    src: Src::Reg(Gpr::Rax),
                },
            ])
        );
    }

    #[test]
    fn unsigned_div_rem_zero_rdx_and_use_div() {
        // `div.un`: quotient from rax, `edx` zeroed (never `cdq`).
        let s = stmt(StmtKind::Binary {
            dst: LocalId(2),
            op: BinaryOp::UDiv,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Local(LocalId(1)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Mov {
                    width: Width::W32,
                    dst: Place::Reg(Gpr::Rax),
                    src: vsrc(0),
                },
                Inst::Mov {
                    width: Width::W32,
                    dst: Place::Reg(Gpr::Rdx),
                    src: Src::Imm(0),
                },
                Inst::Div {
                    width: Width::W32,
                    divisor: vsrc(1),
                },
                Inst::Mov {
                    width: Width::W32,
                    dst: val(2),
                    src: Src::Reg(Gpr::Rax),
                },
            ])
        );
        // `rem.un`: same sequence, remainder from rdx.
        let s = stmt(StmtKind::Binary {
            dst: LocalId(2),
            op: BinaryOp::URem,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Local(LocalId(1)),
        });
        let lowered = lower_one(&s).expect("matches");
        assert!(matches!(lowered[2], Inst::Div { .. }));
        assert_eq!(
            lowered[3],
            Inst::Mov {
                width: Width::W32,
                dst: val(2),
                src: Src::Reg(Gpr::Rdx),
            }
        );
    }

    #[test]
    fn arith_ovf_lowers_to_one_descriptor() {
        // All six checked forms: op and the `.un` signedness pass through
        // on the descriptor; codegen owns the conditional-throw sequence.
        for (op, unsigned) in [
            (BinaryOp::Add, false),
            (BinaryOp::Add, true),
            (BinaryOp::Sub, false),
            (BinaryOp::Sub, true),
            (BinaryOp::Mul, false),
            (BinaryOp::Mul, true),
        ] {
            let s = stmt(StmtKind::BinaryOvf {
                dst: LocalId(2),
                op,
                unsigned,
                lhs: Operand::Local(LocalId(0)),
                rhs: Operand::Local(LocalId(1)),
            });
            assert_eq!(
                lower_one(&s),
                Some(vec![Inst::ArithOvf {
                    op,
                    unsigned,
                    width: Width::W32,
                    dst: val(2),
                    lhs: vsrc(0),
                    rhs: vsrc(1),
                }]),
                "{op:?} unsigned={unsigned}"
            );
        }
        // A 64-bit destination widens the descriptor; a non-Add/Sub/Mul
        // operator matches no rule (the importer never builds one).
        let s = stmt(StmtKind::BinaryOvf {
            dst: LocalId(4),
            op: BinaryOp::Mul,
            unsigned: false,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Const(Const::Int64(3)),
        });
        assert_eq!(
            lower_with(&locals_mixed(), &s),
            Some(vec![Inst::ArithOvf {
                op: BinaryOp::Mul,
                unsigned: false,
                width: Width::W64,
                dst: val(4),
                lhs: vsrc(0),
                rhs: Src::Imm(3),
            }])
        );
    }

    #[test]
    fn shifts_lower_to_the_shift_descriptor() {
        // `shr` (signed) is arithmetic (`sar`); `shr.un` is logical
        // (`shr`). 64-bit value, 32-bit count.
        for (op, expected) in [
            (BinaryOp::Shl, ShiftOp::Shl),
            (BinaryOp::Shr, ShiftOp::Sar),
            (BinaryOp::UShr, ShiftOp::Shr),
        ] {
            let s = stmt(StmtKind::Binary {
                dst: LocalId(4),
                op,
                lhs: Operand::Local(LocalId(0)),
                rhs: Operand::Local(LocalId(3)),
            });
            assert_eq!(
                lower_with(&locals_mixed(), &s),
                Some(vec![Inst::Shift {
                    op: expected,
                    width: Width::W64,
                    dst: val(4),
                    lhs: vsrc(0),
                    rhs: vsrc(3),
                }])
            );
        }
        // A constant count rides along as an immediate.
        let s = stmt(StmtKind::Binary {
            dst: LocalId(2),
            op: BinaryOp::Shl,
            lhs: Operand::Local(LocalId(3)),
            rhs: Operand::Const(Const::Int32(3)),
        });
        assert_eq!(
            lower_with(&locals_mixed(), &s),
            Some(vec![Inst::Shift {
                op: ShiftOp::Shl,
                width: Width::W32,
                dst: val(2),
                lhs: vsrc(3),
                rhs: Src::Imm(3),
            }])
        );
    }

    #[test]
    fn neg_not_lower_to_the_unary_descriptor() {
        for (op, name) in [(UnaryOp::Neg, "neg"), (UnaryOp::Not, "not")] {
            let s = stmt(StmtKind::Unary {
                dst: LocalId(2),
                op,
                src: Operand::Local(LocalId(0)),
            });
            assert_eq!(
                lower_one(&s),
                Some(vec![Inst::Unary {
                    op,
                    width: Width::W32,
                    dst: val(2),
                    src: vsrc(0),
                }]),
                "{name}"
            );
        }
    }

    #[test]
    fn compare_as_value_lowers_to_cmp_setcc() {
        // `ceq`: cmp + sete. The unsigned `cgt.un` maps to `seta`.
        for (op, cc) in [(BinaryOp::Eq, CondCode::Eq), (BinaryOp::UGt, CondCode::UGt)] {
            let s = stmt(StmtKind::Binary {
                dst: LocalId(2),
                op,
                lhs: Operand::Local(LocalId(0)),
                rhs: Operand::Local(LocalId(1)),
            });
            assert_eq!(
                lower_one(&s),
                Some(vec![
                    Inst::Cmp {
                        width: Width::W32,
                        lhs: vsrc(0),
                        rhs: vsrc(1),
                    },
                    Inst::Setcc { cc, dst: val(2) },
                ])
            );
        }
    }

    #[test]
    fn conv_rules_truncate_and_extend() {
        // conv.i4 from a 64-bit source: a truncating 32-bit mov.
        let s = stmt(StmtKind::Conv {
            dst: LocalId(2),
            to: Type::Int32,
            overflow: false,
            unsigned: false,
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_with(&locals_mixed(), &s),
            Some(vec![Inst::Mov {
                width: Width::W32,
                dst: val(2),
                src: vsrc(0),
            }])
        );
        // conv.u8 / conv.i8 from a 32-bit source: MovExt, the `unsigned`
        // flag selecting zero- vs sign-extension.
        for (unsigned, signed) in [(true, false), (false, true)] {
            let s = stmt(StmtKind::Conv {
                dst: LocalId(4),
                to: Type::Int64,
                overflow: false,
                unsigned,
                src: Operand::Local(LocalId(3)),
            });
            assert_eq!(
                lower_with(&locals_mixed(), &s),
                Some(vec![Inst::MovExt {
                    dst: val(4),
                    src: vsrc(3),
                    signed,
                }])
            );
        }
        // Same-width conversions match no rule (the importer drops them).
        let s = stmt(StmtKind::Conv {
            dst: LocalId(4),
            to: Type::Int64,
            overflow: false,
            unsigned: false,
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(lower_with(&locals_mixed(), &s), None);
    }

    #[test]
    fn conv_ovf_lowers_to_the_checked_descriptor() {
        // conv.ovf.i4 of an i64 source: one ConvOvf descriptor carrying
        // the source width, target range, and both signednesses.
        let s = stmt(StmtKind::ConvOvf {
            dst: LocalId(2),
            dst_bits: 32,
            signed_dst: true,
            unsigned_src: false,
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_with(&locals_mixed(), &s),
            Some(vec![Inst::ConvOvf {
                width_src: Width::W64,
                width_dst: 32,
                signed_dst: true,
                unsigned_src: false,
                dst: val(2),
                src: vsrc(0),
            }])
        );
        // conv.ovf.u2.un of a 32-bit source.
        let s = stmt(StmtKind::ConvOvf {
            dst: LocalId(2),
            dst_bits: 16,
            signed_dst: false,
            unsigned_src: true,
            src: Operand::Local(LocalId(3)),
        });
        assert_eq!(
            lower_with(&locals_mixed(), &s),
            Some(vec![Inst::ConvOvf {
                width_src: Width::W32,
                width_dst: 16,
                signed_dst: false,
                unsigned_src: true,
                dst: val(2),
                src: vsrc(3),
            }])
        );
        // A float source matches no rule (the importer emits helper
        // calls — a float-typed src has no GPR width).
        let locals = vec![
            hir::Local {
                ty: Type::Double,
                kind: hir::LocalKind::IlLocal(0),
                pinned: false,
            },
            hir::Local {
                ty: Type::Int32,
                kind: hir::LocalKind::IlLocal(1),
                pinned: false,
            },
        ];
        let s = stmt(StmtKind::ConvOvf {
            dst: LocalId(1),
            dst_bits: 32,
            signed_dst: true,
            unsigned_src: false,
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(lower_with(&locals, &s), None);
    }

    #[test]
    fn ckfinite_lowers_to_the_check_descriptor() {
        // ckfinite of a double: one CkFinite descriptor at FWidth::D.
        let locals = vec![
            hir::Local {
                ty: Type::Double,
                kind: hir::LocalKind::IlLocal(0),
                pinned: false,
            },
            hir::Local {
                ty: Type::Double,
                kind: hir::LocalKind::IlLocal(1),
                pinned: false,
            },
        ];
        let s = stmt(StmtKind::CkFinite {
            dst: LocalId(1),
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_with(&locals, &s),
            Some(vec![Inst::CkFinite {
                width: FWidth::D,
                dst: XmmPlace::Val(Val(LocalId(1))),
                src: XmmSrc::Val(Val(LocalId(0))),
            }])
        );
        // Of a float32 constant: the Bits source at FWidth::S.
        let s = stmt(StmtKind::CkFinite {
            dst: LocalId(1),
            src: Operand::Const(Const::Float(1.5)),
        });
        // dst must be Float-typed for the rule? dst type is not guarded;
        // keep the double locals — the src width drives the descriptor.
        assert_eq!(
            lower_with(&locals, &s),
            Some(vec![Inst::CkFinite {
                width: FWidth::S,
                dst: XmmPlace::Val(Val(LocalId(1))),
                src: XmmSrc::Bits(1.5f32.to_bits() as u64),
            }])
        );
    }

    #[test]
    fn byref_call_arg_materializes_with_lea() {
        let sig = rokajit::ir::CallSig {
            ret: Type::Void,
            args: vec![Type::ByRef],
            has_this: false,
        };
        let s = stmt(StmtKind::Call {
            dst: None,
            target: CallTarget::Direct(handle(0x42)),
            sig,
            args: vec![Operand::AddrOf(LocalId(0))],
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Lea {
                    dst: Place::Reg(Gpr::Rdi),
                    addr: Amode::FrameSlot(LocalId(0)),
                },
                Inst::CallDirect {
                    method: handle(0x42),
                },
            ])
        );
    }

    #[test]
    fn localloc_lowers_to_one_descriptor() {
        // Value form: the size operand passes through as a Src; codegen
        // owns the rounding, the rsp adjustment, and the zero-init call.
        let s = stmt(StmtKind::LocAlloc {
            dst: LocalId(2),
            size: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::LocAlloc {
                dst: val(2),
                size: vsrc(0),
            }])
        );
        // A constant size is an immediate source.
        let s = stmt(StmtKind::LocAlloc {
            dst: LocalId(2),
            size: Operand::Const(Const::NativeInt(64)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::LocAlloc {
                dst: val(2),
                size: Src::Imm(64),
            }])
        );
    }

    #[test]
    fn cpblk_initblk_lower_to_one_descriptor_each() {
        // `cpblk`: addresses keep the block-op shapes (a frame slot or a
        // pointer value); the size is a value or constant source.
        let s = stmt(StmtKind::BlockCopyDyn {
            dst_addr: Operand::Local(LocalId(0)),
            src_addr: Operand::AddrOf(LocalId(1)),
            size: Operand::Local(LocalId(2)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::BlockCopyDyn {
                dst: BlockAddr::Val(Val(LocalId(0))),
                src: BlockAddr::FrameSlot(LocalId(1)),
                size: vsrc(2),
            }])
        );
        let s = stmt(StmtKind::BlockCopyDyn {
            dst_addr: Operand::Local(LocalId(0)),
            src_addr: Operand::Local(LocalId(1)),
            size: Operand::Const(Const::NativeInt(24)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::BlockCopyDyn {
                dst: BlockAddr::Val(Val(LocalId(0))),
                src: BlockAddr::Val(Val(LocalId(1))),
                size: Src::Imm(24),
            }])
        );
        // `initblk`: the fill is a value or immediate source.
        let s = stmt(StmtKind::BlockFillDyn {
            dst_addr: Operand::Local(LocalId(0)),
            fill: Operand::Const(Const::Int32(0x7F)),
            size: Operand::Local(LocalId(2)),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::BlockFillDyn {
                dst: BlockAddr::Val(Val(LocalId(0))),
                fill: Src::Imm(0x7F),
                size: vsrc(2),
            }])
        );
        // A constant address matches no rule (the LIR flattener
        // materializes it into a ByRef temp first).
        let s = stmt(StmtKind::BlockCopyDyn {
            dst_addr: Operand::Const(Const::NativeInt(0x1000)),
            src_addr: Operand::Local(LocalId(1)),
            size: Operand::Local(LocalId(2)),
        });
        assert_eq!(lower_one(&s), None);
    }

    #[test]
    fn compare_branch_materializes_flags() {
        let s = stmt(StmtKind::Branch {
            cond: BranchCond::Cmp {
                op: BinaryOp::Lt,
                lhs: Operand::Local(LocalId(0)),
                rhs: Operand::Const(Const::Int32(2)),
            },
            target: BlockId(2),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Cmp {
                    width: Width::W32,
                    lhs: vsrc(0),
                    rhs: Src::Imm(2),
                },
                Inst::Jcc {
                    cc: CondCode::Lt,
                    target: Label(BlockId(2)),
                },
            ])
        );
        // The unsigned form maps onto the carry-based condition code.
        let s = stmt(StmtKind::Branch {
            cond: BranchCond::Cmp {
                op: BinaryOp::ULt,
                lhs: Operand::Local(LocalId(0)),
                rhs: Operand::Local(LocalId(1)),
            },
            target: BlockId(1),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Cmp {
                    width: Width::W32,
                    lhs: vsrc(0),
                    rhs: vsrc(1),
                },
                Inst::Jcc {
                    cc: CondCode::ULt,
                    target: Label(BlockId(1)),
                },
            ])
        );
    }

    #[test]
    fn true_false_branches_compare_against_zero() {
        let s = stmt(StmtKind::Branch {
            cond: BranchCond::True(Operand::Local(LocalId(0))),
            target: BlockId(1),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Cmp {
                    width: Width::W32,
                    lhs: vsrc(0),
                    rhs: Src::Imm(0),
                },
                Inst::Jcc {
                    cc: CondCode::Ne,
                    target: Label(BlockId(1)),
                },
            ])
        );
        let s = stmt(StmtKind::Branch {
            cond: BranchCond::False(Operand::Local(LocalId(0))),
            target: BlockId(1),
        });
        assert!(matches!(
            lower_one(&s).as_deref(),
            Some([
                _,
                Inst::Jcc {
                    cc: CondCode::Eq,
                    ..
                }
            ])
        ));
    }

    #[test]
    fn jump_lowers_to_jmp() {
        let s = stmt(StmtKind::Jump { target: BlockId(1) });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::Jmp {
                target: Label(BlockId(1))
            }])
        );
    }

    #[test]
    fn switch_lowers_to_a_compare_chain() {
        // switch (v0) { 0: B2, 1: B3, 2: B3, default: B1 } — one cmp/je
        // pair per case (duplicate targets keep their pair), then the
        // default jmp.
        let s = stmt(StmtKind::Switch {
            value: Operand::Local(LocalId(0)),
            targets: vec![BlockId(2), BlockId(3), BlockId(3)],
            default: BlockId(1),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Cmp {
                    width: Width::W32,
                    lhs: vsrc(0),
                    rhs: Src::Imm(0),
                },
                Inst::Jcc {
                    cc: CondCode::Eq,
                    target: Label(BlockId(2)),
                },
                Inst::Cmp {
                    width: Width::W32,
                    lhs: vsrc(0),
                    rhs: Src::Imm(1),
                },
                Inst::Jcc {
                    cc: CondCode::Eq,
                    target: Label(BlockId(3)),
                },
                Inst::Cmp {
                    width: Width::W32,
                    lhs: vsrc(0),
                    rhs: Src::Imm(2),
                },
                Inst::Jcc {
                    cc: CondCode::Eq,
                    target: Label(BlockId(3)),
                },
                Inst::Jmp {
                    target: Label(BlockId(1)),
                },
            ])
        );
    }

    #[test]
    fn direct_call_moves_args_per_the_abi() {
        let sig = rokajit::ir::CallSig {
            ret: Type::Int32,
            args: vec![Type::Int32],
            has_this: false,
        };
        let s = stmt(StmtKind::Call {
            dst: Some(LocalId(2)),
            target: CallTarget::Direct(handle(0x42)),
            sig,
            args: vec![Operand::Temp(LocalId(1))],
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Mov {
                    width: Width::W32,
                    dst: Place::Reg(regs::INT_ARG_REGS[0]),
                    src: vsrc(1),
                },
                Inst::CallDirect {
                    method: handle(0x42),
                },
                Inst::Mov {
                    width: Width::W32,
                    dst: val(2),
                    src: Src::Reg(Gpr::Rax),
                },
            ])
        );
        assert_eq!(regs::INT_ARG_REGS[0], Gpr::Rdi);
    }

    #[test]
    fn void_direct_call_has_no_result_move() {
        let sig = rokajit::ir::CallSig {
            ret: Type::Void,
            args: vec![Type::Int32, Type::Int32],
            has_this: false,
        };
        let s = stmt(StmtKind::Call {
            dst: None,
            target: CallTarget::Direct(handle(0x42)),
            sig,
            args: vec![Operand::Const(Const::Int32(1)), Operand::Local(LocalId(0))],
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Mov {
                    width: Width::W32,
                    dst: Place::Reg(Gpr::Rdi),
                    src: Src::Imm(1),
                },
                Inst::Mov {
                    width: Width::W32,
                    dst: Place::Reg(Gpr::Rsi),
                    src: vsrc(0),
                },
                Inst::CallDirect {
                    method: handle(0x42),
                },
            ])
        );
    }

    #[test]
    fn seven_int_args_overflow_to_an_outgoing_stack_store() {
        // Seven integer arguments (step_10.9: stack args are supported):
        // the first six take the GPR arg registers, the seventh stores to
        // the outgoing area at [rsp+0].
        let sig = rokajit::ir::CallSig {
            ret: Type::Int32,
            args: vec![Type::Int32; 7],
            has_this: false,
        };
        let s = stmt(StmtKind::Call {
            dst: Some(LocalId(2)),
            target: CallTarget::Direct(handle(0x42)),
            sig,
            args: vec![Operand::Local(LocalId(0)); 7],
        });
        let insts = lower_one(&s).expect("matches");
        assert_eq!(insts.len(), 9, "seven arg moves, the call, the result");
        assert_eq!(
            insts[6],
            Inst::StoreStackArg {
                width: Width::W32,
                offset: 0,
                src: vsrc(0),
            }
        );
        assert!(matches!(insts[7], Inst::CallDirect { .. }));
    }

    #[test]
    fn calls_outside_the_subset_match_no_rule() {
        let sig = |n: usize| rokajit::ir::CallSig {
            ret: Type::Int32,
            args: vec![Type::Int32; n],
            has_this: false,
        };
        // A float argument moves into xmm0 (step_10.2: the FP ABI).
        let s = stmt(StmtKind::Call {
            dst: Some(LocalId(2)),
            target: CallTarget::Direct(handle(0x42)),
            sig: rokajit::ir::CallSig {
                ret: Type::Int32,
                args: vec![Type::Float],
                has_this: false,
            },
            args: vec![Operand::Local(LocalId(3))],
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::MovF {
                    width: crate::inst::FWidth::S,
                    dst: crate::inst::XmmPlace::Reg(regs::Xmm::Xmm0),
                    src: crate::inst::XmmSrc::Val(Val(LocalId(3))),
                },
                Inst::CallDirect {
                    method: handle(0x42),
                },
                Inst::Mov {
                    width: Width::W32,
                    dst: val(2),
                    src: Src::Reg(Gpr::Rax),
                },
            ])
        );
        // Virtual dispatch is not a direct call.
        let s = stmt(StmtKind::Call {
            dst: Some(LocalId(2)),
            target: CallTarget::Virtual {
                method: handle(0x42),
            },
            sig: sig(0),
            args: Vec::new(),
        });
        assert_eq!(lower_one(&s), None);
    }

    #[test]
    fn call_indirect_reads_the_pointer_after_the_argument_moves() {
        // step_10.12: `calli` / vtable dispatch — the argument moves are a
        // direct call's, then `call r11` on the pointer operand, then the
        // result. The pointer reads LAST: the argument registers it might
        // pool-occupy are already filled, and codegen's spill makes its
        // frame slot the source of truth.
        let sig = rokajit::ir::CallSig {
            ret: Type::Int32,
            args: vec![Type::Int32],
            has_this: false,
        };
        let s = stmt(StmtKind::Call {
            dst: Some(LocalId(2)),
            target: CallTarget::Indirect(std::boxed::Box::new(Operand::Local(LocalId(0)))),
            sig,
            args: vec![Operand::Local(LocalId(1))],
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Mov {
                    width: Width::W32,
                    dst: Place::Reg(Gpr::Rdi),
                    src: vsrc(1),
                },
                Inst::CallReg { target: vsrc(0) },
                Inst::Mov {
                    width: Width::W32,
                    dst: val(2),
                    src: Src::Reg(Gpr::Rax),
                },
            ])
        );
        // A constant target (ldftn) lowers to the immediate source form.
        let s = stmt(StmtKind::Call {
            dst: None,
            target: CallTarget::Indirect(std::boxed::Box::new(Operand::Const(Const::NativeInt(
                0x7777,
            )))),
            sig: rokajit::ir::CallSig {
                ret: Type::Void,
                args: Vec::new(),
                has_this: false,
            },
            args: Vec::new(),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![Inst::CallReg {
                target: Src::Imm(0x7777),
            }])
        );
    }

    #[test]
    fn returns_emit_the_per_block_epilog() {
        let s = stmt(StmtKind::Return {
            value: Some(Operand::Temp(LocalId(1))),
        });
        assert_eq!(
            lower_one(&s),
            Some(vec![
                Inst::Mov {
                    width: Width::W32,
                    dst: Place::Reg(Gpr::Rax),
                    src: vsrc(1),
                },
                Inst::Leave,
                Inst::Ret,
            ])
        );
        let s = stmt(StmtKind::Return { value: None });
        assert_eq!(lower_one(&s), Some(vec![Inst::Leave, Inst::Ret]));
    }

    #[test]
    fn prolog_is_the_frame_contract_sequence() {
        let prolog = lower_frame(&FrameReq, &Cx::new(&locals(), &StructLayouts::new()))
            .expect("frame rule matches");
        assert_eq!(
            prolog,
            vec![
                Inst::Push { reg: Gpr::Rbp },
                Inst::Mov {
                    width: Width::W64,
                    dst: Place::Reg(Gpr::Rbp),
                    src: Src::Reg(Gpr::Rsp),
                },
                Inst::AllocFrame,
            ]
        );
    }

    #[test]
    fn unmatched_statement_is_unsupported_at_the_method_driver() {
        // An array length read has no rule yet (arrays: a later pack).
        let method = lir::Method {
            blocks: vec![lir::Block {
                id: BlockId(0),
                stmts: vec![stmt(StmtKind::ArrLen {
                    dst: LocalId(1),
                    array: Operand::Local(LocalId(0)),
                })],
            }],
            locals: locals(),
            eh_regions: Vec::new(),
            num_args: 0,
            num_il_locals: 3,
            struct_layouts: StructLayouts::new(),
            generics_context: None,
        };
        assert!(matches!(
            lower_method(&method),
            Err(CompileError::Unsupported(_))
        ));
    }

    // --- step_10.4: the object pack rules ---

    /// Locals: Ref 0 (`this`), Int32 1 (an int result), Ref 2 (a ref
    /// result), ByRef 3 (a field-address temp), Double 4 (a float slot).
    fn locals_obj() -> Vec<hir::Local> {
        let l = |ty: Type, i: u32| hir::Local {
            ty,
            kind: hir::LocalKind::IlLocal(i),
            pinned: false,
        };
        vec![
            l(Type::Ref, 0),
            l(Type::Int32, 1),
            l(Type::Ref, 2),
            l(Type::ByRef, 3),
            l(Type::Double, 4),
        ]
    }

    fn lower_obj(s: &lir::Stmt) -> Option<Vec<Inst>> {
        lower_stmt(s, &Cx::new(&locals_obj(), &StructLayouts::new()))
    }

    #[test]
    fn load_lowers_to_load_mem() {
        // t1 := [this + 16] (an Int32 field read).
        let s = stmt(StmtKind::Load {
            dst: LocalId(1),
            addr: Operand::Local(LocalId(0)),
            offset: 16,
            ty: Type::Int32,
            access: MemAccess::Natural,
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::LoadMem {
                width: Width::W32,
                dst: val(1),
                addr: vsrc(0),
                disp: 16,
            }])
        );
        // A Ref-typed load is 64-bit.
        let s = stmt(StmtKind::Load {
            dst: LocalId(2),
            addr: Operand::Local(LocalId(0)),
            offset: 24,
            ty: Type::Ref,
            access: MemAccess::Natural,
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::LoadMem {
                width: Width::W64,
                dst: val(2),
                addr: vsrc(0),
                disp: 24,
            }])
        );
        // A float field load: `movsd` from memory (step_10.10).
        let s = stmt(StmtKind::Load {
            dst: LocalId(4),
            addr: Operand::Local(LocalId(0)),
            offset: 8,
            ty: Type::Double,
            access: MemAccess::Natural,
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::LoadMemF {
                width: FWidth::D,
                dst: xval(4),
                addr: vsrc(0),
                disp: 8,
            }])
        );
        // A sub-Int32 field load: zero- or sign-extended per the metadata
        // type (step_10.10).
        let s = stmt(StmtKind::Load {
            dst: LocalId(1),
            addr: Operand::Local(LocalId(0)),
            offset: 4,
            ty: Type::Int32,
            access: MemAccess::U16,
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::LoadMemNarrow {
                size: 2,
                signed: false,
                dst: val(1),
                addr: vsrc(0),
                disp: 4,
            }])
        );
        let s = stmt(StmtKind::Load {
            dst: LocalId(1),
            addr: Operand::Local(LocalId(0)),
            offset: 4,
            ty: Type::Int32,
            access: MemAccess::I8,
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::LoadMemNarrow {
                size: 1,
                signed: true,
                dst: val(1),
                addr: vsrc(0),
                disp: 4,
            }])
        );
    }

    #[test]
    fn store_lowers_to_store_mem() {
        // [this + 16] := t1 — width from the source operand.
        let s = stmt(StmtKind::Store {
            addr: Operand::Local(LocalId(0)),
            offset: 16,
            src: Operand::Local(LocalId(1)),
            access: MemAccess::Natural,
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::StoreMem {
                width: Width::W32,
                addr: vsrc(0),
                disp: 16,
                src: vsrc(1),
            }])
        );
        // A constant source rides along as an immediate (W64 for null).
        let s = stmt(StmtKind::Store {
            addr: Operand::Local(LocalId(0)),
            offset: 8,
            src: Operand::Const(Const::Int32(0)),
            access: MemAccess::Natural,
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::StoreMem {
                width: Width::W32,
                addr: vsrc(0),
                disp: 8,
                src: Src::Imm(0),
            }])
        );
        // A float field store: `movsd` to memory (step_10.10).
        let s = stmt(StmtKind::Store {
            addr: Operand::Local(LocalId(0)),
            offset: 8,
            src: Operand::Local(LocalId(4)),
            access: MemAccess::Natural,
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::StoreMemF {
                width: FWidth::D,
                addr: vsrc(0),
                disp: 8,
                src: xsrc(4),
            }])
        );
        // A sub-Int32 field store: only the low bytes write (step_10.10).
        let s = stmt(StmtKind::Store {
            addr: Operand::Local(LocalId(0)),
            offset: 4,
            src: Operand::Local(LocalId(1)),
            access: MemAccess::U8,
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::StoreMemNarrow {
                size: 1,
                addr: vsrc(0),
                disp: 4,
                src: vsrc(1),
            }])
        );
    }

    #[test]
    fn null_check_lowers_to_the_trap_load() {
        let s = stmt(StmtKind::NullCheck {
            arg: Operand::Local(LocalId(0)),
        });
        assert_eq!(lower_obj(&s), Some(vec![Inst::NullCheck { addr: vsrc(0) }]));
        // A constant receiver (ldnull) still checks — it must fault.
        let s = stmt(StmtKind::NullCheck {
            arg: Operand::Const(Const::NullRef),
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::NullCheck { addr: Src::Imm(0) }])
        );
    }

    // --- step_10.8: the array pack ---

    #[test]
    fn bounds_check_lowers_to_the_descriptor() {
        // An Int32 index compares 32-bit; a native-int index 64-bit (the
        // length load zero-extends, so the wide compare is exact).
        let mut ls = locals_obj();
        ls.push(hir::Local {
            ty: Type::NativeInt,
            kind: hir::LocalKind::Temp,
            pinned: false,
        });
        let s = stmt(StmtKind::BoundsCheck {
            array: Operand::Local(LocalId(0)),
            index: Operand::Local(LocalId(1)),
        });
        assert_eq!(
            lower_with(&ls, &s),
            Some(vec![Inst::BoundsCheck {
                index: vsrc(1),
                index_wide: false,
                array: vsrc(0),
            }])
        );
        let s = stmt(StmtKind::BoundsCheck {
            array: Operand::Local(LocalId(0)),
            index: Operand::Local(LocalId(5)),
        });
        assert_eq!(
            lower_with(&ls, &s),
            Some(vec![Inst::BoundsCheck {
                index: vsrc(5),
                index_wide: true,
                array: vsrc(0),
            }])
        );
    }

    /// The step_10.8 element-access matrix, exercised through the plain
    /// typed-memory rules with a ByRef-temp address at offset 0 — exactly
    /// the operands the flattener's ArrElemAddr expansion produces, with
    /// zero array knowledge in the rules (the 10.13 ldind/stind
    /// contract). Locals: Int32 0, Int64 1, NativeInt 2, Float 3,
    /// Double 4, Ref 5, ByRef 6 (the address temp).
    fn locals_matrix() -> Vec<hir::Local> {
        let l = |ty: Type, i: u32| hir::Local {
            ty,
            kind: hir::LocalKind::IlLocal(i),
            pinned: false,
        };
        vec![
            l(Type::Int32, 0),
            l(Type::Int64, 1),
            l(Type::NativeInt, 2),
            l(Type::Float, 3),
            l(Type::Double, 4),
            l(Type::Ref, 5),
            l(Type::ByRef, 6),
        ]
    }

    #[test]
    fn element_matrix_loads_cover_every_kind() {
        let addr = Operand::Temp(LocalId(6));
        let load = |dst: u32, ty: Type, access: MemAccess| {
            stmt(StmtKind::Load {
                dst: LocalId(dst),
                addr,
                offset: 0,
                ty,
                access,
            })
        };
        // Sub-Int32 loads: movsx/movzx by cell size and signedness
        // (ECMA-335 §III.1.1.1: I1/I2 sign-extend, U1/U2 zero-extend).
        for (access, size, signed) in [
            (MemAccess::I8, 1, true),
            (MemAccess::U8, 1, false),
            (MemAccess::I16, 2, true),
            (MemAccess::U16, 2, false),
        ] {
            assert_eq!(
                lower_with(&locals_matrix(), &load(0, Type::Int32, access)),
                Some(vec![Inst::LoadMemNarrow {
                    size,
                    signed,
                    dst: val(0),
                    addr: vsrc(6),
                    disp: 0,
                }]),
                "{access:?}"
            );
        }
        // Natural GPR loads (i4/u4, i8, i, ref): width from the type.
        for (dst, ty, w) in [
            (0, Type::Int32, Width::W32),
            (1, Type::Int64, Width::W64),
            (2, Type::NativeInt, Width::W64),
            (5, Type::Ref, Width::W64),
        ] {
            assert_eq!(
                lower_with(&locals_matrix(), &load(dst, ty, MemAccess::Natural)),
                Some(vec![Inst::LoadMem {
                    width: w,
                    dst: val(dst),
                    addr: vsrc(6),
                    disp: 0,
                }]),
                "{ty:?}"
            );
        }
        // Float loads (r4/r8): movss/movsd.
        for (dst, ty, w) in [(3, Type::Float, FWidth::S), (4, Type::Double, FWidth::D)] {
            assert_eq!(
                lower_with(&locals_matrix(), &load(dst, ty, MemAccess::Natural)),
                Some(vec![Inst::LoadMemF {
                    width: w,
                    dst: xval(dst),
                    addr: vsrc(6),
                    disp: 0,
                }]),
                "{ty:?}"
            );
        }
    }

    #[test]
    fn element_matrix_stores_cover_every_kind() {
        let addr = Operand::Temp(LocalId(6));
        let store = |src: u32, access: MemAccess| {
            stmt(StmtKind::Store {
                addr,
                offset: 0,
                src: Operand::Temp(LocalId(src)),
                access,
            })
        };
        // Sub-Int32 stores: only the low bytes write (extension is a
        // store-time no-op).
        for (access, size) in [
            (MemAccess::I8, 1),
            (MemAccess::U8, 1),
            (MemAccess::I16, 2),
            (MemAccess::U16, 2),
        ] {
            assert_eq!(
                lower_with(&locals_matrix(), &store(0, access)),
                Some(vec![Inst::StoreMemNarrow {
                    size,
                    addr: vsrc(6),
                    disp: 0,
                    src: vsrc(0),
                }]),
                "{access:?}"
            );
        }
        // Natural GPR stores (i4, i8, i, ref): width from the source.
        for (src, w) in [
            (0, Width::W32),
            (1, Width::W64),
            (2, Width::W64),
            (5, Width::W64),
        ] {
            assert_eq!(
                lower_with(&locals_matrix(), &store(src, MemAccess::Natural)),
                Some(vec![Inst::StoreMem {
                    width: w,
                    addr: vsrc(6),
                    disp: 0,
                    src: vsrc(src),
                }]),
                "src local {src}"
            );
        }
        // Float stores (r4/r8): movss/movsd.
        for (src, w) in [(3, FWidth::S), (4, FWidth::D)] {
            assert_eq!(
                lower_with(&locals_matrix(), &store(src, MemAccess::Natural)),
                Some(vec![Inst::StoreMemF {
                    width: w,
                    addr: vsrc(6),
                    disp: 0,
                    src: xsrc(src),
                }]),
                "src local {src}"
            );
        }
    }

    #[test]
    fn field_addr_add_lowers_through_the_arith_rule() {
        // The flatten-emitted `ByRef dst := Ref lhs + NativeInt offset`:
        // the existing arith rule covers it (ByRef is W64).
        let s = stmt(StmtKind::Binary {
            dst: LocalId(3),
            op: BinaryOp::Add,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Const(Const::NativeInt(16)),
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::Arith {
                op: ArithOp::Add,
                width: Width::W64,
                dst: val(3),
                lhs: vsrc(0),
                rhs: Src::Imm(16),
            }])
        );
    }

    // --- step_10.6: the EH shapes ---

    #[test]
    fn throw_lowers_to_helper_call_and_nop() {
        // The exception object (a Ref) into rdi, the never-returning
        // CORINFO_HELP_THROW call, then the region-padding NOP.
        let s = stmt(StmtKind::Throw {
            exception: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![
                Inst::Mov {
                    width: Width::W64,
                    dst: Place::Reg(Gpr::Rdi),
                    src: vsrc(0),
                },
                Inst::CallHelper {
                    id: rokajit_ee::enums::CorInfoHelpFunc::THROW,
                },
                Inst::Nop,
            ])
        );
    }

    #[test]
    fn leave_lowers_to_a_jump_and_the_catch_form_is_the_funclet_return() {
        // Main body / finally handler: a plain jump.
        let s = stmt(StmtKind::Leave { target: BlockId(3) });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::Jmp {
                target: Label(BlockId(3))
            }])
        );
        // Inside a catch region (codegen's dispatch): the funclet return
        // — the resume address in rax, then the epilog.
        assert_eq!(
            catch_leave(BlockId(2)),
            vec![
                Inst::LeaLabel {
                    dst: Gpr::Rax,
                    target: Label(BlockId(2)),
                },
                Inst::FuncletEpilog,
            ]
        );
    }

    #[test]
    fn call_finally_lowers_to_call_nop_jmp() {
        let s = stmt(StmtKind::CallFinally {
            funclet: BlockId(4),
            continuation: BlockId(1),
        });
        assert_eq!(
            lower_obj(&s),
            Some(vec![
                Inst::CallLabel {
                    target: Label(BlockId(4)),
                },
                Inst::Nop,
                Inst::Jmp {
                    target: Label(BlockId(1)),
                },
            ])
        );
    }

    #[test]
    fn catch_arg_stores_rdi_into_the_destination() {
        let s = stmt(StmtKind::CatchArg { dst: LocalId(2) });
        assert_eq!(
            lower_obj(&s),
            Some(vec![Inst::Mov {
                width: Width::W64,
                dst: val(2),
                src: Src::Reg(Gpr::Rdi),
            }])
        );
    }

    #[test]
    fn end_finally_lowers_to_the_funclet_epilog() {
        let s = stmt(StmtKind::EndFinally);
        assert_eq!(lower_obj(&s), Some(vec![Inst::FuncletEpilog]));
    }

    #[test]
    fn rethrow_lowers_to_the_helper_call_plus_nop() {
        let s = stmt(StmtKind::Rethrow);
        assert_eq!(
            lower_obj(&s),
            Some(vec![
                Inst::CallHelper {
                    id: rokajit_ee::enums::CorInfoHelpFunc::RETHROW,
                },
                Inst::Nop,
            ])
        );
    }

    // --- end-to-end: fib's exact IL bytes → descriptor sequence ---

    #[test]
    fn fib_lowers_end_to_end() {
        use rokajit::pipeline::MethodInfo;
        use rokajit_ee::enums::CorInfoType;
        use rokajit_ee::mock::{MockEe, MockSig};

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
            generics_context: None,
            generics_context_keep_alive: false,
            args: ee.make_method_sig(&fib_sig),
            locals: ee.make_locals_sig(&[]),
        };
        let hir = rokajit::pipeline::import(&info, &ee).expect("imports");
        let hir = rokajit::pipeline::morph(hir).expect("morphs");
        let lir = rokajit::pipeline::lower(hir, &crate::X64Target).expect("lowers to LIR");
        // The recursive call's method handle, as the importer resolved it.
        let fib_handle = lir
            .blocks
            .iter()
            .flat_map(|b| &b.stmts)
            .find_map(|s| match &s.kind {
                StmtKind::Call {
                    target: CallTarget::Direct(m),
                    ..
                } => Some(*m),
                _ => None,
            })
            .expect("fib calls itself");
        let m = lower_method(&lir).expect("lowers to descriptors");

        assert_eq!(
            m.prolog,
            vec![
                Inst::Push { reg: Gpr::Rbp },
                Inst::Mov {
                    width: Width::W64,
                    dst: Place::Reg(Gpr::Rbp),
                    src: Src::Reg(Gpr::Rsp),
                },
                Inst::AllocFrame,
            ]
        );
        assert_eq!(m.blocks.len(), 3);

        // Block 0: the `n < 2` guard, flags materialized; the else edge
        // is a fallthrough, so no jmp follows.
        assert_eq!(
            m.blocks[0].insts,
            vec![
                Inst::Cmp {
                    width: Width::W32,
                    lhs: vsrc(0),
                    rhs: Src::Imm(2),
                },
                Inst::Jcc {
                    cc: CondCode::Lt,
                    target: Label(BlockId(2)),
                },
            ]
        );

        // Block 1: fib(n-1) + fib(n-2) — arg setup in RDI, results out
        // of RAX, per-block epilog on the return.
        let call1 = [
            Inst::Arith {
                op: ArithOp::Sub,
                width: Width::W32,
                dst: val(1),
                lhs: vsrc(0),
                rhs: Src::Imm(1),
            },
            Inst::Mov {
                width: Width::W32,
                dst: Place::Reg(Gpr::Rdi),
                src: vsrc(1),
            },
            Inst::CallDirect { method: fib_handle },
            Inst::Mov {
                width: Width::W32,
                dst: val(2),
                src: Src::Reg(Gpr::Rax),
            },
        ];
        let call2 = [
            Inst::Arith {
                op: ArithOp::Sub,
                width: Width::W32,
                dst: val(3),
                lhs: vsrc(0),
                rhs: Src::Imm(2),
            },
            Inst::Mov {
                width: Width::W32,
                dst: Place::Reg(Gpr::Rdi),
                src: vsrc(3),
            },
            Inst::CallDirect { method: fib_handle },
            Inst::Mov {
                width: Width::W32,
                dst: val(4),
                src: Src::Reg(Gpr::Rax),
            },
        ];
        let tail = [
            Inst::Arith {
                op: ArithOp::Add,
                width: Width::W32,
                dst: val(5),
                lhs: vsrc(2),
                rhs: vsrc(4),
            },
            Inst::Mov {
                width: Width::W32,
                dst: Place::Reg(Gpr::Rax),
                src: vsrc(5),
            },
            Inst::Leave,
            Inst::Ret,
        ];
        let expected: Vec<Inst> = [call1.as_slice(), call2.as_slice(), tail.as_slice()].concat();
        assert_eq!(m.blocks[1].insts, expected);

        // Block 2: return n.
        assert_eq!(
            m.blocks[2].insts,
            vec![
                Inst::Mov {
                    width: Width::W32,
                    dst: Place::Reg(Gpr::Rax),
                    src: vsrc(0),
                },
                Inst::Leave,
                Inst::Ret,
            ]
        );
    }

    // --- step_10.2: the float pack rules ---

    use crate::inst::{ArithFOp, FWidth, XmmPlace, XmmSrc};

    /// Locals: Double 0/1, Int32 2 (compare dst), Float 3, Int64 4,
    /// Double 5 (float dst).
    fn locals_f() -> Vec<hir::Local> {
        let l = |ty: Type, i: u32| hir::Local {
            ty,
            kind: hir::LocalKind::IlLocal(i),
            pinned: false,
        };
        vec![
            l(Type::Double, 0),
            l(Type::Double, 1),
            l(Type::Int32, 2),
            l(Type::Float, 3),
            l(Type::Int64, 4),
            l(Type::Double, 5),
        ]
    }

    fn lower_f(s: &lir::Stmt) -> Option<Vec<Inst>> {
        lower_stmt(s, &Cx::new(&locals_f(), &StructLayouts::new()))
    }

    fn xval(i: u32) -> XmmPlace {
        XmmPlace::Val(Val(LocalId(i)))
    }

    fn xsrc(i: u32) -> XmmSrc {
        XmmSrc::Val(Val(LocalId(i)))
    }

    #[test]
    fn float_copy_and_const_lower_to_movsd_and_constf() {
        let s = stmt(StmtKind::Copy {
            dst: LocalId(5),
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::MovF {
                width: FWidth::D,
                dst: xval(5),
                src: xsrc(0),
            }])
        );
        // A float constant becomes its bit pattern.
        let s = stmt(StmtKind::Copy {
            dst: LocalId(3),
            src: Operand::Const(Const::Float(-0.5)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::ConstF {
                width: FWidth::S,
                dst: xval(3),
                bits: u64::from((-0.5f32).to_bits()),
            }])
        );
    }

    #[test]
    fn float_arith_lowers_to_the_scalar_sse_descriptor() {
        for (op, aop) in [
            (BinaryOp::Add, ArithFOp::Add),
            (BinaryOp::Sub, ArithFOp::Sub),
            (BinaryOp::Mul, ArithFOp::Mul),
            (BinaryOp::Div, ArithFOp::Div),
            (BinaryOp::MinF, ArithFOp::Min),
            (BinaryOp::MaxF, ArithFOp::Max),
        ] {
            let s = stmt(StmtKind::Binary {
                dst: LocalId(5),
                op,
                lhs: Operand::Local(LocalId(0)),
                rhs: Operand::Local(LocalId(1)),
            });
            assert_eq!(
                lower_f(&s),
                Some(vec![Inst::ArithF {
                    op: aop,
                    width: FWidth::D,
                    dst: xval(5),
                    lhs: xsrc(0),
                    rhs: xsrc(1),
                }]),
                "{op:?}"
            );
        }
        // A float constant operand carries its bits inline.
        let s = stmt(StmtKind::Binary {
            dst: LocalId(5),
            op: BinaryOp::Add,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Const(Const::Double(1.0)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::ArithF {
                op: ArithFOp::Add,
                width: FWidth::D,
                dst: xval(5),
                lhs: xsrc(0),
                rhs: XmmSrc::Bits(1.0f64.to_bits()),
            }])
        );
        // rem has no SSE form (the importer expands it to a helper call).
        let s = stmt(StmtKind::Binary {
            dst: LocalId(5),
            op: BinaryOp::Rem,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Local(LocalId(1)),
        });
        assert_eq!(lower_f(&s), None);
    }

    #[test]
    fn float_neg_and_not() {
        let s = stmt(StmtKind::Unary {
            dst: LocalId(5),
            op: UnaryOp::Neg,
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::NegF {
                width: FWidth::D,
                dst: xval(5),
                src: xsrc(0),
            }])
        );
        // `not` has no float form (the importer rejects it; the backend
        // agrees).
        let s = stmt(StmtKind::Unary {
            dst: LocalId(5),
            op: UnaryOp::Not,
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(lower_f(&s), None);
    }

    #[test]
    fn float_compare_value_lowers_to_ucomisd_setccf() {
        for op in [
            BinaryOp::Eq,
            BinaryOp::Gt,
            BinaryOp::UGt,
            BinaryOp::Lt,
            BinaryOp::ULt,
        ] {
            let s = stmt(StmtKind::Binary {
                dst: LocalId(2),
                op,
                lhs: Operand::Local(LocalId(0)),
                rhs: Operand::Local(LocalId(1)),
            });
            assert_eq!(
                lower_f(&s),
                Some(vec![
                    Inst::CmpF {
                        width: FWidth::D,
                        lhs: xsrc(0),
                        rhs: xsrc(1),
                    },
                    Inst::SetccF {
                        op,
                        dst: Place::Val(Val(LocalId(2))),
                    },
                ]),
                "{op:?}"
            );
        }
    }

    #[test]
    fn float_branch_lowers_to_ucomisd_jccf() {
        let s = stmt(StmtKind::Branch {
            cond: BranchCond::Cmp {
                op: BinaryOp::UGt,
                lhs: Operand::Local(LocalId(0)),
                rhs: Operand::Local(LocalId(1)),
            },
            target: BlockId(2),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![
                Inst::CmpF {
                    width: FWidth::D,
                    lhs: xsrc(0),
                    rhs: xsrc(1),
                },
                Inst::JccF {
                    op: BinaryOp::UGt,
                    target: Label(BlockId(2)),
                },
            ])
        );
    }

    #[test]
    fn float_convs_lower_to_the_cvt_descriptors() {
        // conv.r8 of an i32: cvtsi2sd, 32-bit source.
        let s = stmt(StmtKind::Conv {
            dst: LocalId(5),
            to: Type::Double,
            overflow: false,
            unsigned: false,
            src: Operand::Local(LocalId(2)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::CvtIntToF {
                width: FWidth::D,
                src_w64: false,
                dst: xval(5),
                src: vsrc(2),
            }])
        );
        // conv.r4 of an i64: cvtsi2ss, 64-bit source.
        let s = stmt(StmtKind::Conv {
            dst: LocalId(3),
            to: Type::Float,
            overflow: false,
            unsigned: false,
            src: Operand::Local(LocalId(4)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::CvtIntToF {
                width: FWidth::S,
                src_w64: true,
                dst: xval(3),
                src: vsrc(4),
            }])
        );
        // conv.i4 of a double: cvttsd2si, 32-bit destination.
        let s = stmt(StmtKind::Conv {
            dst: LocalId(2),
            to: Type::Int32,
            overflow: false,
            unsigned: false,
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::CvtFToInt {
                src_width: FWidth::D,
                dst_w64: false,
                dst: Place::Val(Val(LocalId(2))),
                src: xsrc(0),
            }])
        );
        // conv.u4 of a float: the 64-bit form (the low half is the result).
        let s = stmt(StmtKind::Conv {
            dst: LocalId(2),
            to: Type::Int32,
            overflow: false,
            unsigned: true,
            src: Operand::Local(LocalId(3)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::CvtFToInt {
                src_width: FWidth::S,
                dst_w64: true,
                dst: Place::Val(Val(LocalId(2))),
                src: xsrc(3),
            }])
        );
        // conv.i8 of a double: 64-bit destination.
        let s = stmt(StmtKind::Conv {
            dst: LocalId(4),
            to: Type::Int64,
            overflow: false,
            unsigned: false,
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::CvtFToInt {
                src_width: FWidth::D,
                dst_w64: true,
                dst: Place::Val(Val(LocalId(4))),
                src: xsrc(0),
            }])
        );
        // conv.i of a double (step_11.9): NativeInt — the same signed
        // 64-bit form.
        let s = stmt(StmtKind::Conv {
            dst: LocalId(4),
            to: Type::NativeInt,
            overflow: false,
            unsigned: false,
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::CvtFToInt {
                src_width: FWidth::D,
                dst_w64: true,
                dst: Place::Val(Val(LocalId(4))),
                src: xsrc(0),
            }])
        );
        // conv.r4 of a double / conv.r8 of a float: cvt between widths.
        let s = stmt(StmtKind::Conv {
            dst: LocalId(3),
            to: Type::Float,
            overflow: false,
            unsigned: false,
            src: Operand::Local(LocalId(0)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::CvtFToF {
                to: FWidth::S,
                dst: xval(3),
                src: xsrc(0),
            }])
        );
    }

    #[test]
    fn conv_r_un_of_an_i64_lowers_to_the_u64_fixup() {
        // conv.r.un of a 64-bit operand (unsigned flag + W64 source):
        // the branchy CvtU64ToF expansion.
        let s = stmt(StmtKind::Conv {
            dst: LocalId(5),
            to: Type::Double,
            overflow: false,
            unsigned: true,
            src: Operand::Local(LocalId(4)),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![Inst::CvtU64ToF {
                width: FWidth::D,
                dst: xval(5),
                src: vsrc(4),
            }])
        );
        // An unsigned 32-bit source matches no rule: the importer
        // pre-zero-extends u32 operands to Int64, so this shape never
        // arrives (a signed cvtsi2s* of the low half would be wrong).
        let s = stmt(StmtKind::Conv {
            dst: LocalId(5),
            to: Type::Double,
            overflow: false,
            unsigned: true,
            src: Operand::Local(LocalId(2)),
        });
        assert_eq!(lower_f(&s), None);
    }

    #[test]
    fn helper_call_lowers_like_a_direct_call() {
        // The float-rem helper: (double, double) -> double.
        let sig = rokajit::ir::CallSig {
            ret: Type::Double,
            args: vec![Type::Double, Type::Double],
            has_this: false,
        };
        let s = stmt(StmtKind::Call {
            dst: Some(LocalId(5)),
            target: CallTarget::Helper(rokajit_ee::enums::CorInfoHelpFunc::DBLREM),
            sig,
            args: vec![Operand::Local(LocalId(0)), Operand::Local(LocalId(1))],
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![
                Inst::MovF {
                    width: FWidth::D,
                    dst: XmmPlace::Reg(regs::Xmm::Xmm0),
                    src: xsrc(0),
                },
                Inst::MovF {
                    width: FWidth::D,
                    dst: XmmPlace::Reg(regs::Xmm::Xmm1),
                    src: xsrc(1),
                },
                Inst::CallHelper {
                    id: rokajit_ee::enums::CorInfoHelpFunc::DBLREM,
                },
                Inst::MovF {
                    width: FWidth::D,
                    dst: xval(5),
                    src: XmmSrc::Reg(regs::Xmm::Xmm0),
                },
            ])
        );
    }

    #[test]
    fn helper_call_with_a_struct_return_stores_the_eightbyte() {
        // step_10.10: the ldtoken conversion helper — (native int) -> a
        // one-eightbyte struct (RuntimeTypeHandle). The raw handle
        // constant goes to rdi; the rax result stores into the struct
        // slot, like any register-passed struct return.
        let c = rokajit_ee::handles::ClassHandle::from_raw(0x9008 as *mut u8 as _).unwrap();
        let mut layouts = StructLayouts::new();
        layouts.insert(
            c,
            rokajit::structs::StructLayout {
                size: 8,
                align: 8,
                gc_cells: vec![],
                sysv: rokajit::structs::SysVPass {
                    passed_in_registers: true,
                    count: 1,
                    classes: [
                        rokajit::structs::SysVClass::IntegerRef,
                        rokajit::structs::SysVClass::Integer,
                    ],
                    sizes: [8, 0],
                    offsets: [0, 0],
                },
            },
        );
        let locals = vec![
            hir::Local {
                ty: Type::NativeInt,
                kind: hir::LocalKind::Temp,
                pinned: false,
            },
            hir::Local {
                ty: Type::Struct(c),
                kind: hir::LocalKind::Temp,
                pinned: false,
            },
        ];
        let s = stmt(StmtKind::Call {
            dst: Some(LocalId(1)),
            target: CallTarget::Helper(
                rokajit_ee::enums::CorInfoHelpFunc::TYPEHANDLE_TO_RUNTIMETYPEHANDLE,
            ),
            sig: rokajit::ir::CallSig {
                ret: Type::Struct(c),
                args: vec![Type::NativeInt],
                has_this: false,
            },
            args: vec![Operand::Local(LocalId(0))],
        });
        assert_eq!(
            lower_stmt(&s, &Cx::new(&locals, &layouts)),
            Some(vec![
                Inst::Mov {
                    width: Width::W64,
                    dst: Place::Reg(Gpr::Rdi),
                    src: vsrc(0),
                },
                Inst::CallHelper {
                    id: rokajit_ee::enums::CorInfoHelpFunc::TYPEHANDLE_TO_RUNTIMETYPEHANDLE,
                },
                Inst::StoreEightbyte {
                    local: LocalId(1),
                    offset: 0,
                    size: 8,
                    src: crate::inst::EbReg::Gpr(Gpr::Rax),
                },
            ])
        );
    }

    #[test]
    fn mixed_signature_call_interleaves_the_register_classes() {
        // (int, double, float, long) -> int: rdi, xmm0, xmm1, rsi.
        let sig = rokajit::ir::CallSig {
            ret: Type::Int32,
            args: vec![Type::Int32, Type::Double, Type::Float, Type::Int64],
            has_this: false,
        };
        let s = stmt(StmtKind::Call {
            dst: Some(LocalId(2)),
            target: CallTarget::Direct(handle(0x42)),
            sig,
            args: vec![
                Operand::Local(LocalId(2)),
                Operand::Local(LocalId(0)),
                Operand::Local(LocalId(3)),
                Operand::Local(LocalId(4)),
            ],
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![
                Inst::Mov {
                    width: Width::W32,
                    dst: Place::Reg(Gpr::Rdi),
                    src: vsrc(2),
                },
                Inst::MovF {
                    width: FWidth::D,
                    dst: XmmPlace::Reg(regs::Xmm::Xmm0),
                    src: xsrc(0),
                },
                Inst::MovF {
                    width: FWidth::S,
                    dst: XmmPlace::Reg(regs::Xmm::Xmm1),
                    src: xsrc(3),
                },
                Inst::Mov {
                    width: Width::W64,
                    dst: Place::Reg(Gpr::Rsi),
                    src: vsrc(4),
                },
                Inst::CallDirect {
                    method: handle(0x42),
                },
                Inst::Mov {
                    width: Width::W32,
                    dst: val(2),
                    src: Src::Reg(Gpr::Rax),
                },
            ])
        );
    }

    #[test]
    fn float_return_moves_to_xmm0() {
        let s = stmt(StmtKind::Return {
            value: Some(Operand::Local(LocalId(0))),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![
                Inst::MovF {
                    width: FWidth::D,
                    dst: XmmPlace::Reg(regs::Xmm::Xmm0),
                    src: xsrc(0),
                },
                Inst::Leave,
                Inst::Ret,
            ])
        );
        // A float constant return carries its bits.
        let s = stmt(StmtKind::Return {
            value: Some(Operand::Const(Const::Double(1.0))),
        });
        assert_eq!(
            lower_f(&s),
            Some(vec![
                Inst::MovF {
                    width: FWidth::D,
                    dst: XmmPlace::Reg(regs::Xmm::Xmm0),
                    src: XmmSrc::Bits(1.0f64.to_bits()),
                },
                Inst::Leave,
                Inst::Ret,
            ])
        );
    }
}

#[cfg(test)]
mod ptrconv_repro_tests {
    use super::*;
    use rokajit::pipeline::MethodInfo;
    use rokajit_ee::enums::CorInfoType;
    use rokajit_ee::handles::MethodHandle;
    use rokajit_ee::mock::{MockEe, MockSig};

    #[test]
    fn repro_github_19288_byref_conv_compare() {
        // `ldloca.s 0; conv.u; ldc.i4.0; conv.u; bge.un.s +1; nop; ret`
        let mut ee = MockEe::default();
        let sig = MockSig {
            ret: CorInfoType::Void,
            args: vec![],
            has_this: false,
            ret_class: None,
            arg_classes: Vec::new(),
        };
        let info = MethodInfo {
            ftn: MethodHandle::from_raw(std::ptr::dangling_mut::<u8>() as _).unwrap(),
            il: vec![0x12, 0x00, 0xE0, 0x16, 0xE0, 0x33, 0x01, 0x00, 0x2A],
            max_stack: 8,
            eh_count: 0,
            init_locals: false,
            generics_context: None,
            generics_context_keep_alive: false,
            args: ee.make_method_sig(&sig),
            locals: ee.make_locals_sig(&[CorInfoType::Int]),
        };
        let hir = rokajit::pipeline::import(&info, &ee).expect("imports");
        let hir = rokajit::pipeline::morph(hir).expect("morphs");
        let lir = rokajit::pipeline::lower(hir, &crate::X64Target).expect("lowers");
        let cx = Cx::new(&lir.locals, &lir.struct_layouts);
        for (bi, b) in lir.blocks.iter().enumerate() {
            for (si, s) in b.stmts.iter().enumerate() {
                assert!(lower_stmt(s, &cx).is_some(), "stmt {bi}/{si} missed a rule");
            }
        }
        assert!(lower_method(&lir).is_ok(), "whole method lowers");
    }

    #[test]
    fn repro_github_19288_ldarga_of_a_value_class_arg() {
        // The actual test shape: `ldarga.s 0` of a by-value struct arg
        // (PixelData p), conv.u, compare against zero.
        let mut ee = MockEe::default();
        let class = ee.add_class(3, 1, &[], None);
        let sig = MockSig {
            ret: CorInfoType::Void,
            args: vec![CorInfoType::ValueClass],
            has_this: true,
            ret_class: None,
            arg_classes: vec![Some(class)],
        };
        let info = MethodInfo {
            ftn: MethodHandle::from_raw(std::ptr::dangling_mut::<u8>() as _).unwrap(),
            il: vec![0x0F, 0x01, 0xE0, 0x16, 0xE0, 0x33, 0x01, 0x00, 0x2A],
            max_stack: 8,
            eh_count: 0,
            init_locals: false,
            generics_context: None,
            generics_context_keep_alive: false,
            args: ee.make_method_sig(&sig),
            locals: ee.make_locals_sig(&[]),
        };
        let hir = rokajit::pipeline::import(&info, &ee).expect("imports");
        let hir = rokajit::pipeline::morph(hir).expect("morphs");
        let lir = rokajit::pipeline::lower(hir, &crate::X64Target).expect("lowers");
        let cx = Cx::new(&lir.locals, &lir.struct_layouts);
        for (bi, b) in lir.blocks.iter().enumerate() {
            for (si, s) in b.stmts.iter().enumerate() {
                assert!(lower_stmt(s, &cx).is_some(), "stmt {bi}/{si} missed a rule");
            }
        }
        assert!(lower_method(&lir).is_ok(), "whole method lowers");
    }
}
