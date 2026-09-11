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
//! architecture pins them (arg moves per [`crate::regs::INT_ARG_REGS`],
//! returns through `rax`, the `idiv` fixed-register sequence, the
//! `rbp`-based frame contract from `regs.rs`).

use rokajit::error::{CompileError, CompileResult};
use rokajit::ir::lir::{BranchCond, Operand, StmtKind::*};
use rokajit::ir::{lir, BinaryOp, BlockId, Const, LocalId, Type};
use rokajit::lower::{Cx, Label, Val};

use crate::inst::{Amode, ArithOp, CondCode, Inst, Place, Src, Width};
use crate::regs::{self, Gpr};

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

/// The per-ABI argument setup for a direct call: one `mov` per argument
/// into its SysV register ([`regs::INT_ARG_REGS`] order; entry 0 is the
/// implicit `this` when present — the importer/morph already place the
/// receiver first). `None` — no rule match — when an argument needs a
/// stack slot (more than six integer arguments) or a float register.
fn arg_moves(cx: &Cx, args: &[Operand]) -> Option<Vec<Inst>> {
    if args.len() > regs::INT_ARG_REGS.len() {
        return None;
    }
    let mut moves = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().enumerate() {
        moves.push(Inst::Mov {
            width: operand_width(cx, *arg)?,
            dst: Place::Reg(regs::INT_ARG_REGS[i]),
            src: operand_src(*arg)?,
        });
    }
    Some(moves)
}

/// The per-block epilog, shared by both return rules. Matches the
/// prolog the `standard_frame` rule emits: `leave` undoes
/// `push rbp; mov rbp, rsp; sub rsp, frame`.
fn epilog() -> Vec<Inst> {
    vec![Inst::Leave, Inst::Ret]
}

rokajit::lower_rules! {
    /// LIR statement → x64 instruction descriptors, for the fib subset.
    /// Rules are tried in declaration order; the more specific operand
    /// shapes (constants, addresses) precede the general value rules.
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

    /// `if v goto L` — compare against zero.
    rule branch_true: Branch { cond: BranchCond::True(v), target }
        if let (Some(w), Some(s)) = (operand_width(cx, *v), operand_src(*v))
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

    /// `call m(args)` — direct: argument moves per the ABI constants,
    /// then the call, then the result out of `rax`. `?` on the result
    /// width aborts the match: a matched call whose destination temp has
    /// no GPR width is an upstream bug, not a "try the next rule".
    rule call_direct: Call { dst, target: rokajit::ir::CallTarget::Direct(method), args, .. }
        if let Some(moves) = arg_moves(cx, args)
        => |cx| {
            let mut insts = moves;
            insts.push(Inst::CallDirect { method: *method });
            if let Some(d) = dst {
                insts.push(Inst::Mov {
                    width: width_of(cx, *d)?,
                    dst: Place::Val(Val(*d)),
                    src: Src::Reg(Gpr::Rax),
                });
            }
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

/// Whole-method driver: prolog + every block's statements through the
/// rulesets. A statement no rule matches is an `Unsupported` feature,
/// not a panic (error-model decision).
pub fn lower_method(method: &lir::Method) -> CompileResult<LoweredMethod> {
    let cx = Cx::new(&method.locals);
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
    use rokajit::ir::{hir, CallTarget, IlOffset};
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
        lower_stmt(s, &Cx::new(&locals()))
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
        // A float constant matches no rule.
        let s = stmt(StmtKind::Copy {
            dst: LocalId(3),
            src: Operand::Const(Const::Float(1.0)),
        });
        assert_eq!(lower_one(&s), None);
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
        // Float arithmetic matches no rule.
        let s = stmt(StmtKind::Binary {
            dst: LocalId(3),
            op: BinaryOp::Add,
            lhs: Operand::Local(LocalId(3)),
            rhs: Operand::Local(LocalId(3)),
        });
        assert_eq!(lower_one(&s), None);
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
    fn calls_outside_the_subset_match_no_rule() {
        let sig = |n: usize| rokajit::ir::CallSig {
            ret: Type::Int32,
            args: vec![Type::Int32; n],
            has_this: false,
        };
        // Seven integer arguments: the seventh needs a stack slot.
        let s = stmt(StmtKind::Call {
            dst: Some(LocalId(2)),
            target: CallTarget::Direct(handle(0x42)),
            sig: sig(7),
            args: vec![Operand::Local(LocalId(0)); 7],
        });
        assert_eq!(lower_one(&s), None);
        // A float argument needs an XMM register move.
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
        assert_eq!(lower_one(&s), None);
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
        let prolog = lower_frame(&FrameReq, &Cx::new(&locals())).expect("frame rule matches");
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
        let method = lir::Method {
            blocks: vec![lir::Block {
                id: BlockId(0),
                stmts: vec![stmt(StmtKind::NullCheck {
                    arg: Operand::Local(LocalId(0)),
                })],
            }],
            locals: locals(),
            eh_regions: Vec::new(),
            num_args: 0,
            num_il_locals: 3,
        };
        assert!(matches!(
            lower_method(&method),
            Err(CompileError::Unsupported(_))
        ));
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
}
