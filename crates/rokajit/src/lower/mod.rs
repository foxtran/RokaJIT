//! Pipeline stage 3 (step_07.4): HIR → LIR lowering.
//!
//! Two halves, per the frozen pipeline contract (`pipeline.rs`):
//!
//! - [`lower`] — the target-generic flattener: `hir::Method →
//!   lir::Method`. Expression trees become flat statement sequences (one
//!   operation per statement, calls lifted to top level, results in fresh
//!   temps), and branch conditions fold to [`lir::BranchCond`]. This half
//!   is structural — the IR contract (`docs/ir-design.md`, "Lowering may
//!   assume / must produce") fixes the output shape, so there is nothing
//!   to pattern-match here. The `target` parameter only gates legality:
//!   every local's type must have a register class on the target.
//! - the [`lower_rules!`](crate::lower_rules) DSL (`dsl` module) — the
//!   ISLE-style rule language a backend uses to map LIR statements to
//!   machine-instruction descriptors (rokajit-x64's `lower.rs` is the
//!   first consumer). Instruction selection is a machine decision, so it
//!   lives entirely in backend crates.
//!
//! Terms crossing from the core to a backend: [`Val`] (an unallocated LIR
//! value — codegen, step_07.5, binds it to a register or frame slot) and
//! [`Label`] (a branch target as a block reference, resolved at
//! emission). Backend descriptor types distinguish these from physical
//! registers and addressing modes so invalid sequences are
//! unrepresentable (the ISLE typed-terms lesson).
//!
//! Scope: the fib subset (step_07.md), the step_10.1 scalar-cheap pack,
//! and the step_10.2 float pack: every [`BinaryOp`] (arithmetic,
//! `div`/`rem` signed and unsigned, logic, shifts, and compare-as-value —
//! a compare's temp is `Int32`), on integer and float operands alike
//! (float `rem` arrives as a helper call from the importer), `neg`/`not`
//! unary ops, and the `conv.*` nodes (int↔float included). Everything
//! else — loads and stores through byrefs, switches, and EH — fails with
//! [`CompileError::Unsupported`].

mod dsl;

use crate::error::{CompileError, CompileResult};
use crate::ir::{
    hir, lir, BinaryOp, BlockId, CallSig, CallTarget, Const, IlOffset, LocalId, Type,
    IL_OFFSET_NONE,
};
use crate::target::Target;

/// Term type: an unallocated LIR value (temp or local) referenced by a
/// machine-instruction descriptor. Codegen (step_07.5) binds it to a
/// register or frame slot; the byte encoder (step_07.6) never sees one.
/// Distinct from a physical register so a descriptor cannot confuse the
/// two (typed terms, ISLE lesson).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Val(pub LocalId);

/// Term type: a branch target as a block reference. Codegen resolves it
/// to an emission label; native offsets/fixups are the encoder's
/// business (step_07.6).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Label(pub BlockId);

/// The read-only context a ruleset sees: the method's locals table, for
/// type-driven decisions (e.g. 32- vs 64-bit forms). Fallible lookups
/// return `Option` so a ruleset guard treats an out-of-contract
/// reference as "no rule matched" rather than panicking.
pub struct Cx<'a> {
    locals: &'a [hir::Local],
}

impl<'a> Cx<'a> {
    pub fn new(locals: &'a [hir::Local]) -> Self {
        Cx { locals }
    }

    /// The type of a local/arg/temp slot.
    pub fn ty_of(&self, id: LocalId) -> Option<Type> {
        self.locals.get(id.0 as usize).map(|l| l.ty)
    }

    /// The whole locals table, for rulesets that need more than types.
    pub fn locals(&self) -> &'a [hir::Local] {
        self.locals
    }
}

/// Stage entry point (the body of [`crate::pipeline::lower`]).
///
/// Consumes the morphed method and produces LIR per the IR contract:
/// one operation per statement, operands restricted to
/// [`lir::Operand`], calls only as top-level `StmtKind::Call`, branches
/// folded to `BranchCond`, temps defined exactly once before use.
pub fn lower(method: hir::Method, target: &dyn Target) -> CompileResult<lir::Method> {
    if !method.eh_regions.is_empty() {
        return Err(CompileError::Unsupported("EH regions: not yet supported"));
    }
    for local in &method.locals {
        if target.class_of(local.ty).is_none() {
            return Err(CompileError::Unsupported(
                "a local's type has no register class on this target",
            ));
        }
    }
    let mut fx = Flatten {
        locals: method.locals,
    };
    let mut blocks = Vec::with_capacity(method.blocks.len());
    for (i, block) in method.blocks.iter().enumerate() {
        let next = method.blocks.get(i + 1).map(|b| b.id);
        blocks.push(fx.lower_block(block, next)?);
    }
    Ok(lir::Method {
        blocks,
        locals: fx.locals,
        eh_regions: method.eh_regions,
        num_args: method.num_args,
        num_il_locals: method.num_il_locals,
    })
}

/// The flattener's state: the locals table, grown with fresh temps as
/// expression trees are decomposed. Temp ids follow the IR contract's
/// flat `LocalId` namespace (temps come after args and IL locals).
struct Flatten {
    locals: Vec<hir::Local>,
}

impl Flatten {
    fn temp(&mut self, ty: Type) -> LocalId {
        let id = LocalId(self.locals.len() as u32);
        self.locals.push(hir::Local {
            ty,
            kind: hir::LocalKind::Temp,
            pinned: false,
        });
        id
    }

    /// The type an operand carries, from the locals table or the
    /// constant itself. `Internal` on use: the importer guarantees
    /// in-table locals, so `None` here is an upstream bug.
    fn operand_ty(&self, op: &lir::Operand) -> CompileResult<Type> {
        match op {
            lir::Operand::Temp(id) | lir::Operand::Local(id) => self
                .locals
                .get(id.0 as usize)
                .map(|l| l.ty)
                .ok_or(CompileError::Internal(
                    "operand local outside the locals table",
                )),
            lir::Operand::AddrOf(_) => Ok(Type::ByRef),
            lir::Operand::Const(k) => Ok(match k {
                Const::Int32(_) => Type::Int32,
                Const::Int64(_) => Type::Int64,
                Const::NativeInt(_) => Type::NativeInt,
                Const::Float(_) => Type::Float,
                Const::Double(_) => Type::Double,
                Const::NullRef => Type::Ref,
            }),
        }
    }

    fn push(out: &mut Vec<lir::Stmt>, il_offset: IlOffset, kind: lir::StmtKind) {
        out.push(lir::Stmt { il_offset, kind });
    }

    fn lower_block(
        &mut self,
        block: &hir::Block,
        next: Option<BlockId>,
    ) -> CompileResult<lir::Block> {
        let mut stmts = Vec::new();
        for stmt in &block.stmts {
            match &stmt.kind {
                hir::StmtKind::Store { dst, value } => {
                    let src = self.flatten_expr(value, &mut stmts, stmt.il_offset)?;
                    Self::push(
                        &mut stmts,
                        stmt.il_offset,
                        lir::StmtKind::Copy { dst: *dst, src },
                    );
                }
                hir::StmtKind::StoreInd { .. } => {
                    return Err(CompileError::Unsupported(
                        "store through a byref (stind/stfld): not yet supported",
                    ));
                }
                hir::StmtKind::Eval(expr) => {
                    self.flatten_eval(expr, &mut stmts, stmt.il_offset)?;
                }
            }
        }
        self.lower_terminator(&block.terminator, next, &mut stmts)?;
        Ok(lir::Block {
            id: block.id,
            stmts,
        })
    }

    /// `Eval` of a void call lowers to a `Call` statement with no
    /// destination; any other discarded expression still flattens (its
    /// side effects — calls inside the tree — must survive).
    fn flatten_eval(
        &mut self,
        expr: &hir::Expr,
        out: &mut Vec<lir::Stmt>,
        il: IlOffset,
    ) -> CompileResult<()> {
        if let hir::Expr::Call { target, sig, args } = expr {
            self.flatten_call(target, sig, args, out, il)?;
        } else {
            self.flatten_expr(expr, out, il)?;
        }
        Ok(())
    }

    /// Tree → flat statements; returns the operand holding the value.
    /// Children flatten depth-first in field order, preserving the IL
    /// push order the importer encoded (ir-design.md, "Evaluation order").
    fn flatten_expr(
        &mut self,
        expr: &hir::Expr,
        out: &mut Vec<lir::Stmt>,
        il: IlOffset,
    ) -> CompileResult<lir::Operand> {
        match expr {
            hir::Expr::Const(k) => Ok(lir::Operand::Const(*k)),
            hir::Expr::Local(id) => Ok(lir::Operand::Local(*id)),
            hir::Expr::LocalAddr(id) => Ok(lir::Operand::AddrOf(*id)),
            hir::Expr::Binary { op, lhs, rhs } => {
                let lhs = self.flatten_expr(lhs, out, il)?;
                let rhs = self.flatten_expr(rhs, out, il)?;
                // A compare's result is Int32 (ECMA-335 III.1.5); every
                // other binary op has its (integer) operands' type — for
                // shifts that is the value operand's, per the importer.
                let ty = if is_compare(*op) {
                    Type::Int32
                } else {
                    self.operand_ty(&lhs)?
                };
                let dst = self.temp(ty);
                Self::push(
                    out,
                    il,
                    lir::StmtKind::Binary {
                        dst,
                        op: *op,
                        lhs,
                        rhs,
                    },
                );
                Ok(lir::Operand::Temp(dst))
            }
            hir::Expr::Call { target, sig, args } => {
                match self.flatten_call(target, sig, args, out, il)? {
                    Some(dst) => Ok(lir::Operand::Temp(dst)),
                    // The importer's well-typedness guarantee makes a void
                    // call in value position unreachable.
                    None => Err(CompileError::Internal("void call used as a value")),
                }
            }
            hir::Expr::Load { .. } | hir::Expr::FieldAddr { .. } => Err(CompileError::Unsupported(
                "load through a byref / field access: not yet supported",
            )),
            hir::Expr::StaticFieldAddr { .. } => Err(CompileError::Unsupported(
                "static fields: not yet supported",
            )),
            hir::Expr::Unary { op, arg } => {
                let src = self.flatten_expr(arg, out, il)?;
                let ty = self.operand_ty(&src)?;
                let dst = self.temp(ty);
                Self::push(out, il, lir::StmtKind::Unary { dst, op: *op, src });
                Ok(lir::Operand::Temp(dst))
            }
            hir::Expr::Conv {
                to,
                overflow,
                unsigned,
                arg,
            } => {
                // Checked (`.ovf`) conversions need OverflowException
                // sites; the importer only builds unchecked ones.
                if *overflow {
                    return Err(CompileError::Unsupported("checked (ovf) conversion"));
                }
                let src = self.flatten_expr(arg, out, il)?;
                let dst = self.temp(*to);
                Self::push(
                    out,
                    il,
                    lir::StmtKind::Conv {
                        dst,
                        to: *to,
                        overflow: *overflow,
                        unsigned: *unsigned,
                        src,
                    },
                );
                Ok(lir::Operand::Temp(dst))
            }
            hir::Expr::NullCheck { .. } => {
                Err(CompileError::Unsupported("null checks: not yet supported"))
            }
            hir::Expr::ArrLen { .. } | hir::Expr::ArrElemAddr { .. } => {
                Err(CompileError::Unsupported("arrays: not yet supported"))
            }
            hir::Expr::Cast { .. } | hir::Expr::Box { .. } => {
                Err(CompileError::Unsupported("cast/box: not yet supported"))
            }
            hir::Expr::StructVal { .. } => {
                Err(CompileError::Unsupported("structs: not yet supported"))
            }
        }
    }

    /// Lifts a call to a top-level LIR statement. Arguments flatten in
    /// list order — the IL push order the importer and morph certify —
    /// and an indirect target's address expression evaluates *after* the
    /// arguments, as `calli` requires. Returns the fresh temp holding the
    /// result, or `None` for `Type::Void`.
    fn flatten_call(
        &mut self,
        target: &CallTarget<hir::Expr>,
        sig: &CallSig,
        args: &[hir::Expr],
        out: &mut Vec<lir::Stmt>,
        il: IlOffset,
    ) -> CompileResult<Option<LocalId>> {
        if args.len() != sig.args.len() + usize::from(sig.has_this) {
            return Err(CompileError::Internal(
                "call argument list does not match its signature",
            ));
        }
        let mut operands = Vec::with_capacity(args.len());
        for arg in args {
            operands.push(self.flatten_expr(arg, out, il)?);
        }
        let target = match target {
            CallTarget::Direct(m) => CallTarget::Direct(*m),
            CallTarget::Virtual { method } => CallTarget::Virtual { method: *method },
            CallTarget::Helper(h) => CallTarget::Helper(*h),
            CallTarget::Indirect(addr) => {
                CallTarget::Indirect(Box::new(self.flatten_expr(addr, out, il)?))
            }
        };
        let dst = if sig.ret == Type::Void {
            None
        } else {
            Some(self.temp(sig.ret))
        };
        Self::push(
            out,
            il,
            lir::StmtKind::Call {
                dst,
                target,
                sig: sig.clone(),
                args: operands,
            },
        );
        Ok(dst)
    }

    /// Terminators become the block's trailing statements. Derived
    /// statements carry `IL_OFFSET_NONE`: HIR records offsets on
    /// statements, not terminators, so there is no source offset to
    /// propagate (recorded in decisions/2026-09-11-lowering-rule-dsl.md).
    fn lower_terminator(
        &mut self,
        term: &hir::Terminator,
        next: Option<BlockId>,
        out: &mut Vec<lir::Stmt>,
    ) -> CompileResult<()> {
        match term {
            hir::Terminator::Jump { target } => {
                // A jump to the next block in layout order is a no-op.
                if next != Some(*target) {
                    Self::push(out, IL_OFFSET_NONE, lir::StmtKind::Jump { target: *target });
                }
            }
            hir::Terminator::Branch { cond, then, else_ } => {
                let cond = match cond {
                    hir::Expr::Binary { op, lhs, rhs } if is_compare(*op) => {
                        let lhs = self.flatten_expr(lhs, out, IL_OFFSET_NONE)?;
                        let rhs = self.flatten_expr(rhs, out, IL_OFFSET_NONE)?;
                        lir::BranchCond::Cmp { op: *op, lhs, rhs }
                    }
                    _ => {
                        let value = self.flatten_expr(cond, out, IL_OFFSET_NONE)?;
                        lir::BranchCond::True(value)
                    }
                };
                Self::push(
                    out,
                    IL_OFFSET_NONE,
                    lir::StmtKind::Branch {
                        cond,
                        target: *then,
                    },
                );
                if next != Some(*else_) {
                    Self::push(out, IL_OFFSET_NONE, lir::StmtKind::Jump { target: *else_ });
                }
            }
            hir::Terminator::Return { value } => {
                let value = value
                    .as_ref()
                    .map(|e| self.flatten_expr(e, out, IL_OFFSET_NONE))
                    .transpose()?;
                Self::push(out, IL_OFFSET_NONE, lir::StmtKind::Return { value });
            }
            hir::Terminator::Switch { .. } => {
                return Err(CompileError::Unsupported("switch: not yet supported"));
            }
            hir::Terminator::Throw { .. }
            | hir::Terminator::Leave { .. }
            | hir::Terminator::EndFinally => {
                return Err(CompileError::Unsupported(
                    "EH control flow: not yet supported",
                ));
            }
        }
        Ok(())
    }
}

/// The comparison half of [`BinaryOp`] (value-producing `ceq`/`clt` forms
/// and branch conditions share these operators).
fn is_compare(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::ULt
            | BinaryOp::Le
            | BinaryOp::ULe
            | BinaryOp::Gt
            | BinaryOp::UGt
            | BinaryOp::Ge
            | BinaryOp::UGe
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{hir, lir, BlockId, CallSig, Const, LocalId};
    use crate::pipeline::MethodInfo;
    use crate::target::{CallAbi, RegClassId, RegisterClass};
    use rokajit_ee::enums::CorInfoType;
    use rokajit_ee::handles::MethodHandle;
    use rokajit_ee::mock::{MockEe, MockSig};
    use rokajit_ffi as ffi;

    /// A minimal target for driver tests: every non-struct type lands in
    /// one register class; calls are not classified (07.5's job).
    struct MockTarget;

    impl Target for MockTarget {
        fn pointer_size(&self) -> u8 {
            8
        }
        fn register_classes(&self) -> &'static [RegisterClass] {
            &[]
        }
        fn class_of(&self, ty: Type) -> Option<RegClassId> {
            match ty {
                Type::Struct(_) | Type::Void => None,
                _ => Some(RegClassId(0)),
            }
        }
        fn classify_call(&self, _sig: &CallSig) -> CompileResult<CallAbi> {
            Err(CompileError::Unsupported("mock target"))
        }
        fn call_site_stack_alignment(&self) -> u32 {
            16
        }
    }

    fn local(ty: Type, kind: hir::LocalKind) -> hir::Local {
        hir::Local {
            ty,
            kind,
            pinned: false,
        }
    }

    fn int_arg(i: u32) -> hir::Local {
        local(Type::Int32, hir::LocalKind::IlArg(i))
    }

    /// `int f(int n)` shape: one arg, no IL locals.
    fn method_with(block: hir::Block) -> hir::Method {
        hir::Method {
            blocks: vec![block],
            locals: vec![int_arg(0)],
            eh_regions: Vec::new(),
            num_args: 1,
            num_il_locals: 0,
        }
    }

    fn block(id: u32, stmts: Vec<hir::Stmt>, terminator: hir::Terminator) -> hir::Block {
        hir::Block {
            id: BlockId(id),
            stmts,
            terminator,
        }
    }

    fn hstmt(kind: hir::StmtKind) -> hir::Stmt {
        hir::Stmt {
            il_offset: IlOffset(3),
            kind,
        }
    }

    fn lower_ok(m: hir::Method) -> lir::Method {
        lower(m, &MockTarget).expect("lowers")
    }

    #[test]
    fn store_of_tree_flattens_to_binary_then_copy() {
        // stloc.0 of (ldarg.0 + 5): the tree decomposes to one Binary
        // statement feeding a Copy, preserving the source IL offset.
        let m = method_with(block(
            0,
            vec![hstmt(hir::StmtKind::Store {
                dst: LocalId(0),
                value: hir::Expr::Binary {
                    op: BinaryOp::Add,
                    lhs: Box::new(hir::Expr::Local(LocalId(0))),
                    rhs: Box::new(hir::Expr::Const(Const::Int32(5))),
                },
            })],
            hir::Terminator::Return { value: None },
        ));
        let m = lower_ok(m);
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 3);
        match &stmts[0].kind {
            lir::StmtKind::Binary { dst, op, lhs, rhs } => {
                assert_eq!(*dst, LocalId(1), "fresh temp after the one arg");
                assert_eq!(*op, BinaryOp::Add);
                assert_eq!(*lhs, lir::Operand::Local(LocalId(0)));
                assert_eq!(*rhs, lir::Operand::Const(Const::Int32(5)));
            }
            _ => panic!("expected Binary"),
        }
        assert_eq!(stmts[0].il_offset, IlOffset(3), "IL offset propagates");
        match &stmts[1].kind {
            lir::StmtKind::Copy { dst, src } => {
                assert_eq!(*dst, LocalId(0));
                assert_eq!(*src, lir::Operand::Temp(LocalId(1)));
            }
            _ => panic!("expected Copy"),
        }
        assert!(matches!(
            stmts[2].kind,
            lir::StmtKind::Return { value: None }
        ));
        // The temp landed in the locals table, typed like its source.
        assert_eq!(m.locals.len(), 2);
        assert_eq!(m.locals[1].ty, Type::Int32);
        assert_eq!(m.locals[1].kind, hir::LocalKind::Temp);
    }

    #[test]
    fn nested_calls_lift_in_push_order() {
        // return fib(n-1) + fib(n-2): both calls become top-level
        // statements, in tree (IL push) order, before the add.
        let fib = MethodHandle::from_raw(0x42usize as ffi::CORINFO_METHOD_HANDLE).unwrap();
        let sig = CallSig {
            ret: Type::Int32,
            args: vec![Type::Int32],
            has_this: false,
        };
        let call = |sub_by: i32| hir::Expr::Call {
            target: CallTarget::Direct(fib),
            sig: sig.clone(),
            args: vec![hir::Expr::Binary {
                op: BinaryOp::Sub,
                lhs: Box::new(hir::Expr::Local(LocalId(0))),
                rhs: Box::new(hir::Expr::Const(Const::Int32(sub_by))),
            }],
        };
        let m = method_with(block(
            0,
            Vec::new(),
            hir::Terminator::Return {
                value: Some(hir::Expr::Binary {
                    op: BinaryOp::Add,
                    lhs: Box::new(call(1)),
                    rhs: Box::new(call(2)),
                }),
            },
        ));
        let m = lower_ok(m);
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 6, "sub, call, sub, call, add, return");
        // Temps: t1 = n-1, t2 = fib(t1), t3 = n-2, t4 = fib(t3), t5 = t2 + t4.
        assert!(matches!(
            stmts[0].kind,
            lir::StmtKind::Binary {
                dst: LocalId(1),
                op: BinaryOp::Sub,
                ..
            }
        ));
        match &stmts[1].kind {
            lir::StmtKind::Call {
                dst, target, args, ..
            } => {
                assert_eq!(*dst, Some(LocalId(2)));
                assert!(matches!(target, CallTarget::Direct(_)));
                assert_eq!(args.as_slice(), &[lir::Operand::Temp(LocalId(1))]);
            }
            _ => panic!("expected Call"),
        }
        match &stmts[3].kind {
            lir::StmtKind::Call { dst, args, .. } => {
                assert_eq!(*dst, Some(LocalId(4)));
                assert_eq!(args.as_slice(), &[lir::Operand::Temp(LocalId(3))]);
            }
            _ => panic!("expected Call"),
        }
        match &stmts[4].kind {
            lir::StmtKind::Binary { dst, op, lhs, rhs } => {
                assert_eq!(*dst, LocalId(5));
                assert_eq!(*op, BinaryOp::Add);
                assert_eq!(*lhs, lir::Operand::Temp(LocalId(2)));
                assert_eq!(*rhs, lir::Operand::Temp(LocalId(4)));
            }
            _ => panic!("expected Binary"),
        }
        assert!(matches!(
            stmts[5].kind,
            lir::StmtKind::Return {
                value: Some(lir::Operand::Temp(LocalId(5)))
            }
        ));
        // Terminator-derived statements carry no IL offset.
        assert_eq!(stmts[5].il_offset, IL_OFFSET_NONE);
    }

    #[test]
    fn branch_folds_compare_and_elides_fallthrough_jump() {
        // if (n < 2) goto B2; ...B1... ; B2: ... — B1 is next in layout,
        // so no Jump statement follows the Branch.
        let cond = hir::Expr::Binary {
            op: BinaryOp::Lt,
            lhs: Box::new(hir::Expr::Local(LocalId(0))),
            rhs: Box::new(hir::Expr::Const(Const::Int32(2))),
        };
        let mut m = method_with(block(
            0,
            Vec::new(),
            hir::Terminator::Branch {
                cond,
                then: BlockId(2),
                else_: BlockId(1),
            },
        ));
        m.blocks.push(block(
            1,
            Vec::new(),
            hir::Terminator::Return { value: None },
        ));
        m.blocks.push(block(
            2,
            Vec::new(),
            hir::Terminator::Return { value: None },
        ));
        let m = lower_ok(m);
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1, "fallthrough to B1 needs no Jump");
        match &stmts[0].kind {
            lir::StmtKind::Branch { cond, target } => {
                assert_eq!(*target, BlockId(2));
                assert_eq!(
                    *cond,
                    lir::BranchCond::Cmp {
                        op: BinaryOp::Lt,
                        lhs: lir::Operand::Local(LocalId(0)),
                        rhs: lir::Operand::Const(Const::Int32(2)),
                    }
                );
            }
            _ => panic!("expected Branch"),
        }
    }

    #[test]
    fn branch_to_non_next_block_emits_jump() {
        // else_ is not the next block in layout order → explicit Jump.
        let cond = hir::Expr::Local(LocalId(0));
        let mut m = method_with(block(
            0,
            Vec::new(),
            hir::Terminator::Branch {
                cond,
                then: BlockId(1),
                else_: BlockId(2),
            },
        ));
        m.blocks.push(block(
            1,
            Vec::new(),
            hir::Terminator::Return { value: None },
        ));
        m.blocks.push(block(
            2,
            Vec::new(),
            hir::Terminator::Return { value: None },
        ));
        let m = lower_ok(m);
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2);
        assert!(matches!(
            stmts[0].kind,
            lir::StmtKind::Branch {
                cond: lir::BranchCond::True(lir::Operand::Local(LocalId(0))),
                target: BlockId(1),
            }
        ));
        assert!(matches!(
            stmts[1].kind,
            lir::StmtKind::Jump { target: BlockId(2) }
        ));
    }

    #[test]
    fn unsupported_nodes_fail_with_unsupported() {
        // One representative per rejected family: a load through a byref
        // in an expression, an indirect store, and a switch terminator.
        let load = method_with(block(
            0,
            Vec::new(),
            hir::Terminator::Return {
                value: Some(hir::Expr::Load {
                    addr: Box::new(hir::Expr::Local(LocalId(0))),
                    offset: 0,
                    ty: Type::Int32,
                }),
            },
        ));
        assert!(matches!(
            lower(load, &MockTarget),
            Err(CompileError::Unsupported(_))
        ));

        let store_ind = method_with(block(
            0,
            vec![hstmt(hir::StmtKind::StoreInd {
                addr: hir::Expr::Local(LocalId(0)),
                offset: 0,
                value: hir::Expr::Const(Const::Int32(0)),
            })],
            hir::Terminator::Return { value: None },
        ));
        assert!(matches!(
            lower(store_ind, &MockTarget),
            Err(CompileError::Unsupported(_))
        ));

        let switch = method_with(block(
            0,
            Vec::new(),
            hir::Terminator::Switch {
                value: hir::Expr::Local(LocalId(0)),
                targets: Vec::new(),
                default: BlockId(0),
            },
        ));
        assert!(matches!(
            lower(switch, &MockTarget),
            Err(CompileError::Unsupported(_))
        ));
    }

    #[test]
    fn struct_local_is_rejected_by_the_target_gate() {
        let mut m = method_with(block(
            0,
            Vec::new(),
            hir::Terminator::Return { value: None },
        ));
        let class = rokajit_ee::handles::ClassHandle::from_raw(1usize as ffi::CORINFO_CLASS_HANDLE)
            .unwrap();
        m.locals
            .push(local(Type::Struct(class), hir::LocalKind::IlLocal(0)));
        m.num_il_locals = 1;
        assert!(matches!(
            lower(m, &MockTarget),
            Err(CompileError::Unsupported(_))
        ));
    }

    // --- end-to-end via the importer's MockEe fixtures (as morph's tests) ---

    const FIB_TOKEN: u32 = 0x0600_0001;

    fn fib_fixture(il: &[u8]) -> (MockEe, MethodInfo) {
        let mut ee = MockEe::default();
        let fib_sig = MockSig {
            ret: CorInfoType::Int,
            args: vec![CorInfoType::Int],
            has_this: false,
        };
        ee.add_method(FIB_TOKEN, fib_sig.clone());
        let info = MethodInfo {
            ftn: MethodHandle::from_raw(1usize as ffi::CORINFO_METHOD_HANDLE).unwrap(),
            il: il.to_vec(),
            max_stack: 8,
            eh_count: 0,
            init_locals: false,
            args: ee.make_method_sig(&fib_sig),
            locals: ee.make_locals_sig(&[]),
        };
        (ee, info)
    }

    #[test]
    fn fib_il_lowers_end_to_end() {
        // The exact Fib_ bytes from tests/bin/fib.dll.
        let il = [
            0x02, 0x18, 0x32, 0x12, 0x02, 0x17, 0x59, 0x28, 0x01, 0x00, 0x00, 0x06, 0x02, 0x18,
            0x59, 0x28, 0x01, 0x00, 0x00, 0x06, 0x58, 0x2A, 0x02, 0x2A,
        ];
        let (ee, info) = fib_fixture(&il);
        let hir = crate::import::import(&info, &ee).expect("imports");
        let hir = crate::morph::morph(hir).expect("morphs");
        let m = lower(hir, &MockTarget).expect("lowers");

        assert_eq!(m.blocks.len(), 3);
        // Block 0: the `n < 2` guard; the else target (block 1) is next
        // in layout, so the block is exactly one Branch.
        let b0 = &m.blocks[0].stmts;
        assert_eq!(b0.len(), 1);
        assert!(matches!(
            b0[0].kind,
            lir::StmtKind::Branch {
                cond: lir::BranchCond::Cmp {
                    op: BinaryOp::Lt,
                    ..
                },
                target: BlockId(2),
            }
        ));
        // Block 1: sub, call, sub, call, add, return — one operation per
        // statement, temps threaded in push order.
        let b1 = &m.blocks[1].stmts;
        assert_eq!(b1.len(), 6);
        assert!(matches!(b1[1].kind, lir::StmtKind::Call { .. }));
        assert!(matches!(b1[3].kind, lir::StmtKind::Call { .. }));
        assert!(matches!(
            b1[4].kind,
            lir::StmtKind::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
        assert!(matches!(b1[5].kind, lir::StmtKind::Return { .. }));
        // Block 2: return n — a Copy-free Return of the argument.
        let b2 = &m.blocks[2].stmts;
        assert_eq!(b2.len(), 1);
        assert!(matches!(
            b2[0].kind,
            lir::StmtKind::Return {
                value: Some(lir::Operand::Local(LocalId(0)))
            }
        ));
        // One arg + five temps.
        assert_eq!(m.locals.len(), 6);
    }

    #[test]
    fn pipeline_stage_delegates() {
        let m = crate::pipeline::lower(
            method_with(block(
                0,
                Vec::new(),
                hir::Terminator::Return { value: None },
            )),
            &MockTarget,
        );
        assert!(m.is_ok());
    }

    // --- step_10.1: scalar-cheap pack flattening ---

    /// `return <expr>` with one arg, no IL locals.
    fn lower_ret(expr: hir::Expr) -> lir::Method {
        lower_ok(method_with(block(
            0,
            Vec::new(),
            hir::Terminator::Return { value: Some(expr) },
        )))
    }

    #[test]
    fn compare_as_value_produces_an_int32_temp() {
        // return (arg0 < arg0): the Binary statement's temp is Int32 even
        // though the operands are — the compare-result typing rule.
        let m = lower_ret(hir::Expr::Binary {
            op: BinaryOp::Lt,
            lhs: Box::new(hir::Expr::Local(LocalId(0))),
            rhs: Box::new(hir::Expr::Local(LocalId(0))),
        });
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2);
        match &stmts[0].kind {
            lir::StmtKind::Binary { dst, op, .. } => {
                assert_eq!(*op, BinaryOp::Lt);
                assert_eq!(m.locals[dst.0 as usize].ty, Type::Int32);
            }
            _ => panic!("expected Binary"),
        }
        assert!(matches!(
            stmts[1].kind,
            lir::StmtKind::Return {
                value: Some(lir::Operand::Temp(_))
            }
        ));
    }

    #[test]
    fn logic_and_shift_ops_flatten_with_the_value_type() {
        // and/or/xor/shl/shr/shr.un/div family all become Binary statements.
        for op in [
            BinaryOp::And,
            BinaryOp::Or,
            BinaryOp::Xor,
            BinaryOp::Shl,
            BinaryOp::Shr,
            BinaryOp::UShr,
            BinaryOp::Div,
            BinaryOp::UDiv,
            BinaryOp::URem,
        ] {
            let m = lower_ret(hir::Expr::Binary {
                op,
                lhs: Box::new(hir::Expr::Local(LocalId(0))),
                rhs: Box::new(hir::Expr::Const(Const::Int32(1))),
            });
            match &m.blocks[0].stmts[0].kind {
                lir::StmtKind::Binary { op: got, .. } => assert_eq!(*got, op),
                _ => panic!("expected Binary for {op:?}"),
            }
        }
        // A shift's result temp carries the *value* operand's type. With a
        // 64-bit local the temp is Int64 even though the count is Int32.
        let mut m = method_with(block(
            0,
            Vec::new(),
            hir::Terminator::Return {
                value: Some(hir::Expr::Binary {
                    op: BinaryOp::Shl,
                    lhs: Box::new(hir::Expr::Local(LocalId(1))),
                    rhs: Box::new(hir::Expr::Local(LocalId(0))),
                }),
            },
        ));
        m.locals
            .push(local(Type::Int64, hir::LocalKind::IlLocal(0)));
        m.num_il_locals = 1;
        let m = lower_ok(m);
        match &m.blocks[0].stmts[0].kind {
            lir::StmtKind::Binary { dst, .. } => {
                assert_eq!(m.locals[dst.0 as usize].ty, Type::Int64)
            }
            _ => panic!("expected Binary"),
        }
    }

    #[test]
    fn unary_and_conv_flatten_to_their_statements() {
        let m = lower_ret(hir::Expr::Unary {
            op: crate::ir::UnaryOp::Neg,
            arg: Box::new(hir::Expr::Local(LocalId(0))),
        });
        assert!(matches!(
            m.blocks[0].stmts[0].kind,
            lir::StmtKind::Unary {
                op: crate::ir::UnaryOp::Neg,
                ..
            }
        ));
        // The neg temp takes the operand's type.
        match &m.blocks[0].stmts[0].kind {
            lir::StmtKind::Unary { dst, .. } => {
                assert_eq!(m.locals[dst.0 as usize].ty, Type::Int32)
            }
            _ => unreachable!(),
        }

        let m = lower_ret(hir::Expr::Conv {
            to: Type::Int64,
            overflow: false,
            unsigned: true,
            arg: Box::new(hir::Expr::Local(LocalId(0))),
        });
        match &m.blocks[0].stmts[0].kind {
            lir::StmtKind::Conv {
                dst,
                to,
                overflow,
                unsigned,
                src,
            } => {
                assert_eq!(*to, Type::Int64);
                assert!(!overflow && *unsigned);
                assert_eq!(*src, lir::Operand::Local(LocalId(0)));
                assert_eq!(m.locals[dst.0 as usize].ty, Type::Int64);
            }
            _ => panic!("expected Conv"),
        }

        // A checked conversion is rejected.
        let m = lower(
            method_with(block(
                0,
                Vec::new(),
                hir::Terminator::Return {
                    value: Some(hir::Expr::Conv {
                        to: Type::Int32,
                        overflow: true,
                        unsigned: false,
                        arg: Box::new(hir::Expr::Local(LocalId(0))),
                    }),
                },
            )),
            &MockTarget,
        );
        assert!(matches!(m, Err(CompileError::Unsupported(_))));
    }

    // --- step_10.2: float pack flattening ---
    #[test]
    fn float_ops_flatten_with_the_right_temp_types() {
        // A float binary op's temp carries the operand type (Double);
        // a float compare's temp is Int32 (ECMA-335 III.1.5).
        let mut m = method_with(block(
            0,
            Vec::new(),
            hir::Terminator::Return {
                value: Some(hir::Expr::Binary {
                    op: BinaryOp::Lt,
                    lhs: Box::new(hir::Expr::Binary {
                        op: BinaryOp::Div,
                        lhs: Box::new(hir::Expr::Local(LocalId(1))),
                        rhs: Box::new(hir::Expr::Const(Const::Double(2.0))),
                    }),
                    rhs: Box::new(hir::Expr::Const(Const::Double(0.0))),
                }),
            },
        ));
        m.locals
            .push(local(Type::Double, hir::LocalKind::IlLocal(0)));
        m.num_il_locals = 1;
        let m = lower_ok(m);
        let stmts = &m.blocks[0].stmts;
        // Div statement, then the compare, then the return.
        match &stmts[0].kind {
            lir::StmtKind::Binary { dst, op, .. } => {
                assert_eq!(*op, BinaryOp::Div);
                assert_eq!(m.locals[dst.0 as usize].ty, Type::Double);
            }
            _ => panic!("expected the Div"),
        }
        match &stmts[1].kind {
            lir::StmtKind::Binary { dst, op, .. } => {
                assert_eq!(*op, BinaryOp::Lt);
                assert_eq!(m.locals[dst.0 as usize].ty, Type::Int32);
            }
            _ => panic!("expected the compare"),
        }
    }

    #[test]
    fn float_conv_and_unary_flatten_like_ints() {
        let m = lower_ret(hir::Expr::Conv {
            to: Type::Double,
            overflow: false,
            unsigned: false,
            arg: Box::new(hir::Expr::Local(LocalId(0))),
        });
        match &m.blocks[0].stmts[0].kind {
            lir::StmtKind::Conv { dst, to, .. } => {
                assert_eq!(*to, Type::Double);
                assert_eq!(m.locals[dst.0 as usize].ty, Type::Double);
            }
            _ => panic!("expected Conv"),
        }
        let m = lower_ret(hir::Expr::Unary {
            op: crate::ir::UnaryOp::Neg,
            arg: Box::new(hir::Expr::Const(Const::Float(1.0))),
        });
        match &m.blocks[0].stmts[0].kind {
            lir::StmtKind::Unary { dst, .. } => {
                assert_eq!(m.locals[dst.0 as usize].ty, Type::Float);
            }
            _ => panic!("expected Unary"),
        }
    }
}
