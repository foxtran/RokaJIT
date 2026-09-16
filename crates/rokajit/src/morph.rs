//! Pipeline stage 2 (step_07.3): morph-lite — the pre-lowering
//! normalization the IR contract (`RokaJIT-internal/docs/ir-design.md`)
//! requires between import and lowering.
//!
//! For the fib subset, both concerns the step names check out as
//! **already normalized at import**, so this stage certifies the shape
//! rather than transforming:
//!
//! - **Call argument setup.** Argument order is structural in HIR: trees
//!   evaluate operands in field order, which is IL push order
//!   (ir-design.md "Evaluation order"), and [`hir::Expr::Call`] carries
//!   an explicit `args: Vec<Expr>` — the receiver first when
//!   [`CallSig::has_this`], then the declared args — exactly the operand
//!   list `lir::StmtKind::Call` and `Target::classify_call` (entry 0 is
//!   the implicit `this`) expect. The importer builds that shape
//!   directly, so there is nothing to rewrite. Morph checks the one
//!   invariant the types don't make structural — the arg list is
//!   complete (`args.len() == sig.args.len() + has_this`) — and reports
//!   a violation as an importer bug ([`CompileError::Internal`]).
//!
//! - **Return normalization.** Verified against the contract: it defines
//!   no single-exit shape. `return` is an ordinary block terminator in
//!   HIR ([`hir::Terminator::Return`]) and an ordinary statement in LIR
//!   (`lir::StmtKind::Return`), so a method with several `ret` sites is
//!   already in normalized form — lowering emits a per-block epilog.
//!   Merging exits (RyuJIT's `fgMergeReturns`) serves RyuJIT's shared
//!   epilog/GC-info machinery, not any requirement of this IR; per the
//!   step's scope rule, no code for its own sake.
//!
//! Nothing else belongs in morph-lite: no copy propagation, no folding,
//! no DCE. A transformation that doesn't change what lowering can
//! express isn't here.

use crate::error::{CompileError, CompileResult};
use crate::ir::{hir, CallTarget};

/// Stage entry point (the body of [`crate::pipeline::morph`]). Consumes
/// the imported method and returns it unchanged once the normalized
/// shape is certified — morph-lite rewrites nothing for the fib subset.
pub fn morph(method: hir::Method) -> CompileResult<hir::Method> {
    for block in &method.blocks {
        for stmt in &block.stmts {
            match &stmt.kind {
                hir::StmtKind::Store { value, .. } => certify_expr(value)?,
                hir::StmtKind::StoreInd { addr, value, .. } => {
                    certify_expr(addr)?;
                    certify_expr(value)?;
                }
                hir::StmtKind::BlockZero { addr, .. } => certify_expr(addr)?,
                hir::StmtKind::BlockCopyDyn { dst, src, size } => {
                    certify_expr(dst)?;
                    certify_expr(src)?;
                    certify_expr(size)?;
                }
                hir::StmtKind::BlockFillDyn { dst, fill, size } => {
                    certify_expr(dst)?;
                    certify_expr(fill)?;
                    certify_expr(size)?;
                }
                hir::StmtKind::BoundsCheck { array, index } => {
                    certify_expr(array)?;
                    certify_expr(index)?;
                }
                hir::StmtKind::Eval(expr) => certify_expr(expr)?,
            }
        }
        match &block.terminator {
            hir::Terminator::Branch { cond, .. } => certify_expr(cond)?,
            hir::Terminator::Switch { value, .. } => certify_expr(value)?,
            hir::Terminator::Return {
                value: Some(value), ..
            } => certify_expr(value)?,
            hir::Terminator::Throw { exception } => certify_expr(exception)?,
            hir::Terminator::EndFilter { value } => certify_expr(value)?,
            // The step_10.6 EH terminators carry no expressions: `leave`,
            // a `CallFinally` step, `endfinally`, and `rethrow` reference
            // blocks only (or nothing at all).
            hir::Terminator::Leave { .. }
            | hir::Terminator::CallFinally { .. }
            | hir::Terminator::Rethrow
            | hir::Terminator::EndFinally => {}
            _ => {}
        }
    }
    Ok(method)
}

/// Walks a tree depth-first checking that every call's explicit argument
/// list is complete — declared args plus the receiver when `has_this`.
/// Argument *order* needs no check: it is IL push order by construction
/// (the tree's field order), which is the order lowering flattens.
fn certify_expr(expr: &hir::Expr) -> CompileResult<()> {
    match expr {
        hir::Expr::Const(_)
        | hir::Expr::Local(_)
        | hir::Expr::LocalAddr(_)
        | hir::Expr::StaticFieldAddr { .. }
        | hir::Expr::CatchArg => {}
        hir::Expr::Load { addr, .. } => certify_expr(addr)?,
        hir::Expr::FieldAddr { obj, .. } => certify_expr(obj)?,
        hir::Expr::Unary { arg, .. } => certify_expr(arg)?,
        hir::Expr::Binary { lhs, rhs, .. } | hir::Expr::BinaryOvf { lhs, rhs, .. } => {
            certify_expr(lhs)?;
            certify_expr(rhs)?;
        }
        hir::Expr::Conv { arg, .. } => certify_expr(arg)?,
        hir::Expr::ConvOvf { arg, .. } | hir::Expr::CkFinite { arg } => certify_expr(arg)?,
        hir::Expr::Call { target, sig, args } => {
            if let CallTarget::Indirect(addr) = target {
                certify_expr(addr)?;
            }
            for arg in args {
                certify_expr(arg)?;
            }
            if args.len() != sig.args.len() + usize::from(sig.has_this) {
                return Err(CompileError::Internal(
                    "call argument list does not match its signature",
                ));
            }
        }
        hir::Expr::NullCheck { arg } => certify_expr(arg)?,
        hir::Expr::ArrLen { array } => certify_expr(array)?,
        hir::Expr::ArrElemAddr { array, index, .. } => {
            certify_expr(array)?;
            certify_expr(index)?;
        }
        hir::Expr::Cast { arg, .. } | hir::Expr::Box { arg, .. } => certify_expr(arg)?,
        hir::Expr::StructVal { addr, .. } => certify_expr(addr)?,
        hir::Expr::LocAlloc { size } => certify_expr(size)?,
        hir::Expr::FtnAddr { entry, .. } => certify_expr(entry)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BlockId, CallSig, Const, LocalId, Type};
    use crate::pipeline::MethodInfo;
    use crate::structs::StructLayouts;
    use rokajit_ee::enums::CorInfoType;
    use rokajit_ee::handles::MethodHandle;
    use rokajit_ee::mock::{MockEe, MockSig};
    use rokajit_ffi as ffi;

    const FIB_TOKEN: u32 = 0x0600_0001;
    const VOID_TOKEN: u32 = 0x0600_0002;
    const INST_TOKEN: u32 = 0x0600_0003;

    fn sig(ret: CorInfoType, args: &[CorInfoType]) -> MockSig {
        MockSig {
            ret,
            args: args.to_vec(),
            has_this: false,
            ret_class: None,
            arg_classes: Vec::new(),
        }
    }

    /// The importer's fixtures, mirrored: a MockEe resolving the three
    /// canned tokens plus a `MethodInfo` for an entry method with no IL
    /// locals.
    fn fixture(il: &[u8], entry: &MockSig) -> (MockEe, MethodInfo) {
        let mut ee = MockEe::default();
        ee.add_method(FIB_TOKEN, sig(CorInfoType::Int, &[CorInfoType::Int]));
        ee.add_method(
            VOID_TOKEN,
            sig(CorInfoType::Void, &[CorInfoType::Int, CorInfoType::Int]),
        );
        ee.add_method(
            INST_TOKEN,
            MockSig {
                ret: CorInfoType::Int,
                args: vec![CorInfoType::Int],
                has_this: true,
                ret_class: None,
                arg_classes: Vec::new(),
            },
        );
        let info = MethodInfo {
            ftn: MethodHandle::from_raw(1usize as ffi::CORINFO_METHOD_HANDLE).unwrap(),
            il: il.to_vec(),
            max_stack: 8,
            eh_count: 0,
            init_locals: false,
            generics_context: None,
            generics_context_keep_alive: false,
            args: ee.make_method_sig(entry),
            locals: ee.make_locals_sig(&[]),
        };
        (ee, info)
    }

    /// HIR in → normalized HIR out, `int f(int n)` shape.
    fn morph_ii(il: &[u8]) -> hir::Method {
        let (ee, info) = fixture(il, &sig(CorInfoType::Int, &[CorInfoType::Int]));
        morph(crate::import::import(&info, &ee).expect("imports")).expect("morphs")
    }

    // --- assertion helpers (same shapes as the importer's tests) ---

    fn as_local(e: &hir::Expr) -> LocalId {
        match e {
            hir::Expr::Local(id) => *id,
            _ => panic!("expected Expr::Local"),
        }
    }

    fn as_i32(e: &hir::Expr) -> i32 {
        match e {
            hir::Expr::Const(Const::Int32(v)) => *v,
            _ => panic!("expected Expr::Const(Int32)"),
        }
    }

    fn as_binary(e: &hir::Expr) -> (crate::ir::BinaryOp, &hir::Expr, &hir::Expr) {
        match e {
            hir::Expr::Binary { op, lhs, rhs } => (*op, lhs, rhs),
            _ => panic!("expected Expr::Binary"),
        }
    }

    fn as_call(e: &hir::Expr) -> (&CallTarget<hir::Expr>, &CallSig, &[hir::Expr]) {
        match e {
            hir::Expr::Call { target, sig, args } => (target, sig, args),
            _ => panic!("expected Expr::Call"),
        }
    }

    fn return_value(m: &hir::Method, block: usize) -> &hir::Expr {
        match &m.blocks[block].terminator {
            hir::Terminator::Return { value: Some(v) } => v,
            _ => panic!("expected Return with a value"),
        }
    }

    #[test]
    fn recursive_call_is_already_normalized() {
        // The exact Fib_ bytes from tests/bin/fib.dll (see the importer's
        // fib_end_to_end): recursive calls nest inside the add tree.
        let il = [
            0x02, 0x18, 0x32, 0x12, 0x02, 0x17, 0x59, 0x28, 0x01, 0x00, 0x00, 0x06, 0x02, 0x18,
            0x59, 0x28, 0x01, 0x00, 0x00, 0x06, 0x58, 0x2A, 0x02, 0x2A,
        ];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Int]));
        let hir_in = crate::import::import(&info, &ee).expect("imports");
        let (blocks, locals) = (hir_in.blocks.len(), hir_in.locals.len());
        let m = morph(hir_in).expect("morphs");

        // Morph is shape-preserving: same containers, same locals.
        assert_eq!(m.blocks.len(), blocks);
        assert_eq!(m.locals.len(), locals);
        assert_eq!(m.num_args, 1);

        // Block 1: return fib(n-1) + fib(n-2). Both calls carry explicit
        // arg lists in IL push order — the normalized shape lowering
        // expects, unchanged by morph.
        let (op, lhs, rhs) = as_binary(return_value(&m, 1));
        assert_eq!(op, crate::ir::BinaryOp::Add);
        for (call, sub_by) in [(lhs, 1), (rhs, 2)] {
            let (target, sig, args) = as_call(call);
            assert!(
                matches!(target, CallTarget::Direct(_)),
                "recursive call resolves to a direct target"
            );
            assert_eq!(sig.ret, Type::Int32);
            assert_eq!(sig.args, vec![Type::Int32]);
            assert!(!sig.has_this);
            assert_eq!(args.len(), 1);
            let (op, lhs, rhs) = as_binary(&args[0]);
            assert_eq!(op, crate::ir::BinaryOp::Sub);
            assert_eq!(as_local(lhs), LocalId(0));
            assert_eq!(as_i32(rhs), sub_by);
        }
    }

    #[test]
    fn multi_exit_returns_are_the_contract_shape() {
        // ldarg.0; brtrue.s L; ldc.i4.0; ret; L: ldc.i4.1; ret — two ret
        // sites. The contract permits multi-exit, so morph must not merge
        // them into a single-exit shape.
        let m = morph_ii(&[0x02, 0x2D, 0x02, 0x16, 0x2A, 0x17, 0x2A]);
        assert_eq!(m.blocks.len(), 3);
        let returns = m
            .blocks
            .iter()
            .filter(|b| matches!(b.terminator, hir::Terminator::Return { .. }))
            .count();
        assert_eq!(returns, 2, "morph keeps both ret sites");
        assert_eq!(as_i32(return_value(&m, 1)), 0);
        assert_eq!(as_i32(return_value(&m, 2)), 1);
    }

    #[test]
    fn void_call_args_keep_push_order() {
        // ldc.i4.1; ldc.i4.2; call void(int,int); ldarg.0; ret.
        let m = morph_ii(&[0x17, 0x18, 0x28, 0x02, 0x00, 0x00, 0x06, 0x02, 0x2A]);
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::Eval(call) => {
                let (_, sig, args) = as_call(call);
                assert_eq!(sig.ret, Type::Void);
                assert_eq!(args.len(), sig.args.len());
                assert_eq!(as_i32(&args[0]), 1);
                assert_eq!(as_i32(&args[1]), 2);
            }
            _ => panic!("expected StmtKind::Eval"),
        }
    }

    #[test]
    fn instance_call_keeps_receiver_first() {
        // ldarg.0 (this); ldarg.1; call int inst(int); ret — the receiver
        // heads the explicit arg list, matching classify_call's entry 0.
        let il = [0x02, 0x03, 0x28, 0x03, 0x00, 0x00, 0x06, 0x2A];
        let entry = MockSig {
            ret: CorInfoType::Int,
            args: vec![CorInfoType::Int],
            has_this: true,
            ret_class: None,
            arg_classes: Vec::new(),
        };
        let (ee, info) = fixture(&il, &entry);
        let m = morph(crate::import::import(&info, &ee).expect("imports")).expect("morphs");
        let (_, sig, args) = as_call(return_value(&m, 0));
        assert!(sig.has_this);
        assert_eq!(args.len(), sig.args.len() + 1);
        assert_eq!(as_local(&args[0]), LocalId(0));
        assert_eq!(as_local(&args[1]), LocalId(1));
    }

    #[test]
    fn incomplete_call_arg_list_is_an_importer_bug() {
        // Hand-built HIR whose call is missing its one declared argument:
        // the shape morph certifies, so this must be an Internal (ICE),
        // never silently passed to lowering.
        let target = MethodHandle::from_raw(2usize as ffi::CORINFO_METHOD_HANDLE).unwrap();
        let method = hir::Method {
            blocks: vec![hir::Block {
                id: BlockId(0),
                stmts: Vec::new(),
                terminator: hir::Terminator::Return {
                    value: Some(hir::Expr::Call {
                        target: CallTarget::Direct(target),
                        sig: CallSig {
                            ret: Type::Int32,
                            args: vec![Type::Int32],
                            has_this: false,
                        },
                        args: Vec::new(),
                    }),
                },
            }],
            locals: Vec::new(),
            eh_regions: Vec::new(),
            num_args: 0,
            num_il_locals: 0,
            struct_layouts: StructLayouts::new(),
            generics_context: None,
        };
        assert!(matches!(morph(method), Err(CompileError::Internal(_))));
    }

    #[test]
    fn eh_nodes_certify() {
        // The step_10.6 shapes: a Throw whose exception is a (complete)
        // call tree, a CatchArg entry store, and the no-expression
        // Leave / CallFinally / EndFinally terminators.
        let target = MethodHandle::from_raw(2usize as ffi::CORINFO_METHOD_HANDLE).unwrap();
        let method = hir::Method {
            blocks: vec![
                hir::Block {
                    id: BlockId(0),
                    stmts: Vec::new(),
                    terminator: hir::Terminator::Throw {
                        exception: hir::Expr::Call {
                            target: CallTarget::Direct(target),
                            sig: CallSig {
                                ret: Type::Ref,
                                args: Vec::new(),
                                has_this: false,
                            },
                            args: Vec::new(),
                        },
                    },
                },
                hir::Block {
                    id: BlockId(1),
                    stmts: Vec::new(),
                    terminator: hir::Terminator::CallFinally {
                        funclet: BlockId(3),
                        continuation: BlockId(2),
                    },
                },
                hir::Block {
                    id: BlockId(2),
                    stmts: Vec::new(),
                    terminator: hir::Terminator::Leave { target: BlockId(0) },
                },
                hir::Block {
                    id: BlockId(3),
                    stmts: vec![hir::Stmt {
                        il_offset: crate::ir::IlOffset(0),
                        kind: hir::StmtKind::Store {
                            dst: LocalId(0),
                            value: hir::Expr::CatchArg,
                        },
                    }],
                    terminator: hir::Terminator::EndFinally,
                },
            ],
            locals: Vec::new(),
            eh_regions: Vec::new(),
            num_args: 0,
            num_il_locals: 0,
            struct_layouts: StructLayouts::new(),
            generics_context: None,
        };
        morph(method).expect("the EH shapes certify");
    }

    #[test]
    fn array_nodes_certify() {
        // The step_10.8 shapes: a BoundsCheck statement, and ArrLen /
        // ArrElemAddr (with its baked element size) trees.
        let method = hir::Method {
            blocks: vec![hir::Block {
                id: BlockId(0),
                stmts: vec![hir::Stmt {
                    il_offset: crate::ir::IlOffset(0),
                    kind: hir::StmtKind::BoundsCheck {
                        array: hir::Expr::Local(LocalId(0)),
                        index: hir::Expr::Const(Const::Int32(0)),
                    },
                }],
                terminator: hir::Terminator::Return {
                    value: Some(hir::Expr::ArrLen {
                        array: Box::new(hir::Expr::Load {
                            addr: Box::new(hir::Expr::ArrElemAddr {
                                array: Box::new(hir::Expr::Local(LocalId(0))),
                                index: Box::new(hir::Expr::Const(Const::Int32(1))),
                                elem: Type::Int32,
                                elem_size: 4,
                            }),
                            offset: 0,
                            ty: Type::Int32,
                            access: crate::ir::MemAccess::Natural,
                        }),
                    }),
                },
            }],
            locals: Vec::new(),
            eh_regions: Vec::new(),
            num_args: 0,
            num_il_locals: 0,
            struct_layouts: StructLayouts::new(),
            generics_context: None,
        };
        morph(method).expect("the array shapes certify");
    }

    #[test]
    fn pipeline_stage_delegates() {
        // The 07.1 driver wiring: pipeline::morph is no longer a stub.
        let m = crate::pipeline::morph(hir::Method {
            blocks: vec![hir::Block {
                id: BlockId(0),
                stmts: Vec::new(),
                terminator: hir::Terminator::Return { value: None },
            }],
            locals: Vec::new(),
            eh_regions: Vec::new(),
            num_args: 0,
            num_il_locals: 0,
            struct_layouts: StructLayouts::new(),
            generics_context: None,
        });
        assert!(m.is_ok());
    }
}
