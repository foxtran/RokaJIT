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
//! unary ops, and the `conv.*` nodes (int↔float included). The step_10.4
//! object pack adds: loads and stores through byrefs at a constant offset
//! (`ldfld`/`stfld` — [`lir::StmtKind::Load`]/[`lir::StmtKind::Store`]),
//! field addresses (`FieldAddr` flattens to a `ByRef`-typed `Add` of the
//! object and the EE-supplied offset), and the explicit, trap-based null
//! check (flattened to [`lir::StmtKind::NullCheck`], yielding the checked
//! value unchanged). The step_10.6 EH pack lowers `throw`/`leave`/
//! `endfinally`/call-finally terminators and the catch-handler entry
//! store to their LIR forms and carries the region table through. The
//! step_10.8 array pack expands `ArrLen` to the length load (offset 8,
//! doubling as the null check) and `ArrElemAddr` to the
//! index-scale-plus-header `mul`/`add` chain ending in a ByRef temp, and
//! passes `BoundsCheck` statements through to LIR. Everything else —
//! switches — fails with
//! [`CompileError::Unsupported`].

mod dsl;

use crate::error::{CompileError, CompileResult};
use crate::ir::{
    hir, lir, BinaryOp, BlockId, CallSig, CallTarget, Const, IlOffset, LocalId, Type,
    IL_OFFSET_NONE,
};
use crate::structs::StructLayouts;
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
/// type-driven decisions (e.g. 32- vs 64-bit forms), and the struct
/// layout side table (step_10.9), for struct sizes and SysV
/// classifications. Fallible lookups return `Option` so a ruleset guard
/// treats an out-of-contract reference as "no rule matched" rather than
/// panicking.
pub struct Cx<'a> {
    locals: &'a [hir::Local],
    layouts: &'a StructLayouts,
}

impl<'a> Cx<'a> {
    pub fn new(locals: &'a [hir::Local], layouts: &'a StructLayouts) -> Self {
        Cx { locals, layouts }
    }

    /// The type of a local/arg/temp slot.
    pub fn ty_of(&self, id: LocalId) -> Option<Type> {
        self.locals.get(id.0 as usize).map(|l| l.ty)
    }

    /// The whole locals table, for rulesets that need more than types.
    pub fn locals(&self) -> &'a [hir::Local] {
        self.locals
    }

    /// The layout of a value class (step_10.9).
    pub fn layout_of(
        &self,
        class: rokajit_ee::handles::ClassHandle,
    ) -> Option<&crate::structs::StructLayout> {
        self.layouts.get(&class)
    }

    /// The whole struct layout side table (call classification).
    pub fn layouts(&self) -> &'a StructLayouts {
        self.layouts
    }
}

/// Stage entry point (the body of [`crate::pipeline::lower`]).
///
/// Consumes the morphed method and produces LIR per the IR contract:
/// one operation per statement, operands restricted to
/// [`lir::Operand`], calls only as top-level `StmtKind::Call`, branches
/// folded to `BranchCond`, temps defined exactly once before use.
pub fn lower(method: hir::Method, target: &dyn Target) -> CompileResult<lir::Method> {
    for local in &method.locals {
        if target.class_of(local.ty, &method.struct_layouts).is_none() {
            return Err(CompileError::Unsupported(
                "a local's type has no register class on this target",
            ));
        }
    }
    let mut fx = Flatten {
        locals: method.locals,
        layouts: &method.struct_layouts,
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
        struct_layouts: method.struct_layouts,
        generics_context: method.generics_context,
    })
}

/// The flattener's state: the locals table, grown with fresh temps as
/// expression trees are decomposed, and the struct layout side table
/// (step_10.9 — the register-passed classification drives call results).
/// Temp ids follow the IR contract's flat `LocalId` namespace (temps come
/// after args and IL locals).
struct Flatten<'a> {
    locals: Vec<hir::Local>,
    layouts: &'a StructLayouts,
}

impl Flatten<'_> {
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
                Const::NullRef | Const::FrozenRef(_) => Type::Ref,
            }),
        }
    }

    fn push(out: &mut Vec<lir::Stmt>, il_offset: IlOffset, kind: lir::StmtKind) {
        out.push(lir::Stmt { il_offset, kind });
    }

    /// float → unsigned integer (step_10.11), the saturating semantics
    /// RyuJIT implements since .NET 9 (measured against the reference
    /// JIT): NaN and negatives → 0, in-range values truncate toward zero,
    /// values at/above 2^N saturate to the target's maximum. The sequence
    /// is lowerxarch.cpp's pre-AVX512 lowering:
    ///
    /// ```text
    /// f = maxs(src, 0.0)               // negatives *and* NaN → +0 (the
    ///                                  // second operand wins on NaN)
    /// r = cvtt(f)                      // signed 64-bit; INT64_MIN on
    ///                                  // overflow (f ≥ 2^63)
    /// // 64-bit targets only:
    /// n = cvtt(f - 2^64)               // the wrapped (negative-domain)
    ///                                  // value; the subtraction is exact
    ///                                  // whenever f ≥ 2^63
    /// c = r | (n & (r >> 63))          // wrapped bits iff r overflowed
    /// // both widths:
    /// mask = clt.un(f, 2^N) - 1        // all-ones iff f ≥ 2^N; f is
    ///                                  // never NaN, so clt.un is a plain
    ///                                  // ordered compare here
    /// result = c | mask                // saturate at the target's max
    /// ```
    fn lower_conv_f_to_uint(
        &mut self,
        to: Type,
        fty: Type,
        src: lir::Operand,
        out: &mut Vec<lir::Stmt>,
        il: IlOffset,
    ) -> CompileResult<lir::Operand> {
        let fconst = |v: f64| {
            lir::Operand::Const(match fty {
                Type::Float => Const::Float(v as f32),
                _ => Const::Double(v),
            })
        };
        let wide = !matches!(to, Type::Int32);
        let f = self.temp(fty);
        Self::push(
            out,
            il,
            lir::StmtKind::Binary {
                dst: f,
                op: BinaryOp::MaxF,
                lhs: src,
                rhs: fconst(0.0),
            },
        );
        let r = self.temp(Type::Int64);
        Self::push(
            out,
            il,
            lir::StmtKind::Conv {
                dst: r,
                to: Type::Int64,
                overflow: false,
                unsigned: false,
                src: lir::Operand::Temp(f),
            },
        );
        let limit = fconst(if wide {
            18446744073709551616.0 // 2^64
        } else {
            4294967296.0 // 2^32
        });
        let c = if wide {
            let w = self.temp(fty);
            Self::push(
                out,
                il,
                lir::StmtKind::Binary {
                    dst: w,
                    op: BinaryOp::Sub,
                    lhs: lir::Operand::Temp(f),
                    rhs: limit,
                },
            );
            let n = self.temp(Type::Int64);
            Self::push(
                out,
                il,
                lir::StmtKind::Conv {
                    dst: n,
                    to: Type::Int64,
                    overflow: false,
                    unsigned: false,
                    src: lir::Operand::Temp(w),
                },
            );
            let s = self.temp(Type::Int64);
            Self::push(
                out,
                il,
                lir::StmtKind::Binary {
                    dst: s,
                    op: BinaryOp::Shr,
                    lhs: lir::Operand::Temp(r),
                    rhs: lir::Operand::Const(Const::Int32(63)),
                },
            );
            let a = self.temp(Type::Int64);
            Self::push(
                out,
                il,
                lir::StmtKind::Binary {
                    dst: a,
                    op: BinaryOp::And,
                    lhs: lir::Operand::Temp(n),
                    rhs: lir::Operand::Temp(s),
                },
            );
            let c = self.temp(Type::Int64);
            Self::push(
                out,
                il,
                lir::StmtKind::Binary {
                    dst: c,
                    op: BinaryOp::Or,
                    lhs: lir::Operand::Temp(r),
                    rhs: lir::Operand::Temp(a),
                },
            );
            c
        } else {
            // A 32-bit target keeps the low half of the 64-bit conversion
            // (exact for f < 2^32).
            let c = self.temp(Type::Int32);
            Self::push(
                out,
                il,
                lir::StmtKind::Conv {
                    dst: c,
                    to: Type::Int32,
                    overflow: false,
                    unsigned: false,
                    src: lir::Operand::Temp(r),
                },
            );
            c
        };
        let lt = self.temp(Type::Int32);
        Self::push(
            out,
            il,
            lir::StmtKind::Binary {
                dst: lt,
                op: BinaryOp::ULt,
                lhs: lir::Operand::Temp(f),
                rhs: limit,
            },
        );
        let m32 = self.temp(Type::Int32);
        Self::push(
            out,
            il,
            lir::StmtKind::Binary {
                dst: m32,
                op: BinaryOp::Sub,
                lhs: lir::Operand::Temp(lt),
                rhs: lir::Operand::Const(Const::Int32(1)),
            },
        );
        let dst = self.temp(to);
        let mask = if wide {
            let m64 = self.temp(Type::Int64);
            Self::push(
                out,
                il,
                lir::StmtKind::Conv {
                    dst: m64,
                    to: Type::Int64,
                    overflow: false,
                    unsigned: false,
                    src: lir::Operand::Temp(m32),
                },
            );
            m64
        } else {
            m32
        };
        Self::push(
            out,
            il,
            lir::StmtKind::Binary {
                dst,
                op: BinaryOp::Or,
                lhs: lir::Operand::Temp(c),
                rhs: lir::Operand::Temp(mask),
            },
        );
        Ok(lir::Operand::Temp(dst))
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
                    // The catch-handler entry store (step_10.6): the
                    // funclet's incoming throwable lands in `dst`; there
                    // is no tree to flatten.
                    if let hir::Expr::CatchArg = value {
                        if self.locals[dst.0 as usize].ty != Type::Ref {
                            return Err(CompileError::Internal(
                                "CatchArg store to a non-Ref local",
                            ));
                        }
                        Self::push(
                            &mut stmts,
                            stmt.il_offset,
                            lir::StmtKind::CatchArg { dst: *dst },
                        );
                        continue;
                    }
                    let src = self.flatten_expr(value, &mut stmts, stmt.il_offset)?;
                    // A struct store is a block copy into the destination's
                    // frame slot (struct values are addresses, step_10.9);
                    // frame destinations never need a GC barrier.
                    if let Type::Struct(class) = self.locals[dst.0 as usize].ty {
                        Self::push(
                            &mut stmts,
                            stmt.il_offset,
                            lir::StmtKind::BlockCopy {
                                dst_addr: lir::Operand::AddrOf(*dst),
                                dst_offset: 0,
                                src_addr: src,
                                class,
                            },
                        );
                    } else {
                        Self::push(
                            &mut stmts,
                            stmt.il_offset,
                            lir::StmtKind::Copy { dst: *dst, src },
                        );
                    }
                }
                hir::StmtKind::StoreInd {
                    addr,
                    offset,
                    value,
                    access,
                } => {
                    // Store through a byref at a constant offset (`stfld`).
                    // Address first, then the value — the IL push order the
                    // importer encoded (`stfld`: obj pushed before value).
                    let addr = self.flatten_expr(addr, &mut stmts, stmt.il_offset)?;
                    let src = self.flatten_expr(value, &mut stmts, stmt.il_offset)?;
                    if let Some(class) = struct_class_of(value) {
                        // A struct store through a computed address
                        // (`stobj`/`cpobj`/struct `stfld`/`stsfld`): a
                        // block copy.
                        let addr = self.block_addr_value(addr, &mut stmts, stmt.il_offset);
                        Self::push(
                            &mut stmts,
                            stmt.il_offset,
                            lir::StmtKind::BlockCopy {
                                dst_addr: addr,
                                dst_offset: *offset,
                                src_addr: src,
                                class,
                            },
                        );
                    } else {
                        let addr = self.addr_value(addr, &mut stmts, stmt.il_offset);
                        Self::push(
                            &mut stmts,
                            stmt.il_offset,
                            lir::StmtKind::Store {
                                addr,
                                offset: *offset,
                                src,
                                access: *access,
                            },
                        );
                    }
                }
                hir::StmtKind::BlockZero { addr, class } => {
                    let addr = self.flatten_expr(addr, &mut stmts, stmt.il_offset)?;
                    let addr = self.block_addr_value(addr, &mut stmts, stmt.il_offset);
                    Self::push(
                        &mut stmts,
                        stmt.il_offset,
                        lir::StmtKind::BlockZero {
                            dst_addr: addr,
                            class: *class,
                        },
                    );
                }
                hir::StmtKind::BoundsCheck { array, index } => {
                    // The bounds check (step_10.8): a statement consuming
                    // the array and index operands (IL order: array first).
                    let array = self.flatten_expr(array, &mut stmts, stmt.il_offset)?;
                    let index = self.flatten_expr(index, &mut stmts, stmt.il_offset)?;
                    Self::push(
                        &mut stmts,
                        stmt.il_offset,
                        lir::StmtKind::BoundsCheck { array, index },
                    );
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

    /// A memory-access address as a pointer *value*: `AddrOf(l)` (a
    /// `ldloca`-shaped address — struct field access through a byref,
    /// step_10.9) materializes into a fresh ByRef temp first, because the
    /// LIR `Load`/`Store` rules consume address values, not address-of
    /// forms. Anything else passes through unchanged.
    fn addr_value(
        &mut self,
        addr: lir::Operand,
        out: &mut Vec<lir::Stmt>,
        il: IlOffset,
    ) -> lir::Operand {
        match addr {
            lir::Operand::AddrOf(l) => {
                let dst = self.temp(Type::ByRef);
                Self::push(
                    out,
                    il,
                    lir::StmtKind::Copy {
                        dst,
                        src: lir::Operand::AddrOf(l),
                    },
                );
                lir::Operand::Temp(dst)
            }
            _ => addr,
        }
    }

    /// A block-op address (`BlockCopy`/`BlockZero`, struct values): the
    /// x64 block rules take a frame slot or a pointer-typed slot, so a
    /// *constant* address (a static field's frozen address, step_10.7)
    /// materializes into a fresh ByRef temp first. `AddrOf` and slot
    /// operands pass straight through.
    fn block_addr_value(
        &mut self,
        addr: lir::Operand,
        out: &mut Vec<lir::Stmt>,
        il: IlOffset,
    ) -> lir::Operand {
        match addr {
            lir::Operand::Const(_) => {
                let dst = self.temp(Type::ByRef);
                Self::push(out, il, lir::StmtKind::Copy { dst, src: addr });
                lir::Operand::Temp(dst)
            }
            _ => addr,
        }
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
            // The importer builds CatchArg only as the value of a catch
            // handler's synthesized entry store, handled in `lower_block`.
            hir::Expr::CatchArg => Err(CompileError::Internal(
                "CatchArg outside a catch handler's entry store",
            )),
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
                    // A struct call result lives in the call's destination
                    // temp; the value is its address (step_10.9).
                    Some(dst) if matches!(sig.ret, Type::Struct(_)) => {
                        Ok(lir::Operand::AddrOf(dst))
                    }
                    Some(dst) => Ok(lir::Operand::Temp(dst)),
                    // The importer's well-typedness guarantee makes a void
                    // call in value position unreachable.
                    None => Err(CompileError::Internal("void call used as a value")),
                }
            }
            hir::Expr::Load {
                addr,
                offset,
                ty,
                access,
            } => {
                // Load through a byref at a constant offset (`ldfld`).
                let addr = self.flatten_expr(addr, out, il)?;
                let addr = self.addr_value(addr, out, il);
                let dst = self.temp(*ty);
                Self::push(
                    out,
                    il,
                    lir::StmtKind::Load {
                        dst,
                        addr,
                        offset: *offset,
                        ty: *ty,
                        access: *access,
                    },
                );
                Ok(lir::Operand::Temp(dst))
            }
            hir::Expr::FieldAddr { obj, offset, .. } => {
                // A field address is `obj + offset`, a managed byref: the
                // temp is ByRef-typed, so it is automatically an interior-
                // pointer GC root at every safepoint. An AddrOf object
                // (a `ldloca`-shaped receiver) materializes into a byref
                // temp first — the Binary rules consume values, not
                // address-of forms.
                let obj = self.flatten_expr(obj, out, il)?;
                let obj = self.addr_value(obj, out, il);
                let dst = self.temp(Type::ByRef);
                Self::push(
                    out,
                    il,
                    lir::StmtKind::Binary {
                        dst,
                        op: BinaryOp::Add,
                        lhs: obj,
                        rhs: lir::Operand::Const(Const::NativeInt(*offset as isize)),
                    },
                );
                Ok(lir::Operand::Temp(dst))
            }
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
                // A float source with an unsigned integer target is the
                // saturating conversion (step_10.11) — expanded to a
                // statement sequence, not a single Conv.
                let src_ty = self.operand_ty(&src)?;
                if *unsigned
                    && matches!(src_ty, Type::Float | Type::Double)
                    && matches!(*to, Type::Int32 | Type::Int64 | Type::NativeInt)
                {
                    return self.lower_conv_f_to_uint(*to, src_ty, src, out, il);
                }
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
            hir::Expr::NullCheck { arg } => {
                // The explicit, trap-based null check (step_10.4): the
                // checked value is the result — the statement exists purely
                // for its fault.
                let arg = self.flatten_expr(arg, out, il)?;
                Self::push(out, il, lir::StmtKind::NullCheck { arg });
                Ok(arg)
            }
            hir::Expr::ArrLen { array } => {
                // The length sits at offset 8 (corinfo.h's CORINFO_Array
                // layout); the load doubles as the null check — a null
                // array faults here, the hardware fault translated to the
                // NRE (step_10.4's trap model).
                let array = self.flatten_expr(array, out, il)?;
                let array = self.addr_value(array, out, il);
                let dst = self.temp(Type::Int32);
                Self::push(
                    out,
                    il,
                    lir::StmtKind::Load {
                        dst,
                        addr: array,
                        offset: crate::ir::ARRAY_LENGTH_OFFSET,
                        ty: Type::Int32,
                        access: crate::ir::MemAccess::Natural,
                    },
                );
                Ok(lir::Operand::Temp(dst))
            }
            hir::Expr::ArrElemAddr {
                array,
                index,
                elem_size,
                ..
            } => {
                // Element address: `array + 16 + index * elem_size`
                // (corinfo.h's CORINFO_Array layout) — the FieldAddr
                // shape: the address temp is ByRef-typed, so it is
                // automatically an interior-pointer GC root.
                let array = self.flatten_expr(array, out, il)?;
                let array = self.addr_value(array, out, il);
                let index = self.flatten_expr(index, out, il)?;
                // A 32-bit index zero-extends to native width (the bounds
                // check already proved 0 <= index < len).
                let index = if self.operand_ty(&index)? == Type::Int32 {
                    let dst = self.temp(Type::NativeInt);
                    Self::push(
                        out,
                        il,
                        lir::StmtKind::Conv {
                            dst,
                            to: Type::NativeInt,
                            overflow: false,
                            unsigned: true,
                            src: index,
                        },
                    );
                    lir::Operand::Temp(dst)
                } else {
                    index
                };
                let scaled = self.temp(Type::NativeInt);
                Self::push(
                    out,
                    il,
                    lir::StmtKind::Binary {
                        dst: scaled,
                        op: BinaryOp::Mul,
                        lhs: index,
                        rhs: lir::Operand::Const(Const::NativeInt(*elem_size as isize)),
                    },
                );
                let offset = self.temp(Type::NativeInt);
                Self::push(
                    out,
                    il,
                    lir::StmtKind::Binary {
                        dst: offset,
                        op: BinaryOp::Add,
                        lhs: lir::Operand::Temp(scaled),
                        rhs: lir::Operand::Const(Const::NativeInt(
                            crate::ir::ARRAY_DATA_OFFSET as isize,
                        )),
                    },
                );
                let dst = self.temp(Type::ByRef);
                Self::push(
                    out,
                    il,
                    lir::StmtKind::Binary {
                        dst,
                        op: BinaryOp::Add,
                        lhs: array,
                        rhs: lir::Operand::Temp(offset),
                    },
                );
                Ok(lir::Operand::Temp(dst))
            }
            hir::Expr::Cast { .. } | hir::Expr::Box { .. } => {
                Err(CompileError::Unsupported("cast/box: not yet supported"))
            }
            hir::Expr::StructVal { addr, .. } => {
                // A struct value IS its address (step_10.9): in LIR every
                // struct-typed value is a ByRef operand naming the memory
                // the value occupies. A constant address (a struct-typed
                // static's frozen address, step_10.7) materializes first.
                let addr = self.flatten_expr(addr, out, il)?;
                Ok(self.block_addr_value(addr, out, il))
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
        let dst = match sig.ret {
            Type::Void => None,
            Type::Struct(class) => {
                if self.layouts[&class].sysv.passed_in_registers {
                    Some(self.temp(sig.ret))
                } else {
                    // A non-register-passed struct returns through the
                    // hidden retbuf the importer already passed as an
                    // argument — the call has no register result.
                    None
                }
            }
            _ => Some(self.temp(sig.ret)),
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
                if let Some(class) = value.as_ref().and_then(struct_class_of) {
                    // A register-passed struct return (step_10.9): the
                    // value's address; the ABI distribution into
                    // rax/rdx/xmm0/xmm1 is the backend's lowering. The
                    // non-register-passed form never reaches here — the
                    // importer rewrote it to a block copy through the
                    // hidden retbuf pointer plus a plain pointer return.
                    let addr = value
                        .as_ref()
                        .map(|e| self.flatten_expr(e, out, IL_OFFSET_NONE))
                        .transpose()?
                        .ok_or(CompileError::Internal("struct return without a value"))?;
                    Self::push(
                        out,
                        IL_OFFSET_NONE,
                        lir::StmtKind::ReturnStruct { addr, class },
                    );
                } else {
                    let value = value
                        .as_ref()
                        .map(|e| self.flatten_expr(e, out, IL_OFFSET_NONE))
                        .transpose()?;
                    Self::push(out, IL_OFFSET_NONE, lir::StmtKind::Return { value });
                }
            }
            hir::Terminator::Switch { .. } => {
                return Err(CompileError::Unsupported("switch: not yet supported"));
            }
            hir::Terminator::Throw { exception } => {
                // The exception tree flattens like a branch condition
                // (its calls/traps evaluate first); the Throw consumes
                // the operand.
                let exception = self.flatten_expr(exception, out, IL_OFFSET_NONE)?;
                Self::push(out, IL_OFFSET_NONE, lir::StmtKind::Throw { exception });
            }
            hir::Terminator::Leave { target } => {
                Self::push(
                    out,
                    IL_OFFSET_NONE,
                    lir::StmtKind::Leave { target: *target },
                );
            }
            hir::Terminator::CallFinally {
                funclet,
                continuation,
            } => {
                Self::push(
                    out,
                    IL_OFFSET_NONE,
                    lir::StmtKind::CallFinally {
                        funclet: *funclet,
                        continuation: *continuation,
                    },
                );
            }
            hir::Terminator::EndFinally => {
                Self::push(out, IL_OFFSET_NONE, lir::StmtKind::EndFinally);
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

/// The value class of a struct-typed expression (step_10.9), or `None`.
/// Struct values reach lowering only as `StructVal` (address-shaped) or a
/// struct-returning call — never as a bare `Local` (the importer builds
/// the address form directly).
fn struct_class_of(expr: &hir::Expr) -> Option<rokajit_ee::handles::ClassHandle> {
    match expr {
        hir::Expr::StructVal { class, .. } => Some(*class),
        hir::Expr::Call { sig, .. } => match sig.ret {
            Type::Struct(class) => Some(class),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{hir, lir, BlockId, CallSig, Const, LocalId};
    use crate::pipeline::MethodInfo;
    use crate::structs::StructLayouts;
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
        fn class_of(&self, ty: Type, layouts: &StructLayouts) -> Option<RegClassId> {
            match ty {
                Type::Struct(class) if layouts.contains_key(&class) => Some(RegClassId(0)),
                Type::Struct(_) | Type::Void => None,
                _ => Some(RegClassId(0)),
            }
        }
        fn classify_call(
            &self,
            _sig: &CallSig,
            _layouts: &StructLayouts,
        ) -> CompileResult<CallAbi> {
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
            struct_layouts: StructLayouts::new(),
            generics_context: None,
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
        // Switches remain outside the lowering subset.
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

    // --- step_10.4: the object pack flattening ---

    /// An instance-method shape: one Ref arg (`this`).
    fn method_with_ref_arg(block: hir::Block) -> hir::Method {
        let mut m = method_with(block);
        m.locals[0] = local(Type::Ref, hir::LocalKind::IlArg(0));
        m
    }

    #[test]
    fn ldfld_load_flattens_to_null_check_then_load() {
        // return this.x (an Int32 field at offset 8): NullCheck yields the
        // checked receiver unchanged; the Load reads through it.
        let m = lower_ok(method_with_ref_arg(block(
            0,
            Vec::new(),
            hir::Terminator::Return {
                value: Some(hir::Expr::Load {
                    addr: Box::new(hir::Expr::NullCheck {
                        arg: Box::new(hir::Expr::Local(LocalId(0))),
                    }),
                    offset: 8,
                    ty: Type::Int32,
                    access: crate::ir::MemAccess::Natural,
                }),
            },
        )));
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 3, "null check, load, return");
        assert!(matches!(
            stmts[0].kind,
            lir::StmtKind::NullCheck {
                arg: lir::Operand::Local(LocalId(0))
            }
        ));
        match &stmts[1].kind {
            lir::StmtKind::Load {
                dst,
                addr,
                offset,
                ty,
                access,
            } => {
                assert_eq!(*dst, LocalId(1), "fresh temp after the one arg");
                assert_eq!(*addr, lir::Operand::Local(LocalId(0)));
                assert_eq!(*offset, 8);
                assert_eq!(*ty, Type::Int32);
                assert_eq!(*access, crate::ir::MemAccess::Natural);
                assert_eq!(m.locals[1].ty, Type::Int32);
            }
            _ => panic!("expected Load"),
        }
    }

    #[test]
    fn stfld_flattens_address_then_value_in_push_order() {
        // this.x = 42 — the address operand flattens before the value.
        let m = lower_ok(method_with_ref_arg(block(
            0,
            vec![hstmt(hir::StmtKind::StoreInd {
                addr: hir::Expr::NullCheck {
                    arg: Box::new(hir::Expr::Local(LocalId(0))),
                },
                offset: 8,
                value: hir::Expr::Const(Const::Int32(42)),
                access: crate::ir::MemAccess::Natural,
            })],
            hir::Terminator::Return { value: None },
        )));
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 3, "null check, store, return");
        assert!(matches!(stmts[0].kind, lir::StmtKind::NullCheck { .. }));
        match &stmts[1].kind {
            lir::StmtKind::Store {
                addr,
                offset,
                src,
                access,
            } => {
                assert_eq!(*addr, lir::Operand::Local(LocalId(0)));
                assert_eq!(*offset, 8);
                assert_eq!(*src, lir::Operand::Const(Const::Int32(42)));
                assert_eq!(*access, crate::ir::MemAccess::Natural);
            }
            _ => panic!("expected Store"),
        }
    }

    #[test]
    fn field_addr_flattens_to_a_byref_add() {
        // &this.x: obj + the EE offset; the temp is ByRef-typed (an
        // interior-pointer GC root).
        let field = rokajit_ee::handles::FieldHandle::from_raw(0x300usize as _).unwrap();
        let m = lower_ok(method_with_ref_arg(block(
            0,
            vec![hstmt(hir::StmtKind::Store {
                dst: LocalId(0),
                value: hir::Expr::FieldAddr {
                    obj: Box::new(hir::Expr::NullCheck {
                        arg: Box::new(hir::Expr::Local(LocalId(0))),
                    }),
                    field,
                    offset: 12,
                },
            })],
            hir::Terminator::Return { value: None },
        )));
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 4, "null check, add, copy, return");
        assert!(matches!(stmts[0].kind, lir::StmtKind::NullCheck { .. }));
        match &stmts[1].kind {
            lir::StmtKind::Binary { dst, op, lhs, rhs } => {
                assert_eq!(*dst, LocalId(1));
                assert_eq!(*op, BinaryOp::Add);
                assert_eq!(*lhs, lir::Operand::Local(LocalId(0)));
                assert_eq!(*rhs, lir::Operand::Const(Const::NativeInt(12)));
                assert_eq!(
                    m.locals[dst.0 as usize].ty,
                    Type::ByRef,
                    "the address temp is a byref root"
                );
            }
            _ => panic!("expected Binary(Add)"),
        }
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

    // --- step_10.9: struct flattening ---

    use crate::structs::{StructLayout, SysVClass, SysVPass};
    use rokajit_ee::handles::ClassHandle;

    fn class(raw: usize) -> ClassHandle {
        ClassHandle::from_raw(raw as ffi::CORINFO_CLASS_HANDLE).unwrap()
    }

    fn layout(passed: bool) -> StructLayout {
        let mut sysv = SysVPass::memory();
        sysv.passed_in_registers = passed;
        if passed {
            sysv.count = 1;
            sysv.classes[0] = SysVClass::Integer;
            sysv.sizes[0] = 8;
        }
        StructLayout {
            size: 8,
            align: 8,
            gc_cells: vec![],
            sysv,
        }
    }

    /// A method with two struct locals of class `c` (ids 0, 1) whose
    /// layout is in the side table.
    fn struct_method(c: ClassHandle, block: hir::Block) -> hir::Method {
        let mut layouts = StructLayouts::new();
        layouts.insert(c, layout(false));
        hir::Method {
            blocks: vec![block],
            locals: vec![
                local(Type::Struct(c), hir::LocalKind::IlLocal(0)),
                local(Type::Struct(c), hir::LocalKind::IlLocal(1)),
            ],
            eh_regions: Vec::new(),
            num_args: 0,
            num_il_locals: 2,
            struct_layouts: layouts,
            generics_context: None,
        }
    }

    #[test]
    fn struct_store_lowers_to_a_block_copy() {
        let c = class(0x51);
        let m = struct_method(
            c,
            block(
                0,
                vec![hstmt(hir::StmtKind::Store {
                    dst: LocalId(1),
                    value: hir::Expr::StructVal {
                        addr: Box::new(hir::Expr::LocalAddr(LocalId(0))),
                        class: c,
                    },
                })],
                hir::Terminator::Return { value: None },
            ),
        );
        let m = lower_ok(m);
        match &m.blocks[0].stmts[0].kind {
            lir::StmtKind::BlockCopy {
                dst_addr,
                dst_offset,
                src_addr,
                class,
            } => {
                assert_eq!(*dst_addr, lir::Operand::AddrOf(LocalId(1)));
                assert_eq!(*dst_offset, 0);
                assert_eq!(*src_addr, lir::Operand::AddrOf(LocalId(0)));
                assert_eq!(*class, c);
            }
            _ => panic!("expected BlockCopy"),
        }
    }

    #[test]
    fn struct_storeind_lowers_to_a_block_copy_with_the_field_offset() {
        let c = class(0x52);
        let m = struct_method(
            c,
            block(
                0,
                vec![hstmt(hir::StmtKind::StoreInd {
                    addr: hir::Expr::Local(LocalId(0)),
                    offset: 8,
                    value: hir::Expr::StructVal {
                        addr: Box::new(hir::Expr::LocalAddr(LocalId(1))),
                        class: c,
                    },
                    access: crate::ir::MemAccess::Natural,
                })],
                hir::Terminator::Return { value: None },
            ),
        );
        let m = lower_ok(m);
        match &m.blocks[0].stmts[0].kind {
            lir::StmtKind::BlockCopy {
                dst_addr,
                dst_offset,
                src_addr,
                class,
            } => {
                assert_eq!(*dst_addr, lir::Operand::Local(LocalId(0)));
                assert_eq!(*dst_offset, 8);
                assert_eq!(*src_addr, lir::Operand::AddrOf(LocalId(1)));
                assert_eq!(*class, c);
            }
            _ => panic!("expected BlockCopy"),
        }
    }

    #[test]
    fn initobj_lowers_to_block_zero() {
        let c = class(0x53);
        let m = struct_method(
            c,
            block(
                0,
                vec![hstmt(hir::StmtKind::BlockZero {
                    addr: hir::Expr::LocalAddr(LocalId(0)),
                    class: c,
                })],
                hir::Terminator::Return { value: None },
            ),
        );
        let m = lower_ok(m);
        match &m.blocks[0].stmts[0].kind {
            lir::StmtKind::BlockZero { dst_addr, class } => {
                assert_eq!(*dst_addr, lir::Operand::AddrOf(LocalId(0)));
                assert_eq!(*class, c);
            }
            _ => panic!("expected BlockZero"),
        }
    }

    #[test]
    fn struct_return_lowers_to_return_struct_of_the_address() {
        let c = class(0x54);
        let m = struct_method(
            c,
            block(
                0,
                Vec::new(),
                hir::Terminator::Return {
                    value: Some(hir::Expr::StructVal {
                        addr: Box::new(hir::Expr::LocalAddr(LocalId(0))),
                        class: c,
                    }),
                },
            ),
        );
        let m = lower_ok(m);
        match &m.blocks[0].stmts[0].kind {
            lir::StmtKind::ReturnStruct { addr, class } => {
                assert_eq!(*addr, lir::Operand::AddrOf(LocalId(0)));
                assert_eq!(*class, c);
            }
            _ => panic!("expected ReturnStruct"),
        }
    }

    #[test]
    fn register_passed_struct_call_result_is_the_dst_temps_address() {
        let c = class(0x55);
        let mut m = struct_method(
            c,
            block(
                0,
                Vec::new(),
                hir::Terminator::Return {
                    value: Some(hir::Expr::Call {
                        target: CallTarget::Direct(
                            MethodHandle::from_raw(0x42usize as ffi::CORINFO_METHOD_HANDLE)
                                .unwrap(),
                        ),
                        sig: CallSig {
                            ret: Type::Struct(c),
                            args: vec![],
                            has_this: false,
                        },
                        args: vec![],
                    }),
                },
            ),
        );
        m.struct_layouts.insert(c, layout(true));
        let m = lower_ok(m);
        let stmts = &m.blocks[0].stmts;
        // The call's destination is a fresh struct temp (id 2, after the
        // two IL locals); the ReturnStruct reads its address.
        match &stmts[0].kind {
            lir::StmtKind::Call { dst, .. } => assert_eq!(*dst, Some(LocalId(2))),
            _ => panic!("expected Call"),
        }
        assert!(matches!(m.locals[2].ty, Type::Struct(_)));
        match &stmts[1].kind {
            lir::StmtKind::ReturnStruct { addr, .. } => {
                assert_eq!(*addr, lir::Operand::AddrOf(LocalId(2)))
            }
            _ => panic!("expected ReturnStruct"),
        }
    }

    #[test]
    fn non_register_passed_struct_call_has_no_destination() {
        // The retbuf temp the importer passed receives the value; the LIR
        // call has no register result.
        let c = class(0x56);
        let m = struct_method(
            c,
            block(
                0,
                vec![hstmt(hir::StmtKind::Eval(hir::Expr::Call {
                    target: CallTarget::Direct(
                        MethodHandle::from_raw(0x42usize as ffi::CORINFO_METHOD_HANDLE).unwrap(),
                    ),
                    sig: CallSig {
                        ret: Type::Struct(c),
                        args: vec![Type::ByRef],
                        has_this: false,
                    },
                    args: vec![hir::Expr::LocalAddr(LocalId(0))],
                }))],
                hir::Terminator::Return { value: None },
            ),
        );
        let m = lower_ok(m);
        match &m.blocks[0].stmts[0].kind {
            lir::StmtKind::Call { dst, .. } => assert_eq!(*dst, None),
            _ => panic!("expected Call"),
        }
    }

    // --- end-to-end via the importer's MockEe fixtures (as morph's tests) ---

    const FIB_TOKEN: u32 = 0x0600_0001;

    fn fib_fixture(il: &[u8]) -> (MockEe, MethodInfo) {
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
            ftn: MethodHandle::from_raw(1usize as ffi::CORINFO_METHOD_HANDLE).unwrap(),
            il: il.to_vec(),
            max_stack: 8,
            eh_count: 0,
            init_locals: false,
            generics_context: None,
            generics_context_keep_alive: false,
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

    // --- step_10.6: EH lowering ---

    #[test]
    fn throw_flattens_its_exception_tree() {
        // Throw of (arg0 + 1): the tree flattens to a Binary statement
        // feeding the Throw, exactly like a branch condition.
        let m = lower_ok(method_with(block(
            0,
            Vec::new(),
            hir::Terminator::Throw {
                exception: hir::Expr::Binary {
                    op: BinaryOp::Add,
                    lhs: Box::new(hir::Expr::Local(LocalId(0))),
                    rhs: Box::new(hir::Expr::Const(Const::Int32(1))),
                },
            },
        )));
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2);
        assert!(matches!(stmts[0].kind, lir::StmtKind::Binary { .. }));
        assert!(matches!(
            stmts[1].kind,
            lir::StmtKind::Throw {
                exception: lir::Operand::Temp(LocalId(1))
            }
        ));
    }

    #[test]
    fn leave_callfinally_endfinally_lower_and_regions_carry_through() {
        let mut m = method_with(block(
            0,
            Vec::new(),
            hir::Terminator::Leave { target: BlockId(2) },
        ));
        m.blocks.push(block(
            1,
            Vec::new(),
            hir::Terminator::CallFinally {
                funclet: BlockId(3),
                continuation: BlockId(2),
            },
        ));
        m.blocks.push(block(
            2,
            Vec::new(),
            hir::Terminator::Return { value: None },
        ));
        m.blocks
            .push(block(3, Vec::new(), hir::Terminator::EndFinally));
        m.eh_regions.push(hir::EhRegion {
            kind: hir::EhRegionKind::Finally,
            try_start: BlockId(0),
            try_end: BlockId(1),
            handler_start: BlockId(3),
            handler_end: BlockId(4),
        });
        let m = lower_ok(m);
        assert_eq!(m.eh_regions.len(), 1, "the region table carries over");
        assert!(matches!(
            m.blocks[0].stmts[0].kind,
            lir::StmtKind::Leave { target: BlockId(2) }
        ));
        assert!(matches!(
            m.blocks[1].stmts[0].kind,
            lir::StmtKind::CallFinally {
                funclet: BlockId(3),
                continuation: BlockId(2)
            }
        ));
        assert!(matches!(
            m.blocks[3].stmts[0].kind,
            lir::StmtKind::EndFinally
        ));
    }

    #[test]
    fn catch_arg_store_lowers_to_the_lir_form() {
        // Store { dst, CatchArg } → lir CatchArg { dst } — no tree walk.
        let mut m = method_with(block(
            0,
            vec![hstmt(hir::StmtKind::Store {
                dst: LocalId(0),
                value: hir::Expr::CatchArg,
            })],
            hir::Terminator::Return { value: None },
        ));
        m.locals[0] = local(Type::Ref, hir::LocalKind::Temp);
        let m = lower_ok(m);
        assert!(matches!(
            m.blocks[0].stmts[0].kind,
            lir::StmtKind::CatchArg { dst: LocalId(0) }
        ));
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

    // --- step_10.8: array flattening ---

    #[test]
    fn arr_len_lowers_to_the_length_load() {
        // return ldlen(this) — a natural Int32 load at offset 8 through
        // the array; the load doubles as the null check.
        let m = lower_ok(method_with_ref_arg(block(
            0,
            Vec::new(),
            hir::Terminator::Return {
                value: Some(hir::Expr::ArrLen {
                    array: Box::new(hir::Expr::Local(LocalId(0))),
                }),
            },
        )));
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2, "the load, the return");
        match &stmts[0].kind {
            lir::StmtKind::Load {
                dst,
                addr,
                offset,
                ty,
                access,
            } => {
                assert_eq!(*dst, LocalId(1), "fresh temp after the one arg");
                assert_eq!(*addr, lir::Operand::Local(LocalId(0)));
                assert_eq!(*offset, 8, "corinfo.h's CORINFO_Array length offset");
                assert_eq!(*ty, Type::Int32);
                assert_eq!(*access, crate::ir::MemAccess::Natural);
                assert_eq!(m.locals[1].ty, Type::Int32);
            }
            _ => panic!("expected the length Load"),
        }
    }

    #[test]
    fn arr_elem_addr_lowers_to_the_mul_add_chain() {
        // return this[i] for an i4 element: the address is
        // array + 16 + zero-extended index * 4, ending in a ByRef temp
        // (the FieldAddr shape — an interior-pointer GC root), and the
        // load reads through it at offset 0.
        let m = lower_ok(method_with_ref_arg(block(
            0,
            Vec::new(),
            hir::Terminator::Return {
                value: Some(hir::Expr::Load {
                    addr: Box::new(hir::Expr::ArrElemAddr {
                        array: Box::new(hir::Expr::Local(LocalId(0))),
                        index: Box::new(hir::Expr::Const(Const::Int32(2))),
                        elem: Type::Int32,
                        elem_size: 4,
                    }),
                    offset: 0,
                    ty: Type::Int32,
                    access: crate::ir::MemAccess::Natural,
                }),
            },
        )));
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 6, "conv, mul, add, add, load, return");
        // The Int32 index zero-extends to native width first.
        match &stmts[0].kind {
            lir::StmtKind::Conv {
                dst,
                to,
                unsigned,
                src,
                ..
            } => {
                assert_eq!(*to, Type::NativeInt);
                assert!(unsigned);
                assert_eq!(*src, lir::Operand::Const(Const::Int32(2)));
                assert_eq!(*dst, LocalId(1));
            }
            _ => panic!("expected the widening Conv"),
        }
        match &stmts[1].kind {
            lir::StmtKind::Binary { dst, op, lhs, rhs } => {
                assert_eq!(*op, BinaryOp::Mul);
                assert_eq!(*lhs, lir::Operand::Temp(LocalId(1)));
                assert_eq!(*rhs, lir::Operand::Const(Const::NativeInt(4)));
                assert_eq!(*dst, LocalId(2));
                assert_eq!(m.locals[2].ty, Type::NativeInt);
            }
            _ => panic!("expected the scaling Mul"),
        }
        match &stmts[2].kind {
            lir::StmtKind::Binary { dst, op, lhs, rhs } => {
                assert_eq!(*op, BinaryOp::Add);
                assert_eq!(*lhs, lir::Operand::Temp(LocalId(2)));
                assert_eq!(
                    *rhs,
                    lir::Operand::Const(Const::NativeInt(16)),
                    "the data offset"
                );
                assert_eq!(*dst, LocalId(3));
            }
            _ => panic!("expected the header Add"),
        }
        match &stmts[3].kind {
            lir::StmtKind::Binary { dst, op, lhs, rhs } => {
                assert_eq!(*op, BinaryOp::Add);
                assert_eq!(*lhs, lir::Operand::Local(LocalId(0)), "the array");
                assert_eq!(*rhs, lir::Operand::Temp(LocalId(3)));
                assert_eq!(*dst, LocalId(4));
                assert_eq!(
                    m.locals[4].ty,
                    Type::ByRef,
                    "the address temp is a byref root"
                );
            }
            _ => panic!("expected the address Add"),
        }
        match &stmts[4].kind {
            lir::StmtKind::Load {
                addr,
                offset,
                ty,
                access,
                ..
            } => {
                assert_eq!(*addr, lir::Operand::Temp(LocalId(4)));
                assert_eq!(*offset, 0);
                assert_eq!(*ty, Type::Int32);
                assert_eq!(*access, crate::ir::MemAccess::Natural);
            }
            _ => panic!("expected the element Load"),
        }
    }

    #[test]
    fn arr_elem_addr_of_a_native_index_skips_the_conv() {
        // A NativeInt index needs no widening: mul, add, add.
        let mut m = method_with_ref_arg(block(
            0,
            Vec::new(),
            hir::Terminator::Return {
                value: Some(hir::Expr::Load {
                    addr: Box::new(hir::Expr::ArrElemAddr {
                        array: Box::new(hir::Expr::Local(LocalId(0))),
                        index: Box::new(hir::Expr::Local(LocalId(1))),
                        elem: Type::Ref,
                        elem_size: 8,
                    }),
                    offset: 0,
                    ty: Type::Ref,
                    access: crate::ir::MemAccess::Natural,
                }),
            },
        ));
        m.locals
            .push(local(Type::NativeInt, hir::LocalKind::IlLocal(0)));
        m.num_il_locals = 1;
        let m = lower_ok(m);
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 5, "mul, add, add, load, return");
        match &stmts[0].kind {
            lir::StmtKind::Binary { op, lhs, rhs, .. } => {
                assert_eq!(*op, BinaryOp::Mul);
                assert_eq!(*lhs, lir::Operand::Local(LocalId(1)));
                assert_eq!(*rhs, lir::Operand::Const(Const::NativeInt(8)));
            }
            _ => panic!("expected the scaling Mul"),
        }
    }

    #[test]
    fn bounds_check_flattens_to_the_lir_statement() {
        let m = lower_ok(method_with_ref_arg(block(
            0,
            vec![hstmt(hir::StmtKind::BoundsCheck {
                array: hir::Expr::Local(LocalId(0)),
                index: hir::Expr::Const(Const::Int32(0)),
            })],
            hir::Terminator::Return { value: None },
        )));
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2, "the check, the return");
        match &stmts[0].kind {
            lir::StmtKind::BoundsCheck { array, index } => {
                assert_eq!(*array, lir::Operand::Local(LocalId(0)));
                assert_eq!(*index, lir::Operand::Const(Const::Int32(0)));
            }
            _ => panic!("expected BoundsCheck"),
        }
        assert_eq!(stmts[0].il_offset, IlOffset(3), "the IL offset propagates");
    }

    // --- step_10.11: float -> unsigned integer (saturating) ---

    /// `double f(double d)` shape: one Double arg, no IL locals.
    fn method_with_double_arg(block: hir::Block) -> hir::Method {
        let mut m = method_with(block);
        m.locals[0] = local(Type::Double, hir::LocalKind::IlArg(0));
        m
    }

    #[test]
    fn conv_u8_from_float_expands_to_the_saturating_sequence() {
        // return (ulong)arg0 — the lowerxarch.cpp sequence: maxs clamp,
        // the signed cvtt of both the clamped and the 2^64-wrapped value,
        // the overflow blend, and the saturation mask.
        let m = lower_ok(method_with_double_arg(block(
            0,
            Vec::new(),
            hir::Terminator::Return {
                value: Some(hir::Expr::Conv {
                    to: Type::Int64,
                    overflow: false,
                    unsigned: true,
                    arg: Box::new(hir::Expr::Local(LocalId(0))),
                }),
            },
        )));
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 12, "the expansion, then the return");
        match &stmts[0].kind {
            lir::StmtKind::Binary { dst, op, lhs, rhs } => {
                assert_eq!(*op, BinaryOp::MaxF);
                assert_eq!(*dst, LocalId(1));
                assert_eq!(*lhs, lir::Operand::Local(LocalId(0)));
                assert_eq!(*rhs, lir::Operand::Const(Const::Double(0.0)));
            }
            _ => panic!("expected the maxs clamp"),
        }
        // r = cvtt(f): a SIGNED Conv node, the 64-bit form.
        match &stmts[1].kind {
            lir::StmtKind::Conv {
                dst,
                to,
                unsigned,
                src,
                ..
            } => {
                assert_eq!(*to, Type::Int64);
                assert!(!unsigned);
                assert_eq!(*dst, LocalId(2));
                assert_eq!(*src, lir::Operand::Temp(LocalId(1)));
            }
            _ => panic!("expected the signed cvtt"),
        }
        // w = f - 2^64, then n = cvtt(w).
        match &stmts[2].kind {
            lir::StmtKind::Binary { op, rhs, .. } => {
                assert_eq!(*op, BinaryOp::Sub);
                assert_eq!(
                    *rhs,
                    lir::Operand::Const(Const::Double(18446744073709551616.0))
                );
            }
            _ => panic!("expected the 2^64 wrap subtract"),
        }
        assert!(matches!(
            stmts[3].kind,
            lir::StmtKind::Conv {
                to: Type::Int64,
                unsigned: false,
                ..
            }
        ));
        // The overflow blend: s = r >> 63 (arithmetic), a = n & s,
        // c = r | a.
        match &stmts[4].kind {
            lir::StmtKind::Binary { op, rhs, .. } => {
                assert_eq!(*op, BinaryOp::Shr);
                assert_eq!(*rhs, lir::Operand::Const(Const::Int32(63)));
            }
            _ => panic!("expected the sign-mask shift"),
        }
        assert!(matches!(
            stmts[5].kind,
            lir::StmtKind::Binary {
                op: BinaryOp::And,
                ..
            }
        ));
        assert!(matches!(
            stmts[6].kind,
            lir::StmtKind::Binary {
                op: BinaryOp::Or,
                ..
            }
        ));
        // The saturation mask: clt.un against 2^64, minus one.
        match &stmts[7].kind {
            lir::StmtKind::Binary { dst, op, rhs, .. } => {
                assert_eq!(*op, BinaryOp::ULt);
                assert_eq!(m.locals[dst.0 as usize].ty, Type::Int32);
                assert_eq!(
                    *rhs,
                    lir::Operand::Const(Const::Double(18446744073709551616.0))
                );
            }
            _ => panic!("expected the limit compare"),
        }
        match &stmts[8].kind {
            lir::StmtKind::Binary { op, rhs, .. } => {
                assert_eq!(*op, BinaryOp::Sub);
                assert_eq!(*rhs, lir::Operand::Const(Const::Int32(1)));
            }
            _ => panic!("expected the mask bias"),
        }
        // The mask widens to 64 bits and ORs into the result.
        assert!(matches!(
            stmts[9].kind,
            lir::StmtKind::Conv {
                to: Type::Int64,
                unsigned: false,
                ..
            }
        ));
        match &stmts[10].kind {
            lir::StmtKind::Binary { dst, op, .. } => {
                assert_eq!(*op, BinaryOp::Or);
                assert_eq!(m.locals[dst.0 as usize].ty, Type::Int64);
            }
            _ => panic!("expected the result Or"),
        }
        assert!(matches!(
            stmts[11].kind,
            lir::StmtKind::Return {
                value: Some(lir::Operand::Temp(LocalId(10)))
            }
        ));
    }

    #[test]
    fn conv_u4_from_float_saturates_at_2_to_the_32() {
        // return (uint)arg0 — the 32-bit target keeps the low half of the
        // 64-bit conversion and saturates at 2^32: no wrap subtract.
        let m = lower_ok(method_with_double_arg(block(
            0,
            Vec::new(),
            hir::Terminator::Return {
                value: Some(hir::Expr::Conv {
                    to: Type::Int32,
                    overflow: false,
                    unsigned: true,
                    arg: Box::new(hir::Expr::Local(LocalId(0))),
                }),
            },
        )));
        let stmts = &m.blocks[0].stmts;
        assert_eq!(
            stmts.len(),
            7,
            "clamp, cvtt64, low half, compare, mask, or, return"
        );
        match &stmts[2].kind {
            lir::StmtKind::Conv { to, src, .. } => {
                assert_eq!(*to, Type::Int32);
                assert_eq!(*src, lir::Operand::Temp(LocalId(2)));
            }
            _ => panic!("expected the low-half truncation"),
        }
        match &stmts[3].kind {
            lir::StmtKind::Binary { op, rhs, .. } => {
                assert_eq!(*op, BinaryOp::ULt);
                assert_eq!(*rhs, lir::Operand::Const(Const::Double(4294967296.0)));
            }
            _ => panic!("expected the 2^32 compare"),
        }
        assert!(matches!(
            stmts[5].kind,
            lir::StmtKind::Binary {
                op: BinaryOp::Or,
                ..
            }
        ));
        // No Int64 mask widening on the 32-bit path.
        assert_eq!(m.locals[6].ty, Type::Int32);
    }

    #[test]
    fn conv_u1_from_int_keeps_the_shift_pair() {
        // A non-float unsigned narrow is untouched by step_10.11:
        // truncate to 32 bits, shl/ushr pair (conv_narrow at import).
        let m = lower_ret(hir::Expr::Binary {
            op: BinaryOp::UShr,
            lhs: Box::new(hir::Expr::Binary {
                op: BinaryOp::Shl,
                lhs: Box::new(hir::Expr::Local(LocalId(0))),
                rhs: Box::new(hir::Expr::Const(Const::Int32(24))),
            }),
            rhs: Box::new(hir::Expr::Const(Const::Int32(24))),
        });
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 3, "shl, ushr, return");
        assert!(matches!(
            stmts[0].kind,
            lir::StmtKind::Binary {
                op: BinaryOp::Shl,
                ..
            }
        ));
        assert!(matches!(
            stmts[1].kind,
            lir::StmtKind::Binary {
                op: BinaryOp::UShr,
                ..
            }
        ));
    }
}
