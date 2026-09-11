//! Pipeline stage 1 (step_07.2): CIL bytes → [`hir::Method`].
//!
//! Scope is the fib subset (`RokaJIT-internal/docs/step_07.md`, verified
//! against `tests/bin/fib.dll`): `ldarg`/`ldloc`/`stloc` in all widths,
//! the `ldc.i4` family, `add`/`sub`/`mul`/`rem` (fib's `Main` uses `rem`),
//! the integer compare-branch family `beq`..`blt.un` plus `brfalse`/
//! `brtrue`/`br` (short and long forms — fib itself uses `blt.s`, there is
//! no `clt`), `call`, and `ret`. Anything else is
//! [`CompileError::Unsupported`]; malformed IL is [`CompileError::BadIl`].
//! The importer never panics: every operand read is bounds-checked.
//!
//! Stack discipline (ECMA-335 §III): the evaluation stack is simulated
//! statically, with types propagated. It must be **empty at every block
//! boundary** — values crossing a boundary are legal IL but need temp
//! materialization, which is a later step; they are rejected as
//! `Unsupported` (so the ir-design stack-height invariant holds vacuously
//! for everything the importer accepts).
//!
//! EE queries consumed (via `&dyn EeInfo`): `resolve_token`,
//! `get_call_info`, and signature walking (`get_arg_type`/`get_arg_next`,
//! bounded by `numArgs` — the real EE's `getArgNext` never returns null, so
//! stepping past `numArgs` walks off the signature blob). The entry
//! method's argument and local signatures come from
//! [`MethodInfo::args`]/[`MethodInfo::locals`] (the frozen pipeline
//! contract), not from `get_method_sig`.

use std::collections::{BTreeSet, HashMap};

use rokajit_ee::ee_info::{zeroed_out, EeInfo};
use rokajit_ee::enums::{CallInfoFlags, CorInfoType};
use rokajit_ee::handles::{ArgListHandle, MethodHandle};
use rokajit_ffi as ffi;

use crate::error::{CompileError, CompileResult};
use crate::ir::{hir, BinaryOp, BlockId, CallSig, CallTarget, Const, IlOffset, LocalId, Type};
use crate::pipeline::MethodInfo;

/// Stage entry point (the body of [`crate::pipeline::import`]).
pub fn import(info: &MethodInfo, ee: &dyn EeInfo) -> CompileResult<hir::Method> {
    if info.eh_count > 0 {
        return Err(CompileError::Unsupported("EH regions"));
    }
    if info.il.is_empty() {
        return Err(CompileError::BadIl("empty IL stream"));
    }
    check_call_conv(info.args.callConv)?;

    // The locals table: IL args (with `this` first when present), then the
    // IL locals from the locals signature. Temps don't exist yet at import.
    let mut local_types = Vec::new();
    if info.args.callConv & ffi::CorInfoCallConv_CORINFO_CALLCONV_HASTHIS != 0 {
        // Class instance method: `this` is an object reference. (Value-type
        // instance methods take a byref `this` — out of scope with structs.)
        local_types.push(Type::Ref);
    }
    local_types.extend(sig_arg_types(&info.args, ee)?);
    let num_args = local_types.len() as u32;
    local_types.extend(sig_arg_types(&info.locals, ee)?);
    let num_il_locals = local_types.len() as u32 - num_args;
    let ret_ty = ir_type_raw(info.args.retType())?;

    let insns = decode(&info.il)?;
    let leaders = find_leaders(&info.il, &insns)?;
    let block_of = leaders
        .iter()
        .enumerate()
        .map(|(i, &offset)| (offset, i as u32))
        .collect();
    let mut importer = BlockImport {
        ee,
        info,
        local_types: &local_types,
        num_args,
        num_il_locals,
        ret_ty,
        block_of,
        expected_depth: HashMap::new(),
        stack: Vec::new(),
    };
    let mut blocks = Vec::with_capacity(leaders.len());
    for b in 0..leaders.len() {
        blocks.push(importer.import_block(b, &leaders, &insns)?);
    }
    // Every block was imported assuming an empty entry stack; a predecessor
    // that recorded a non-zero depth means values cross a boundary.
    for &depth in importer.expected_depth.values() {
        if depth != 0 {
            return Err(CompileError::Unsupported(
                "evaluation-stack values crossing a block boundary",
            ));
        }
    }

    let locals = local_types
        .iter()
        .enumerate()
        .map(|(i, &ty)| {
            let kind = if (i as u32) < num_args {
                hir::LocalKind::IlArg(i as u32)
            } else {
                hir::LocalKind::IlLocal(i as u32 - num_args)
            };
            hir::Local {
                ty,
                kind,
                pinned: false,
            }
        })
        .collect();
    Ok(hir::Method {
        blocks,
        locals,
        eh_regions: Vec::new(),
        num_args,
        num_il_locals,
    })
}

/// Maps an EE type to the IR's evaluation-stack vocabulary (ECMA-335
/// §III.1.1.1: the sub-Int32 metadata types normalize to Int32).
fn ir_type(ty: CorInfoType) -> CompileResult<Type> {
    Ok(match ty {
        CorInfoType::Void => Type::Void,
        CorInfoType::Bool
        | CorInfoType::Char
        | CorInfoType::Byte
        | CorInfoType::UByte
        | CorInfoType::Short
        | CorInfoType::UShort
        | CorInfoType::Int
        | CorInfoType::UInt => Type::Int32,
        CorInfoType::Long | CorInfoType::ULong => Type::Int64,
        CorInfoType::NativeInt | CorInfoType::NativeUInt | CorInfoType::Ptr => Type::NativeInt,
        CorInfoType::Float => Type::Float,
        CorInfoType::Double => Type::Double,
        CorInfoType::Class => Type::Ref,
        CorInfoType::ByRef => Type::ByRef,
        CorInfoType::ValueClass => {
            return Err(CompileError::Unsupported("value types in signatures"));
        }
        CorInfoType::Undef => {
            return Err(CompileError::BadIl("CORINFO_TYPE_UNDEF in signature"));
        }
    })
}

fn ir_type_raw(raw: ffi::CorInfoType) -> CompileResult<Type> {
    match CorInfoType::from_raw(raw) {
        Some(ty) => ir_type(ty),
        None => Err(CompileError::BadIl("CorInfoType outside the header set")),
    }
}

fn check_call_conv(call_conv: ffi::CorInfoCallConv) -> CompileResult<()> {
    if call_conv & ffi::CorInfoCallConv_CORINFO_CALLCONV_GENERIC != 0 {
        return Err(CompileError::Unsupported("generic methods"));
    }
    if call_conv & ffi::CorInfoCallConv_CORINFO_CALLCONV_MASK
        != ffi::CorInfoCallConv_CORINFO_CALLCONV_DEFAULT
    {
        return Err(CompileError::Unsupported("non-default calling convention"));
    }
    Ok(())
}

/// Walks a signature's argument list, bounded by `numArgs` (see the module
/// docs: `getArgNext` is not an end-of-list signal on the real EE).
fn sig_arg_types(sig: &ffi::CORINFO_SIG_INFO, ee: &dyn EeInfo) -> CompileResult<Vec<Type>> {
    let mut types = Vec::with_capacity(sig.numArgs() as usize);
    let mut cursor = ArgListHandle::from_raw(sig.args);
    for _ in 0..sig.numArgs() {
        let Some(arg) = cursor else {
            return Err(CompileError::BadIl("sig arg list shorter than numArgs"));
        };
        let (ty, _value_class) = ee.get_arg_type(sig, arg);
        types.push(ir_type(ty)?);
        cursor = ee.get_arg_next(arg);
    }
    Ok(types)
}

/// One decoded instruction — pass 1 output, pass 2 input.
struct Insn {
    op: Op,
    offset: u32,
    size: u32,
}

/// The fib-subset opcode set, operands already decoded. Branch targets are
/// absolute IL offsets, range-checked at decode time.
#[derive(Copy, Clone)]
enum Op {
    Nop,
    LdArg(u16),
    LdLoc(u16),
    StLoc(u16),
    LdcI4(i32),
    Add,
    Sub,
    Mul,
    Rem,
    Br {
        target: u32,
    },
    /// `brfalse`/`brtrue`: one operand, compared against zero.
    BrZero {
        op: BinaryOp,
        target: u32,
    },
    /// `beq`..`blt.un`: two operands.
    BrCmp {
        op: BinaryOp,
        target: u32,
    },
    Call(u32),
    Ret,
}

/// The compare ops of `beq`..`blt.un` in opcode order (0x2E..=0x37 short,
/// 0x3B..=0x44 long). Equality has no signedness; the `.un` inequality forms
/// map to the `U*` unsigned ops.
const BR_CMP_OPS: [BinaryOp; 10] = [
    BinaryOp::Eq,
    BinaryOp::Ge,
    BinaryOp::Gt,
    BinaryOp::Le,
    BinaryOp::Lt,
    BinaryOp::Ne,
    BinaryOp::UGe,
    BinaryOp::UGt,
    BinaryOp::ULe,
    BinaryOp::ULt,
];

/// Bounds-checked cursor over the IL stream.
struct Reader<'a> {
    il: &'a [u8],
    ip: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> CompileResult<&'a [u8]> {
        let end = self
            .ip
            .checked_add(n)
            .filter(|&end| end <= self.il.len())
            .ok_or(CompileError::BadIl("truncated instruction operand"))?;
        let bytes = &self.il[self.ip..end];
        self.ip = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> CompileResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> CompileResult<i8> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> CompileResult<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> CompileResult<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> CompileResult<i32> {
        Ok(self.u32()? as i32)
    }
}

fn branch_target(il_len: usize, next_ip: usize, delta: i32) -> CompileResult<u32> {
    let target = next_ip as i64 + i64::from(delta);
    if target < 0 || target >= il_len as i64 {
        return Err(CompileError::BadIl("branch target outside the IL stream"));
    }
    Ok(target as u32)
}

/// Pass 1: linear decode. Any opcode outside the fib subset is
/// `Unsupported` — including inside what would prove to be unreachable
/// code, since pass 1 cannot know that yet.
fn decode(il: &[u8]) -> CompileResult<Vec<Insn>> {
    let mut insns = Vec::new();
    let mut r = Reader { il, ip: 0 };
    while r.ip < il.len() {
        let offset = r.ip as u32;
        let opcode = r.u8()?;
        let op = match opcode {
            0x00 => Op::Nop,
            0x02..=0x05 => Op::LdArg(u16::from(opcode - 0x02)),
            0x06..=0x09 => Op::LdLoc(u16::from(opcode - 0x06)),
            0x0A..=0x0D => Op::StLoc(u16::from(opcode - 0x0A)),
            0x0E => Op::LdArg(u16::from(r.u8()?)),
            0x11 => Op::LdLoc(u16::from(r.u8()?)),
            0x13 => Op::StLoc(u16::from(r.u8()?)),
            0x15..=0x1E => Op::LdcI4(i32::from(opcode) - 0x16),
            0x1F => Op::LdcI4(i32::from(r.i8()?)),
            0x20 => Op::LdcI4(r.i32()?),
            0x28 => Op::Call(r.u32()?),
            0x2A => Op::Ret,
            0x2B => {
                let d = r.i8()?;
                Op::Br {
                    target: branch_target(il.len(), r.ip, i32::from(d))?,
                }
            }
            0x2C | 0x2D => {
                let d = r.i8()?;
                Op::BrZero {
                    // brfalse branches on == 0, brtrue on != 0.
                    op: if opcode == 0x2C {
                        BinaryOp::Eq
                    } else {
                        BinaryOp::Ne
                    },
                    target: branch_target(il.len(), r.ip, i32::from(d))?,
                }
            }
            0x2E..=0x37 => {
                let d = r.i8()?;
                Op::BrCmp {
                    op: BR_CMP_OPS[usize::from(opcode - 0x2E)],
                    target: branch_target(il.len(), r.ip, i32::from(d))?,
                }
            }
            0x38 => {
                let d = r.i32()?;
                Op::Br {
                    target: branch_target(il.len(), r.ip, d)?,
                }
            }
            0x39 | 0x3A => {
                let d = r.i32()?;
                Op::BrZero {
                    op: if opcode == 0x39 {
                        BinaryOp::Eq
                    } else {
                        BinaryOp::Ne
                    },
                    target: branch_target(il.len(), r.ip, d)?,
                }
            }
            0x3B..=0x44 => {
                let d = r.i32()?;
                Op::BrCmp {
                    op: BR_CMP_OPS[usize::from(opcode - 0x3B)],
                    target: branch_target(il.len(), r.ip, d)?,
                }
            }
            0x58 => Op::Add,
            0x59 => Op::Sub,
            0x5A => Op::Mul,
            0x5D => Op::Rem,
            0xFE => match r.u8()? {
                0x09 => Op::LdArg(r.u16()?),
                0x0C => Op::LdLoc(r.u16()?),
                0x0E => Op::StLoc(r.u16()?),
                _ => {
                    return Err(CompileError::Unsupported(
                        "0xFE-prefixed opcode outside the fib subset",
                    ));
                }
            },
            _ => return Err(CompileError::Unsupported("opcode outside the fib subset")),
        };
        insns.push(Insn {
            op,
            offset,
            size: r.ip as u32 - offset,
        });
    }
    Ok(insns)
}

/// Computes block-start offsets (in layout order) and validates branch
/// targets: every target must land on an instruction boundary, conditional
/// branches must have a fallthrough, and code after an unconditional
/// transfer must be a branch target (i.e. reachable).
fn find_leaders(il: &[u8], insns: &[Insn]) -> CompileResult<Vec<u32>> {
    let mut is_boundary = vec![false; il.len()];
    for insn in insns {
        is_boundary[insn.offset as usize] = true;
    }

    let mut leaders: BTreeSet<u32> = BTreeSet::from([0]);
    for insn in insns {
        match insn.op {
            Op::Br { target } | Op::BrZero { target, .. } | Op::BrCmp { target, .. } => {
                if !is_boundary[target as usize] {
                    return Err(CompileError::BadIl(
                        "branch target is not an instruction boundary",
                    ));
                }
                leaders.insert(target);
            }
            _ => {}
        }
        if matches!(insn.op, Op::BrZero { .. } | Op::BrCmp { .. }) {
            let fallthrough = insn.offset + insn.size;
            if fallthrough as usize >= il.len() {
                return Err(CompileError::BadIl(
                    "conditional branch falls off the end of the method",
                ));
            }
            leaders.insert(fallthrough);
        }
    }
    for (i, insn) in insns.iter().enumerate() {
        if matches!(insn.op, Op::Br { .. } | Op::Ret) {
            if let Some(next) = insns.get(i + 1) {
                if !leaders.contains(&next.offset) {
                    return Err(CompileError::BadIl(
                        "unreachable IL after an unconditional control transfer",
                    ));
                }
            }
        }
    }
    Ok(leaders.into_iter().collect())
}

/// Per-compilation state for the block-import pass: the eval-stack
/// simulation plus everything the instruction handlers need.
struct BlockImport<'a> {
    ee: &'a dyn EeInfo,
    info: &'a MethodInfo,
    local_types: &'a [Type],
    num_args: u32,
    num_il_locals: u32,
    ret_ty: Type,
    /// Leader offset → block index in layout order.
    block_of: HashMap<u32, u32>,
    /// Stack depth each block entry requires, as told by its predecessors.
    expected_depth: HashMap<u32, usize>,
    stack: Vec<(Type, hir::Expr)>,
}

fn binary(op: BinaryOp, lhs: hir::Expr, rhs: hir::Expr) -> hir::Expr {
    hir::Expr::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

impl BlockImport<'_> {
    fn push(&mut self, ty: Type, expr: hir::Expr) -> CompileResult<()> {
        if self.stack.len() >= self.info.max_stack as usize {
            return Err(CompileError::BadIl("evaluation stack deeper than maxStack"));
        }
        self.stack.push((ty, expr));
        Ok(())
    }

    fn pop(&mut self) -> CompileResult<(Type, hir::Expr)> {
        self.stack
            .pop()
            .ok_or(CompileError::BadIl("evaluation stack underflow"))
    }

    /// Pops one operand that must be an integer stack type.
    fn pop_int(&mut self) -> CompileResult<(Type, hir::Expr)> {
        let (ty, value) = self.pop()?;
        if !matches!(ty, Type::Int32 | Type::Int64 | Type::NativeInt) {
            return Err(CompileError::BadIl("operand must be an integer"));
        }
        Ok((ty, value))
    }

    fn arith(&mut self, op: BinaryOp) -> CompileResult<()> {
        let (rt, rhs) = self.pop_int()?;
        let (lt, lhs) = self.pop_int()?;
        if lt != rt {
            return Err(CompileError::BadIl("binary operand type mismatch"));
        }
        self.push(lt, binary(op, lhs, rhs))
    }

    fn note_depth(&mut self, leader: u32, depth: usize) -> CompileResult<()> {
        match self.expected_depth.entry(leader) {
            std::collections::hash_map::Entry::Occupied(e) => {
                if *e.get() != depth {
                    return Err(CompileError::BadIl(
                        "inconsistent stack depth at a merge point",
                    ));
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(depth);
            }
        }
        Ok(())
    }

    fn block_id(&self, leader: u32) -> CompileResult<BlockId> {
        self.block_of
            .get(&leader)
            .map(|&b| BlockId(b))
            .ok_or(CompileError::Internal("branch to a non-leader offset"))
    }

    fn il_local_id(&self, index: u32) -> CompileResult<LocalId> {
        if index >= self.num_il_locals {
            return Err(CompileError::BadIl("local index out of range"));
        }
        Ok(LocalId(self.num_args + index))
    }

    fn branch(
        &mut self,
        cond: hir::Expr,
        target: u32,
        fallthrough: u32,
    ) -> CompileResult<hir::Terminator> {
        self.note_depth(target, self.stack.len())?;
        self.note_depth(fallthrough, self.stack.len())?;
        Ok(hir::Terminator::Branch {
            cond,
            then: self.block_id(target)?,
            else_: self.block_id(fallthrough)?,
        })
    }

    fn ret(&mut self) -> CompileResult<hir::Terminator> {
        if self.ret_ty == Type::Void {
            if !self.stack.is_empty() {
                return Err(CompileError::BadIl(
                    "stack not empty at ret of a void method",
                ));
            }
            return Ok(hir::Terminator::Return { value: None });
        }
        let (ty, value) = self.pop()?;
        if ty != self.ret_ty {
            return Err(CompileError::BadIl("return value type mismatch"));
        }
        if !self.stack.is_empty() {
            return Err(CompileError::BadIl(
                "more than the return value on the stack at ret",
            ));
        }
        Ok(hir::Terminator::Return { value: Some(value) })
    }

    /// `call` (0x28): resolve the token, take the EE's verdict on how the
    /// call is performed, and pop the call-site signature's arguments.
    fn call(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        // Built as RyuJIT's impResolveToken builds it (importer.cpp:70): the
        // method context, the compilation scope, and the token-kind IN hint
        // (a zero tokenType hard-faults the real EE — step_05 finding).
        let mut resolved = zeroed_out(|t: &mut ffi::CORINFO_RESOLVED_TOKEN| {
            t.tokenContext = self.info.ftn.as_raw() as ffi::CORINFO_CONTEXT_HANDLE;
            t.tokenScope = self.info.args.scope;
            t.token = token;
            t.tokenType = ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Method;
        });
        self.ee.resolve_token(&mut resolved);
        let call = self
            .ee
            .get_call_info(&mut resolved, None, self.info.ftn, CallInfoFlags::EMPTY);
        if call.kind != ffi::CORINFO_CALL_KIND_CORINFO_CALL {
            return Err(CompileError::Unsupported("non-direct call kind"));
        }
        check_call_conv(call.sig.callConv)?;
        let has_this = call.sig.callConv & ffi::CorInfoCallConv_CORINFO_CALLCONV_HASTHIS != 0;
        let ret = ir_type_raw(call.sig.retType())?;
        let arg_types = sig_arg_types(&call.sig, self.ee)?;

        let mut args = Vec::with_capacity(arg_types.len() + usize::from(has_this));
        for &expected in arg_types.iter().rev() {
            let (ty, value) = self.pop()?;
            if ty != expected {
                return Err(CompileError::BadIl("call argument type mismatch"));
            }
            args.push(value);
        }
        args.reverse();
        if has_this {
            let (ty, this) = self.pop()?;
            if !matches!(ty, Type::Ref | Type::ByRef) {
                return Err(CompileError::BadIl("`this` must be a reference"));
            }
            args.insert(0, this);
        }
        let Some(method) = MethodHandle::from_raw(call.hMethod) else {
            return Err(CompileError::BadIl(
                "get_call_info returned a null method handle",
            ));
        };
        let expr = hir::Expr::Call {
            target: CallTarget::Direct(method),
            sig: CallSig {
                ret,
                args: arg_types,
                has_this,
            },
            args,
        };
        if ret == Type::Void {
            stmts.push(hir::Stmt {
                il_offset,
                kind: hir::StmtKind::Eval(expr),
            });
        } else {
            self.push(ret, expr)?;
        }
        Ok(())
    }

    /// Imports the instructions of block `b` (leader `leaders[b]`). The
    /// stack starts empty — a later pass over `expected_depth` proves that
    /// assumption against every predecessor, so any leftover from the
    /// previous block is discarded here.
    fn import_block(
        &mut self,
        b: usize,
        leaders: &[u32],
        insns: &[Insn],
    ) -> CompileResult<hir::Block> {
        self.stack.clear();
        let start = leaders[b];
        let end = leaders
            .get(b + 1)
            .copied()
            .unwrap_or(self.info.il.len() as u32);
        let first = insns.partition_point(|i| i.offset < start);
        let last = insns.partition_point(|i| i.offset < end);

        let mut stmts = Vec::new();
        let mut terminator = None;
        for insn in &insns[first..last] {
            let il_offset = IlOffset(insn.offset);
            match insn.op {
                Op::Nop => {}
                Op::LdArg(index) => {
                    let index = u32::from(index);
                    if index >= self.num_args {
                        return Err(CompileError::BadIl("argument index out of range"));
                    }
                    self.push(
                        self.local_types[index as usize],
                        hir::Expr::Local(LocalId(index)),
                    )?;
                }
                Op::LdLoc(index) => {
                    let id = self.il_local_id(u32::from(index))?;
                    self.push(self.local_types[id.0 as usize], hir::Expr::Local(id))?;
                }
                Op::StLoc(index) => {
                    let id = self.il_local_id(u32::from(index))?;
                    let (ty, value) = self.pop()?;
                    if ty != self.local_types[id.0 as usize] {
                        return Err(CompileError::BadIl("stloc type mismatch"));
                    }
                    stmts.push(hir::Stmt {
                        il_offset,
                        kind: hir::StmtKind::Store { dst: id, value },
                    });
                }
                Op::LdcI4(v) => self.push(Type::Int32, hir::Expr::Const(Const::Int32(v)))?,
                Op::Add => self.arith(BinaryOp::Add)?,
                Op::Sub => self.arith(BinaryOp::Sub)?,
                Op::Mul => self.arith(BinaryOp::Mul)?,
                Op::Rem => self.arith(BinaryOp::Rem)?,
                Op::Call(token) => self.call(token, &mut stmts, il_offset)?,
                Op::Br { target } => {
                    self.note_depth(target, self.stack.len())?;
                    terminator = Some(hir::Terminator::Jump {
                        target: self.block_id(target)?,
                    });
                }
                Op::BrZero { op, target } => {
                    let (ty, value) = self.pop_int()?;
                    let zero = match ty {
                        Type::Int32 => Const::Int32(0),
                        Type::Int64 => Const::Int64(0),
                        Type::NativeInt => Const::NativeInt(0),
                        _ => return Err(CompileError::Internal("pop_int returned non-integer")),
                    };
                    let cond = binary(op, value, hir::Expr::Const(zero));
                    terminator = Some(self.branch(cond, target, insn.offset + insn.size)?);
                }
                Op::BrCmp { op, target } => {
                    let (rt, rhs) = self.pop_int()?;
                    let (lt, lhs) = self.pop_int()?;
                    if lt != rt {
                        return Err(CompileError::BadIl("compare operand type mismatch"));
                    }
                    terminator =
                        Some(self.branch(binary(op, lhs, rhs), target, insn.offset + insn.size)?);
                }
                Op::Ret => {
                    terminator = Some(self.ret()?);
                }
            }
        }
        // Leader construction guarantees a terminator lands block-final; a
        // block without one falls through to the next.
        let terminator = match terminator {
            Some(t) => t,
            None => {
                let Some(&next) = leaders.get(b + 1) else {
                    return Err(CompileError::BadIl("IL falls off the end of the method"));
                };
                self.note_depth(next, self.stack.len())?;
                hir::Terminator::Jump {
                    target: BlockId(b as u32 + 1),
                }
            }
        };
        Ok(hir::Block {
            id: BlockId(b as u32),
            stmts,
            terminator,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rokajit_ee::mock::{MockEe, MockSig};

    const FIB_TOKEN: u32 = 0x0600_0001;
    const VOID_TOKEN: u32 = 0x0600_0002;
    const INST_TOKEN: u32 = 0x0600_0003;

    fn sig(ret: CorInfoType, args: &[CorInfoType]) -> MockSig {
        MockSig {
            ret,
            args: args.to_vec(),
            has_this: false,
        }
    }

    /// A MockEe that resolves the three canned method tokens, plus a
    /// `MethodInfo` for the entry method described by `entry`/`locals`.
    fn fixture_full(
        il: &[u8],
        entry: &MockSig,
        locals: &[CorInfoType],
        max_stack: u32,
        eh_count: u32,
    ) -> (MockEe, MethodInfo) {
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
            },
        );
        let info = MethodInfo {
            ftn: MethodHandle::from_raw(1usize as ffi::CORINFO_METHOD_HANDLE).unwrap(),
            il: il.to_vec(),
            max_stack,
            eh_count,
            init_locals: false,
            args: ee.make_method_sig(entry),
            locals: ee.make_locals_sig(locals),
        };
        (ee, info)
    }

    fn fixture(il: &[u8], entry: &MockSig, locals: &[CorInfoType]) -> (MockEe, MethodInfo) {
        fixture_full(il, entry, locals, 8, 0)
    }

    /// `int f(int n)` shape.
    fn import_ii(il: &[u8]) -> CompileResult<hir::Method> {
        let (ee, info) = fixture(il, &sig(CorInfoType::Int, &[CorInfoType::Int]), &[]);
        import(&info, &ee)
    }

    // --- assertion helpers (the HIR types deliberately have no Debug) ---

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

    fn as_binary(e: &hir::Expr) -> (BinaryOp, &hir::Expr, &hir::Expr) {
        match e {
            hir::Expr::Binary { op, lhs, rhs } => (*op, lhs, rhs),
            _ => panic!("expected Expr::Binary"),
        }
    }

    fn as_call(e: &hir::Expr) -> (&CallSig, &[hir::Expr]) {
        match e {
            hir::Expr::Call { sig, args, .. } => (sig, args),
            _ => panic!("expected Expr::Call"),
        }
    }

    fn return_value(m: &hir::Method, block: usize) -> &hir::Expr {
        match &m.blocks[block].terminator {
            hir::Terminator::Return { value: Some(v) } => v,
            _ => panic!("expected Return with a value"),
        }
    }

    fn store(stmt: &hir::Stmt) -> (LocalId, &hir::Expr) {
        match &stmt.kind {
            hir::StmtKind::Store { dst, value } => (*dst, value),
            _ => panic!("expected StmtKind::Store"),
        }
    }

    // --- the acceptance test: hand-encoded fib (the exact Fib_ bytes from
    // --- tests/bin/fib.dll) ---

    #[test]
    fn fib_end_to_end() {
        // 02 18 32 12 | 02 17 59 28 01000006 | 02 18 59 28 01000006 58 2A | 02 2A
        // ldarg.0; ldc.i4.2; blt.s +18; ldarg.0; ldc.i4.1; sub; call fib;
        // ldarg.0; ldc.i4.2; sub; call fib; add; ret; ldarg.0; ret
        let il = [
            0x02, 0x18, 0x32, 0x12, 0x02, 0x17, 0x59, 0x28, 0x01, 0x00, 0x00, 0x06, 0x02, 0x18,
            0x59, 0x28, 0x01, 0x00, 0x00, 0x06, 0x58, 0x2A, 0x02, 0x2A,
        ];
        let m = import_ii(&il).expect("fib imports");

        assert_eq!(m.num_args, 1);
        assert_eq!(m.num_il_locals, 0);
        assert_eq!(m.locals.len(), 1);
        assert_eq!(m.locals[0].ty, Type::Int32);
        assert_eq!(m.locals[0].kind, hir::LocalKind::IlArg(0));
        assert_eq!(m.blocks.len(), 3);
        assert!(m.eh_regions.is_empty());

        // Block 0: if (n < 2) goto block 2 else block 1. The compare is a
        // tree on the Branch terminator; no statements.
        assert!(m.blocks[0].stmts.is_empty());
        match &m.blocks[0].terminator {
            hir::Terminator::Branch { cond, then, else_ } => {
                let (op, lhs, rhs) = as_binary(cond);
                assert_eq!(op, BinaryOp::Lt);
                assert_eq!(as_local(lhs), LocalId(0));
                assert_eq!(as_i32(rhs), 2);
                assert_eq!(*then, BlockId(2));
                assert_eq!(*else_, BlockId(1));
            }
            _ => panic!("block 0: expected Branch"),
        }

        // Block 1: return fib(n-1) + fib(n-2) — calls nest inside the add
        // tree in IL push order.
        assert!(m.blocks[1].stmts.is_empty());
        let (op, lhs, rhs) = as_binary(return_value(&m, 1));
        assert_eq!(op, BinaryOp::Add);
        for (call, sub_by) in [(lhs, 1), (rhs, 2)] {
            let (sig, args) = as_call(call);
            assert_eq!(
                sig,
                &CallSig {
                    ret: Type::Int32,
                    args: vec![Type::Int32],
                    has_this: false
                }
            );
            assert_eq!(args.len(), 1);
            let (op, lhs, rhs) = as_binary(&args[0]);
            assert_eq!(op, BinaryOp::Sub);
            assert_eq!(as_local(lhs), LocalId(0));
            assert_eq!(as_i32(rhs), sub_by);
        }

        // Block 2: return n.
        assert_eq!(as_local(return_value(&m, 2)), LocalId(0));
    }

    #[test]
    fn main_end_to_end() {
        // The exact Main bytes: ldc.i4.s 20; call fib; ldc.i4 256; rem; ret.
        let il = [
            0x1F, 0x14, 0x28, 0x01, 0x00, 0x00, 0x06, 0x20, 0x00, 0x01, 0x00, 0x00, 0x5D, 0x2A,
        ];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        let m = import(&info, &ee).expect("Main imports");
        assert_eq!(m.blocks.len(), 1);
        let (op, call, modulus) = as_binary(return_value(&m, 0));
        assert_eq!(op, BinaryOp::Rem);
        assert_eq!(as_i32(&as_call(call).1[0]), 20);
        assert_eq!(as_i32(modulus), 256);
    }

    // --- one test per opcode form ---

    #[test]
    fn ldarg_all_widths() {
        let four = &[CorInfoType::Int; 4];
        // ldarg.0..3 (0x02..0x05): store each into ldloc.0..3, then ret 0.
        let il = [0x02, 0x0A, 0x03, 0x0B, 0x04, 0x0C, 0x05, 0x0D, 0x16, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, four), four);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 4);
        for (i, stmt) in stmts.iter().enumerate() {
            let (dst, value) = store(stmt);
            assert_eq!(dst, LocalId(4 + i as u32));
            assert_eq!(as_local(value), LocalId(i as u32));
            assert_eq!(stmt.il_offset, IlOffset(2 * i as u32 + 1));
        }

        // ldarg.s (0x0E) and wide ldarg (0xFE 0x09), arg 5 of 6.
        for il in [
            &[0x0E, 0x05, 0x0A, 0x16, 0x2A][..],
            &[0xFE, 0x09, 0x05, 0x00, 0x0A, 0x16, 0x2A][..],
        ] {
            let (ee, info) = fixture(
                il,
                &sig(CorInfoType::Int, &[CorInfoType::Int; 6]),
                &[CorInfoType::Int],
            );
            let m = import(&info, &ee).expect("imports");
            let (dst, value) = store(&m.blocks[0].stmts[0]);
            assert_eq!(dst, LocalId(6));
            assert_eq!(as_local(value), LocalId(5));
        }
    }

    #[test]
    fn ldloc_stloc_all_widths() {
        let six = &[CorInfoType::Int; 6];
        // ldloc.0..3 (0x06..0x09) + stloc.0..3 (0x0A..0x0D).
        let il = [0x06, 0x0A, 0x07, 0x0B, 0x08, 0x0C, 0x09, 0x0D, 0x16, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), six);
        let m = import(&info, &ee).expect("imports");
        for (i, stmt) in m.blocks[0].stmts.iter().enumerate() {
            let (dst, value) = store(stmt);
            assert_eq!(dst, LocalId(i as u32));
            assert_eq!(as_local(value), LocalId(i as u32));
        }

        // ldloc.s (0x11) / stloc.s (0x13) and wide forms (0xFE 0C/0E), local 4/5.
        for (il, index) in [
            (&[0x11, 0x04, 0x13, 0x04, 0x16, 0x2A][..], 4u32),
            (
                &[0xFE, 0x0C, 0x05, 0x00, 0xFE, 0x0E, 0x05, 0x00, 0x16, 0x2A][..],
                5,
            ),
        ] {
            let (ee, info) = fixture(il, &sig(CorInfoType::Int, &[]), six);
            let m = import(&info, &ee).expect("imports");
            let (dst, value) = store(&m.blocks[0].stmts[0]);
            assert_eq!(dst, LocalId(index));
            assert_eq!(as_local(value), LocalId(index));
        }
    }

    #[test]
    fn ldc_i4_all_forms() {
        // ldc.i4.m1..ldc.i4.8 (0x15..0x1E): ldc; stloc.0; ldc.i4.0; ret.
        for opcode in 0x15u8..=0x1E {
            let il = [opcode, 0x0A, 0x16, 0x2A];
            let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[CorInfoType::Int]);
            let m = import(&info, &ee).expect("imports");
            let (_, value) = store(&m.blocks[0].stmts[0]);
            assert_eq!(as_i32(value), i32::from(opcode) - 0x16);
        }
        // ldc.i4.s (0x1F) sign-extends; ldc.i4 (0x20) is a full i32.
        let il = [
            0x1F, 0xF6, 0x0A, 0x20, 0x78, 0x56, 0x34, 0x12, 0x0B, 0x16, 0x2A,
        ];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[CorInfoType::Int; 2]);
        let m = import(&info, &ee).expect("imports");
        assert_eq!(as_i32(store(&m.blocks[0].stmts[0]).1), -10);
        assert_eq!(as_i32(store(&m.blocks[0].stmts[1]).1), 0x1234_5678);
    }

    #[test]
    fn arithmetic_ops() {
        for (opcode, expected) in [
            (0x58, BinaryOp::Add),
            (0x59, BinaryOp::Sub),
            (0x5A, BinaryOp::Mul),
            (0x5D, BinaryOp::Rem),
        ] {
            // ldarg.0; ldarg.1; op; ret.
            let il = [0x02, 0x03, opcode, 0x2A];
            let (ee, info) = fixture(
                &il,
                &sig(CorInfoType::Int, &[CorInfoType::Int, CorInfoType::Int]),
                &[],
            );
            let m = import(&info, &ee).expect("imports");
            let (op, lhs, rhs) = as_binary(return_value(&m, 0));
            assert_eq!(op, expected);
            assert_eq!(as_local(lhs), LocalId(0));
            assert_eq!(as_local(rhs), LocalId(1));
        }
    }

    #[test]
    fn compare_branches_all_forms() {
        // ldarg.0; ldarg.1; bXX L; else: ldc.i4.0; ret; L: ldc.i4.1; ret.
        for (i, &expected) in BR_CMP_OPS.iter().enumerate() {
            for (branch, long) in [(0x2E + i as u8, false), (0x3B + i as u8, true)] {
                let mut il = vec![0x02, 0x03, branch];
                let delta: i32 = 2;
                if long {
                    il.extend_from_slice(&delta.to_le_bytes());
                } else {
                    il.push(delta as u8);
                }
                il.extend_from_slice(&[0x16, 0x2A, 0x17, 0x2A]);
                let m = import_ii2(&il);
                assert_eq!(m.blocks.len(), 3);
                match &m.blocks[0].terminator {
                    hir::Terminator::Branch { cond, then, else_ } => {
                        let (op, lhs, rhs) = as_binary(cond);
                        assert_eq!(op, expected);
                        assert_eq!(as_local(lhs), LocalId(0));
                        assert_eq!(as_local(rhs), LocalId(1));
                        assert_eq!(*then, BlockId(2));
                        assert_eq!(*else_, BlockId(1));
                    }
                    _ => panic!("expected Branch"),
                }
                assert_eq!(as_i32(return_value(&m, 1)), 0);
                assert_eq!(as_i32(return_value(&m, 2)), 1);
            }
        }
    }

    /// `int f(int a, int b)` shape.
    fn import_ii2(il: &[u8]) -> hir::Method {
        let (ee, info) = fixture(
            il,
            &sig(CorInfoType::Int, &[CorInfoType::Int, CorInfoType::Int]),
            &[],
        );
        import(&info, &ee).expect("imports")
    }

    #[test]
    fn brfalse_brtrue_forms() {
        // ldarg.0; brfalse/brtrue L; ldc.i4.0; ret; L: ldc.i4.1; ret.
        for (il, expected) in [
            (
                &[0x02, 0x2C, 0x02, 0x16, 0x2A, 0x17, 0x2A][..],
                BinaryOp::Eq,
            ),
            (
                &[0x02, 0x2D, 0x02, 0x16, 0x2A, 0x17, 0x2A][..],
                BinaryOp::Ne,
            ),
            (
                &[0x02, 0x39, 0x02, 0, 0, 0, 0x16, 0x2A, 0x17, 0x2A][..],
                BinaryOp::Eq,
            ),
            (
                &[0x02, 0x3A, 0x02, 0, 0, 0, 0x16, 0x2A, 0x17, 0x2A][..],
                BinaryOp::Ne,
            ),
        ] {
            let m = import_ii(il).expect("imports");
            match &m.blocks[0].terminator {
                hir::Terminator::Branch { cond, then, else_ } => {
                    let (op, lhs, rhs) = as_binary(cond);
                    assert_eq!(op, expected);
                    assert_eq!(as_local(lhs), LocalId(0));
                    assert_eq!(as_i32(rhs), 0);
                    assert_eq!(*then, BlockId(2));
                    assert_eq!(*else_, BlockId(1));
                }
                _ => panic!("expected Branch"),
            }
        }
    }

    #[test]
    fn br_forms() {
        // br L; L: ldarg.0; ret — short (0x2B) and long (0x38).
        for il in [
            &[0x2B, 0x00, 0x02, 0x2A][..],
            &[0x38, 0x00, 0x00, 0x00, 0x00, 0x02, 0x2A][..],
        ] {
            let m = import_ii(il).expect("imports");
            assert_eq!(m.blocks.len(), 2);
            match &m.blocks[0].terminator {
                hir::Terminator::Jump { target } => assert_eq!(*target, BlockId(1)),
                _ => panic!("expected Jump"),
            }
            assert_eq!(as_local(return_value(&m, 1)), LocalId(0));
        }
    }

    #[test]
    fn nop_and_fallthrough_jump() {
        // nop produces no statement.
        let m = import_ii(&[0x00, 0x02, 0x2A]).expect("imports");
        assert!(m.blocks[0].stmts.is_empty());

        // A block not ending in a terminator falls through to the next:
        // ldarg.0; brfalse.s L; ldc.i4.0; stloc.0; (fallthrough)
        // L: ldc.i4.0; ret.
        let il = [0x02, 0x2C, 0x02, 0x16, 0x0A, 0x16, 0x2A];
        let (ee, info) = fixture(
            &il,
            &sig(CorInfoType::Int, &[CorInfoType::Int]),
            &[CorInfoType::Int],
        );
        let m = import(&info, &ee).expect("imports");
        assert_eq!(m.blocks.len(), 3);
        assert_eq!(m.blocks[1].stmts.len(), 1);
        match &m.blocks[1].terminator {
            hir::Terminator::Jump { target } => assert_eq!(*target, BlockId(2)),
            _ => panic!("expected fallthrough Jump"),
        }
    }

    #[test]
    fn call_void_becomes_eval_statement() {
        // ldc.i4.1; ldc.i4.2; call void(int,int); ldarg.0; ret.
        let il = [0x17, 0x18, 0x28, 0x02, 0x00, 0x00, 0x06, 0x02, 0x2A];
        let m = import_ii(&il).expect("imports");
        assert_eq!(m.blocks[0].stmts.len(), 1);
        let stmt = &m.blocks[0].stmts[0];
        assert_eq!(stmt.il_offset, IlOffset(2));
        match &stmt.kind {
            hir::StmtKind::Eval(call) => {
                let (sig, args) = as_call(call);
                assert_eq!(sig.ret, Type::Void);
                assert_eq!(sig.args, vec![Type::Int32, Type::Int32]);
                assert!(!sig.has_this);
                // IL push order: first declared arg is deepest.
                assert_eq!(as_i32(&args[0]), 1);
                assert_eq!(as_i32(&args[1]), 2);
            }
            _ => panic!("expected StmtKind::Eval"),
        }
        assert_eq!(as_local(return_value(&m, 0)), LocalId(0));
    }

    #[test]
    fn call_instance_pops_this_first() {
        // ldarg.0 (this); ldarg.1; call int inst(int); ret.
        let il = [0x02, 0x03, 0x28, 0x03, 0x00, 0x00, 0x06, 0x2A];
        let entry = MockSig {
            ret: CorInfoType::Int,
            args: vec![CorInfoType::Int],
            has_this: true,
        };
        let (ee, info) = fixture(&il, &entry, &[]);
        let m = import(&info, &ee).expect("imports");
        // `this` is arg 0 and typed Ref.
        assert_eq!(m.num_args, 2);
        assert_eq!(m.locals[0].ty, Type::Ref);
        let (sig, args) = as_call(return_value(&m, 0));
        assert!(sig.has_this);
        assert_eq!(args.len(), 2);
        assert_eq!(as_local(&args[0]), LocalId(0));
        assert_eq!(as_local(&args[1]), LocalId(1));
    }

    #[test]
    fn ret_void() {
        let (ee, info) = fixture(&[0x2A], &sig(CorInfoType::Void, &[]), &[]);
        let m = import(&info, &ee).expect("imports");
        assert!(matches!(
            m.blocks[0].terminator,
            hir::Terminator::Return { value: None }
        ));
    }

    // --- negative tests: clean CompileError, never a panic ---

    #[test]
    fn unknown_opcodes_are_unsupported() {
        // callvirt, an undefined single byte, an unsupported 0xFE form.
        for il in [
            &[0x6F, 0x01, 0x00, 0x00, 0x06, 0x2A][..],
            &[0xFF, 0x2A][..],
            &[0xFE, 0x17, 0x2A][..],
        ] {
            assert!(
                matches!(import_ii(il), Err(CompileError::Unsupported(_))),
                "IL {il:?} must be Unsupported"
            );
        }
    }

    #[test]
    fn stack_underflow_is_bad_il() {
        for il in [
            &[0x58, 0x2A][..],       // add on an empty stack
            &[0x02, 0x58, 0x2A][..], // add on one value
            &[0x0A, 0x16, 0x2A][..], // stloc.0 on an empty stack
            &[0x2A][..],             // ret with no return value
        ] {
            let (ee, info) = fixture(
                il,
                &sig(CorInfoType::Int, &[CorInfoType::Int]),
                &[CorInfoType::Int],
            );
            assert!(
                matches!(import(&info, &ee), Err(CompileError::BadIl(_))),
                "IL {il:?} must be BadIl"
            );
        }
    }

    #[test]
    fn bad_branch_targets_are_bad_il() {
        for il in [
            // Past the end (target 0x12 > len).
            &[0x2B, 0x10, 0x02, 0x2A][..],
            // Before the start (target 2 - 16 < 0).
            &[0x2B, 0xF0, 0x02, 0x2A][..],
            // Mid-instruction: br.s +1 lands inside the ldc.i4 at offset 2.
            &[0x2B, 0x01, 0x20, 0x00, 0x00, 0x00, 0x00, 0x2A][..],
        ] {
            assert!(
                matches!(import_ii(il), Err(CompileError::BadIl(_))),
                "IL {il:?} must be BadIl"
            );
        }
    }

    #[test]
    fn malformed_bodies_are_bad_il() {
        // Falls off the end of the method.
        assert!(matches!(import_ii(&[0x16]), Err(CompileError::BadIl(_))));
        // Conditional branch with no fallthrough (blt.s loops to itself).
        assert!(matches!(
            import_ii(&[0x16, 0x16, 0x32, 0xFE]),
            Err(CompileError::BadIl(_))
        ));
        // Truncated operands.
        assert!(matches!(
            import_ii(&[0x20, 0x01]),
            Err(CompileError::BadIl(_))
        ));
        assert!(matches!(import_ii(&[0x2B]), Err(CompileError::BadIl(_))));
        // Unreachable code after ret that no branch targets.
        assert!(matches!(
            import_ii(&[0x2A, 0x00]),
            Err(CompileError::BadIl(_))
        ));
        // Return type mismatch (Ref value, Int32 signature).
        let (ee, info) = fixture(
            &[0x02, 0x2A],
            &sig(CorInfoType::Int, &[CorInfoType::Class]),
            &[],
        );
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));
        // Stack not empty at ret.
        assert!(matches!(
            import_ii(&[0x02, 0x16, 0x2A]),
            Err(CompileError::BadIl(_))
        ));
        // Call argument type mismatch (Ref pushed, Int32 declared).
        let il = [0x02, 0x28, 0x01, 0x00, 0x00, 0x06, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));
    }

    #[test]
    fn index_out_of_range_is_bad_il() {
        // ldarg.1 with one argument.
        assert!(matches!(
            import_ii(&[0x03, 0x2A]),
            Err(CompileError::BadIl(_))
        ));
        // ldloc.0 / stloc.0 with no locals.
        assert!(matches!(
            import_ii(&[0x06, 0x2A]),
            Err(CompileError::BadIl(_))
        ));
        assert!(matches!(
            import_ii(&[0x16, 0x0A, 0x16, 0x2A]),
            Err(CompileError::BadIl(_))
        ));
    }

    #[test]
    fn max_stack_is_enforced() {
        // maxStack 1, but add needs two values.
        let (ee, info) = fixture_full(
            &[0x02, 0x02, 0x58, 0x2A],
            &sig(CorInfoType::Int, &[CorInfoType::Int]),
            &[],
            1,
            0,
        );
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));
    }

    #[test]
    fn stack_values_crossing_a_boundary_are_unsupported() {
        // ldc.i4.0; br.s L; L: ldc.i4.0; stloc.0; ldc.i4.0; ret — the branch
        // carries one stack value into L.
        let il = [0x16, 0x2B, 0x00, 0x16, 0x0A, 0x16, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[CorInfoType::Int]);
        assert!(matches!(
            import(&info, &ee),
            Err(CompileError::Unsupported(_))
        ));
    }

    #[test]
    fn eh_regions_are_unsupported() {
        let (ee, info) = fixture_full(&[0x2A], &sig(CorInfoType::Void, &[]), &[], 8, 1);
        assert!(matches!(
            import(&info, &ee),
            Err(CompileError::Unsupported(_))
        ));
    }
}
