//! Pipeline stage 1 (step_07.2): CIL bytes → [`hir::Method`].
//!
//! Scope is the fib subset (`RokaJIT-internal/docs/step_07.md`), the
//! step_10.1 scalar-cheap pack (`RokaJIT-internal/docs/step_10.1.md`), and
//! the step_10.2 float pack (`RokaJIT-internal/docs/step_10.2.md`):
//! `ldarg`/`ldloc`/`stloc` in all widths, the `ldc.i4` family plus
//! `ldc.r4`/`ldc.r8`, `add`/`sub`/`mul`/`div`/`rem` (integers and floats;
//! float `rem` expands to the `CORINFO_HELP_FLTREM`/`DBLREM` helper call)
//! and the unsigned `div.un`/`rem.un`, the logic ops `and`/`or`/`xor`/
//! `neg`/`not` (`neg` accepts floats), the shifts `shl`/`shr`/`shr.un`,
//! compare-as-value `ceq`/`cgt`/`cgt.un`/`clt`/`clt.un` (integers,
//! floats, and the reference forms), the integer conversions
//! `conv.i1`/`i2`/`i4`/`i8`/`u4`/`u8` (float sources truncate toward
//! zero; `conv.u8` from a float is out) plus `conv.r4`/`conv.r8`,
//! `dup`/`pop`, `ldloca`/`ldarga`/`starg` (short and wide forms),
//! `ldnull`, `ldstr` (resolved through the EE's `constructStringLiteral`
//! to a frozen-ref constant; the IAT_PVALUE/PPVALUE indirection forms are
//! out), the compare-branch family `beq`..`blt.un` plus `brfalse`/
//! `brtrue`/`br` (short and long forms; the null-check forms now also
//! accept references, and the compare forms floats), `call`, and `ret`.
//! The step_10.4 object pack adds `callvirt` (scoped: the EE must
//! devirtualize to a direct call — a real vtable dispatch is
//! `Unsupported`), instance field access `ldfld`/`stfld`/`ldflda` (static
//! fields are out), and `newobj` (EE allocation helper + a direct
//! constructor call; the reference-field store goes through the EE's
//! checked-write-barrier helper). Anything else is
//! [`CompileError::Unsupported`]; malformed IL is
//! [`CompileError::BadIl`]. The importer never panics: every operand read
//! is bounds-checked.
//!
//! Stack discipline (ECMA-335 §III): the evaluation stack is simulated
//! statically, with types propagated. It must be **empty at every block
//! boundary** — values crossing a boundary are legal IL but need temp
//! materialization, which is a later step; they are rejected as
//! `Unsupported` (so the ir-design stack-height invariant holds vacuously
//! for everything the importer accepts). A value may, however, stay on
//! the stack across a `stloc` *within* a block: a tree that references
//! the store's destination observed the pre-store value, so `stloc`
//! spills every such tree to a temp first (RyuJIT's `impSpillLclRefs`).
//!
//! EE queries consumed (via `&dyn EeInfo`): `resolve_token`,
//! `get_call_info`, `construct_string_literal` (for `ldstr`), the field
//! queries (`get_field_offset`/`get_field_type`/`is_field_static`),
//! `embed_class_handle`, `init_class`, and `get_new_helper` (the object
//! pack), and
//! signature walking (`get_arg_type`/`get_arg_next`,
//! bounded by `numArgs` — the real EE's `getArgNext` never returns null, so
//! stepping past `numArgs` walks off the signature blob). The entry
//! method's argument and local signatures come from
//! [`MethodInfo::args`]/[`MethodInfo::locals`] (the frozen pipeline
//! contract), not from `get_method_sig`.

use std::collections::{BTreeSet, HashMap};

use rokajit_ee::ee_info::{zeroed_out, EeInfo};
use rokajit_ee::enums::{
    CallInfoFlags, CorInfoHelpFunc, CorInfoInitClassResult, CorInfoType, InfoAccessType,
};
use rokajit_ee::handles::{
    ArgListHandle, ClassHandle, ContextHandle, FieldHandle, MethodHandle, ModuleHandle,
};
use rokajit_ffi as ffi;

use crate::error::{CompileError, CompileResult};
use crate::ir::{
    hir, BinaryOp, BlockId, CallSig, CallTarget, Const, IlOffset, LocalId, Type, UnaryOp,
};
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
    // IL locals from the locals signature. The importer appends its own
    // temps (the stloc interference spill — see `BlockImport::stloc`) after
    // the IL locals.
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
        local_types,
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

    let locals = importer
        .local_types
        .iter()
        .enumerate()
        .map(|(i, &ty)| {
            let kind = if (i as u32) < num_args {
                hir::LocalKind::IlArg(i as u32)
            } else if (i as u32) < num_args + num_il_locals {
                hir::LocalKind::IlLocal(i as u32 - num_args)
            } else {
                hir::LocalKind::Temp
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

/// The supported opcode set, operands already decoded. Branch targets are
/// absolute IL offsets, range-checked at decode time.
#[derive(Copy, Clone)]
enum Op {
    Nop,
    LdArg(u16),
    LdLoc(u16),
    StLoc(u16),
    /// `ldarga` — address of an argument, a `ByRef` value.
    LdArgA(u16),
    /// `starg` — store into an argument slot.
    StArg(u16),
    /// `ldloca` — address of a local, a `ByRef` value.
    LdLoca(u16),
    LdcI4(i32),
    /// `ldc.i8` — a 64-bit integer constant.
    LdcI8(i64),
    /// `ldc.r4` — a single-precision constant (the f32 bit pattern
    /// decodes straight from the operand).
    LdcR4(f32),
    /// `ldc.r8`.
    LdcR8(f64),
    /// `ldstr` — a metadata string token (0x70xxxxxx, the #US heap).
    LdStr(u32),
    LdNull,
    Dup,
    Pop,
    /// Same-type numeric binary ops: arithmetic, `div`/`rem` and their
    /// unsigned forms, and the bitwise logic ops. Float operands are
    /// valid for `add`/`sub`/`mul`/`div`/`rem` only (ECMA-335 §III.1.5);
    /// float `rem` expands to the EE helper call at import.
    Binary(BinaryOp),
    /// `shl`/`shr`/`shr.un`: the shift count's type is independent of the
    /// value's (ECMA-335 §III.1.5), so these are not [`Op::Binary`].
    Shift(BinaryOp),
    /// `neg`/`not`.
    Unary(UnaryOp),
    /// `ceq`/`cgt`/`cgt.un`/`clt`/`clt.un` — compare producing an Int32
    /// value (as opposed to the branch-folded compares).
    Compare(BinaryOp),
    /// `conv.i1`/`i2`/`i4`/`i8`/`u4`/`u8` (unchecked forms only).
    Conv(ConvKind),
    Br {
        target: u32,
    },
    /// `brfalse`/`brtrue`: one operand, compared against zero/null.
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
    /// `callvirt` — same resolution as `call`, but the receiver is
    /// null-checked (ECMA-335 §III.4.2: NullReferenceException on a null
    /// `this` even when the EE devirtualizes to a direct call).
    CallVirt(u32),
    /// `newobj` — allocation through the EE's `getNewHelper` helper plus a
    /// direct constructor call.
    NewObj(u32),
    /// `ldfld` — instance field load (field metadata token).
    LdFld(u32),
    /// `ldflda` — instance field address, a `ByRef` value.
    LdFldA(u32),
    /// `stfld` — instance field store; a reference-typed field stores
    /// through the EE's checked-write-barrier helper (the GC must be told).
    StFld(u32),
    Ret,
}

/// The `conv.*` opcodes of the scalar-cheap pack plus the float pack's
/// `conv.r4`/`conv.r8`. `I1`/`I2` carry the truncation width the IR's
/// type vocabulary cannot (eval-stack types normalize at Int32, ECMA-335
/// §III.1.1.1) — the importer expands them to shift pairs
/// ([`BlockImport::conv_narrow`]).
#[derive(Copy, Clone)]
enum ConvKind {
    I1,
    I2,
    I4,
    I8,
    U4,
    U8,
    R4,
    R8,
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

    fn u64(&mut self) -> CompileResult<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
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

/// Pass 1: linear decode. Any opcode outside the supported set is
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
            0x0F => Op::LdArgA(u16::from(r.u8()?)),
            0x10 => Op::StArg(u16::from(r.u8()?)),
            0x11 => Op::LdLoc(u16::from(r.u8()?)),
            0x12 => Op::LdLoca(u16::from(r.u8()?)),
            0x13 => Op::StLoc(u16::from(r.u8()?)),
            0x14 => Op::LdNull,
            0x15..=0x1E => Op::LdcI4(i32::from(opcode) - 0x16),
            0x1F => Op::LdcI4(i32::from(r.i8()?)),
            0x20 => Op::LdcI4(r.i32()?),
            0x21 => Op::LdcI8(r.u64()? as i64),
            0x22 => Op::LdcR4(f32::from_bits(r.u32()?)),
            0x23 => Op::LdcR8(f64::from_bits(r.u64()?)),
            0x25 => Op::Dup,
            0x26 => Op::Pop,
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
            0x58 => Op::Binary(BinaryOp::Add),
            0x59 => Op::Binary(BinaryOp::Sub),
            0x5A => Op::Binary(BinaryOp::Mul),
            0x5B => Op::Binary(BinaryOp::Div),
            0x5C => Op::Binary(BinaryOp::UDiv),
            0x5D => Op::Binary(BinaryOp::Rem),
            0x5E => Op::Binary(BinaryOp::URem),
            0x5F => Op::Binary(BinaryOp::And),
            0x60 => Op::Binary(BinaryOp::Or),
            0x61 => Op::Binary(BinaryOp::Xor),
            0x62 => Op::Shift(BinaryOp::Shl),
            0x63 => Op::Shift(BinaryOp::Shr),
            0x64 => Op::Shift(BinaryOp::UShr),
            0x65 => Op::Unary(UnaryOp::Neg),
            0x66 => Op::Unary(UnaryOp::Not),
            0x67 => Op::Conv(ConvKind::I1),
            0x68 => Op::Conv(ConvKind::I2),
            0x69 => Op::Conv(ConvKind::I4),
            0x6A => Op::Conv(ConvKind::I8),
            0x6B => Op::Conv(ConvKind::R4),
            0x6C => Op::Conv(ConvKind::R8),
            0x6D => Op::Conv(ConvKind::U4),
            0x6E => Op::Conv(ConvKind::U8),
            0x6F => Op::CallVirt(r.u32()?),
            0x72 => Op::LdStr(r.u32()?),
            0x73 => Op::NewObj(r.u32()?),
            0x7B => Op::LdFld(r.u32()?),
            0x7C => Op::LdFldA(r.u32()?),
            0x7D => Op::StFld(r.u32()?),
            0xFE => match r.u8()? {
                0x01 => Op::Compare(BinaryOp::Eq),
                0x02 => Op::Compare(BinaryOp::Gt),
                0x03 => Op::Compare(BinaryOp::UGt),
                0x04 => Op::Compare(BinaryOp::Lt),
                0x05 => Op::Compare(BinaryOp::ULt),
                0x09 => Op::LdArg(r.u16()?),
                0x0A => Op::LdArgA(r.u16()?),
                0x0B => Op::StArg(r.u16()?),
                0x0C => Op::LdLoc(r.u16()?),
                0x0D => Op::LdLoca(r.u16()?),
                0x0E => Op::StLoc(r.u16()?),
                _ => {
                    return Err(CompileError::Unsupported(
                        "0xFE-prefixed opcode outside the supported set",
                    ));
                }
            },
            _ => {
                return Err(CompileError::Unsupported(
                    "opcode outside the supported set",
                ))
            }
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
    /// Types of the flat locals namespace (args, IL locals, then importer
    /// temps — grown by [`BlockImport::temp`]).
    local_types: Vec<Type>,
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

/// Does the tree read local `id` anywhere? Drives the stloc interference
/// spill (`BlockImport::stloc`): a tree on the evaluation stack that
/// references the store's destination must keep the pre-store value.
fn references_local(expr: &hir::Expr, id: LocalId) -> bool {
    match expr {
        hir::Expr::Const(_) | hir::Expr::StaticFieldAddr { .. } => false,
        hir::Expr::Local(l) | hir::Expr::LocalAddr(l) => *l == id,
        hir::Expr::Load { addr, .. } => references_local(addr, id),
        hir::Expr::FieldAddr { obj, .. } => references_local(obj, id),
        hir::Expr::Unary { arg, .. } => references_local(arg, id),
        hir::Expr::Binary { lhs, rhs, .. } => {
            references_local(lhs, id) || references_local(rhs, id)
        }
        hir::Expr::Conv { arg, .. } => references_local(arg, id),
        hir::Expr::Call { target, args, .. } => {
            let target_ref = match target {
                CallTarget::Indirect(addr) => references_local(addr, id),
                _ => false,
            };
            target_ref || args.iter().any(|a| references_local(a, id))
        }
        hir::Expr::NullCheck { arg } => references_local(arg, id),
        hir::Expr::ArrLen { array } => references_local(array, id),
        hir::Expr::ArrElemAddr { array, index, .. } => {
            references_local(array, id) || references_local(index, id)
        }
        hir::Expr::Cast { arg, .. } | hir::Expr::Box { arg, .. } => references_local(arg, id),
        hir::Expr::StructVal { addr, .. } => references_local(addr, id),
    }
}

/// Does evaluating the tree have an observable effect beyond producing a
/// value — a call, or a trapping `div`/`rem` (divide-by-zero and the
/// `int.MinValue / -1` overflow are exceptions, so a discarded `x / y`
/// must still execute)? Drives `pop`: a tree with an effect becomes an
/// `Eval` statement; anything else is dropped.
fn must_eval(expr: &hir::Expr) -> bool {
    match expr {
        hir::Expr::Const(_) | hir::Expr::Local(_) | hir::Expr::LocalAddr(_) => false,
        hir::Expr::StaticFieldAddr { .. } => false,
        hir::Expr::Load { .. } => true, // can fault (null byref)
        hir::Expr::FieldAddr { obj, .. } => must_eval(obj),
        hir::Expr::Unary { arg, .. } => must_eval(arg),
        hir::Expr::Binary { op, lhs, rhs } => {
            matches!(
                op,
                BinaryOp::Div | BinaryOp::UDiv | BinaryOp::Rem | BinaryOp::URem
            ) || must_eval(lhs)
                || must_eval(rhs)
        }
        hir::Expr::Conv { arg, .. } => must_eval(arg),
        hir::Expr::Call { .. } => true,
        hir::Expr::NullCheck { .. } => true, // the check itself can fault
        hir::Expr::ArrLen { .. } => true,    // faults on a null array
        hir::Expr::ArrElemAddr { array, index, .. } => must_eval(array) || must_eval(index),
        // Not built by the importer yet, but a cast can throw and a box
        // allocates — both observable.
        hir::Expr::Cast { .. } | hir::Expr::Box { .. } => true,
        hir::Expr::StructVal { addr, .. } => must_eval(addr),
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
        // ECMA-335 §III.1.5: `add`/`sub`/`mul`/`div`/`rem` accept
        // same-type float pairs; the unsigned and bitwise forms are
        // integer-only.
        let float_ok = matches!(
            op,
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Rem
        );
        let (rt, rhs) = self.pop()?;
        let (lt, lhs) = self.pop()?;
        let int = matches!(lt, Type::Int32 | Type::Int64 | Type::NativeInt);
        let fp = matches!(lt, Type::Float | Type::Double);
        if lt != rt || !(int || (float_ok && fp)) {
            return Err(CompileError::BadIl("binary operand type mismatch"));
        }
        if fp && op == BinaryOp::Rem {
            // Float `rem` has no SSE form: it is a call to the EE's
            // fmod/fmodf helper (CORINFO_HELP_FLTREM/DBLREM) — RyuJIT's
            // morph.cpp GT_MOD lowering does exactly this.
            let helper = if lt == Type::Float {
                CorInfoHelpFunc::FLTREM
            } else {
                CorInfoHelpFunc::DBLREM
            };
            return self.push(
                lt,
                hir::Expr::Call {
                    target: CallTarget::Helper(helper),
                    sig: CallSig {
                        ret: lt,
                        args: vec![lt, lt],
                        has_this: false,
                    },
                    args: vec![lhs, rhs],
                },
            );
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

    /// A fresh importer temp, after the IL locals in the flat namespace.
    fn temp(&mut self, ty: Type) -> LocalId {
        let id = LocalId(self.local_types.len() as u32);
        self.local_types.push(ty);
        id
    }

    /// `stloc`/`starg`: the value pops normally, but any tree still on the
    /// stack that references the destination loaded the local *before* this
    /// store — ECMA-335 gives it the old value. Materialize every such
    /// tree into a temp ahead of the store (RyuJIT's `impSpillLclRefs`,
    /// importer.cpp:413), so the tree reads can't observe the new value.
    fn store_local(
        &mut self,
        id: LocalId,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let (ty, value) = self.pop()?;
        if ty != self.local_types[id.0 as usize] {
            return Err(CompileError::BadIl("store type mismatch"));
        }
        for i in 0..self.stack.len() {
            let (entry_ty, _) = &self.stack[i];
            let entry_ty = *entry_ty;
            if references_local(&self.stack[i].1, id) {
                let tmp = self.temp(entry_ty);
                let value = std::mem::replace(&mut self.stack[i].1, hir::Expr::Local(tmp));
                stmts.push(hir::Stmt {
                    il_offset,
                    kind: hir::StmtKind::Store { dst: tmp, value },
                });
            }
        }
        stmts.push(hir::Stmt {
            il_offset,
            kind: hir::StmtKind::Store { dst: id, value },
        });
        Ok(())
    }

    /// `dup`: copy the top of the evaluation stack. A trivial tree copies
    /// outright; anything else is spilled to a temp first so its effects
    /// (calls, trapping divides) still happen exactly once.
    fn dup(&mut self, stmts: &mut Vec<hir::Stmt>, il_offset: IlOffset) -> CompileResult<()> {
        let (ty, value) = self.pop()?;
        match value {
            hir::Expr::Const(k) => {
                self.push(ty, hir::Expr::Const(k))?;
                self.push(ty, hir::Expr::Const(k))?;
            }
            hir::Expr::Local(id) => {
                self.push(ty, hir::Expr::Local(id))?;
                self.push(ty, hir::Expr::Local(id))?;
            }
            hir::Expr::LocalAddr(id) => {
                self.push(ty, hir::Expr::LocalAddr(id))?;
                self.push(ty, hir::Expr::LocalAddr(id))?;
            }
            value => {
                let tmp = self.temp(ty);
                stmts.push(hir::Stmt {
                    il_offset,
                    kind: hir::StmtKind::Store { dst: tmp, value },
                });
                self.push(ty, hir::Expr::Local(tmp))?;
                self.push(ty, hir::Expr::Local(tmp))?;
            }
        }
        Ok(())
    }

    /// `pop`: discard the top of the stack. A tree whose evaluation has an
    /// observable effect (a call, or a `div`/`rem` that can trap) is still
    /// evaluated — it becomes an `Eval` statement.
    fn pop_value(&mut self, stmts: &mut Vec<hir::Stmt>, il_offset: IlOffset) -> CompileResult<()> {
        let (_ty, value) = self.pop()?;
        if must_eval(&value) {
            stmts.push(hir::Stmt {
                il_offset,
                kind: hir::StmtKind::Eval(value),
            });
        }
        Ok(())
    }

    /// `ceq`/`cgt`/`cgt.un`/`clt`/`clt.un`: pop two operands, push the
    /// Int32 result. Integer operands must agree in type; the reference
    /// forms (`ceq` and `cgt.un` only, ECMA-335 §III.1.5) also accept
    /// `Ref`/`ByRef`/`NativeInt` operands in any combination (a `null`
    /// literal compares against both references and pointers).
    /// Same-type float pairs take every form; on floats the plain forms
    /// are the *ordered* predicates (false on NaN) and the `.un` forms
    /// the unordered ones (true on NaN) — table III.4.
    fn compare(&mut self, op: BinaryOp) -> CompileResult<()> {
        let (rt, rhs) = self.pop()?;
        let (lt, lhs) = self.pop()?;
        let int = |t: Type| matches!(t, Type::Int32 | Type::Int64 | Type::NativeInt);
        let ptr = |t: Type| matches!(t, Type::Ref | Type::ByRef | Type::NativeInt);
        let fp = |t: Type| matches!(t, Type::Float | Type::Double);
        let ok = if (int(lt) && int(rt)) || (fp(lt) && fp(rt)) {
            // Same-type numeric pairs (int or float).
            lt == rt
        } else {
            matches!(op, BinaryOp::Eq | BinaryOp::UGt) && ptr(lt) && ptr(rt)
        };
        if !ok {
            return Err(CompileError::BadIl("compare operand type mismatch"));
        }
        self.push(Type::Int32, binary(op, lhs, rhs))
    }

    /// `shl`/`shr`/`shr.un`: the count is `Int32`/`NativeInt` regardless of
    /// the value's type; the result has the value's type. The hardware
    /// masks the count (31/63 by operand width), matching ECMA-335's
    /// masking rule, so no mask tree is built.
    fn shift(&mut self, op: BinaryOp) -> CompileResult<()> {
        let (ct, count) = self.pop_int()?;
        if !matches!(ct, Type::Int32 | Type::NativeInt) {
            return Err(CompileError::BadIl(
                "shift count must be int32 or native int",
            ));
        }
        let (vt, value) = self.pop_int()?;
        self.push(vt, binary(op, value, count))
    }

    /// `conv.*` (unchecked): stack-type transitions of the scalar-cheap
    /// and float packs. Same-width conversions are the identity;
    /// `conv.i1`/`conv.i2` expand to shift pairs (`conv_narrow`); the rest
    /// become `hir::Expr::Conv` nodes whose `unsigned` flag selects sign-
    /// vs zero-extension at lowering. Float sources truncate toward zero
    /// (`cvtt*`); `conv.r4`/`conv.r8` convert from any numeric operand.
    fn conv(&mut self, kind: ConvKind) -> CompileResult<()> {
        let (ty, value) = self.pop()?;
        // Pointer conversions (`conv.i`/`conv.u` from a byref) stay out.
        let int = matches!(ty, Type::Int32 | Type::Int64 | Type::NativeInt);
        let fp = matches!(ty, Type::Float | Type::Double);
        if !int && !fp {
            return Err(CompileError::Unsupported(
                "conv from a non-numeric operand (pointers)",
            ));
        }
        match kind {
            ConvKind::I1 => self.conv_narrow(8, ty, value),
            ConvKind::I2 => self.conv_narrow(16, ty, value),
            // conv.i4/u4: truncate to 32 bits (identity on an Int32).
            ConvKind::I4 | ConvKind::U4 => {
                if ty == Type::Int32 {
                    self.push(Type::Int32, value)
                } else {
                    let unsigned = matches!(kind, ConvKind::U4);
                    self.push(
                        Type::Int32,
                        hir::Expr::Conv {
                            to: Type::Int32,
                            overflow: false,
                            unsigned,
                            arg: Box::new(value),
                        },
                    )
                }
            }
            // conv.i8/u8: extend to 64 bits (identity on a 64-bit
            // operand). conv.u8 from a float needs the unsigned-overflow
            // fixup sequence — outside the pack.
            ConvKind::I8 | ConvKind::U8 => {
                if fp && matches!(kind, ConvKind::U8) {
                    return Err(CompileError::Unsupported("conv.u8 from a float operand"));
                }
                if ty == Type::Int64 || ty == Type::NativeInt {
                    self.push(ty, value)
                } else {
                    let unsigned = matches!(kind, ConvKind::U8);
                    self.push(
                        Type::Int64,
                        hir::Expr::Conv {
                            to: Type::Int64,
                            overflow: false,
                            unsigned,
                            arg: Box::new(value),
                        },
                    )
                }
            }
            // conv.r4/r8: to float from any numeric operand (identity
            // when already at the target width).
            ConvKind::R4 | ConvKind::R8 => {
                let to = if matches!(kind, ConvKind::R4) {
                    Type::Float
                } else {
                    Type::Double
                };
                if ty == to {
                    self.push(to, value)
                } else {
                    self.push(
                        to,
                        hir::Expr::Conv {
                            to,
                            overflow: false,
                            unsigned: false,
                            arg: Box::new(value),
                        },
                    )
                }
            }
        }
    }

    /// `conv.i1`/`conv.i2`: truncate to `bits` then sign-extend, as the
    /// shift pair `(v << (32 - bits)) >> (32 - bits)` at 32 bits. The IR's
    /// type vocabulary normalizes sub-Int32 types away (ECMA-335
    /// §III.1.1.1), so the narrowing cannot be a `Conv` node; the shift
    /// expansion is exact (arithmetic `shr` replicates the sign bit). A
    /// non-Int32 operand converts to Int32 first — for a float source
    /// that is the truncating `cvtt*` conversion, after which the low
    /// `bits` behave as for integers.
    fn conv_narrow(&mut self, bits: u32, ty: Type, value: hir::Expr) -> CompileResult<()> {
        let value = if ty == Type::Int32 {
            value
        } else {
            // A wider operand narrows to 32 bits first; the low `bits`
            // survive either way.
            hir::Expr::Conv {
                to: Type::Int32,
                overflow: false,
                unsigned: false,
                arg: Box::new(value),
            }
        };
        let sh = hir::Expr::Const(Const::Int32((32 - bits) as i32));
        let shifted = binary(BinaryOp::Shl, value, sh);
        let back = binary(
            BinaryOp::Shr,
            shifted,
            hir::Expr::Const(Const::Int32((32 - bits) as i32)),
        );
        self.push(Type::Int32, back)
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

    /// `ldstr` (0x72): the EE constructs and interns the literal — RyuJIT's
    /// `GT_CNS_STR` morph path goes through `constructStringLiteral`
    /// (morph.cpp:6775), and so do we, resolved against the compilation
    /// scope (decisions/2026-09-12-ldstr-and-gc-roots.md). `IAT_VALUE`
    /// hands back the frozen object reference directly: an IR ref
    /// constant. The handle-cell indirection forms (R2R-style) need load
    /// and relocation plumbing tier 0 doesn't have — a later step.
    fn ldstr(&mut self, token: u32) -> CompileResult<()> {
        let Some(module) = ModuleHandle::from_raw(self.info.args.scope) else {
            return Err(CompileError::BadIl("ldstr with a null module scope"));
        };
        let (access, value) = self.ee.construct_string_literal(module, token);
        match access {
            InfoAccessType::Value => {
                let Some(ptr) = value else {
                    return Err(CompileError::Internal(
                        "construct_string_literal: IAT_VALUE with a null pointer",
                    ));
                };
                self.push(
                    Type::Ref,
                    hir::Expr::Const(Const::FrozenRef(ptr.as_ptr() as u64)),
                )
            }
            _ => Err(CompileError::Unsupported(
                "ldstr through a handle-cell indirection (IAT_PVALUE/PPVALUE)",
            )),
        }
    }

    /// `call` (0x28) / `callvirt` (0x6F): resolve the token, take the EE's
    /// verdict on how the call is performed, and pop the call-site
    /// signature's arguments. `flags` carries the callvirt distinction to
    /// the EE (`CORINFO_CALLINFO_CALLVIRT`); `null_check_this` wraps the
    /// receiver in an explicit null check — required for callvirt
    /// (ECMA-335 §III.4.2: NullReferenceException on a null `this` even
    /// for a non-virtual target) and forbidden for `call` (§III.4.1
    /// tolerates a null `this`).
    fn call(
        &mut self,
        token: u32,
        flags: CallInfoFlags,
        null_check_this: bool,
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
            .get_call_info(&mut resolved, None, self.info.ftn, flags);
        if call.kind != ffi::CORINFO_CALL_KIND_CORINFO_CALL {
            // Direct calls only: for callvirt the EE devirtualizes
            // non-virtual and provably-final targets; anything else is a
            // real vtable/interface dispatch — a later step.
            return Err(CompileError::Unsupported("non-direct call kind"));
        }
        check_call_conv(call.sig.callConv)?;
        let has_this = call.sig.callConv & ffi::CorInfoCallConv_CORINFO_CALLCONV_HASTHIS != 0;
        if null_check_this && !has_this {
            return Err(CompileError::BadIl("callvirt on a static method"));
        }
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
            let this = if null_check_this {
                hir::Expr::NullCheck {
                    arg: Box::new(this),
                }
            } else {
                this
            };
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

    /// Field-token resolution shared by `ldfld`/`stfld`/`ldflda`
    /// (step_10.4): the token-kind hint is `CORINFO_TOKENKIND_Field`
    /// (importer.cpp `impResolveToken` for the field opcodes), an
    /// unresolved token is BadIl, and static fields are out of the pack.
    /// Returns the field handle and the EE-supplied instance offset.
    fn resolve_instance_field(&mut self, token: u32) -> CompileResult<(FieldHandle, u32)> {
        let mut resolved = zeroed_out(|t: &mut ffi::CORINFO_RESOLVED_TOKEN| {
            t.tokenContext = self.info.ftn.as_raw() as ffi::CORINFO_CONTEXT_HANDLE;
            t.tokenScope = self.info.args.scope;
            t.token = token;
            t.tokenType = ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Field;
        });
        self.ee.resolve_token(&mut resolved);
        let Some(field) = FieldHandle::from_raw(resolved.hField) else {
            return Err(CompileError::BadIl("field token did not resolve"));
        };
        if self.ee.is_field_static(field) {
            return Err(CompileError::Unsupported(
                "static fields: not yet supported",
            ));
        }
        Ok((field, self.ee.get_field_offset(field)))
    }

    /// The IR type of a field, gated to the 10.4 pack: the full-width
    /// integers, native ints/pointers, and object references. The
    /// sub-Int32 metadata types need width-correct loads/stores and
    /// floats/structs need machinery the pack doesn't have.
    fn field_ir_type(&self, field: FieldHandle) -> CompileResult<Type> {
        let (ty, _value_class) = self.ee.get_field_type(field);
        match ty {
            CorInfoType::Int | CorInfoType::UInt => Ok(Type::Int32),
            CorInfoType::Long | CorInfoType::ULong => Ok(Type::Int64),
            CorInfoType::NativeInt | CorInfoType::NativeUInt | CorInfoType::Ptr => {
                Ok(Type::NativeInt)
            }
            CorInfoType::Class => Ok(Type::Ref),
            _ => Err(CompileError::Unsupported(
                "field type outside the 10.4 object pack",
            )),
        }
    }

    /// The receiver of a field access must be a class reference; a byref
    /// receiver means a value-type field access (out of the pack).
    fn pop_field_receiver(&mut self) -> CompileResult<hir::Expr> {
        let (ty, obj) = self.pop()?;
        if ty != Type::Ref {
            return Err(CompileError::Unsupported(
                "field access on a non-class receiver (value types)",
            ));
        }
        Ok(obj)
    }

    /// `ldfld` (0x7B): the load's address is the null-checked receiver —
    /// the null check is explicit and trap-based (RyuJIT's model: a load
    /// through the pointer, the hardware fault translated by the EE; the
    /// offset-folding optimization is deliberately not tier 0's).
    fn ldfld(&mut self, token: u32) -> CompileResult<()> {
        let (field, offset) = self.resolve_instance_field(token)?;
        let ty = self.field_ir_type(field)?;
        let obj = self.pop_field_receiver()?;
        self.push(
            ty,
            hir::Expr::Load {
                addr: Box::new(hir::Expr::NullCheck { arg: Box::new(obj) }),
                offset,
                ty,
            },
        )
    }

    /// `ldflda` (0x7C): the field's address — type-agnostic (an address
    /// carries no field type), so the field-type gate does not apply.
    fn ldflda(&mut self, token: u32) -> CompileResult<()> {
        let (field, offset) = self.resolve_instance_field(token)?;
        let obj = self.pop_field_receiver()?;
        self.push(
            Type::ByRef,
            hir::Expr::FieldAddr {
                obj: Box::new(hir::Expr::NullCheck { arg: Box::new(obj) }),
                field,
                offset,
            },
        )
    }

    /// `stfld` (0x7D): an indirect store for a non-reference field. A
    /// reference-typed field must inform the GC: the store goes through
    /// the EE's checked write barrier (`JIT_CheckedWriteBarrier(dst,
    /// src)` — `CORINFO_HELP_CHECKED_ASSIGN_REF`), whose two arguments are
    /// exactly SysV's first two integer registers. The barrier itself
    /// must never see a null destination (an AV inside the helper would
    /// not translate to a NullReferenceException), so the destination
    /// address is built off the null-checked receiver.
    fn stfld(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let (field, offset) = self.resolve_instance_field(token)?;
        let ty = self.field_ir_type(field)?;
        let (vt, value) = self.pop()?;
        if vt != ty {
            return Err(CompileError::BadIl("stfld value type mismatch"));
        }
        let obj = hir::Expr::NullCheck {
            arg: Box::new(self.pop_field_receiver()?),
        };
        let kind = if ty == Type::Ref {
            hir::StmtKind::Eval(hir::Expr::Call {
                target: CallTarget::Helper(CorInfoHelpFunc::CHECKED_ASSIGN_REF),
                sig: CallSig {
                    ret: Type::Void,
                    args: vec![Type::ByRef, Type::Ref],
                    has_this: false,
                },
                args: vec![
                    hir::Expr::FieldAddr {
                        obj: Box::new(obj),
                        field,
                        offset,
                    },
                    value,
                ],
            })
        } else {
            hir::StmtKind::StoreInd {
                addr: obj,
                offset,
                value,
            }
        };
        stmts.push(hir::Stmt { il_offset, kind });
        Ok(())
    }

    /// `newobj` (0x73): allocate through the EE's `getNewHelper` helper
    /// (the `CORINFO_HELP_NEWFAST`/`NEWSFAST` families — the
    /// single-argument `(MethodTable*) -> Object*` forms), then run the
    /// constructor as a direct call on the fresh object, and push the
    /// object. The class
    /// handle is embedded as a raw `NativeInt` constant: a MethodTable* is
    /// not an object reference and must never be GC-rooted as one.
    fn newobj(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let mut resolved = zeroed_out(|t: &mut ffi::CORINFO_RESOLVED_TOKEN| {
            t.tokenContext = self.info.ftn.as_raw() as ffi::CORINFO_CONTEXT_HANDLE;
            t.tokenScope = self.info.args.scope;
            t.token = token;
            t.tokenType = ffi::CorInfoTokenKind_CORINFO_TOKENKIND_NewObj;
        });
        self.ee.resolve_token(&mut resolved);
        let Some(class) = ClassHandle::from_raw(resolved.hClass) else {
            return Err(CompileError::BadIl(
                "newobj token did not resolve to a class",
            ));
        };

        // The static-constructor trigger: RyuJIT's newobj import queries
        // initClass with no field (not a field-trigger query), the method
        // being compiled, and the class as the context
        // (importer.cpp CEE_NEWOBJ / corinfo.h:2775's contract). A
        // USE_HELPER verdict emits the CORINFO_HELP_INITCLASS call before
        // the allocation. The context is a *tagged* class context
        // (MAKE_CLASSCONTEXT) — an untagged class handle is interpreted
        // as a method context by the EE and crashes it.
        let context = ContextHandle::from_class(class);
        let init = self.ee.init_class(None, Some(self.info.ftn), context);

        // The allocation's MethodTable* operand, embedded directly; an
        // indirection cell (R2R-style) needs load/reloc plumbing tier 0
        // doesn't have.
        let (embedded, indirection) = self.ee.embed_class_handle(class);
        let (Some(class), None) = (embedded, indirection) else {
            return Err(CompileError::Unsupported(
                "class handle through an indirection cell",
            ));
        };
        let class_const = || hir::Expr::Const(Const::NativeInt(class.as_raw() as isize));

        if init.contains(CorInfoInitClassResult::USE_HELPER) {
            stmts.push(hir::Stmt {
                il_offset,
                kind: hir::StmtKind::Eval(hir::Expr::Call {
                    target: CallTarget::Helper(CorInfoHelpFunc::INITCLASS),
                    sig: CallSig {
                        ret: Type::Void,
                        args: vec![Type::NativeInt],
                        has_this: false,
                    },
                    args: vec![class_const()],
                }),
            });
        }

        let (helper, _has_side_effects) = self.ee.get_new_helper(&resolved, self.info.ftn);
        // The single-argument (MethodTable*) -> Object* class-alloc helpers,
        // minus the FINALIZE forms (a finalizable newobj needs the stack
        // spill of RyuJIT's "finalizable newobj spill" — a later step) and
        // NEWSFAST_ALIGN8_VC (boxed value classes — out with structs). The
        // EE answers NEWSFAST for a plain small class (jitinterface.cpp
        // getNewHelperStatic); NEWFAST is its slow fallback.
        if !matches!(
            helper,
            CorInfoHelpFunc::NEWFAST
                | CorInfoHelpFunc::NEWFAST_MAYBEFROZEN
                | CorInfoHelpFunc::NEWSFAST
                | CorInfoHelpFunc::NEWSFAST_ALIGN8
        ) {
            return Err(CompileError::Unsupported(
                "allocation helper outside the newobj set",
            ));
        }

        // The object lands in a fresh Ref temp — automatically a GC root
        // (frame-resident, zero-initialized) across both safepoints (the
        // allocation and the constructor call). The constructor is a
        // direct call, so no null check wraps the fresh object:
        // JIT_New* never returns null.
        let t_obj = self.temp(Type::Ref);
        stmts.push(hir::Stmt {
            il_offset,
            kind: hir::StmtKind::Store {
                dst: t_obj,
                value: hir::Expr::Call {
                    target: CallTarget::Helper(helper),
                    sig: CallSig {
                        ret: Type::Ref,
                        args: vec![Type::NativeInt],
                        has_this: false,
                    },
                    args: vec![class_const()],
                },
            },
        });

        // The constructor: a direct instance call whose `this` is the
        // fresh object, not a stack value (importer.cpp CEE_NEWOBJ's
        // newObjThisPtr).
        let call = self
            .ee
            .get_call_info(&mut resolved, None, self.info.ftn, CallInfoFlags::EMPTY);
        if call.kind != ffi::CORINFO_CALL_KIND_CORINFO_CALL {
            return Err(CompileError::Unsupported("non-direct call kind"));
        }
        check_call_conv(call.sig.callConv)?;
        if call.sig.callConv & ffi::CorInfoCallConv_CORINFO_CALLCONV_HASTHIS == 0 {
            return Err(CompileError::BadIl("newobj on a static method"));
        }
        let ret = ir_type_raw(call.sig.retType())?;
        if ret != Type::Void {
            return Err(CompileError::BadIl("a constructor must return void"));
        }
        let arg_types = sig_arg_types(&call.sig, self.ee)?;
        let mut args = Vec::with_capacity(arg_types.len() + 1);
        for &expected in arg_types.iter().rev() {
            let (ty, value) = self.pop()?;
            if ty != expected {
                return Err(CompileError::BadIl("call argument type mismatch"));
            }
            args.push(value);
        }
        args.reverse();
        args.insert(0, hir::Expr::Local(t_obj));
        let Some(ctor) = MethodHandle::from_raw(call.hMethod) else {
            return Err(CompileError::BadIl(
                "get_call_info returned a null method handle",
            ));
        };
        stmts.push(hir::Stmt {
            il_offset,
            kind: hir::StmtKind::Eval(hir::Expr::Call {
                target: CallTarget::Direct(ctor),
                sig: CallSig {
                    ret: Type::Void,
                    args: arg_types,
                    has_this: true,
                },
                args,
            }),
        });
        self.push(Type::Ref, hir::Expr::Local(t_obj))
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
                    self.store_local(id, &mut stmts, il_offset)?;
                }
                Op::LdArgA(index) => {
                    let index = u32::from(index);
                    if index >= self.num_args {
                        return Err(CompileError::BadIl("argument index out of range"));
                    }
                    self.push(Type::ByRef, hir::Expr::LocalAddr(LocalId(index)))?;
                }
                Op::StArg(index) => {
                    let index = u32::from(index);
                    if index >= self.num_args {
                        return Err(CompileError::BadIl("argument index out of range"));
                    }
                    self.store_local(LocalId(index), &mut stmts, il_offset)?;
                }
                Op::LdLoca(index) => {
                    let id = self.il_local_id(u32::from(index))?;
                    self.push(Type::ByRef, hir::Expr::LocalAddr(id))?;
                }
                Op::LdcI4(v) => self.push(Type::Int32, hir::Expr::Const(Const::Int32(v)))?,
                Op::LdcI8(v) => self.push(Type::Int64, hir::Expr::Const(Const::Int64(v)))?,
                Op::LdcR4(v) => self.push(Type::Float, hir::Expr::Const(Const::Float(v)))?,
                Op::LdcR8(v) => self.push(Type::Double, hir::Expr::Const(Const::Double(v)))?,
                Op::LdStr(token) => self.ldstr(token)?,
                Op::LdNull => self.push(Type::Ref, hir::Expr::Const(Const::NullRef))?,
                Op::Dup => self.dup(&mut stmts, il_offset)?,
                Op::Pop => self.pop_value(&mut stmts, il_offset)?,
                Op::Binary(op) => self.arith(op)?,
                Op::Shift(op) => self.shift(op)?,
                Op::Compare(op) => self.compare(op)?,
                Op::Unary(op) => {
                    let (ty, value) = self.pop()?;
                    // `neg` accepts integers and floats; `not` is
                    // integer-only (ECMA-335 §III.1.5).
                    let numeric = matches!(
                        ty,
                        Type::Int32 | Type::Int64 | Type::NativeInt | Type::Float | Type::Double
                    );
                    if !numeric || (op == UnaryOp::Not && matches!(ty, Type::Float | Type::Double))
                    {
                        return Err(CompileError::BadIl("unary operand type mismatch"));
                    }
                    self.push(
                        ty,
                        hir::Expr::Unary {
                            op,
                            arg: Box::new(value),
                        },
                    )?;
                }
                Op::Conv(kind) => self.conv(kind)?,
                Op::Call(token) => {
                    self.call(token, CallInfoFlags::EMPTY, false, &mut stmts, il_offset)?
                }
                Op::CallVirt(token) => {
                    self.call(token, CallInfoFlags::CALLVIRT, true, &mut stmts, il_offset)?
                }
                Op::NewObj(token) => self.newobj(token, &mut stmts, il_offset)?,
                Op::LdFld(token) => self.ldfld(token)?,
                Op::LdFldA(token) => self.ldflda(token)?,
                Op::StFld(token) => self.stfld(token, &mut stmts, il_offset)?,
                Op::Br { target } => {
                    self.note_depth(target, self.stack.len())?;
                    terminator = Some(hir::Terminator::Jump {
                        target: self.block_id(target)?,
                    });
                }
                Op::BrZero { op, target } => {
                    // brfalse/brtrue: integers compare against zero,
                    // references against null (`ldnull; brfalse` is the
                    // canonical null check).
                    let (ty, value) = self.pop()?;
                    let zero = match ty {
                        Type::Int32 => Const::Int32(0),
                        Type::Int64 => Const::Int64(0),
                        Type::NativeInt => Const::NativeInt(0),
                        Type::Ref | Type::ByRef => Const::NullRef,
                        _ => {
                            return Err(CompileError::BadIl(
                                "brfalse/brtrue operand must be an integer or reference",
                            ));
                        }
                    };
                    let cond = binary(op, value, hir::Expr::Const(zero));
                    terminator = Some(self.branch(cond, target, insn.offset + insn.size)?);
                }
                Op::BrCmp { op, target } => {
                    let (rt, rhs) = self.pop()?;
                    let (lt, lhs) = self.pop()?;
                    // Integer operands must agree in type; `beq`/`bne.un`
                    // additionally accept reference pairs, and same-type
                    // float pairs take every form (ECMA-335 §III.1.5; on
                    // floats the plain forms are ordered, `.un` unordered).
                    let int = |t: Type| matches!(t, Type::Int32 | Type::Int64 | Type::NativeInt);
                    let ptr = |t: Type| matches!(t, Type::Ref | Type::ByRef);
                    let fp = |t: Type| matches!(t, Type::Float | Type::Double);
                    let ok = if (int(lt) && int(rt)) || (fp(lt) && fp(rt)) {
                        // Same-type numeric pairs (int or float).
                        lt == rt
                    } else {
                        matches!(op, BinaryOp::Eq | BinaryOp::Ne) && ptr(lt) && ptr(rt)
                    };
                    if !ok {
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

    // --- stloc interference spill (RyuJIT's impSpillLclRefs) ---

    #[test]
    fn stloc_spills_stack_trees_referencing_the_destination() {
        // ldloc.0; ldloc.0; ldc.i4.1; add; stloc.0; ret — the first ldloc.0
        // stays on the stack across the store to local 0. IL semantics give
        // it the *pre-store* value, so the importer must snapshot it into a
        // temp ahead of the store; the ret then returns the temp.
        let il = [0x06, 0x06, 0x17, 0x58, 0x0A, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[CorInfoType::Int]);
        let m = import(&info, &ee).expect("imports");
        assert_eq!(m.blocks.len(), 1);
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2, "spill store, then the stloc itself");
        let (dst, value) = store(&stmts[0]);
        assert_eq!(dst, LocalId(1), "the spill temp follows the one IL local");
        assert_eq!(as_local(value), LocalId(0));
        let (dst, value) = store(&stmts[1]);
        assert_eq!(dst, LocalId(0));
        let (op, lhs, rhs) = as_binary(value);
        assert_eq!(op, BinaryOp::Add);
        assert_eq!(as_local(lhs), LocalId(0));
        assert_eq!(as_i32(rhs), 1);
        assert_eq!(as_local(return_value(&m, 0)), LocalId(1));
        // The temp is in the locals table, typed and tagged like one.
        assert_eq!(m.locals.len(), 2);
        assert_eq!(m.locals[1].ty, Type::Int32);
        assert_eq!(m.locals[1].kind, hir::LocalKind::Temp);
    }

    #[test]
    fn stloc_without_interference_does_not_spill() {
        // ldloc.1; ldloc.0; ldc.i4.1; add; stloc.0; ret — the value under
        // the stloc references local 1, not the destination: no temp, no
        // spill store.
        let il = [0x07, 0x06, 0x17, 0x58, 0x0A, 0x2A];
        let (ee, info) = fixture(
            &il,
            &sig(CorInfoType::Int, &[]),
            &[CorInfoType::Int, CorInfoType::Int],
        );
        let m = import(&info, &ee).expect("imports");
        assert_eq!(m.locals.len(), 2, "no temp created");
        assert_eq!(m.blocks[0].stmts.len(), 1);
        assert_eq!(as_local(return_value(&m, 0)), LocalId(1));
    }

    /// The acceptance test for the step_08 FibLoop fix: the exact FibLoop
    /// bytes (see tests/bin, `runtime/src/tests/JIT/CodeGenBringUpTests/
    /// FibLoop.cs`) — an iterative loop whose body Roslyn compiles to an
    /// eval-stack value (`curr + next`) live across `stloc.0`.
    #[test]
    fn fibloop_end_to_end() {
        // 16 0A | 17 0B | 16 0C | 2B 0A | 06 07 58 07 0A 0B 08 17 58 0C |
        // 08 02 32 F2 | 06 2A
        // curr=0; next=1; i=0; br CHECK;
        // LOOP: curr+next (stack); curr=next; next=<stack>; i++;
        // CHECK: if (i < x) goto LOOP; return curr
        let il = [
            0x16, 0x0A, 0x17, 0x0B, 0x16, 0x0C, 0x2B, 0x0A, 0x06, 0x07, 0x58, 0x07, 0x0A, 0x0B,
            0x08, 0x17, 0x58, 0x0C, 0x08, 0x02, 0x32, 0xF2, 0x06, 0x2A,
        ];
        let (ee, info) = fixture(
            &il,
            &sig(CorInfoType::Int, &[CorInfoType::Int]),
            &[CorInfoType::Int, CorInfoType::Int, CorInfoType::Int],
        );
        let m = import(&info, &ee).expect("FibLoop imports");

        // Locals: arg x, IL locals curr/next/i, then the spill temp.
        assert_eq!(m.num_args, 1);
        assert_eq!(m.num_il_locals, 3);
        assert_eq!(m.locals.len(), 5);
        assert_eq!(m.locals[4].kind, hir::LocalKind::Temp);
        assert_eq!(m.blocks.len(), 4);

        // Block 1 (the loop body): the add result must be snapshotted into
        // the temp BEFORE curr is overwritten, then next takes the temp.
        let stmts = &m.blocks[1].stmts;
        assert_eq!(stmts.len(), 4);
        let (dst, value) = store(&stmts[0]);
        assert_eq!(dst, LocalId(4), "interference spill of curr+next");
        let (op, lhs, rhs) = as_binary(value);
        assert_eq!(op, BinaryOp::Add);
        assert_eq!(as_local(lhs), LocalId(1));
        assert_eq!(as_local(rhs), LocalId(2));
        assert_eq!(stmts[0].il_offset, IlOffset(0x0C), "the stloc.0 offset");
        let (dst, value) = store(&stmts[1]);
        assert_eq!(dst, LocalId(1), "curr = next");
        assert_eq!(as_local(value), LocalId(2));
        let (dst, value) = store(&stmts[2]);
        assert_eq!(dst, LocalId(2), "next = the snapshotted sum");
        assert_eq!(as_local(value), LocalId(4));
        let (dst, value) = store(&stmts[3]);
        assert_eq!(dst, LocalId(3), "i++");
        let (op, _, _) = as_binary(value);
        assert_eq!(op, BinaryOp::Add);

        // Block 2: the backward conditional branch (blt.s -14) to block 1.
        match &m.blocks[2].terminator {
            hir::Terminator::Branch { cond, then, else_ } => {
                let (op, lhs, rhs) = as_binary(cond);
                assert_eq!(op, BinaryOp::Lt);
                assert_eq!(as_local(lhs), LocalId(3));
                assert_eq!(as_local(rhs), LocalId(0));
                assert_eq!(*then, BlockId(1));
                assert_eq!(*else_, BlockId(3));
            }
            _ => panic!("block 2: expected Branch"),
        }
        // Block 3: return curr.
        assert_eq!(as_local(return_value(&m, 3)), LocalId(1));
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
    fn ldc_i8_pushes_an_int64_constant() {
        // ldc.i8 long.MinValue; stloc.0; ldc.i4.0; ret.
        let mut il = vec![0x21];
        il.extend_from_slice(&i64::MIN.to_le_bytes());
        il.extend_from_slice(&[0x0A, 0x16, 0x2A]);
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[CorInfoType::Long]);
        let m = import(&info, &ee).expect("imports");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(0));
        assert!(matches!(value, hir::Expr::Const(Const::Int64(i64::MIN))));
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
        // castclass, an undefined single byte, an unsupported 0xFE form.
        for il in [
            &[0x74, 0x01, 0x00, 0x00, 0x06, 0x2A][..],
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

    // --- step_10.1: the scalar-cheap pack ---

    fn as_unary(e: &hir::Expr) -> (UnaryOp, &hir::Expr) {
        match e {
            hir::Expr::Unary { op, arg } => (*op, arg),
            _ => panic!("expected Expr::Unary"),
        }
    }

    fn as_conv(e: &hir::Expr) -> (Type, bool, bool, &hir::Expr) {
        match e {
            hir::Expr::Conv {
                to,
                overflow,
                unsigned,
                arg,
            } => (*to, *overflow, *unsigned, arg),
            _ => panic!("expected Expr::Conv"),
        }
    }

    fn as_local_addr(e: &hir::Expr) -> LocalId {
        match e {
            hir::Expr::LocalAddr(id) => *id,
            _ => panic!("expected Expr::LocalAddr"),
        }
    }

    #[test]
    fn dup_copies_trivial_trees() {
        // ldc.i4.3; dup; add; ret — both copies are the same constant.
        let m = import_ii(&[0x19, 0x25, 0x58, 0x2A]).expect("imports");
        assert!(m.blocks[0].stmts.is_empty(), "no spill for a constant");
        let (op, lhs, rhs) = as_binary(return_value(&m, 0));
        assert_eq!(op, BinaryOp::Add);
        assert_eq!(as_i32(lhs), 3);
        assert_eq!(as_i32(rhs), 3);

        // ldarg.0; dup; add; ret — a local reference copies too.
        let m = import_ii(&[0x02, 0x25, 0x58, 0x2A]).expect("imports");
        assert!(m.blocks[0].stmts.is_empty());
        let (op, lhs, rhs) = as_binary(return_value(&m, 0));
        assert_eq!(op, BinaryOp::Add);
        assert_eq!(as_local(lhs), LocalId(0));
        assert_eq!(as_local(rhs), LocalId(0));
    }

    #[test]
    fn dup_of_a_call_spills_to_a_temp() {
        // call fib; dup; add; ret — the call must evaluate exactly once:
        // a store to a fresh temp, then two reads of it.
        let il = [0x02, 0x28, 0x01, 0x00, 0x00, 0x06, 0x25, 0x58, 0x2A];
        let m = import_ii(&il).expect("imports");
        assert_eq!(m.locals.len(), 2, "arg + the dup spill temp");
        assert_eq!(m.locals[1].kind, hir::LocalKind::Temp);
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1);
        let (dst, value) = store(&stmts[0]);
        assert_eq!(dst, LocalId(1));
        assert!(matches!(value, hir::Expr::Call { .. }));
        let (op, lhs, rhs) = as_binary(return_value(&m, 0));
        assert_eq!(op, BinaryOp::Add);
        assert_eq!(as_local(lhs), LocalId(1));
        assert_eq!(as_local(rhs), LocalId(1));
    }

    #[test]
    fn pop_drops_pure_trees_but_evaluates_effects() {
        // ldc.i4.1; pop; ldarg.0; ret — a pure tree is dropped outright.
        let m = import_ii(&[0x17, 0x26, 0x02, 0x2A]).expect("imports");
        assert!(m.blocks[0].stmts.is_empty());

        // call fib; pop; ldarg.0; ret — the call survives as an Eval.
        let il = [0x02, 0x28, 0x01, 0x00, 0x00, 0x06, 0x26, 0x02, 0x2A];
        let m = import_ii(&il).expect("imports");
        assert_eq!(m.blocks[0].stmts.len(), 1);
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::Eval(e) => assert!(matches!(e, hir::Expr::Call { .. })),
            _ => panic!("expected StmtKind::Eval"),
        }

        // ldarg.0; ldarg.0; div; pop; ldc.i4.0; ret — a trapping divide
        // must still execute: it becomes an Eval of the div tree.
        let il = [0x02, 0x02, 0x5B, 0x26, 0x16, 0x2A];
        let m = import_ii(&il).expect("imports");
        assert_eq!(m.blocks[0].stmts.len(), 1);
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::Eval(e) => assert_eq!(as_binary(e).0, BinaryOp::Div),
            _ => panic!("expected StmtKind::Eval of the divide"),
        }
    }

    #[test]
    fn ldloca_ldarga_starg_forms() {
        // ldloca.s 0; pop; ldc.i4.0; ret — address of local 0 is a ByRef.
        let il = [0x12, 0x00, 0x26, 0x16, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[CorInfoType::Int]);
        let m = import(&info, &ee).expect("imports");
        assert!(m.blocks[0].stmts.is_empty(), "pop of an address is pure");

        // The wide ldloca (0xFE 0D) decodes identically.
        let il = [0xFE, 0x0D, 0x00, 0x00, 0x26, 0x16, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[CorInfoType::Int]);
        import(&info, &ee).expect("wide ldloca imports");

        // ldarga.s 0 (and wide 0xFE 0A); starg.s 0 (and wide 0xFE 0B):
        // starg.s of an int into arg 0 stores through the locals table.
        for il in [
            &[0x0F, 0x00, 0x26, 0x16, 0x2A][..], // ldarga.s 0; pop
            &[0xFE, 0x0A, 0x00, 0x00, 0x26, 0x16, 0x2A][..], // ldarga 0; pop
        ] {
            let m = import_ii(il).expect("ldarga imports");
            assert!(m.blocks[0].stmts.is_empty());
        }
        for il in [
            &[0x17, 0x10, 0x00, 0x02, 0x2A][..], // ldc.i4.1; starg.s 0; ldarg.0; ret
            &[0x17, 0xFE, 0x0B, 0x00, 0x00, 0x02, 0x2A][..], // wide starg
        ] {
            let m = import_ii(il).expect("starg imports");
            let stmts = &m.blocks[0].stmts;
            assert_eq!(stmts.len(), 1);
            let (dst, value) = store(&stmts[0]);
            assert_eq!(dst, LocalId(0), "starg writes the argument slot");
            assert_eq!(as_i32(value), 1);
            assert_eq!(as_local(return_value(&m, 0)), LocalId(0));
        }
    }

    #[test]
    fn starg_spills_stack_trees_referencing_the_argument() {
        // ldarg.0; ldc.i4.1; starg.s 0; ret — the ldarg.0 under the store
        // read arg 0 *before* the store: it is snapshotted to a temp (the
        // same spill discipline as stloc) and is what `ret` returns.
        let il = [0x02, 0x17, 0x10, 0x00, 0x2A];
        let m = import_ii(&il).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2, "spill store, then the starg itself");
        let (dst, value) = store(&stmts[0]);
        assert_eq!(dst, LocalId(1), "the spill temp follows the one arg");
        assert_eq!(as_local(value), LocalId(0));
        let (dst, value) = store(&stmts[1]);
        assert_eq!(dst, LocalId(0));
        assert_eq!(as_i32(value), 1);
        assert_eq!(as_local(return_value(&m, 0)), LocalId(1));
    }

    #[test]
    fn ldarga_of_a_ref_arg_is_typed_byref() {
        // ldarga.s 0; starg.s 1 — byref copy between two byref args.
        let il = [0x0F, 0x00, 0x10, 0x01, 0x16, 0x2A];
        let (ee, info) = fixture(
            &il,
            &sig(CorInfoType::Int, &[CorInfoType::ByRef, CorInfoType::ByRef]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(1));
        assert_eq!(as_local_addr(value), LocalId(0));
    }

    #[test]
    fn compare_ops_produce_int32_values() {
        // ldarg.0; ldarg.1; cXX; ret — ceq/cgt/cgt.un/clt/clt.un.
        for (opcode2, expected) in [
            (0x01, BinaryOp::Eq),
            (0x02, BinaryOp::Gt),
            (0x03, BinaryOp::UGt),
            (0x04, BinaryOp::Lt),
            (0x05, BinaryOp::ULt),
        ] {
            let il = [0x02, 0x03, 0xFE, opcode2, 0x2A];
            let m = import_ii2(&il);
            let (op, lhs, rhs) = as_binary(return_value(&m, 0));
            assert_eq!(op, expected);
            assert_eq!(as_local(lhs), LocalId(0));
            assert_eq!(as_local(rhs), LocalId(1));
        }
    }

    #[test]
    fn ceq_accepts_references_and_null() {
        // ldarg.0; ldnull; ceq; ret — a Ref-vs-null compare, Int32 result.
        let il = [0x02, 0x14, 0xFE, 0x01, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
        let m = import(&info, &ee).expect("imports");
        let (op, lhs, rhs) = as_binary(return_value(&m, 0));
        assert_eq!(op, BinaryOp::Eq);
        assert_eq!(as_local(lhs), LocalId(0));
        assert!(matches!(rhs, hir::Expr::Const(Const::NullRef)));

        // clt on references is BadIl (ECMA-335 §III.1.5: only ceq/cgt.un).
        let il = [0x02, 0x02, 0xFE, 0x04, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));
    }

    #[test]
    fn logic_and_unary_ops() {
        // ldarg.0; ldarg.1; op; ret for and/or/xor; ldarg.0; neg/not; ret.
        for (opcode, expected) in [
            (0x5F, BinaryOp::And),
            (0x60, BinaryOp::Or),
            (0x61, BinaryOp::Xor),
        ] {
            let il = [0x02, 0x03, opcode, 0x2A];
            let m = import_ii2(&il);
            assert_eq!(as_binary(return_value(&m, 0)).0, expected);
        }
        for (opcode, expected) in [(0x65, UnaryOp::Neg), (0x66, UnaryOp::Not)] {
            let il = [0x02, opcode, 0x2A];
            let m = import_ii(&il).expect("imports");
            let (op, arg) = as_unary(return_value(&m, 0));
            assert_eq!(op, expected);
            assert_eq!(as_local(arg), LocalId(0));
        }
    }

    #[test]
    fn shifts_take_an_independent_count_type() {
        // The legal mixed case: value i64, count i32 (ECMA-335 §III.1.5).
        for (opcode, expected) in [
            (0x62, BinaryOp::Shl),
            (0x63, BinaryOp::Shr),
            (0x64, BinaryOp::UShr),
        ] {
            let il = [0x02, 0x03, opcode, 0x2A];
            let (ee, info) = fixture(
                &il,
                &sig(CorInfoType::Long, &[CorInfoType::Long, CorInfoType::Int]),
                &[],
            );
            let m = import(&info, &ee).expect("imports");
            let (op, lhs, rhs) = as_binary(return_value(&m, 0));
            assert_eq!(op, expected);
            assert_eq!(as_local(lhs), LocalId(0));
            assert_eq!(as_local(rhs), LocalId(1));
        }
        // i64 count is rejected.
        let il = [0x02, 0x03, 0x62, 0x2A];
        let (ee, info) = fixture(
            &il,
            &sig(CorInfoType::Long, &[CorInfoType::Long, CorInfoType::Long]),
            &[],
        );
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));
    }

    #[test]
    fn div_family_ops() {
        for (opcode, expected) in [
            (0x5B, BinaryOp::Div),
            (0x5C, BinaryOp::UDiv),
            (0x5D, BinaryOp::Rem),
            (0x5E, BinaryOp::URem),
        ] {
            let il = [0x02, 0x03, opcode, 0x2A];
            let m = import_ii2(&il);
            assert_eq!(as_binary(return_value(&m, 0)).0, expected);
        }
    }

    #[test]
    fn conv_widening_and_truncation_nodes() {
        // conv.i4 of an i64 arg: Conv node to Int32 (truncation).
        let il = [0x02, 0x69, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Long]), &[]);
        let m = import(&info, &ee).expect("imports");
        let (to, overflow, unsigned, arg) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::Int32);
        assert!(!overflow && !unsigned);
        assert_eq!(as_local(arg), LocalId(0));

        // conv.u4 of an i64 arg: same truncation, unsigned flag set.
        let il = [0x02, 0x6D, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Long]), &[]);
        let m = import(&info, &ee).expect("imports");
        let (_, _, unsigned, _) = as_conv(return_value(&m, 0));
        assert!(unsigned);

        // conv.i8 of an i32 arg: Conv node to Int64, signed.
        let il = [0x02, 0x6A, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Long, &[CorInfoType::Int]), &[]);
        let m = import(&info, &ee).expect("imports");
        let (to, _, unsigned, arg) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::Int64);
        assert!(!unsigned);
        assert_eq!(as_local(arg), LocalId(0));

        // conv.u8: unsigned extension.
        let il = [0x02, 0x6E, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Long, &[CorInfoType::Int]), &[]);
        let m = import(&info, &ee).expect("imports");
        let (to, _, unsigned, _) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::Int64);
        assert!(unsigned);

        // Same-width conversions are the identity: no node at all.
        let m = import_ii(&[0x02, 0x69, 0x2A]).expect("conv.i4 of i32");
        assert_eq!(as_local(return_value(&m, 0)), LocalId(0));
        let (ee, info) = fixture(
            &[0x02, 0x6A, 0x2A],
            &sig(CorInfoType::Long, &[CorInfoType::Long]),
            &[],
        );
        let m = import(&info, &ee).expect("conv.i8 of i64");
        assert_eq!(as_local(return_value(&m, 0)), LocalId(0));
    }

    #[test]
    fn conv_i1_i2_expand_to_shift_pairs() {
        // ldarg.0; conv.i1; ret — ((v << 24) >> 24) at 32 bits.
        let m = import_ii(&[0x02, 0x67, 0x2A]).expect("imports");
        let (op, shl, count) = as_binary(return_value(&m, 0));
        assert_eq!(op, BinaryOp::Shr, "arithmetic shift right re-sign-extends");
        assert_eq!(as_i32(count), 24);
        let (op, value, count) = as_binary(shl);
        assert_eq!(op, BinaryOp::Shl);
        assert_eq!(as_local(value), LocalId(0));
        assert_eq!(as_i32(count), 24);

        // conv.i2: shift by 16.
        let m = import_ii(&[0x02, 0x68, 0x2A]).expect("imports");
        let (_, _, count) = as_binary(return_value(&m, 0));
        assert_eq!(as_i32(count), 16);

        // A 64-bit operand truncates to 32 bits first: the shl's value is
        // a Conv-to-Int32 node.
        let (ee, info) = fixture(
            &[0x02, 0x67, 0x2A],
            &sig(CorInfoType::Int, &[CorInfoType::Long]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (_, shl, _) = as_binary(return_value(&m, 0));
        let (_, value, _) = as_binary(shl);
        let (to, _, _, arg) = as_conv(value);
        assert_eq!(to, Type::Int32);
        assert_eq!(as_local(arg), LocalId(0));
    }

    #[test]
    fn ldnull_pushes_a_null_ref_and_branches_on_it() {
        // ldnull; stloc.0; ldloc.0; brfalse.s L; ldc.i4.1; ret; L: ldc.i4.2; ret
        // — a Ref local holding null, branched on directly.
        let il = [0x14, 0x0A, 0x06, 0x2C, 0x02, 0x17, 0x2A, 0x18, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[CorInfoType::Class]);
        let m = import(&info, &ee).expect("imports");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(0));
        assert!(matches!(value, hir::Expr::Const(Const::NullRef)));
        match &m.blocks[0].terminator {
            hir::Terminator::Branch { cond, .. } => {
                let (op, lhs, rhs) = as_binary(cond);
                assert_eq!(op, BinaryOp::Eq);
                assert_eq!(as_local(lhs), LocalId(0));
                assert!(matches!(rhs, hir::Expr::Const(Const::NullRef)));
            }
            _ => panic!("expected Branch"),
        }

        // beq on two refs is legal; blt on refs is not.
        let il = [0x02, 0x03, 0x2E, 0x02, 0x16, 0x2A, 0x17, 0x2A];
        let (ee, info) = fixture(
            &il,
            &sig(CorInfoType::Int, &[CorInfoType::Class, CorInfoType::Class]),
            &[],
        );
        import(&info, &ee).expect("beq on refs imports");
        let il = [0x02, 0x03, 0x32, 0x02, 0x16, 0x2A, 0x17, 0x2A];
        let (ee, info) = fixture(
            &il,
            &sig(CorInfoType::Int, &[CorInfoType::Class, CorInfoType::Class]),
            &[],
        );
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));
    }

    // --- step_10.3: ldstr ---

    #[test]
    fn ldstr_pushes_a_frozen_ref_constant() {
        // ldstr 0x70000001; stloc.0; ldloc.0; ldnull; ceq; ret — a Ref
        // local holding the literal, compared against null.
        let il = [
            0x72, 0x01, 0x00, 0x00, 0x70, 0x0A, 0x06, 0x14, 0xFE, 0x01, 0x2A,
        ];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[CorInfoType::Class]);
        let m = import(&info, &ee).expect("imports");
        assert_eq!(m.locals[0].ty, Type::Ref, "the IL local is a reference");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(0));
        match value {
            hir::Expr::Const(Const::FrozenRef(addr)) => {
                // The mock cans 0x5AFE_0000 + token, deterministic per
                // token (the interning shape: identical literals,
                // identical references).
                assert_eq!(*addr, 0x5AFE_0000 + 0x7000_0001);
            }
            _ => panic!("expected Const::FrozenRef"),
        }
        // The literal flows into a ceq against null like any Ref value.
        let (op, lhs, rhs) = as_binary(return_value(&m, 0));
        assert_eq!(op, BinaryOp::Eq);
        assert_eq!(as_local(lhs), LocalId(0));
        assert!(matches!(rhs, hir::Expr::Const(Const::NullRef)));

        // The same token in a second method yields the same address.
        let m2 = import(&info, &ee).expect("imports");
        let (_, value2) = store(&m2.blocks[0].stmts[0]);
        assert!(
            matches!(value2, hir::Expr::Const(Const::FrozenRef(a)) if *a == 0x5AFE_0000 + 0x7000_0001)
        );
    }

    // --- step_10.2: the float pack ---

    fn as_const_float(e: &hir::Expr) -> Const {
        match e {
            hir::Expr::Const(k @ (Const::Float(_) | Const::Double(_))) => *k,
            _ => panic!("expected a float Expr::Const"),
        }
    }

    /// `double f(double, double)` shape.
    fn import_dd(il: &[u8]) -> hir::Method {
        let (ee, info) = fixture(
            il,
            &sig(
                CorInfoType::Double,
                &[CorInfoType::Double, CorInfoType::Double],
            ),
            &[],
        );
        import(&info, &ee).expect("imports")
    }

    #[test]
    fn ldc_r4_r8_push_typed_constants() {
        // ldc.r4 2.5; stloc.0 (float local); ldc.r8 -1.25; stloc.1; ...
        let il = [
            0x22, 0x00, 0x00, 0x20, 0x40, 0x0A, // ldc.r4 2.5; stloc.0
            0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF4, 0xBF,
            0x0B, // ldc.r8 -1.25; stloc.1
            0x16, 0x2A, // ldc.i4.0; ret
        ];
        let (ee, info) = fixture(
            &il,
            &sig(CorInfoType::Int, &[]),
            &[CorInfoType::Float, CorInfoType::Double],
        );
        let m = import(&info, &ee).expect("imports");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(0));
        assert_eq!(m.locals[0].ty, Type::Float);
        assert_eq!(as_const_float(value), Const::Float(2.5));
        let (dst, value) = store(&m.blocks[0].stmts[1]);
        assert_eq!(dst, LocalId(1));
        assert_eq!(m.locals[1].ty, Type::Double);
        assert_eq!(as_const_float(value), Const::Double(-1.25));
    }

    #[test]
    fn float_arith_and_neg_type_like_the_operands() {
        // ldarg.0; ldarg.1; add; ret — Double + Double.
        for (opcode, expected) in [
            (0x58, BinaryOp::Add),
            (0x59, BinaryOp::Sub),
            (0x5A, BinaryOp::Mul),
            (0x5B, BinaryOp::Div),
        ] {
            let il = [0x02, 0x03, opcode, 0x2A];
            let m = import_dd(&il);
            let (op, lhs, rhs) = as_binary(return_value(&m, 0));
            assert_eq!(op, expected);
            assert_eq!(as_local(lhs), LocalId(0));
            assert_eq!(as_local(rhs), LocalId(1));
        }
        // ldarg.0; neg; ret — float neg.
        let m = import_dd(&[0x02, 0x65, 0x2A]);
        let (op, arg) = as_unary(return_value(&m, 0));
        assert_eq!(op, UnaryOp::Neg);
        assert_eq!(as_local(arg), LocalId(0));

        // and/div.un/not on floats are BadIl.
        for il in [
            &[0x02, 0x03, 0x5F, 0x2A][..], // and
            &[0x02, 0x03, 0x5C, 0x2A][..], // div.un
            &[0x02, 0x66, 0x2A][..],       // not
        ] {
            let (ee, info) = fixture(
                il,
                &sig(
                    CorInfoType::Double,
                    &[CorInfoType::Double, CorInfoType::Double],
                ),
                &[],
            );
            assert!(
                matches!(import(&info, &ee), Err(CompileError::BadIl(_))),
                "IL {il:?} must be rejected"
            );
        }
        // Mixed float/double arithmetic is BadIl.
        let (ee, info) = fixture(
            &[0x02, 0x03, 0x58, 0x2A],
            &sig(
                CorInfoType::Double,
                &[CorInfoType::Double, CorInfoType::Float],
            ),
            &[],
        );
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));
    }

    #[test]
    fn float_rem_is_the_ee_helper_call() {
        // ldarg.0; ldarg.1; rem; ret — doubles: CORINFO_HELP_DBLREM.
        let m = import_dd(&[0x02, 0x03, 0x5D, 0x2A]);
        match return_value(&m, 0) {
            hir::Expr::Call { target, sig, args } => {
                assert!(matches!(
                    target,
                    CallTarget::Helper(h) if *h == CorInfoHelpFunc::DBLREM
                ));
                assert_eq!(
                    sig,
                    &CallSig {
                        ret: Type::Double,
                        args: vec![Type::Double, Type::Double],
                        has_this: false
                    }
                );
                assert_eq!(as_local(&args[0]), LocalId(0));
                assert_eq!(as_local(&args[1]), LocalId(1));
            }
            _ => panic!("expected the rem helper call"),
        }
        // floats: CORINFO_HELP_FLTREM.
        let (ee, info) = fixture(
            &[0x02, 0x03, 0x5D, 0x2A],
            &sig(
                CorInfoType::Float,
                &[CorInfoType::Float, CorInfoType::Float],
            ),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        match return_value(&m, 0) {
            hir::Expr::Call { target, sig, .. } => {
                assert!(matches!(
                    target,
                    CallTarget::Helper(h) if *h == CorInfoHelpFunc::FLTREM
                ));
                assert_eq!(sig.ret, Type::Float);
            }
            _ => panic!("expected the rem helper call"),
        }
    }

    #[test]
    fn float_compares_and_branches() {
        // ldarg.0; ldarg.1; cXX; ret — all five forms on doubles.
        for (opcode2, expected) in [
            (0x01, BinaryOp::Eq),
            (0x02, BinaryOp::Gt),
            (0x03, BinaryOp::UGt),
            (0x04, BinaryOp::Lt),
            (0x05, BinaryOp::ULt),
        ] {
            let il = [0x02, 0x03, 0xFE, opcode2, 0x2A];
            let (ee, info) = fixture(
                &il,
                &sig(
                    CorInfoType::Int,
                    &[CorInfoType::Double, CorInfoType::Double],
                ),
                &[],
            );
            let m = import(&info, &ee).expect("imports");
            let (op, lhs, rhs) = as_binary(return_value(&m, 0));
            assert_eq!(op, expected);
            assert_eq!(as_local(lhs), LocalId(0));
            assert_eq!(as_local(rhs), LocalId(1));
        }
        // ldarg.0; ldarg.1; bgt.s L; ... — float compare branches import.
        let il = [0x02, 0x03, 0x30, 0x02, 0x16, 0x2A, 0x17, 0x2A];
        let (ee, info) = fixture(
            &il,
            &sig(
                CorInfoType::Int,
                &[CorInfoType::Double, CorInfoType::Double],
            ),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        match &m.blocks[0].terminator {
            hir::Terminator::Branch { cond, .. } => {
                assert_eq!(as_binary(cond).0, BinaryOp::Gt);
            }
            _ => panic!("expected Branch"),
        }
    }

    #[test]
    fn conv_to_and_from_floats() {
        // ldarg.0; conv.r8; ret — int to double.
        let (ee, info) = fixture(
            &[0x02, 0x6C, 0x2A],
            &sig(CorInfoType::Double, &[CorInfoType::Int]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (to, overflow, unsigned, arg) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::Double);
        assert!(!overflow && !unsigned);
        assert_eq!(as_local(arg), LocalId(0));

        // ldarg.0; conv.r4; ret — double to float.
        let (ee, info) = fixture(
            &[0x02, 0x6B, 0x2A],
            &sig(CorInfoType::Float, &[CorInfoType::Double]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (to, _, _, _) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::Float);

        // ldarg.0; conv.i4; ret — double to int (truncation).
        let (ee, info) = fixture(
            &[0x02, 0x69, 0x2A],
            &sig(CorInfoType::Int, &[CorInfoType::Double]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (to, _, _, _) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::Int32);

        // ldarg.0; conv.u4; ret — the unsigned flag rides along.
        let (ee, info) = fixture(
            &[0x02, 0x6D, 0x2A],
            &sig(CorInfoType::Int, &[CorInfoType::Double]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (to, _, unsigned, _) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::Int32);
        assert!(unsigned);

        // conv.u8 from a float is outside the pack.
        let (ee, info) = fixture(
            &[0x02, 0x6E, 0x2A],
            &sig(CorInfoType::Long, &[CorInfoType::Double]),
            &[],
        );
        assert!(matches!(
            import(&info, &ee),
            Err(CompileError::Unsupported(_))
        ));
    }

    // --- step_10.4: the object pack ---

    const FIELD_TOKEN: u32 = 0x0400_0001;
    const REF_FIELD_TOKEN: u32 = 0x0400_0002;
    const CTOR_TOKEN: u32 = 0x0600_0004;

    /// `int (this, int)` instance-method entry shape, with the two canned
    /// field tokens registered (an Int32 field at offset 16, a Class field
    /// at offset 24).
    fn object_fixture(il: &[u8]) -> (MockEe, MethodInfo) {
        let entry = MockSig {
            ret: CorInfoType::Int,
            args: vec![CorInfoType::Int],
            has_this: true,
        };
        let (mut ee, info) = fixture(il, &entry, &[]);
        ee.add_field(FIELD_TOKEN, CorInfoType::Int, 16);
        ee.add_field(REF_FIELD_TOKEN, CorInfoType::Class, 24);
        (ee, info)
    }

    fn as_null_check(e: &hir::Expr) -> &hir::Expr {
        match e {
            hir::Expr::NullCheck { arg } => arg,
            _ => panic!("expected Expr::NullCheck"),
        }
    }

    #[test]
    fn callvirt_null_checks_this_and_passes_callvirt() {
        // ldarg.0 (this); ldarg.1; callvirt int inst(int); ret.
        let il = [0x02, 0x03, 0x6F, 0x03, 0x00, 0x00, 0x06, 0x2A];
        let (ee, info) = object_fixture(&il);
        let m = import(&info, &ee).expect("imports");
        let (sig, args) = as_call(return_value(&m, 0));
        assert!(sig.has_this);
        assert_eq!(args.len(), 2);
        assert_eq!(as_local(as_null_check(&args[0])), LocalId(0));
        assert_eq!(as_local(&args[1]), LocalId(1));
        assert_eq!(
            ee.call_info_flags.borrow().as_slice(),
            [CallInfoFlags::CALLVIRT]
        );

        // `call` (0x28) on the same shape leaves `this` unchecked (ECMA-335
        // §III.4.1 tolerates a null receiver) and passes no flags.
        let il = [0x02, 0x03, 0x28, 0x03, 0x00, 0x00, 0x06, 0x2A];
        let (ee, info) = object_fixture(&il);
        let m = import(&info, &ee).expect("imports");
        let (_, args) = as_call(return_value(&m, 0));
        assert_eq!(as_local(&args[0]), LocalId(0));
        assert_eq!(
            ee.call_info_flags.borrow().as_slice(),
            [CallInfoFlags::EMPTY]
        );
    }

    #[test]
    fn callvirt_gates_static_targets_and_non_direct_kinds() {
        // callvirt of a static method is BadIl.
        let il = [0x02, 0x6F, 0x01, 0x00, 0x00, 0x06, 0x2A];
        let (ee, info) = object_fixture(&il);
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));

        // A real vtable dispatch (the EE declines to devirtualize) is out.
        let il = [0x02, 0x03, 0x6F, 0x03, 0x00, 0x00, 0x06, 0x2A];
        let (mut ee, info) = object_fixture(&il);
        ee.non_direct_calls.insert(INST_TOKEN);
        assert!(matches!(
            import(&info, &ee),
            Err(CompileError::Unsupported(_))
        ));
    }

    #[test]
    fn ldfld_builds_a_null_checked_load() {
        // ldarg.0; ldfld int@16; ret.
        let il = [0x02, 0x7B, 0x01, 0x00, 0x00, 0x04, 0x2A];
        let (ee, info) = object_fixture(&il);
        let m = import(&info, &ee).expect("imports");
        match return_value(&m, 0) {
            hir::Expr::Load { addr, offset, ty } => {
                assert_eq!(*offset, 16, "the EE-supplied offset");
                assert_eq!(*ty, Type::Int32);
                assert_eq!(as_local(as_null_check(addr)), LocalId(0));
            }
            _ => panic!("expected Expr::Load"),
        }
    }

    #[test]
    fn ldflda_pushes_a_byref_field_address() {
        // ldarg.0; ldflda int@16; stloc.0; ldc.i4.0; ret — the address
        // stores into a ByRef local.
        let il = [0x02, 0x7C, 0x01, 0x00, 0x00, 0x04, 0x0A, 0x16, 0x2A];
        let entry = MockSig {
            ret: CorInfoType::Int,
            args: vec![CorInfoType::Int],
            has_this: true,
        };
        let (mut ee, info) = fixture_full(&il, &entry, &[CorInfoType::ByRef], 8, 0);
        ee.add_field(FIELD_TOKEN, CorInfoType::Int, 16);
        let m = import(&info, &ee).expect("imports");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(2), "the ByRef IL local follows this + arg");
        match value {
            hir::Expr::FieldAddr { obj, offset, .. } => {
                assert_eq!(*offset, 16);
                assert_eq!(as_local(as_null_check(obj)), LocalId(0));
            }
            _ => panic!("expected Expr::FieldAddr"),
        }
    }

    #[test]
    fn stfld_of_an_int_builds_store_ind() {
        // ldarg.0; ldarg.1; stfld int@16; ldc.i4.0; ret.
        let il = [0x02, 0x03, 0x7D, 0x01, 0x00, 0x00, 0x04, 0x16, 0x2A];
        let (ee, info) = object_fixture(&il);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1);
        match &stmts[0].kind {
            hir::StmtKind::StoreInd {
                addr,
                offset,
                value,
            } => {
                assert_eq!(*offset, 16);
                assert_eq!(as_local(as_null_check(addr)), LocalId(0));
                assert_eq!(as_local(value), LocalId(1));
            }
            _ => panic!("expected StmtKind::StoreInd"),
        }
    }

    #[test]
    fn stfld_of_a_reference_goes_through_the_write_barrier() {
        // this.ref = value — the checked-write-barrier helper call:
        // CHECKED_ASSIGN_REF(&this.ref, value), the FieldAddr as arg 0.
        let il = [0x02, 0x14, 0x7D, 0x02, 0x00, 0x00, 0x04, 0x16, 0x2A];
        let (ee, info) = object_fixture(&il);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1);
        match &stmts[0].kind {
            hir::StmtKind::Eval(hir::Expr::Call { target, sig, args }) => {
                assert!(
                    matches!(target, CallTarget::Helper(h) if *h == CorInfoHelpFunc::CHECKED_ASSIGN_REF),
                    "the checked-write-barrier helper"
                );
                assert_eq!(
                    sig,
                    &CallSig {
                        ret: Type::Void,
                        args: vec![Type::ByRef, Type::Ref],
                        has_this: false,
                    }
                );
                assert_eq!(args.len(), 2);
                match &args[0] {
                    hir::Expr::FieldAddr { obj, offset, .. } => {
                        assert_eq!(*offset, 24);
                        assert_eq!(as_local(as_null_check(obj)), LocalId(0));
                    }
                    _ => panic!("expected Expr::FieldAddr"),
                }
                assert!(matches!(args[1], hir::Expr::Const(Const::NullRef)));
            }
            _ => panic!("expected the barrier helper Eval"),
        }
    }

    #[test]
    fn newobj_builds_the_alloc_store_and_ctor_eval() {
        // ldc.i4.7; newobj C::.ctor(int); pop; ldc.i4.0; ret.
        let il = [0x1D, 0x73, 0x04, 0x00, 0x00, 0x06, 0x26, 0x16, 0x2A];
        let entry = sig(CorInfoType::Int, &[]);
        let (mut ee, info) = fixture(&il, &entry, &[]);
        ee.add_method(
            CTOR_TOKEN,
            MockSig {
                ret: CorInfoType::Void,
                args: vec![CorInfoType::Int],
                has_this: true,
            },
        );
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2, "alloc store, ctor eval");

        // t_obj = CORINFO_HELP_NEWFAST(class-as-NativeInt-const).
        let (dst, value) = store(&stmts[0]);
        assert_eq!(dst, LocalId(0), "the object temp is the first local");
        assert_eq!(m.locals[0].ty, Type::Ref);
        assert_eq!(m.locals[0].kind, hir::LocalKind::Temp);
        match value {
            hir::Expr::Call { target, sig, args } => {
                assert!(
                    matches!(target, CallTarget::Helper(h) if *h == CorInfoHelpFunc::NEWFAST),
                    "the NEWFAST allocation helper"
                );
                assert_eq!(
                    sig,
                    &CallSig {
                        ret: Type::Ref,
                        args: vec![Type::NativeInt],
                        has_this: false,
                    }
                );
                assert!(
                    matches!(args[0], hir::Expr::Const(Const::NativeInt(_))),
                    "the class handle is a raw pointer constant, never a Ref"
                );
            }
            _ => panic!("expected the allocation helper call"),
        }

        // C::.ctor(t_obj, 7) — `this` is the temp, not a stack value.
        match &stmts[1].kind {
            hir::StmtKind::Eval(hir::Expr::Call { target, sig, args }) => {
                assert!(matches!(target, CallTarget::Direct(_)));
                assert_eq!(
                    sig,
                    &CallSig {
                        ret: Type::Void,
                        args: vec![Type::Int32],
                        has_this: true,
                    }
                );
                assert_eq!(args.len(), 2);
                assert_eq!(as_local(&args[0]), LocalId(0));
                assert_eq!(as_i32(&args[1]), 7);
            }
            _ => panic!("expected the constructor Eval"),
        }
        // `pop` of the pushed object temp is pure: no third statement.
    }

    #[test]
    fn newobj_with_a_cctor_emits_init_class_first() {
        let il = [0x73, 0x04, 0x00, 0x00, 0x06, 0x26, 0x16, 0x2A];
        let entry = sig(CorInfoType::Int, &[]);
        let (mut ee, info) = fixture(&il, &entry, &[]);
        ee.add_method(
            CTOR_TOKEN,
            MockSig {
                ret: CorInfoType::Void,
                args: vec![],
                has_this: true,
            },
        );
        ee.init_class_result = CorInfoInitClassResult::USE_HELPER;
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 3, "initclass, alloc store, ctor eval");
        match &stmts[0].kind {
            hir::StmtKind::Eval(hir::Expr::Call { target, sig, args }) => {
                assert!(
                    matches!(target, CallTarget::Helper(h) if *h == CorInfoHelpFunc::INITCLASS),
                    "the INITCLASS helper"
                );
                assert_eq!(sig.ret, Type::Void);
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected the INITCLASS helper call"),
        }
    }

    #[test]
    fn newobj_rejects_an_unknown_allocation_helper() {
        let il = [0x73, 0x04, 0x00, 0x00, 0x06, 0x26, 0x16, 0x2A];
        let entry = sig(CorInfoType::Int, &[]);
        let (mut ee, info) = fixture(&il, &entry, &[]);
        ee.add_method(
            CTOR_TOKEN,
            MockSig {
                ret: CorInfoType::Void,
                args: vec![],
                has_this: true,
            },
        );
        ee.new_helper = Some(CorInfoHelpFunc::NEWARR_1_PTR);
        assert!(matches!(
            import(&info, &ee),
            Err(CompileError::Unsupported(_))
        ));
    }

    #[test]
    fn field_gates_static_fields_and_out_of_pack_types() {
        // A static field token: ldfld is Unsupported with the named cause.
        let il = [0x02, 0x7B, 0x01, 0x00, 0x00, 0x04, 0x2A];
        let (mut ee, info) = object_fixture(&il);
        ee.fields.get_mut(&FIELD_TOKEN).unwrap().is_static = true;
        let err = import(&info, &ee)
            .err()
            .expect("static field is unsupported");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("static fields")),
            "{err:?}"
        );

        // A Float field: outside the 10.4 pack.
        let (mut ee, info) = object_fixture(&il);
        ee.fields.get_mut(&FIELD_TOKEN).unwrap().ty = CorInfoType::Float;
        let err = import(&info, &ee)
            .err()
            .expect("float field is unsupported");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("object pack")),
            "{err:?}"
        );

        // An unresolvable field token is BadIl.
        let il = [0x02, 0x7B, 0x77, 0x00, 0x00, 0x04, 0x2A];
        let (ee, info) = object_fixture(&il);
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));

        // A byref receiver (a value-type field access) is Unsupported.
        let il = [0x12, 0x00, 0x7B, 0x01, 0x00, 0x00, 0x04, 0x2A];
        let entry = sig(CorInfoType::Int, &[]);
        let (mut ee, info) = fixture(&il, &entry, &[CorInfoType::Int]);
        ee.add_field(FIELD_TOKEN, CorInfoType::Int, 16);
        let err = import(&info, &ee)
            .err()
            .expect("byref receiver is unsupported");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("value types")),
            "{err:?}"
        );
    }
}
