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
//! `conv.i1`/`i2`/`i4`/`i8`/`u1`/`u2`/`u4`/`u8`/`u`/`i` (float sources
//! truncate toward zero, saturating for the unsigned targets —
//! step_10.11) plus
//! `conv.r4`/`conv.r8`,
//! `dup`/`pop`, `ldloca`/`ldarga`/`starg` (short and wide forms),
//! `ldnull`, `ldstr` (resolved through the EE's `constructStringLiteral`
//! to a frozen-ref constant; the IAT_PVALUE/PPVALUE indirection forms are
//! out), the compare-branch family `beq`..`blt.un` plus `brfalse`/
//! `brtrue`/`br` (short and long forms; the null-check forms now also
//! accept references, and the compare forms floats), `call`, and `ret`.
//! The step_10.4 object pack adds `callvirt` (scoped: the EE must
//! devirtualize to a direct call — a real vtable dispatch is
//! `Unsupported`), instance field access `ldfld`/`stfld`/`ldflda`
//! (statics have their own pack), and `newobj` (EE allocation helper + a
//! direct constructor call; the reference-field store goes through the
//! EE's checked-write-barrier helper). The step_10.7 statics pack adds
//! `ldsfld`/`ldsflda`/`stsfld`: the field's address comes from the EE's
//! `getFieldInfo` (a plain static answers `STATIC_ADDRESS` with the
//! final address as an `IAT_VALUE` constant — no layout math JIT-side;
//! a boxed value-class static is one indirection away), the
//! `CORINFO_FLG_FIELD_INITCLASS` flag plus `init_class` drive the static
//! constructor trigger (a `CORINFO_HELP_INITCLASS` call ahead of the
//! access), and a reference-typed `stsfld` reuses the checked write
//! barrier. The step_10.9 value-type pack adds
//! `initobj`/`ldobj`/`stobj`/`cpobj`, structs in signatures (args,
//! returns — including the hidden return buffer — and locals, with SysV
//! AMD64 eightbyte classification), struct instance methods (`this` as a
//! byref), and struct-typed fields (loads yield the field address as a
//! `StructVal`; stores are block copies, GC-embedding structs through the
//! bulk-write-barrier helper). The step_10.6 EH pack adds `throw`,
//! `leave`/`leave.s`, and `endfinally`, plus the EH clause table:
//! typed-catch and finally clauses become contiguous block ranges (main
//! blocks first in IL order — with synthetic `CallFinally` step blocks
//! spliced in — then each clause's handler blocks grouped at the tail), a
//! catch handler's entry block starts with a synthesized store of the
//! exception object ([`hir::Expr::CatchArg`]), and a `leave` that crosses
//! finally handlers becomes a chain of step blocks ending in
//! [`hir::Terminator::CallFinally`] hops. Filter and fault clauses and
//! `rethrow`/`endfilter` are Unsupported. The step_10.10 pack adds
//! `ldtoken` (the token's raw handle embedded via `embed_generic_handle`
//! — IAT_VALUE only — and converted to the RuntimeTypeHandle/
//! RuntimeMethodHandle/RuntimeFieldHandle struct through the
//! TYPEHANDLE_TO_*/METHODDESC_TO_*/FIELDDESC_TO_* helper family) and
//! `sizeof` (a JIT-time constant fold of `get_class_size`). The
//! step_10.8 array pack adds `newarr` (the EE's `getNewArrHelper`
//! allocation), `ldlen` (a length load at offset 8 that doubles as the
//! null check), and `ldelem.*`/`stelem.*`/`ldelem`/`stelem`/`ldelema`
//! (an explicit `BoundsCheck` statement — the RNGCHKFAIL helper throws
//! `IndexOutOfRangeException` — plus the typed load/store through the
//! computed element address; `stelem.ref` and `ldelema` of a reference
//! element go through the `ARRADDR_ST`/`LDELEMA_REF` helpers, which
//! check bounds, covariance, and barriers internally). Anything else
//! is
//! [`CompileError::Unsupported`]; malformed IL is
//! [`CompileError::BadIl`]. The importer never panics: every operand read
//! is bounds-checked.
//!
//! Stack discipline (ECMA-335 §III): the evaluation stack is simulated
//! statically, with types propagated. It must be **empty at every block
//! boundary** — values crossing a boundary are legal IL but need temp
//! materialization, which is a later step; they are rejected as
//! `Unsupported` (so the ir-design stack-height invariant holds vacuously
//! for everything the importer accepts). The one exception is a catch
//! handler's entry block: the VM enters it with the exception object on
//! the stack, modeled as a synthesized depth-1 entry (step_10.6). A
//! value may, however, stay on
//! the stack across a `stloc` *within* a block: a tree that references
//! the store's destination observed the pre-store value, so `stloc`
//! spills every such tree to a temp first (RyuJIT's `impSpillLclRefs`).
//!
//! EE queries consumed (via `&dyn EeInfo`): `resolve_token`,
//! `get_call_info`, `construct_string_literal` (for `ldstr`), the field
//! queries (`get_field_offset`/`get_field_type`/`is_field_static`, plus
//! `get_field_info` for the statics),
//! `embed_class_handle`, `init_class`, and `get_new_helper` (the object
//! pack), `embed_generic_handle`/`get_token_type_as_handle` (ldtoken) and
//! `get_class_size` (sizeof), `get_new_arr_helper`/`is_sd_array`/
//! `as_cor_info_type` (the array pack), and
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
    hir, BinaryOp, BlockId, CallSig, CallTarget, Const, IlOffset, LocalId, MemAccess, Type, UnaryOp,
};
use crate::pipeline::MethodInfo;
use crate::structs::{layout_of, StructLayouts};

/// Stage entry point (the body of [`crate::pipeline::import`]).
pub fn import(info: &MethodInfo, ee: &dyn EeInfo) -> CompileResult<hir::Method> {
    if info.il.is_empty() {
        return Err(CompileError::BadIl("empty IL stream"));
    }
    check_call_conv(info.args.callConv)?;

    // The locals table: IL args (with `this` first when present), then the
    // IL locals from the locals signature. The importer appends its own
    // temps (the stloc interference spill — see `BlockImport::stloc`) after
    // the IL locals. Layout facts for every value class mentioned anywhere
    // in the method are queried once and cached in `struct_layouts`
    // (step_10.9).
    let mut struct_layouts = StructLayouts::new();
    let mut local_types = Vec::new();
    let has_this = info.args.callConv & ffi::CorInfoCallConv_CORINFO_CALLCONV_HASTHIS != 0;
    if has_this {
        // Class instance method: `this` is an object reference; value-type
        // instance methods take a byref `this` (step_10.9 — mutations
        // through it must reach the caller's memory).
        let class = ee.get_method_class(info.ftn);
        local_types.push(if ee.is_value_class(class) {
            Type::ByRef
        } else {
            Type::Ref
        });
    }
    // The hidden return buffer (step_10.9): a method whose own return type
    // is a non-register-passed struct takes an implicit ByRef argument
    // immediately after `this` (the managed convention, clr-abi.md) and
    // returns the buffer address in rax.
    let ret_ty = sig_elem_type(
        CorInfoType::from_raw(info.args.retType()),
        ClassHandle::from_raw(info.args.retTypeClass),
        ee,
        &mut struct_layouts,
    )?;
    let retbuf = match ret_ty {
        Type::Struct(class) if !struct_layouts[&class].sysv.passed_in_registers => {
            let id = LocalId(local_types.len() as u32);
            local_types.push(Type::ByRef);
            Some(id)
        }
        _ => None,
    };
    local_types.extend(sig_arg_types(&info.args, ee, &mut struct_layouts)?);
    let num_args = local_types.len() as u32;
    local_types.extend(sig_arg_types(&info.locals, ee, &mut struct_layouts)?);
    let num_il_locals = local_types.len() as u32 - num_args;

    let insns = decode(&info.il)?;
    let clauses = fetch_clauses(info, ee)?;
    validate_clauses(&clauses, &insns, info.il.len() as u32)?;
    let leaders = find_leaders(&info.il, &insns, &clauses)?;
    let block_of = leaders
        .iter()
        .enumerate()
        .map(|(i, &offset)| (offset, i as u32))
        .collect();
    let catch_entries: BTreeSet<u32> = clauses
        .iter()
        .filter(|c| matches!(c.kind, ClauseKind::Catch { .. }))
        .map(|c| c.handler_start)
        .collect();
    let mut importer = BlockImport {
        ee,
        info,
        local_types,
        num_args,
        num_il_locals,
        ret_ty,
        retbuf,
        struct_layouts,
        block_of,
        expected_depth: HashMap::new(),
        stack: Vec::new(),
        clauses,
        catch_entries,
        chains: Vec::new(),
    };
    let mut blocks = Vec::with_capacity(leaders.len());
    for b in 0..leaders.len() {
        blocks.push(importer.import_block(b, &leaders, &insns)?);
    }
    // Every block was imported assuming an empty entry stack — except a
    // catch handler's entry block, which starts with the synthesized
    // exception push (depth 1). A predecessor that recorded a different
    // depth means values cross a boundary — or, for a catch entry, that
    // something falls or branches into the handler.
    for (&leader, &depth) in &importer.expected_depth {
        if importer.catch_entries.contains(&leader) {
            if depth != 1 {
                return Err(CompileError::BadIl(
                    "inconsistent stack depth at a merge point",
                ));
            }
        } else if depth != 0 {
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
    // With EH clauses the block list is rebuilt (step_10.6): main-area
    // blocks in IL order with the `leave` chains' step blocks spliced in,
    // then each clause's handler blocks grouped at the tail; ids and the
    // region table follow the new order.
    let (mut blocks, eh_regions) = if importer.clauses.is_empty() {
        (blocks, Vec::new())
    } else {
        let ranges: Vec<(u32, u32)> = (0..leaders.len())
            .map(|b| {
                (
                    leaders[b],
                    leaders.get(b + 1).copied().unwrap_or(info.il.len() as u32),
                )
            })
            .collect();
        rebuild_blocks(
            blocks,
            &ranges,
            &importer.clauses,
            &importer.chains,
            &importer.block_of,
        )?
    };
    // The per-method class-init trigger (RyuJIT's morph.cpp:50
    // fgMorphMainInit): a method whose OWN class has a precise
    // (non-beforefieldinit) cctor not yet run starts with the INITCLASS
    // helper call. The field-access/newobj triggers alone miss the
    // same-class and entry-point cases — a static method that never
    // touches its class's fields would run with the cctor unrun (found
    // by JIT/opt/Cloning/loops_with_eh.cs in the 10.8 triage). The query
    // is `initClass(NULL, NULL, method-context)`: with no field, the EE
    // answers for the context's own class, and its same-class
    // NOT_REQUIRED optimization requires a non-null method — so the
    // prolog trigger fires exactly when the cctor genuinely must run
    // (jitinterface.cpp initClass).
    let prolog_init = importer
        .ee
        .init_class(None, None, ContextHandle::from_method(info.ftn));
    if prolog_init.contains(CorInfoInitClassResult::USE_HELPER) {
        let class = importer.ee.get_method_class(info.ftn);
        let mt = importer.embed_class_const(class)?;
        blocks[0].stmts.insert(
            0,
            hir::Stmt {
                il_offset: IlOffset(0),
                kind: hir::StmtKind::Eval(hir::Expr::Call {
                    target: CallTarget::Helper(CorInfoHelpFunc::INITCLASS),
                    sig: CallSig {
                        ret: Type::Void,
                        args: vec![Type::NativeInt],
                        has_this: false,
                    },
                    args: vec![mt],
                }),
            },
        );
    }
    Ok(hir::Method {
        blocks,
        locals,
        eh_regions,
        num_args,
        num_il_locals,
        struct_layouts: importer.struct_layouts,
    })
}

/// Maps an EE type to the IR's evaluation-stack vocabulary (ECMA-335
/// §III.1.1.1: the sub-Int32 metadata types normalize to Int32). `class`
/// is the value-class handle for `CorInfoType::ValueClass` elements
/// (`getArgType`'s second answer, or the signature's `retTypeClass`); the
/// class's layout is queried into `layouts` on first mention.
fn sig_elem_type(
    ty: Option<CorInfoType>,
    class: Option<ClassHandle>,
    ee: &dyn EeInfo,
    layouts: &mut StructLayouts,
) -> CompileResult<Type> {
    let Some(ty) = ty else {
        return Err(CompileError::BadIl("CorInfoType outside the header set"));
    };
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
            let Some(class) = class else {
                return Err(CompileError::BadIl(
                    "value-class signature element without a class handle",
                ));
            };
            layout_of(layouts, ee, class)?;
            Type::Struct(class)
        }
        CorInfoType::Undef => {
            return Err(CompileError::BadIl("CORINFO_TYPE_UNDEF in signature"));
        }
    })
}

/// The stack type and memory shape of a *stored* element of EE type `ty`
/// (a field, or the payload of an unboxed primitive): unlike
/// [`sig_elem_type`] — the signature/eval-stack view — the sub-Int32
/// types keep their cell width in [`MemAccess`], and `ByRef`/`Undef`/
/// `Void` are rejected (no such storage is in the supported set).
fn corinfo_mem_type(ty: CorInfoType) -> CompileResult<(Type, MemAccess)> {
    Ok(match ty {
        CorInfoType::Bool => (Type::Int32, MemAccess::U8),
        CorInfoType::Char => (Type::Int32, MemAccess::U16),
        // CORINFO_TYPE_BYTE is ELEMENT_TYPE_I1 — the SIGNED byte
        // (jitinterface.cpp asCorInfoType's element map); UBYTE is U1.
        CorInfoType::Byte => (Type::Int32, MemAccess::I8),
        CorInfoType::UByte => (Type::Int32, MemAccess::U8),
        CorInfoType::Short => (Type::Int32, MemAccess::I16),
        CorInfoType::UShort => (Type::Int32, MemAccess::U16),
        CorInfoType::Int | CorInfoType::UInt => (Type::Int32, MemAccess::Natural),
        CorInfoType::Long | CorInfoType::ULong => (Type::Int64, MemAccess::Natural),
        CorInfoType::NativeInt | CorInfoType::NativeUInt | CorInfoType::Ptr => {
            (Type::NativeInt, MemAccess::Natural)
        }
        CorInfoType::Float => (Type::Float, MemAccess::Natural),
        CorInfoType::Double => (Type::Double, MemAccess::Natural),
        CorInfoType::Class => (Type::Ref, MemAccess::Natural),
        _ => {
            return Err(CompileError::Unsupported(
                "field type outside the object pack",
            ));
        }
    })
}

/// The natural cell size of a full-width IR type (array element sizes;
/// step_10.8): 4 for Int32/Float, 8 for the 64-bit types.
fn natural_cell_size(ty: Type) -> u32 {
    match ty {
        Type::Int32 | Type::Float => 4,
        Type::Int64 | Type::NativeInt | Type::Ref | Type::ByRef | Type::Double => 8,
        Type::Struct(_) | Type::Void => unreachable!("no natural cell size"),
    }
}

/// A resolved array element kind (step_10.8): either a typed cell (stack
/// type, memory shape, element size in bytes) or a struct element whose
/// layout sits in the side table.
#[derive(Copy, Clone)]
enum ElemKind {
    Cell(Type, MemAccess, u32),
    Struct(ClassHandle),
}

fn check_call_conv(call_conv: ffi::CorInfoCallConv) -> CompileResult<()> {
    if call_conv & ffi::CorInfoCallConv_CORINFO_CALLCONV_GENERIC != 0 {
        return Err(CompileError::Unsupported("generic methods"));
    }
    // CORINFO_CALLCONV_PARAMTYPE (corinfo.h:666): the method is shared
    // generic code and takes a hidden instantiation argument after its
    // declared parameters — e.g. a static method on a generic type
    // (`MyG<T,U>.foo()`). The flag sits above the 4-bit convention mask,
    // so the mask check alone lets it through; without it the compiled
    // body would run with no generic context at all — a silent
    // wrong-result, not a crash.
    if call_conv & ffi::CorInfoCallConv_CORINFO_CALLCONV_PARAMTYPE != 0 {
        return Err(CompileError::Unsupported(
            "generic methods (shared code needs the hidden context argument)",
        ));
    }
    if call_conv & ffi::CorInfoCallConv_CORINFO_CALLCONV_MASK
        != ffi::CorInfoCallConv_CORINFO_CALLCONV_DEFAULT
    {
        return Err(CompileError::Unsupported("non-default calling convention"));
    }
    Ok(())
}

/// Walks a signature's argument list, bounded by `numArgs` (see the module
/// docs: `getArgNext` is not an end-of-list signal on the real EE). The
/// value-class handle `getArgType` reports for struct arguments is
/// captured (step_10.9) and its layout queried into `layouts`.
fn sig_arg_types(
    sig: &ffi::CORINFO_SIG_INFO,
    ee: &dyn EeInfo,
    layouts: &mut StructLayouts,
) -> CompileResult<Vec<Type>> {
    let mut types = Vec::with_capacity(sig.numArgs() as usize);
    let mut cursor = ArgListHandle::from_raw(sig.args);
    for _ in 0..sig.numArgs() {
        let Some(arg) = cursor else {
            return Err(CompileError::BadIl("sig arg list shorter than numArgs"));
        };
        let (ty, value_class) = ee.get_arg_type(sig, arg);
        types.push(sig_elem_type(Some(ty), value_class, ee, layouts)?);
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
    /// `ldtoken` — push the RuntimeHandle struct for a metadata token
    /// (type, method, or field); the raw EE handle converts through the
    /// TYPEHANDLE_TO_* helper family (step_10.10).
    LdToken(u32),
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
    /// `ldsfld` — static field load; the EE's `getFieldInfo` answers the
    /// field's final address directly (step_10.7).
    LdSFld(u32),
    /// `ldsflda` — static field address, a `ByRef` value.
    LdSFldA(u32),
    /// `stsfld` — static field store; a reference-typed field stores
    /// through the checked-write-barrier helper, like `stfld`.
    StSFld(u32),
    /// `cpobj` — struct copy between two addresses (type token).
    CpObj(u32),
    /// `ldobj` — struct load through an address (type token).
    LdObj(u32),
    /// `stobj` — struct store through an address (type token).
    StObj(u32),
    /// `initobj` — zero-init a value-type slot (type token).
    InitObj(u32),
    /// `sizeof` — the type's unmanaged size, a JIT-time constant fold of
    /// the EE's `getClassSize` (step_10.10).
    SizeOf(u32),
    /// `castclass` — throwing cast through the EE's casting helper (the
    /// helper raises `InvalidCastException`; type token).
    CastClass(u32),
    /// `isinst` — null-producing cast through the EE's casting helper.
    IsInst(u32),
    /// `unbox` — boxed value to a byref to its payload (EE `UNBOX`
    /// helper; type token).
    Unbox(u32),
    /// `box` — value to heap object through the EE's `BOX` helper (type
    /// token); a no-op on a non-value class.
    Box(u32),
    /// `unbox.any` — boxed value to the value itself: `unbox` + `ldobj`
    /// for a value class, `castclass` for anything else (type token).
    UnboxAny(u32),
    /// `newarr` — 1-D zero-based array allocation through the EE's
    /// `getNewArrHelper` (array class token; step_10.8).
    NewArr(u32),
    /// `ldlen` — the array's element count (step_10.8).
    LdLen,
    /// `ldelema` — the element's address, an interior `ByRef` (element
    /// class token; step_10.8).
    LdElemA(u32),
    /// `ldelem.*` — the fixed element kinds (0x90..=0x9A), each carrying
    /// its (stack type, cell shape, element size) from
    /// [`LDELEM_FIXED_KINDS`] (step_10.8).
    LdElemK(Type, MemAccess, u32),
    /// `ldelem` — the element load with the element class token (0xA3).
    LdElem(u32),
    /// `stelem.*` — the fixed element kinds (0x9B..=0xA2), from
    /// [`STELEM_FIXED_KINDS`] (step_10.8).
    StElemK(Type, MemAccess, u32),
    /// `stelem` — the element store with the element class token (0xA4).
    StElem(u32),
    /// `throw` — raise the stack-top exception reference.
    Throw,
    /// `leave`/`leave.s` — exit the enclosing protected region(s) for
    /// `target` (absolute IL offset), emptying the evaluation stack.
    Leave {
        target: u32,
    },
    /// `endfinally` — return from a finally funclet.
    EndFinally,
    Ret,
}

/// The `conv.*` opcodes of the scalar-cheap pack plus the float pack's
/// `conv.r4`/`conv.r8`. `I1`/`I2`/`U1`/`U2` carry the truncation width the
/// IR's type vocabulary cannot (eval-stack types normalize at Int32,
/// ECMA-335 §III.1.1.1) — the importer expands them to shift pairs
/// ([`BlockImport::conv_narrow`]); `U1`/`U2` are the zero-extending forms
/// (step_10.7's rider).
#[derive(Copy, Clone)]
enum ConvKind {
    I1,
    I2,
    I4,
    I8,
    U1,
    U2,
    U4,
    U8,
    /// `conv.i` — to native int: sign-extension from Int32, the identity
    /// on a 64-bit operand (step_10.11; integer sources only).
    I,
    /// `conv.u` — to native uint: zero-extension from Int32, the identity
    /// on a 64-bit operand.
    U,
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

/// The fixed `ldelem.*` element kinds in opcode order (0x90..=0x9A: i1,
/// u1, i2, u2, i4, u4, i8, i, r4, r8, ref) as (stack type, cell shape,
/// element size). Sub-Int32 elements normalize to Int32 on the stack but
/// keep their cell width (the field rule, [`corinfo_mem_type`]).
const LDELEM_FIXED_KINDS: [(Type, MemAccess, u32); 11] = [
    (Type::Int32, MemAccess::I8, 1),
    (Type::Int32, MemAccess::U8, 1),
    (Type::Int32, MemAccess::I16, 2),
    (Type::Int32, MemAccess::U16, 2),
    (Type::Int32, MemAccess::Natural, 4),     // i4
    (Type::Int32, MemAccess::Natural, 4),     // u4
    (Type::Int64, MemAccess::Natural, 8),     // i8
    (Type::NativeInt, MemAccess::Natural, 8), // i
    (Type::Float, MemAccess::Natural, 4),     // r4
    (Type::Double, MemAccess::Natural, 8),    // r8
    (Type::Ref, MemAccess::Natural, 8),       // ref
];

/// The fixed `stelem.*` element kinds in opcode order (0x9B..=0xA2: i,
/// i1, i2, i4, i8, r4, r8, ref). Stores write only the low bytes, so the
/// signedness distinction of the load forms doesn't exist here.
const STELEM_FIXED_KINDS: [(Type, MemAccess, u32); 8] = [
    (Type::NativeInt, MemAccess::Natural, 8), // i
    (Type::Int32, MemAccess::I8, 1),          // i1
    (Type::Int32, MemAccess::I16, 2),         // i2
    (Type::Int32, MemAccess::Natural, 4),     // i4
    (Type::Int64, MemAccess::Natural, 8),     // i8
    (Type::Float, MemAccess::Natural, 4),     // r4
    (Type::Double, MemAccess::Natural, 8),    // r8
    (Type::Ref, MemAccess::Natural, 8),       // ref
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
            0x70 => Op::CpObj(r.u32()?),
            0x71 => Op::LdObj(r.u32()?),
            0x72 => Op::LdStr(r.u32()?),
            0x73 => Op::NewObj(r.u32()?),
            0x74 => Op::CastClass(r.u32()?),
            0x75 => Op::IsInst(r.u32()?),
            0x79 => Op::Unbox(r.u32()?),
            0x7A => Op::Throw,
            0x7B => Op::LdFld(r.u32()?),
            0x7C => Op::LdFldA(r.u32()?),
            0x7D => Op::StFld(r.u32()?),
            0x7E => Op::LdSFld(r.u32()?),
            0x7F => Op::LdSFldA(r.u32()?),
            0x80 => Op::StSFld(r.u32()?),
            0x81 => Op::StObj(r.u32()?),
            0x8C => Op::Box(r.u32()?),
            0x8D => Op::NewArr(r.u32()?),
            0x8E => Op::LdLen,
            0x8F => Op::LdElemA(r.u32()?),
            0x90..=0x9A => {
                let (ty, access, size) = LDELEM_FIXED_KINDS[usize::from(opcode - 0x90)];
                Op::LdElemK(ty, access, size)
            }
            0x9B..=0xA2 => {
                let (ty, access, size) = STELEM_FIXED_KINDS[usize::from(opcode - 0x9B)];
                Op::StElemK(ty, access, size)
            }
            0xA3 => Op::LdElem(r.u32()?),
            0xA4 => Op::StElem(r.u32()?),
            0xA5 => Op::UnboxAny(r.u32()?),
            0xD0 => Op::LdToken(r.u32()?),
            0xD1 => Op::Conv(ConvKind::U2),
            0xD2 => Op::Conv(ConvKind::U1),
            0xD3 => Op::Conv(ConvKind::I),
            0xDC => Op::EndFinally,
            0xDD => {
                let d = r.i32()?;
                Op::Leave {
                    target: branch_target(il.len(), r.ip, d)?,
                }
            }
            0xDE => {
                let d = r.i8()?;
                Op::Leave {
                    target: branch_target(il.len(), r.ip, i32::from(d))?,
                }
            }
            0xE0 => Op::Conv(ConvKind::U),
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
                0x11 => {
                    return Err(CompileError::Unsupported("endfilter (EH filter clauses)"));
                }
                0x15 => Op::InitObj(r.u32()?),
                0x1A => return Err(CompileError::Unsupported("rethrow")),
                0x1C => Op::SizeOf(r.u32()?),
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
/// transfer must be a branch target (i.e. reachable). EH region
/// boundaries (try/handler starts and ends) and `leave` targets are
/// block starts too (step_10.6).
fn find_leaders(il: &[u8], insns: &[Insn], clauses: &[Clause]) -> CompileResult<Vec<u32>> {
    let mut is_boundary = vec![false; il.len()];
    for insn in insns {
        is_boundary[insn.offset as usize] = true;
    }

    let mut leaders: BTreeSet<u32> = BTreeSet::from([0]);
    for clause in clauses {
        for boundary in [
            clause.try_start,
            clause.try_end,
            clause.handler_start,
            clause.handler_end,
        ] {
            // A region end at the very end of the method starts no block.
            if boundary as usize == il.len() {
                continue;
            }
            if !is_boundary[boundary as usize] {
                return Err(CompileError::BadIl(
                    "EH region boundary is not an instruction boundary",
                ));
            }
            leaders.insert(boundary);
        }
    }
    for insn in insns {
        match insn.op {
            Op::Br { target }
            | Op::BrZero { target, .. }
            | Op::BrCmp { target, .. }
            | Op::Leave { target } => {
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
        if matches!(
            insn.op,
            Op::Br { .. } | Op::Ret | Op::Throw | Op::Leave { .. } | Op::EndFinally
        ) {
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

/// One EH clause in IL space (step_10.6), translated from
/// `CORINFO_EH_CLAUSE`: unlike the native table (codegencommon.cpp
/// `genReportEH`), the IL form's lengths are LENGTHS, so the half-open
/// ranges here are computed as start + length.
struct Clause {
    kind: ClauseKind,
    try_start: u32,
    try_end: u32,
    handler_start: u32,
    handler_end: u32,
}

enum ClauseKind {
    /// A typed catch; the raw mdToken from the EE, passed through to the
    /// artifact (the VM resolves and type-tests it).
    Catch {
        class_token: u32,
    },
    Finally,
}

/// The method's EH clauses from `get_eh_info` (step_10.6). Only typed
/// catches and finallys are in scope; filters and faults are named
/// Unsupported.
fn fetch_clauses(info: &MethodInfo, ee: &dyn EeInfo) -> CompileResult<Vec<Clause>> {
    let mut clauses = Vec::with_capacity(info.eh_count as usize);
    for i in 0..info.eh_count {
        let raw = ee.get_eh_info(info.ftn, i);
        let flags = raw.Flags;
        if flags & ffi::CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_FILTER != 0 {
            return Err(CompileError::Unsupported("EH filter clauses"));
        }
        if flags & ffi::CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_FAULT != 0 {
            return Err(CompileError::Unsupported("EH fault clauses"));
        }
        // SAMETRY is a JIT→EE table flag the EE never reports here.
        let kind = match flags & !ffi::CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_SAMETRY {
            ffi::CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_NONE => ClauseKind::Catch {
                // SAFETY: a NONE clause's union member is ClassToken.
                class_token: unsafe { raw.__bindgen_anon_1.ClassToken },
            },
            ffi::CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_FINALLY => ClauseKind::Finally,
            _ => return Err(CompileError::BadIl("unknown EH clause flags")),
        };
        let bounds = |start: u32, len: u32| {
            start
                .checked_add(len)
                .ok_or(CompileError::BadIl("EH region outside the IL stream"))
        };
        clauses.push(Clause {
            kind,
            try_start: raw.TryOffset,
            try_end: bounds(raw.TryOffset, raw.TryLength)?,
            handler_start: raw.HandlerOffset,
            handler_end: bounds(raw.HandlerOffset, raw.HandlerLength)?,
        });
    }
    Ok(clauses)
}

/// The EH well-formedness the importer relies on (step_10.6): regions
/// are non-empty and in-bounds, a clause's handler is disjoint from its
/// own try, and every pair of try/handler ranges is disjoint, nested, or
/// — for try ranges only — equal (multiple catch clauses on one try).
/// Each try must also protect at least one instruction that no handler
/// covers (the layout rebuild maps a try to its *main* blocks).
fn validate_clauses(clauses: &[Clause], insns: &[Insn], il_len: u32) -> CompileResult<()> {
    for c in clauses {
        if c.try_start >= c.try_end || c.handler_start >= c.handler_end {
            return Err(CompileError::BadIl("empty EH region"));
        }
        if c.try_end > il_len || c.handler_end > il_len {
            return Err(CompileError::BadIl("EH region outside the IL stream"));
        }
        if c.handler_start < c.try_end && c.try_start < c.handler_end {
            return Err(CompileError::BadIl("EH handler overlaps its own try"));
        }
        let protects_something = insns.iter().any(|insn| {
            insn.offset >= c.try_start
                && insn.offset < c.try_end
                && !clauses
                    .iter()
                    .any(|h| insn.offset >= h.handler_start && insn.offset < h.handler_end)
        });
        if !protects_something {
            return Err(CompileError::BadIl(
                "EH try region protects no instructions",
            ));
        }
    }
    let mut regions = Vec::with_capacity(clauses.len() * 2);
    for c in clauses {
        regions.push((c.try_start, c.try_end, true));
        regions.push((c.handler_start, c.handler_end, false));
    }
    for (i, &(s1, e1, try1)) in regions.iter().enumerate() {
        for &(s2, e2, try2) in &regions[i + 1..] {
            let disjoint = e1 <= s2 || e2 <= s1;
            let equal = s1 == s2 && e1 == e2;
            let nested = (s1 >= s2 && e1 <= e2) || (s2 >= s1 && e2 <= e1);
            if disjoint || nested && (try1 && try2 || !equal) {
                continue;
            }
            return Err(CompileError::BadIl("improperly nested EH regions"));
        }
    }
    Ok(())
}

/// The clause whose HANDLER region is the innermost containing `offset`
/// (step_10.6): catch handler entries are entered with the exception on
/// the stack, and `leave` inside a catch handler is the funclet's
/// return-the-resume-address form.
fn innermost_handler(clauses: &[Clause], offset: u32) -> Option<usize> {
    clauses
        .iter()
        .enumerate()
        .filter(|(_, c)| offset >= c.handler_start && offset < c.handler_end)
        .min_by_key(|(_, c)| c.handler_end - c.handler_start)
        .map(|(i, _)| i)
}

/// The finally clauses a `leave` at `site` must invoke to reach `target`
/// (step_10.6): every finally clause whose try region contains the site
/// but not the target, innermost first.
fn finally_chain(clauses: &[Clause], site: u32, target: u32) -> Vec<usize> {
    let mut chain: Vec<usize> = clauses
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            matches!(c.kind, ClauseKind::Finally)
                && site >= c.try_start
                && site < c.try_end
                && !(target >= c.try_start && target < c.try_end)
        })
        .map(|(i, _)| i)
        .collect();
    // Region validation guarantees proper nesting, so span order is
    // nesting order.
    chain.sort_by_key(|&i| clauses[i].try_end - clauses[i].try_start);
    chain
}

/// A `leave` whose path crosses one or more finally handlers, recorded
/// at import for the layout rebuild (step_10.6).
struct LeaveChain {
    /// The pre-rebuild block index of the leave site.
    source: usize,
    /// The finally clauses to invoke, innermost first.
    hops: Vec<usize>,
    /// The leave's target IL offset (a leader).
    target: u32,
    /// A leave inside a catch handler keeps its `Leave` terminator —
    /// the funclet returns the resume address, which is the first step
    /// block — where a plain-body leave ends its block in a `Jump`.
    from_catch: bool,
}

/// One synthetic block of a leave chain: `hop < hops.len()` is a
/// `CallFinally` step for that hop's clause, `hop == hops.len()` is the
/// chain's final `Leave { target }` block. Spliced immediately after the
/// last main block of the try the hop exits, so its native code lands
/// outside that try but inside the enclosing region (clr-abi.md
/// §Invoking Finallys: the call's return address must not be in the try
/// being exited).
struct Step {
    /// The pre-rebuild main-area block this step splices after.
    anchor: usize,
    /// Span of the try range the hop exits (ordering: innermost first).
    exit_span: u32,
    chain: usize,
    hop: usize,
}

/// Rebuilds the block list for an EH method (step_10.6): main-area
/// blocks in IL order (a block is main-area iff its IL range lies
/// outside every handler region) with the leave chains' step blocks
/// spliced after their anchor blocks, then the handler blocks grouped
/// per clause at the tail (groups in handler IL order, IL order within a
/// group). BlockIds are renumbered to layout order, every terminator
/// target is remapped, and the `eh_regions` table is computed over the
/// new order.
fn rebuild_blocks(
    mut blocks: Vec<hir::Block>,
    ranges: &[(u32, u32)],
    clauses: &[Clause],
    chains: &[LeaveChain],
    block_of: &HashMap<u32, u32>,
) -> CompileResult<(Vec<hir::Block>, Vec<hir::EhRegion>)> {
    let n = blocks.len();
    // EH region boundaries are leaders, so a block lies entirely inside
    // or outside each handler range.
    let handler_of: Vec<Option<usize>> = ranges
        .iter()
        .map(|&(start, _)| {
            clauses
                .iter()
                .position(|c| start >= c.handler_start && start < c.handler_end)
        })
        .collect();
    let is_main = |i: usize| handler_of[i].is_none();
    let in_try = |i: usize, c: &Clause| ranges[i].0 >= c.try_start && ranges[i].0 < c.try_end;
    // The anchor a step block splices after: the last main block of the
    // try being exited. Validation guarantees the try protects at least
    // one non-handler instruction, so the block exists.
    let anchor_of = |c: &Clause| -> CompileResult<usize> {
        (0..n)
            .rfind(|&i| is_main(i) && in_try(i, c))
            .ok_or(CompileError::Internal("EH try region has no main blocks"))
    };

    let mut steps: Vec<Step> = Vec::new();
    for (ch, chain) in chains.iter().enumerate() {
        for (hop, &clause) in chain.hops.iter().enumerate() {
            steps.push(Step {
                anchor: anchor_of(&clauses[clause])?,
                exit_span: clauses[clause].try_end - clauses[clause].try_start,
                chain: ch,
                hop,
            });
        }
        // The final Leave block rides on the last hop's anchor.
        let &last = chain
            .hops
            .last()
            .ok_or(CompileError::Internal("leave chain with no hops"))?;
        steps.push(Step {
            anchor: anchor_of(&clauses[last])?,
            exit_span: clauses[last].try_end - clauses[last].try_start,
            chain: ch,
            hop: chain.hops.len(),
        });
    }

    // The new order: mains, with each anchor's steps right after it
    // (innermost-exiting first), then the handler groups.
    enum Item {
        Orig(usize),
        Step(usize, usize),
    }
    let mut order: Vec<Item> = Vec::with_capacity(n + steps.len());
    for i in 0..n {
        if !is_main(i) {
            continue;
        }
        order.push(Item::Orig(i));
        let mut here: Vec<&Step> = steps.iter().filter(|s| s.anchor == i).collect();
        here.sort_by_key(|s| (s.exit_span, s.chain, s.hop));
        order.extend(here.iter().map(|s| Item::Step(s.chain, s.hop)));
    }
    let mut clause_order: Vec<usize> = (0..clauses.len()).collect();
    clause_order.sort_by_key(|&c| clauses[c].handler_start);
    for &c in &clause_order {
        order.extend((0..n).filter(|&i| handler_of[i] == Some(c)).map(Item::Orig));
    }

    // Renumber: originals and steps get their layout position as the id.
    let mut new_id = vec![0u32; n];
    let mut step_id: HashMap<(usize, usize), u32> = HashMap::new();
    for (pos, item) in order.iter().enumerate() {
        match *item {
            Item::Orig(i) => new_id[i] = pos as u32,
            Item::Step(ch, hop) => {
                step_id.insert((ch, hop), pos as u32);
            }
        }
    }
    let remap = |id: &mut BlockId, new_id: &[u32]| *id = BlockId(new_id[id.0 as usize]);

    let mut out: Vec<hir::Block> = Vec::with_capacity(order.len());
    let mut blocks: Vec<Option<hir::Block>> = blocks.drain(..).map(Some).collect();
    for (pos, item) in order.iter().enumerate() {
        match *item {
            Item::Orig(i) => {
                let mut block = blocks[i].take().expect("one slot per block");
                block.id = BlockId(pos as u32);
                match &mut block.terminator {
                    hir::Terminator::Jump { target } | hir::Terminator::Leave { target } => {
                        remap(target, &new_id)
                    }
                    hir::Terminator::Branch { then, else_, .. } => {
                        remap(then, &new_id);
                        remap(else_, &new_id);
                    }
                    hir::Terminator::Switch {
                        targets, default, ..
                    } => {
                        for target in targets {
                            remap(target, &new_id);
                        }
                        remap(default, &new_id);
                    }
                    hir::Terminator::Return { .. }
                    | hir::Terminator::Throw { .. }
                    | hir::Terminator::EndFinally => {}
                    hir::Terminator::CallFinally { .. } => {
                        return Err(CompileError::Internal(
                            "CallFinally on an imported (non-step) block",
                        ));
                    }
                }
                out.push(block);
            }
            Item::Step(ch, hop) => {
                let chain = &chains[ch];
                let terminator = if hop < chain.hops.len() {
                    let handler_entry = block_of
                        .get(&clauses[chain.hops[hop]].handler_start)
                        .ok_or(CompileError::Internal("handler entry is not a block start"))?;
                    hir::Terminator::CallFinally {
                        funclet: BlockId(new_id[*handler_entry as usize]),
                        continuation: BlockId(step_id[&(ch, hop + 1)]),
                    }
                } else {
                    let target = block_of
                        .get(&chain.target)
                        .ok_or(CompileError::Internal("leave target is not a block start"))?;
                    hir::Terminator::Leave {
                        target: BlockId(new_id[*target as usize]),
                    }
                };
                out.push(hir::Block {
                    id: BlockId(pos as u32),
                    stmts: Vec::new(),
                    terminator,
                });
            }
        }
    }
    // The chain source blocks: a plain-body leave ends Jump(first step);
    // a leave from a catch handler keeps the funclet-returning Leave
    // form, aimed at the first step block (the VM resumes there).
    for (ch, chain) in chains.iter().enumerate() {
        let first_step = BlockId(step_id[&(ch, 0)]);
        let block = &mut out[new_id[chain.source] as usize];
        block.terminator = if chain.from_catch {
            hir::Terminator::Leave { target: first_step }
        } else {
            hir::Terminator::Jump { target: first_step }
        };
    }

    // The region table over the new layout. A try range maps to its run
    // of main blocks — a nested handler's IL moved to the tail — extended
    // over any step blocks trailing its last main block whose hop exits a
    // STRICTLY INNER try (the coincident-try-end case: the call exiting
    // the inner try must still sit inside this region).
    let mut regions = Vec::with_capacity(clauses.len());
    for (ci, c) in clauses.iter().enumerate() {
        let try_start = (0..n)
            .filter(|&i| is_main(i) && in_try(i, c))
            .map(|i| new_id[i])
            .min()
            .ok_or(CompileError::Internal("EH try region has no main blocks"))?;
        let mut try_end = (0..n)
            .filter(|&i| is_main(i) && in_try(i, c))
            .map(|i| new_id[i])
            .max()
            .expect("non-empty above")
            + 1;
        while let Some(Item::Step(ch, hop)) = order.get(try_end as usize) {
            let chain = &chains[*ch];
            let exit_clause = chain.hops[(*hop).min(chain.hops.len() - 1)];
            let exit_try = &clauses[exit_clause];
            let strictly_inner = exit_clause != ci
                && exit_try.try_start >= c.try_start
                && exit_try.try_end <= c.try_end
                && (exit_try.try_start > c.try_start || exit_try.try_end < c.try_end);
            if !strictly_inner {
                break;
            }
            try_end += 1;
        }
        let handler_start = (0..n)
            .filter(|&i| handler_of[i] == Some(ci))
            .map(|i| new_id[i])
            .min()
            .ok_or(CompileError::Internal("EH handler region has no blocks"))?;
        let handler_end = (0..n)
            .filter(|&i| handler_of[i] == Some(ci))
            .map(|i| new_id[i])
            .max()
            .expect("non-empty above")
            + 1;
        regions.push(hir::EhRegion {
            kind: match clauses[ci].kind {
                ClauseKind::Catch { class_token } => hir::EhRegionKind::Catch { class_token },
                ClauseKind::Finally => hir::EhRegionKind::Finally,
            },
            try_start: BlockId(try_start),
            try_end: BlockId(try_end),
            handler_start: BlockId(handler_start),
            handler_end: BlockId(handler_end),
        });
    }
    Ok((out, regions))
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
    /// The hidden return buffer arg-local (a ByRef immediately after
    /// `this`), present when the method's own return type is a
    /// non-register-passed struct (step_10.9).
    retbuf: Option<LocalId>,
    /// Layout facts of every value class mentioned, queried once per class.
    struct_layouts: StructLayouts,
    /// Leader offset → block index in layout order.
    block_of: HashMap<u32, u32>,
    /// Stack depth each block entry requires, as told by its predecessors.
    expected_depth: HashMap<u32, usize>,
    stack: Vec<(Type, hir::Expr)>,
    /// The method's EH clauses in IL space (step_10.6); empty for the
    /// fib subset.
    clauses: Vec<Clause>,
    /// Handler-entry offsets of the catch clauses: those blocks are
    /// entered with the exception object on the eval stack.
    catch_entries: BTreeSet<u32>,
    /// `leave`s whose path crosses finally handlers, for the layout
    /// rebuild (step_10.6).
    chains: Vec<LeaveChain>,
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
        hir::Expr::Const(_) | hir::Expr::StaticFieldAddr { .. } | hir::Expr::CatchArg => false,
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
        hir::Expr::StaticFieldAddr { .. } | hir::Expr::CatchArg => false,
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

    /// Maps an IL argument index (`ldarg`/`ldarga`/`starg`) to its local
    /// slot. The hidden retbuf arg-local (step_10.9) sits between `this`
    /// and the user arguments in the flat namespace but is NOT an IL
    /// argument, so user-arg indices shift past it.
    fn il_arg_id(&self, index: u32) -> CompileResult<LocalId> {
        let this_count =
            u32::from(self.info.args.callConv & ffi::CorInfoCallConv_CORINFO_CALLCONV_HASTHIS != 0);
        let retbuf_count = u32::from(self.retbuf.is_some());
        if index >= self.num_args - retbuf_count {
            return Err(CompileError::BadIl("argument index out of range"));
        }
        Ok(LocalId(
            index + if index >= this_count { retbuf_count } else { 0 },
        ))
    }

    /// A fresh importer temp, after the IL locals in the flat namespace.
    fn temp(&mut self, ty: Type) -> LocalId {
        let id = LocalId(self.local_types.len() as u32);
        self.local_types.push(ty);
        id
    }

    /// The expression a read of local `id` yields: a struct-typed slot
    /// reads as its address (`StructVal` — struct values are memory-backed,
    /// step_10.9), everything else is the plain local read.
    fn local_value_expr(&self, id: LocalId) -> (Type, hir::Expr) {
        let ty = self.local_types[id.0 as usize];
        let expr = match ty {
            Type::Struct(class) => hir::Expr::StructVal {
                addr: Box::new(hir::Expr::LocalAddr(id)),
                class,
            },
            _ => hir::Expr::Local(id),
        };
        (ty, expr)
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
                // Struct values are addresses: the tree re-reads the spill
                // temp through `StructVal`, never a bare `Local`.
                if let Type::Struct(class) = entry_ty {
                    self.stack[i].1 = hir::Expr::StructVal {
                        addr: Box::new(hir::Expr::LocalAddr(tmp)),
                        class,
                    };
                }
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
                // A duplicated struct value re-reads the temp through
                // `StructVal` (struct values are addresses, step_10.9).
                let (ty, expr) = self.local_value_expr(tmp);
                self.push(ty, expr)?;
                let (_, expr) = self.local_value_expr(tmp);
                self.push(ty, expr)?;
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
            // The discarded tree's statement runs before anything later;
            // pending trees below it read their memory first (the
            // statement-before-tree rule — see `spill_stack`).
            self.spill_stack(stmts, il_offset)?;
            stmts.push(hir::Stmt {
                il_offset,
                kind: hir::StmtKind::Eval(value),
            });
        }
        Ok(())
    }

    /// Spills every pending evaluation-stack tree into a fresh temp, in
    /// stack order (bottom first), replacing each with a read of its temp.
    /// Mandatory before pushing a statement while trees are pending: the
    /// statement executes before the terminator/consumer's tree
    /// evaluation, but IL semantics produced those values *before* the
    /// side effect — e.g. `ceq …; initobj x; brfalse` must compare the
    /// pre-initobj memory (found by Runtime_62524's silent wrong answer).
    fn spill_stack(
        &mut self,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        for i in 0..self.stack.len() {
            let (ty, _) = &self.stack[i];
            let ty = *ty;
            let value = std::mem::replace(&mut self.stack[i].1, hir::Expr::Const(Const::Int32(0)));
            // Constants and address-of-local trees can't observe the side
            // effect (an address's identity doesn't change); everything
            // else — reads of locals, memory, call results — must be
            // evaluated before it.
            if matches!(value, hir::Expr::LocalAddr(_) | hir::Expr::Const(_)) {
                self.stack[i].1 = value;
                continue;
            }
            let tmp = self.temp(ty);
            stmts.push(hir::Stmt {
                il_offset,
                kind: hir::StmtKind::Store { dst: tmp, value },
            });
            let (_, expr) = self.local_value_expr(tmp);
            self.stack[i].1 = expr;
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
    /// `conv.i1`/`conv.i2`/`conv.u1`/`conv.u2` expand to shift pairs
    /// (`conv_narrow`); the rest
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
            ConvKind::I1 => self.conv_narrow(8, false, ty, value),
            ConvKind::I2 => self.conv_narrow(16, false, ty, value),
            ConvKind::U1 => self.conv_narrow(8, true, ty, value),
            ConvKind::U2 => self.conv_narrow(16, true, ty, value),
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
            // operand). From a float, conv.u8 is the saturating
            // unsigned conversion — the `unsigned` flag rides the Conv
            // node and the HIR→LIR lowering expands the sequence
            // (step_10.11; `decisions/` entry for the choice).
            ConvKind::I8 | ConvKind::U8 => {
                if ty == Type::Int64 || ty == Type::NativeInt {
                    // A 64-bit operand is the identity; the IL stack type
                    // still becomes int64 (a NativeInt source re-types).
                    self.push(Type::Int64, value)
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
            // conv.i: to native int (step_10.11) — sign-extend from
            // Int32, the identity on a 64-bit operand. From a float the
            // saturating signed conversion is the still-open 10.2 follow-
            // up (see the conv_narrow comment), so it stays unsupported.
            ConvKind::I => {
                if fp {
                    return Err(CompileError::Unsupported("conv.i from a float operand"));
                }
                if ty == Type::Int64 || ty == Type::NativeInt {
                    self.push(Type::NativeInt, value)
                } else {
                    self.push(
                        Type::NativeInt,
                        hir::Expr::Conv {
                            to: Type::NativeInt,
                            overflow: false,
                            unsigned: false,
                            arg: Box::new(value),
                        },
                    )
                }
            }
            // conv.u: to native uint (step_10.7's rider) — zero-extend
            // from Int32, the identity on a 64-bit operand. From a float
            // it is the same saturating unsigned 64-bit truncation as
            // conv.u8 (step_10.11).
            ConvKind::U => {
                if ty == Type::Int64 || ty == Type::NativeInt {
                    self.push(Type::NativeInt, value)
                } else {
                    self.push(
                        Type::NativeInt,
                        hir::Expr::Conv {
                            to: Type::NativeInt,
                            overflow: false,
                            unsigned: true,
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

    /// `conv.i1`/`conv.i2`/`conv.u1`/`conv.u2`: truncate to `bits` then
    /// extend, as the shift pair `(v << (32 - bits)) >> (32 - bits)` at 32
    /// bits — an arithmetic `shr` replicates the sign bit (`conv.i*`),
    /// a logical `shr.un` zero-fills (`conv.u*`). The IR's
    /// type vocabulary normalizes sub-Int32 types away (ECMA-335
    /// §III.1.1.1), so the narrowing cannot be a `Conv` node; the shift
    /// expansion is exact. A non-Int32 operand converts to Int32 first —
    /// for a float source that is the truncating `cvtt*` conversion,
    /// after which the low `bits` behave as for integers.
    ///
    /// Float source, unsigned narrow (step_10.11): RyuJIT's .NET 9+
    /// semantics saturate to the small type's range (measured against the
    /// reference JIT: NaN/negative → 0, above the max → the max), so the
    /// operand is clamped in the float domain first — `maxs(v, 0)` maps
    /// negatives *and* NaN to +0 (the second operand wins on NaN), then
    /// `mins(…, MAX)`. The signed narrows from a float keep the plain
    /// `cvtt*` pre-conversion: matching RyuJIT's signed clamp is the
    /// still-open 10.2 follow-up (`convfloat` mismatch).
    fn conv_narrow(
        &mut self,
        bits: u32,
        unsigned: bool,
        ty: Type,
        value: hir::Expr,
    ) -> CompileResult<()> {
        let fp = matches!(ty, Type::Float | Type::Double);
        let value = if fp && unsigned {
            let max = ((1u64 << bits) - 1) as f64;
            let (zero, limit) = if ty == Type::Float {
                (Const::Float(0.0), Const::Float(max as f32))
            } else {
                (Const::Double(0.0), Const::Double(max))
            };
            let above_zero = binary(BinaryOp::MaxF, value, hir::Expr::Const(zero));
            binary(BinaryOp::MinF, above_zero, hir::Expr::Const(limit))
        } else {
            value
        };
        let value = if ty == Type::Int32 {
            value
        } else {
            // A wider operand narrows to 32 bits first; the low `bits`
            // survive either way. A clamped float converts exactly.
            hir::Expr::Conv {
                to: Type::Int32,
                overflow: false,
                unsigned: false,
                arg: Box::new(value),
            }
        };
        let sh = hir::Expr::Const(Const::Int32((32 - bits) as i32));
        let shifted = binary(BinaryOp::Shl, value, sh);
        let back_shift = if unsigned {
            BinaryOp::UShr
        } else {
            BinaryOp::Shr
        };
        let back = binary(
            back_shift,
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

    fn ret(
        &mut self,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<hir::Terminator> {
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
        if let Some(retbuf) = self.retbuf {
            // The hidden-return-buffer convention (step_10.9): the struct
            // value copies through the retbuf pointer, and the callee
            // returns the buffer address in rax (clr-abi.md). The
            // destination is the caller's frame, so no GC barrier is
            // needed for the copy.
            stmts.push(hir::Stmt {
                il_offset,
                kind: hir::StmtKind::StoreInd {
                    addr: hir::Expr::Local(retbuf),
                    offset: 0,
                    value,
                    access: MemAccess::Natural,
                },
            });
            return Ok(hir::Terminator::Return {
                value: Some(hir::Expr::Local(retbuf)),
            });
        }
        Ok(hir::Terminator::Return { value: Some(value) })
    }

    /// `throw` (0x7A, step_10.6): the pending stack trees evaluate in IL
    /// order before the throw (the spill), then the exception pops — it
    /// must be a reference.
    fn throw(
        &mut self,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<hir::Terminator> {
        self.spill_stack(stmts, il_offset)?;
        let (ty, exception) = self.pop()?;
        if ty != Type::Ref {
            return Err(CompileError::BadIl("throw operand must be a reference"));
        }
        Ok(hir::Terminator::Throw { exception })
    }

    /// `leave`/`leave.s` (step_10.6): the evaluation stack empties
    /// (ECMA-335 §III.2.38) — pending trees still evaluate first, for
    /// their side effects. A leave whose path crosses finally handlers
    /// records a chain for the layout rebuild (which splices in the
    /// `CallFinally` step blocks); anything else is the plain `Leave`
    /// terminator — inside a catch handler, the funclet's
    /// return-the-resume-address form.
    fn leave(
        &mut self,
        b: usize,
        target: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<hir::Terminator> {
        self.spill_stack(stmts, il_offset)?;
        self.stack.clear();
        self.note_depth(target, 0)?;
        let hops = finally_chain(&self.clauses, il_offset.0, target);
        if !hops.is_empty() {
            let from_catch = matches!(
                innermost_handler(&self.clauses, il_offset.0),
                Some(c) if matches!(self.clauses[c].kind, ClauseKind::Catch { .. })
            );
            self.chains.push(LeaveChain {
                source: b,
                hops,
                target,
                from_catch,
            });
        }
        Ok(hir::Terminator::Leave {
            target: self.block_id(target)?,
        })
    }

    /// `endfinally` (0xDC, step_10.6): valid only inside a finally
    /// handler — the funclet's plain return. The stack resets, with
    /// pending trees evaluated first for their side effects.
    fn endfinally(
        &mut self,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<hir::Terminator> {
        match innermost_handler(&self.clauses, il_offset.0) {
            Some(c) if matches!(self.clauses[c].kind, ClauseKind::Finally) => {}
            _ => return Err(CompileError::BadIl("endfinally outside a finally handler")),
        }
        self.spill_stack(stmts, il_offset)?;
        self.stack.clear();
        Ok(hir::Terminator::EndFinally)
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

    /// `ldtoken` (0xD0, step_10.10): push the RuntimeTypeHandle /
    /// RuntimeMethodHandle / RuntimeFieldHandle struct for a metadata
    /// token. RyuJIT's CEE_LDTOKEN shape (importer.cpp:10616): resolve
    /// with the Ldtoken kind hint, embed the raw EE handle through
    /// `embedGenericHandle`, and convert it to the managed handle struct
    /// through the TYPEHANDLE_TO_*/METHODDESC_TO_*/FIELDDESC_TO_* helper
    /// (jithelpers.h:238-240 — CoreCLR's handle structs wrap a managed
    /// object, so the helper call is where the RuntimeType comes from),
    /// returned in `rax` like any one-eightbyte struct. A generic-context
    /// runtime lookup is the shared-generics step; an indirection cell
    /// needs load/reloc plumbing tier 0 doesn't have (the ldstr/newobj
    /// policy — the EE decides, we follow).
    fn ldtoken(&mut self, token: u32) -> CompileResult<()> {
        let mut resolved = zeroed_out(|t: &mut ffi::CORINFO_RESOLVED_TOKEN| {
            t.tokenContext = self.info.ftn.as_raw() as ffi::CORINFO_CONTEXT_HANDLE;
            t.tokenScope = self.info.args.scope;
            t.token = token;
            t.tokenType = ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Ldtoken;
        });
        self.ee.resolve_token(&mut resolved);
        // The handle struct's class — RyuJIT's `gtRetClsHnd`
        // (getTokenTypeAsHandle answers RuntimeTypeHandle /
        // RuntimeMethodHandle / RuntimeFieldHandle according to which of
        // the resolved handles is set, jitinterface.cpp:556).
        let Some(handle_class) = self.ee.get_token_type_as_handle(&resolved) else {
            return Err(CompileError::BadIl("ldtoken token did not resolve"));
        };
        let result = self
            .ee
            .embed_generic_handle(&mut resolved, false, self.info.ftn);
        if result.lookup.lookupKind.needsRuntimeLookup {
            return Err(CompileError::Unsupported(
                "ldtoken with a generic-context runtime lookup",
            ));
        }
        // !needsRuntimeLookup ⇒ the constLookup union member is live
        // (corinfo.h's CORINFO_LOOKUP contract).
        let const_lookup = unsafe { result.lookup.__bindgen_anon_1.constLookup };
        if const_lookup.accessType != ffi::InfoAccessType_IAT_VALUE {
            return Err(CompileError::Unsupported(
                "ldtoken handle through an indirection cell (IAT_PVALUE/PPVALUE)",
            ));
        }
        let handle = unsafe { const_lookup.__bindgen_anon_1.handle };

        // RyuJIT's impTokenToHandle mustRestoreHandle bookkeeping: record
        // the load dependency with the EE (for a field, its owning
        // class's). Notifications only — no codegen effect in-process.
        match result.handleType {
            ffi::CorInfoGenericHandleType_CORINFO_HANDLETYPE_CLASS => {
                if let Some(c) = ClassHandle::from_raw(handle as ffi::CORINFO_CLASS_HANDLE) {
                    self.ee.class_must_be_loaded_before_code_is_run(c);
                }
            }
            ffi::CorInfoGenericHandleType_CORINFO_HANDLETYPE_METHOD => {
                if let Some(m) = MethodHandle::from_raw(handle as ffi::CORINFO_METHOD_HANDLE) {
                    self.ee.method_must_be_loaded_before_code_is_run(m);
                }
            }
            ffi::CorInfoGenericHandleType_CORINFO_HANDLETYPE_FIELD => {
                if let Some(f) = FieldHandle::from_raw(handle as ffi::CORINFO_FIELD_HANDLE) {
                    self.ee
                        .class_must_be_loaded_before_code_is_run(self.ee.get_field_class(f));
                }
            }
            _ => {}
        }

        // The raw-handle → handle-struct conversion helper, by the
        // resolved handle kind (importer.cpp:10634-10641).
        let helper = if !resolved.hMethod.is_null() {
            CorInfoHelpFunc::METHODDESC_TO_STUBRUNTIMEMETHOD
        } else if !resolved.hField.is_null() {
            CorInfoHelpFunc::FIELDDESC_TO_STUBRUNTIMEFIELD
        } else if !resolved.hClass.is_null() {
            CorInfoHelpFunc::TYPEHANDLE_TO_RUNTIMETYPEHANDLE
        } else {
            return Err(CompileError::BadIl("ldtoken token did not resolve"));
        };
        layout_of(&mut self.struct_layouts, self.ee, handle_class)?;
        self.push(
            Type::Struct(handle_class),
            hir::Expr::Call {
                target: CallTarget::Helper(helper),
                sig: CallSig {
                    ret: Type::Struct(handle_class),
                    args: vec![Type::NativeInt],
                    has_this: false,
                },
                args: vec![hir::Expr::Const(Const::NativeInt(handle as isize))],
            },
        )
    }

    /// `sizeof` (0xFE 1C, step_10.10): the type's unmanaged size, folded
    /// to a constant at JIT time — the EE's `getClassSize` is the query
    /// (RyuJIT's CEE_SIZEOF, importer.cpp:10953: no value-type gate;
    /// CoreCLR answers for any type, and C# emits the opcode for unmanaged
    /// types only). The IL stack type is unsigned int32.
    fn sizeof_(&mut self, token: u32) -> CompileResult<()> {
        let (_resolved, class) =
            self.resolve_box_cast_class(token, ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Class)?;
        let size = self.ee.get_class_size(class);
        self.push(Type::Int32, hir::Expr::Const(Const::Int32(size as i32)))
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
        let ret = sig_elem_type(
            CorInfoType::from_raw(call.sig.retType()),
            ClassHandle::from_raw(call.sig.retTypeClass),
            self.ee,
            &mut self.struct_layouts,
        )?;
        let mut arg_types = sig_arg_types(&call.sig, self.ee, &mut self.struct_layouts)?;

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
        if let Type::Struct(class) = ret {
            if !self.struct_layouts[&class].sysv.passed_in_registers {
                // The hidden return buffer (step_10.9): a call returning a
                // non-register-passed struct takes the address of a fresh
                // struct temp as an implicit argument immediately after
                // `this` (the managed convention — NOT the PInvoke
                // first-arg convention). The call runs for its retbuf
                // write; the value is the temp.
                self.spill_stack(stmts, il_offset)?;
                let t = self.temp(ret);
                args.insert(usize::from(has_this), hir::Expr::LocalAddr(t));
                arg_types.insert(0, Type::ByRef);
                stmts.push(hir::Stmt {
                    il_offset,
                    kind: hir::StmtKind::Eval(hir::Expr::Call {
                        target: CallTarget::Direct(method),
                        sig: CallSig {
                            ret,
                            args: arg_types,
                            has_this,
                        },
                        args,
                    }),
                });
                return self.push(
                    ret,
                    hir::Expr::StructVal {
                        addr: Box::new(hir::Expr::LocalAddr(t)),
                        class,
                    },
                );
            }
        }
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
            self.spill_stack(stmts, il_offset)?;
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
    /// unresolved token is BadIl, and static fields stay out of the
    /// instance path (`ldsfld`/`ldsflda`/`stsfld` have their own,
    /// step_10.7).
    /// Returns the field handle, the EE-supplied instance offset, and
    /// whether the field's declaring class is a value class (step_10.9:
    /// such fields accept a ByRef receiver).
    fn resolve_instance_field(&mut self, token: u32) -> CompileResult<(FieldHandle, u32, bool)> {
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
        let declaring_is_value_class =
            ClassHandle::from_raw(resolved.hClass).is_some_and(|c| self.ee.is_value_class(c));
        Ok((
            field,
            self.ee.get_field_offset(field),
            declaring_is_value_class,
        ))
    }

    /// The IR type and memory-access shape of a field (the step_10.4
    /// object pack plus value-class fields, step_10.9, plus the float and
    /// sub-Int32 field widths, step_10.10): the full-width integers,
    /// native ints/pointers, floats, object references, and structs
    /// (whose layout is queried into the side table). The sub-Int32
    /// metadata types are `Int32` on the stack (ECMA-335 §III.1.1.1) but
    /// keep their 1-/2-byte cell — [`MemAccess`] carries that shape.
    fn field_mem_type(&mut self, field: FieldHandle) -> CompileResult<(Type, MemAccess)> {
        let (ty, value_class) = self.ee.get_field_type(field);
        if let CorInfoType::ValueClass = ty {
            let ty = sig_elem_type(Some(ty), value_class, self.ee, &mut self.struct_layouts)?;
            return Ok((ty, MemAccess::Natural));
        }
        corinfo_mem_type(ty)
    }

    /// The receiver of a field access: a class reference (null-checked at
    /// the access), or — when the field's declaring class is a value class
    /// (`byref_ok`) — a byref into the struct OR the struct value itself
    /// (csc emits `ldarg`/`ldloc` + `ldfld` for value reads: the value's
    /// home is the receiver's memory). Byref/value receivers are never
    /// null-checked: byrefs are managed pointers and a value is never
    /// null. Returns the receiver (an address expression for the
    /// value/byref forms) and whether it needs the explicit null check.
    fn pop_field_receiver(
        &mut self,
        byref_ok: bool,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<(hir::Expr, bool)> {
        let (ty, obj) = self.pop()?;
        match ty {
            Type::Ref => Ok((obj, true)),
            Type::ByRef if byref_ok => Ok((obj, false)),
            Type::Struct(class) if byref_ok => {
                let addr = self.struct_addr_of(obj, class, stmts, il_offset);
                Ok((addr, false))
            }
            _ => Err(CompileError::Unsupported(
                "field access on a non-class receiver (value types)",
            )),
        }
    }

    /// `ldfld` (0x7B): the load's address is the null-checked receiver —
    /// the null check is explicit and trap-based (RyuJIT's model: a load
    /// through the pointer, the hardware fault translated by the EE; the
    /// offset-folding optimization is deliberately not tier 0's). A
    /// struct-typed field (step_10.9) loads as its address: the value is a
    /// `StructVal` of the field address.
    fn ldfld(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let (field, offset, byref_ok) = self.resolve_instance_field(token)?;
        let (ty, access) = self.field_mem_type(field)?;
        let (obj, null_check) = self.pop_field_receiver(byref_ok, stmts, il_offset)?;
        let obj = if null_check {
            hir::Expr::NullCheck { arg: Box::new(obj) }
        } else {
            obj
        };
        self.spill_stack(stmts, il_offset)?;
        if let Type::Struct(class) = ty {
            return self.push(
                ty,
                hir::Expr::StructVal {
                    addr: Box::new(hir::Expr::FieldAddr {
                        obj: Box::new(obj),
                        field,
                        offset,
                    }),
                    class,
                },
            );
        }
        self.push(
            ty,
            hir::Expr::Load {
                addr: Box::new(obj),
                offset,
                ty,
                access,
            },
        )
    }

    /// `ldflda` (0x7C): the field's address — type-agnostic (an address
    /// carries no field type), so the field-type gate does not apply.
    fn ldflda(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let (field, offset, byref_ok) = self.resolve_instance_field(token)?;
        let (obj, null_check) = self.pop_field_receiver(byref_ok, stmts, il_offset)?;
        let obj = if null_check {
            hir::Expr::NullCheck { arg: Box::new(obj) }
        } else {
            obj
        };
        self.push(
            Type::ByRef,
            hir::Expr::FieldAddr {
                obj: Box::new(obj),
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
    ///
    /// A struct-typed field (step_10.9) stores as a block copy; when the
    /// struct embeds GC pointers the copy goes through
    /// `CORINFO_HELP_BULK_WRITEBARRIER` (RyuJIT's own heap answer — it
    /// applies the barrier per cell when the destination is in the heap
    /// and is a plain copy otherwise).
    fn stfld(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let (field, offset, byref_ok) = self.resolve_instance_field(token)?;
        let (ty, access) = self.field_mem_type(field)?;
        let (vt, value) = self.pop()?;
        if vt != ty {
            return Err(CompileError::BadIl("stfld value type mismatch"));
        }
        let (obj, null_check) = self.pop_field_receiver(byref_ok, stmts, il_offset)?;
        let obj = if null_check {
            hir::Expr::NullCheck { arg: Box::new(obj) }
        } else {
            obj
        };
        self.spill_stack(stmts, il_offset)?;
        if let Type::Struct(class) = ty {
            let addr = hir::Expr::FieldAddr {
                obj: Box::new(obj),
                field,
                offset,
            };
            return self.store_struct_through(addr, class, value, stmts, il_offset);
        }
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
                access,
            }
        };
        stmts.push(hir::Stmt { il_offset, kind });
        Ok(())
    }

    /// Static-field resolution shared by `ldsfld`/`ldsflda`/`stsfld`
    /// (step_10.7): one `getFieldInfo` query per opcode (the
    /// `CORINFO_ACCESS_GET`/`SET`/`ADDRESS` flag is the only difference —
    /// `flags`, corinfo.h:622). For a plain static the EE answers
    /// `CORINFO_FIELD_STATIC_ADDRESS` (RVA statics:
    /// `..._STATIC_RVA_ADDRESS`, the same shape) with `IAT_VALUE`, and
    /// `fieldLookup.addr` IS the field's final address — the offset into
    /// the statics block is already baked in (jitinterface.cpp:1492
    /// `GetStaticAddressHandle`), embedded here as a raw `NativeInt`
    /// constant with no layout math (the newobj MethodTable* policy).
    /// A value-class static the EE boxes (`STATIC_IN_HEAP`) instead
    /// answers the address of the cell holding the frozen box object —
    /// the field data is one indirection plus the object header away
    /// (importer.cpp:4417). Thread statics, shared-generic/collectible
    /// helper accessors, indirection cells, and access callouts are all
    /// named `Unsupported` (the gates RyuJIT's `CORINFO_FLG_FIELD_STATIC`
    /// check and R2R paths also take).
    ///
    /// Returns the field, the field's address expression, and whether the
    /// class-init trigger fired (see [`BlockImport::maybe_init_class`]).
    fn resolve_static_field(
        &mut self,
        token: u32,
        flags: u32,
    ) -> CompileResult<(FieldHandle, hir::Expr, bool)> {
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
        let info = self.ee.get_field_info(&mut resolved, self.info.ftn, flags);
        if info.fieldFlags & ffi::CORINFO_FIELD_FLAGS_CORINFO_FLG_FIELD_STATIC == 0 {
            // ldsfld/stsfld/ldsflda on an instance field — RyuJIT
            // BADCODEs this too (importer.cpp CEE_LDSFLD).
            return Err(CompileError::BadIl("static access on an instance field"));
        }
        match info.fieldAccessor {
            ffi::CORINFO_FIELD_ACCESSOR_CORINFO_FIELD_STATIC_ADDRESS
            | ffi::CORINFO_FIELD_ACCESSOR_CORINFO_FIELD_STATIC_RVA_ADDRESS => {}
            ffi::CORINFO_FIELD_ACCESSOR_CORINFO_FIELD_STATIC_TLS
            | ffi::CORINFO_FIELD_ACCESSOR_CORINFO_FIELD_STATIC_TLS_MANAGED => {
                return Err(CompileError::Unsupported(
                    "thread-local statics ([ThreadStatic])",
                ));
            }
            ffi::CORINFO_FIELD_ACCESSOR_CORINFO_FIELD_STATIC_SHARED_STATIC_HELPER
            | ffi::CORINFO_FIELD_ACCESSOR_CORINFO_FIELD_STATIC_GENERICS_STATIC_HELPER => {
                return Err(CompileError::Unsupported(
                    "static field of a shared-generic or collectible class (helper accessor)",
                ));
            }
            ffi::CORINFO_FIELD_ACCESSOR_CORINFO_FIELD_STATIC_ADDR_HELPER
            | ffi::CORINFO_FIELD_ACCESSOR_CORINFO_FIELD_STATIC_READYTORUN_HELPER => {
                return Err(CompileError::Unsupported(
                    "static field through an address helper (R2R)",
                ));
            }
            _ => {
                return Err(CompileError::Unsupported(
                    "static field accessor outside the statics pack",
                ));
            }
        }
        if info.accessAllowed != ffi::CorInfoIsAccessAllowedResult_CORINFO_ACCESS_ALLOWED {
            return Err(CompileError::Unsupported(
                "static field needing an access callout",
            ));
        }
        if info.fieldLookup.accessType != ffi::InfoAccessType_IAT_VALUE {
            // The indirection-cell answer is the R2R shape — the
            // ldstr/newobj policy: load+relocation plumbing tier 0
            // doesn't have.
            return Err(CompileError::Unsupported(
                "static field address through an indirection cell (IAT_PVALUE/PPVALUE)",
            ));
        }
        // SAFETY: IAT_VALUE's live union member is `addr`
        // (corinfo.h's CORINFO_CONST_LOOKUP contract).
        let addr = unsafe { info.fieldLookup.__bindgen_anon_1.addr };
        let mut addr_expr = hir::Expr::Const(Const::NativeInt(addr as isize));
        if info.fieldFlags & ffi::CORINFO_FIELD_FLAGS_CORINFO_FLG_FIELD_STATIC_IN_HEAP != 0 {
            // A boxed value-class static: the cell holds the frozen box
            // object; the field data sits past the object header.
            addr_expr = hir::Expr::FieldAddr {
                obj: Box::new(hir::Expr::Load {
                    addr: Box::new(addr_expr),
                    offset: 0,
                    ty: Type::Ref,
                    access: MemAccess::Natural,
                }),
                field,
                offset: 8, // TARGET_POINTER_SIZE
            };
        }
        // The class-init trigger (the semantic core of the pack):
        // getFieldInfo's INITCLASS flag says the class is not yet inited;
        // initClass then decides — queried exactly as RyuJIT's
        // impInitClass does (importer.cpp:3897): the field, the method
        // being compiled, and the *method* context (an untagged or
        // class-tagged context here would crash the EE).
        let mut needs_init = false;
        if info.fieldFlags & ffi::CORINFO_FIELD_FLAGS_CORINFO_FLG_FIELD_INITCLASS != 0 {
            let init = self.ee.init_class(
                Some(field),
                Some(self.info.ftn),
                ContextHandle::from_method(self.info.ftn),
            );
            // DONT_INLINE is an inlining hint; we never inline.
            needs_init = init.contains(CorInfoInitClassResult::USE_HELPER);
        }
        Ok((field, addr_expr, needs_init))
    }

    /// Emits the static-constructor trigger (`CORINFO_HELP_INITCLASS` of
    /// the field's owning class, embedded as a raw `NativeInt` constant —
    /// the newobj emission's exact shape) when the `init_class` verdict
    /// asked for a helper.
    fn maybe_init_class(
        &mut self,
        field: FieldHandle,
        needs_init: bool,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        if !needs_init {
            return Ok(());
        }
        let owner = self.ee.get_field_class(field);
        stmts.push(hir::Stmt {
            il_offset,
            kind: hir::StmtKind::Eval(hir::Expr::Call {
                target: CallTarget::Helper(CorInfoHelpFunc::INITCLASS),
                sig: CallSig {
                    ret: Type::Void,
                    args: vec![Type::NativeInt],
                    has_this: false,
                },
                args: vec![self.embed_class_const(owner)?],
            }),
        });
        Ok(())
    }

    /// `ldsfld` (0x7E): a load through the field's static address. Pending
    /// stack trees evaluate first (the spill), then the `.cctor` trigger
    /// (a beforefieldinit class has no explicit cctor to observe, so the
    /// uniform order is spec-conforming for both init modes), then the
    /// load itself — a tree, like `ldfld`'s. A struct-typed static loads
    /// as its address (the `StructVal` discipline, step_10.9).
    fn ldsfld(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let (field, addr, needs_init) =
            self.resolve_static_field(token, ffi::CORINFO_ACCESS_FLAGS_CORINFO_ACCESS_GET)?;
        let (ty, access) = self.field_mem_type(field)?;
        self.spill_stack(stmts, il_offset)?;
        self.maybe_init_class(field, needs_init, stmts, il_offset)?;
        if let Type::Struct(class) = ty {
            return self.push(
                ty,
                hir::Expr::StructVal {
                    addr: Box::new(addr),
                    class,
                },
            );
        }
        self.push(
            ty,
            hir::Expr::Load {
                addr: Box::new(addr),
                offset: 0,
                ty,
                access,
            },
        )
    }

    /// `ldsflda` (0x7F): the static field's address — type-agnostic, like
    /// `ldflda`.
    fn ldsflda(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let (field, addr, needs_init) =
            self.resolve_static_field(token, ffi::CORINFO_ACCESS_FLAGS_CORINFO_ACCESS_ADDRESS)?;
        self.spill_stack(stmts, il_offset)?;
        self.maybe_init_class(field, needs_init, stmts, il_offset)?;
        self.push(Type::ByRef, addr)
    }

    /// `stsfld` (0x80): a store through the field's static address. The
    /// uniform order is value → `.cctor` → store (RyuJIT's precise-init
    /// order, importer.cpp:9849; ECMA-335's relaxed init makes it
    /// conforming for beforefieldinit classes too): the value materializes
    /// into a temp before the INITCLASS helper runs, so a value with side
    /// effects evaluates ahead of the cctor. A reference-typed field
    /// stores through the checked write barrier (statics blocks are
    /// EE-managed GC roots), a struct-typed one through the block-copy
    /// path (step_10.9), anything else as a plain indirect store.
    fn stsfld(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let (field, addr, needs_init) =
            self.resolve_static_field(token, ffi::CORINFO_ACCESS_FLAGS_CORINFO_ACCESS_SET)?;
        let (ty, access) = self.field_mem_type(field)?;
        let (vt, value) = self.pop()?;
        if vt != ty {
            return Err(CompileError::BadIl("stsfld value type mismatch"));
        }
        self.spill_stack(stmts, il_offset)?;
        let value = if needs_init {
            let t = self.temp(ty);
            stmts.push(hir::Stmt {
                il_offset,
                kind: hir::StmtKind::Store { dst: t, value },
            });
            self.local_value_expr(t).1
        } else {
            value
        };
        self.maybe_init_class(field, needs_init, stmts, il_offset)?;
        if let Type::Struct(class) = ty {
            return self.store_struct_through(addr, class, value, stmts, il_offset);
        }
        let kind = if ty == Type::Ref {
            hir::StmtKind::Eval(hir::Expr::Call {
                target: CallTarget::Helper(CorInfoHelpFunc::CHECKED_ASSIGN_REF),
                sig: CallSig {
                    ret: Type::Void,
                    args: vec![Type::ByRef, Type::Ref],
                    has_this: false,
                },
                args: vec![addr, value],
            })
        } else {
            hir::StmtKind::StoreInd {
                addr,
                offset: 0,
                value,
                access,
            }
        };
        stmts.push(hir::Stmt { il_offset, kind });
        Ok(())
    }

    /// The address of a struct value: `StructVal` unwraps; anything else
    /// (e.g. a register-passed call result) spills to a temp first and the
    /// temp's address is the answer.
    fn struct_addr_of(
        &mut self,
        value: hir::Expr,
        class: ClassHandle,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> hir::Expr {
        match value {
            hir::Expr::StructVal { addr, .. } => *addr,
            value => {
                let t = self.temp(Type::Struct(class));
                stmts.push(hir::Stmt {
                    il_offset,
                    kind: hir::StmtKind::Store { dst: t, value },
                });
                hir::Expr::LocalAddr(t)
            }
        }
    }

    /// A struct store through a computed address (`stobj`/`cpobj`/struct
    /// `stfld`): when the class embeds GC pointers the copy goes through
    /// `CORINFO_HELP_BULK_WRITEBARRIER(dst, src, size)` — GC-correct for
    /// heap destinations, a plain copy for stack ones (the helper checks);
    /// otherwise a plain block copy (`StoreInd` of the `StructVal`).
    fn store_struct_through(
        &mut self,
        addr: hir::Expr,
        class: ClassHandle,
        value: hir::Expr,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let layout = &self.struct_layouts[&class];
        if layout.gc_cells.is_empty() {
            stmts.push(hir::Stmt {
                il_offset,
                kind: hir::StmtKind::StoreInd {
                    addr,
                    offset: 0,
                    value,
                    access: MemAccess::Natural,
                },
            });
            return Ok(());
        }
        let size = layout.size as isize;
        let src = self.struct_addr_of(value, class, stmts, il_offset);
        stmts.push(hir::Stmt {
            il_offset,
            kind: hir::StmtKind::Eval(hir::Expr::Call {
                target: CallTarget::Helper(CorInfoHelpFunc::BULK_WRITEBARRIER),
                sig: CallSig {
                    ret: Type::Void,
                    args: vec![Type::ByRef, Type::ByRef, Type::NativeInt],
                    has_this: false,
                },
                args: vec![addr, src, hir::Expr::Const(Const::NativeInt(size))],
            }),
        });
        Ok(())
    }

    /// Class-token resolution for the struct opcodes (`ldobj`/`stobj`/
    /// `cpobj`/`initobj`): the token-kind hint is `CORINFO_TOKENKIND_Class`
    /// (importer.cpp `impResolveToken`, same as `newobj`'s class side).
    fn resolve_value_class(&mut self, token: u32) -> CompileResult<ClassHandle> {
        let mut resolved = zeroed_out(|t: &mut ffi::CORINFO_RESOLVED_TOKEN| {
            t.tokenContext = self.info.ftn.as_raw() as ffi::CORINFO_CONTEXT_HANDLE;
            t.tokenScope = self.info.args.scope;
            t.token = token;
            t.tokenType = ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Class;
        });
        self.ee.resolve_token(&mut resolved);
        let Some(class) = ClassHandle::from_raw(resolved.hClass) else {
            return Err(CompileError::BadIl("type token did not resolve to a class"));
        };
        if !self.ee.is_value_class(class) {
            return Err(CompileError::Unsupported(
                "initobj/ldobj/stobj/cpobj of a non-value class",
            ));
        }
        layout_of(&mut self.struct_layouts, self.ee, class)?;
        Ok(class)
    }

    /// Pops the address operand of a struct memory opcode: a managed
    /// byref or a native-int pointer (ECMA-335 §III.4.27: `&` or
    /// `native int`).
    fn pop_struct_addr(&mut self) -> CompileResult<hir::Expr> {
        let (ty, addr) = self.pop()?;
        if !matches!(ty, Type::ByRef | Type::NativeInt) {
            return Err(CompileError::BadIl(
                "struct opcode address must be a byref or native int",
            ));
        }
        Ok(addr)
    }

    /// `ldobj` (0x71): the struct value at the address.
    fn ldobj(&mut self, token: u32) -> CompileResult<()> {
        let class = self.resolve_value_class(token)?;
        let addr = self.pop_struct_addr()?;
        self.push(
            Type::Struct(class),
            hir::Expr::StructVal {
                addr: Box::new(addr),
                class,
            },
        )
    }

    /// `stobj` (0x81): the struct value stores through the address.
    fn stobj(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let class = self.resolve_value_class(token)?;
        let (vt, value) = self.pop()?;
        if vt != Type::Struct(class) {
            return Err(CompileError::BadIl("stobj value type mismatch"));
        }
        let addr = self.pop_struct_addr()?;
        self.spill_stack(stmts, il_offset)?;
        self.store_struct_through(addr, class, value, stmts, il_offset)
    }

    /// `cpobj` (0x70): struct copy from the source address to the
    /// destination address.
    fn cpobj(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let class = self.resolve_value_class(token)?;
        let src = self.pop_struct_addr()?;
        let dst = self.pop_struct_addr()?;
        self.spill_stack(stmts, il_offset)?;
        self.store_struct_through(
            dst,
            class,
            hir::Expr::StructVal {
                addr: Box::new(src),
                class,
            },
            stmts,
            il_offset,
        )
    }

    /// `initobj` (0xFE 15): zero-init the slot at the address.
    fn initobj(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let class = self.resolve_value_class(token)?;
        let addr = self.pop_struct_addr()?;
        self.spill_stack(stmts, il_offset)?;
        stmts.push(hir::Stmt {
            il_offset,
            kind: hir::StmtKind::BlockZero { addr, class },
        });
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
        // `newobj` of a value class (step_10.10): no allocation — csc's
        // `new S(args)` is in-place construction (RyuJIT's impImportNewObj
        // valuetype path): a fresh struct temp, zero-initialized (initobj
        // semantics — a conforming constructor overwrites every field),
        // the constructor called on the temp's address, and the temp
        // itself pushed as the value.
        let is_value_class = self.ee.is_value_class(class);

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

        // The class's MethodTable* operand, embedded directly; an
        // indirection cell (R2R-style) needs load/reloc plumbing tier 0
        // doesn't have. The reference path passes it to the allocation
        // helper; both paths pass it to INITCLASS when a cctor runs.
        let (embedded, indirection) = self.ee.embed_class_handle(class);
        let (Some(embedded), None) = (embedded, indirection) else {
            return Err(CompileError::Unsupported(
                "class handle through an indirection cell",
            ));
        };
        let class_const = || hir::Expr::Const(Const::NativeInt(embedded.as_raw() as isize));

        // Pending stack trees (the constructor arguments included) must
        // evaluate before the allocation side effects.
        self.spill_stack(stmts, il_offset)?;
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

        // The value-class construction target, or the reference
        // allocation, prepared before the constructor's arguments pop.
        // `this_arg` is the constructor's receiver; `result` is the
        // value the `newobj` pushes.
        let (this_arg, result) = if is_value_class {
            layout_of(&mut self.struct_layouts, self.ee, class)?;
            let t = self.temp(Type::Struct(class));
            stmts.push(hir::Stmt {
                il_offset,
                kind: hir::StmtKind::BlockZero {
                    addr: hir::Expr::LocalAddr(t),
                    class,
                },
            });
            (
                hir::Expr::LocalAddr(t),
                hir::Expr::StructVal {
                    addr: Box::new(hir::Expr::LocalAddr(t)),
                    class,
                },
            )
        } else {
            let (helper, _has_side_effects) = self.ee.get_new_helper(&resolved, self.info.ftn);
            // The single-argument (MethodTable*) -> Object* class-alloc
            // helpers, minus the FINALIZE forms (a finalizable newobj
            // needs the stack spill of RyuJIT's "finalizable newobj
            // spill" — a later step) and NEWSFAST_ALIGN8_VC (boxed value
            // classes — out with structs). The EE answers NEWSFAST for a
            // plain small class (jitinterface.cpp getNewHelperStatic);
            // NEWFAST is its slow fallback.
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

            // The object lands in a fresh Ref temp — automatically a GC
            // root (frame-resident, zero-initialized) across both
            // safepoints (the allocation and the constructor call).
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
            (hir::Expr::Local(t_obj), hir::Expr::Local(t_obj))
        };

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
        let ret = sig_elem_type(
            CorInfoType::from_raw(call.sig.retType()),
            ClassHandle::from_raw(call.sig.retTypeClass),
            self.ee,
            &mut self.struct_layouts,
        )?;
        if ret != Type::Void {
            return Err(CompileError::BadIl("a constructor must return void"));
        }
        let arg_types = sig_arg_types(&call.sig, self.ee, &mut self.struct_layouts)?;
        let mut args = Vec::with_capacity(arg_types.len() + 1);
        for &expected in arg_types.iter().rev() {
            let (ty, value) = self.pop()?;
            if ty != expected {
                return Err(CompileError::BadIl("call argument type mismatch"));
            }
            args.push(value);
        }
        args.reverse();
        // The constructor: a direct instance call whose `this` is the
        // fresh object/temp, not a stack value (importer.cpp CEE_NEWOBJ's
        // newObjThisPtr). No null check wraps it: JIT_New* never returns
        // null, and a fresh struct temp is never null.
        args.insert(0, this_arg);
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
        let result_ty = if is_value_class {
            Type::Struct(class)
        } else {
            Type::Ref
        };
        self.push(result_ty, result)
    }

    /// Resolves a class metadata token for the box/cast opcodes. `kind`
    /// is the token-kind hint `impResolveToken` sets (importer.cpp:
    /// `CORINFO_TOKENKIND_Box` for `box`, `CORINFO_TOKENKIND_Casting` for
    /// `castclass`/`isinst`, `CORINFO_TOKENKIND_Class` for the unbox
    /// forms).
    fn resolve_box_cast_class(
        &mut self,
        token: u32,
        kind: ffi::CorInfoTokenKind,
    ) -> CompileResult<(ffi::CORINFO_RESOLVED_TOKEN, ClassHandle)> {
        let mut resolved = zeroed_out(|t: &mut ffi::CORINFO_RESOLVED_TOKEN| {
            t.tokenContext = self.info.ftn.as_raw() as ffi::CORINFO_CONTEXT_HANDLE;
            t.tokenScope = self.info.args.scope;
            t.token = token;
            t.tokenType = kind;
        });
        self.ee.resolve_token(&mut resolved);
        let Some(class) = ClassHandle::from_raw(resolved.hClass) else {
            return Err(CompileError::BadIl("type token did not resolve to a class"));
        };
        Ok((resolved, class))
    }

    /// Embeds a class handle as a raw `NativeInt` constant (a MethodTable*
    /// is not an object reference and is never GC-rooted as one — the
    /// newobj rule); an indirection cell needs load/reloc plumbing tier 0
    /// doesn't have.
    fn embed_class_const(&mut self, class: ClassHandle) -> CompileResult<hir::Expr> {
        let (embedded, indirection) = self.ee.embed_class_handle(class);
        let (Some(class), None) = (embedded, indirection) else {
            return Err(CompileError::Unsupported(
                "class handle through an indirection cell",
            ));
        };
        Ok(hir::Expr::Const(Const::NativeInt(class.as_raw() as isize)))
    }

    /// Pops the object operand of a cast/unbox: a reference (`null`
    /// included).
    fn pop_object(&mut self) -> CompileResult<hir::Expr> {
        let (ty, obj) = self.pop()?;
        if ty != Type::Ref {
            return Err(CompileError::BadIl(
                "cast/unbox operand must be a reference",
            ));
        }
        Ok(obj)
    }

    /// `isinst` (0x75) / `castclass` (0x74): the EE picks the casting
    /// helper (`getCastingHelper` — the ISINSTANCEOF*/CHKCAST* families,
    /// all `(MethodTable*, Object*) -> Object*`), and the helper answers
    /// the type test. `castclass` failure raises `InvalidCastException`
    /// from the helper's own throwing path; nothing is open-coded.
    fn cast(&mut self, token: u32, throwing: bool) -> CompileResult<()> {
        let (resolved, class) =
            self.resolve_box_cast_class(token, ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Casting)?;
        self.cast_from_resolved(&resolved, class, throwing)
    }

    /// The cast helper call on an already-resolved token; also the
    /// non-value-class tail of `unbox.any` (the throwing form).
    fn cast_from_resolved(
        &mut self,
        resolved: &ffi::CORINFO_RESOLVED_TOKEN,
        class: ClassHandle,
        throwing: bool,
    ) -> CompileResult<()> {
        let helper = self.ee.get_casting_helper(resolved, throwing);
        if !matches!(
            helper,
            CorInfoHelpFunc::ISINSTANCEOFANY
                | CorInfoHelpFunc::ISINSTANCEOFCLASS
                | CorInfoHelpFunc::ISINSTANCEOFINTERFACE
                | CorInfoHelpFunc::ISINSTANCEOFARRAY
                | CorInfoHelpFunc::CHKCASTANY
                | CorInfoHelpFunc::CHKCASTCLASS
                | CorInfoHelpFunc::CHKCASTINTERFACE
                | CorInfoHelpFunc::CHKCASTARRAY
        ) {
            return Err(CompileError::Unsupported(
                "casting helper outside the isinst/castclass set",
            ));
        }
        let mt = self.embed_class_const(class)?;
        let obj = self.pop_object()?;
        self.push(
            Type::Ref,
            hir::Expr::Call {
                target: CallTarget::Helper(helper),
                sig: CallSig {
                    ret: Type::Ref,
                    args: vec![Type::NativeInt, Type::Ref],
                    has_this: false,
                },
                args: vec![mt, obj],
            },
        )
    }

    /// `box` (0x8C): allocate the box and copy the value through the EE's
    /// `BOX` helper — `CastHelpers.Box(MethodTable*, ref byte)` —
    /// never an inline allocate/copy sequence (tier 0: correct helper
    /// selection, not reimplemented boxing). Boxing a non-value class is
    /// the ECMA-335 no-op form (the reference passes through). The class
    /// passed to the helper is `getTypeForBox`'s answer (boxing
    /// `Nullable<T>` produces a boxed `T`).
    fn box_(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let (_resolved, class) =
            self.resolve_box_cast_class(token, ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Box)?;
        if !self.ee.is_value_class(class) {
            // `box` of a reference type is a NOP (importer.cpp CEE_BOX).
            let (ty, value) = self.pop()?;
            if ty != Type::Ref {
                return Err(CompileError::BadIl("box operand type mismatch"));
            }
            return self.push(Type::Ref, value);
        }
        let helper = self.ee.get_box_helper(class);
        match helper {
            CorInfoHelpFunc::BOX => {}
            CorInfoHelpFunc::BOX_NULLABLE => {
                return Err(CompileError::Unsupported("box of Nullable<T>"));
            }
            _ => {
                return Err(CompileError::Unsupported("box helper outside the box set"));
            }
        }
        let boxed = self.ee.get_type_for_box(class);
        let mt = self.embed_class_const(boxed)?;

        // The box operand pops first; the remaining pending stack trees
        // then spill (IL order: they were produced before the operand),
        // and only then does the operand's own temp store evaluate it —
        // the stobj ordering pattern.
        let (ty, value) = self.pop()?;
        let expected = sig_elem_type(
            Some(self.ee.as_cor_info_type(class)),
            Some(class),
            self.ee,
            &mut self.struct_layouts,
        )?;
        if ty != expected {
            return Err(CompileError::BadIl("box operand type mismatch"));
        }
        self.spill_stack(stmts, il_offset)?;
        // The helper's second argument is the address of the value's
        // bytes: a struct value IS its address (step_10.9), a scalar
        // spills to a temp whose address is taken.
        let addr = if let Type::Struct(class) = ty {
            self.struct_addr_of(value, class, stmts, il_offset)
        } else {
            match value {
                hir::Expr::Local(id) => hir::Expr::LocalAddr(id),
                value => {
                    let t = self.temp(ty);
                    stmts.push(hir::Stmt {
                        il_offset,
                        kind: hir::StmtKind::Store { dst: t, value },
                    });
                    hir::Expr::LocalAddr(t)
                }
            }
        };
        self.push(
            Type::Ref,
            hir::Expr::Call {
                target: CallTarget::Helper(CorInfoHelpFunc::BOX),
                sig: CallSig {
                    ret: Type::Ref,
                    args: vec![Type::NativeInt, Type::ByRef],
                    has_this: false,
                },
                args: vec![mt, addr],
            },
        )
    }

    /// The `CORINFO_HELP_UNBOX` call shared by `unbox` and `unbox.any`:
    /// `CastHelpers.Unbox(MethodTable*, object) -> ref byte` — the EE
    /// helper both checks the type (NullReferenceException /
    /// InvalidCastException from its own throwing paths) and computes the
    /// payload address, so the boxed-value layout offset is never
    /// materialized JIT-side.
    fn unbox_payload_call(&mut self, class: ClassHandle) -> CompileResult<hir::Expr> {
        let helper = self.ee.get_un_box_helper(class);
        match helper {
            CorInfoHelpFunc::UNBOX => {}
            CorInfoHelpFunc::UNBOX_NULLABLE => {
                return Err(CompileError::Unsupported("unbox of Nullable<T>"));
            }
            _ => {
                return Err(CompileError::Unsupported(
                    "unbox helper outside the unbox set",
                ));
            }
        }
        let mt = self.embed_class_const(class)?;
        let obj = self.pop_object()?;
        Ok(hir::Expr::Call {
            target: CallTarget::Helper(CorInfoHelpFunc::UNBOX),
            sig: CallSig {
                ret: Type::ByRef,
                args: vec![Type::NativeInt, Type::Ref],
                has_this: false,
            },
            args: vec![mt, obj],
        })
    }

    /// `unbox` (0x79): the boxed value's payload address, an interior
    /// `ByRef`. The temp the result lands in is a frame-resident byref
    /// slot — reported in the GC slot table with the interior flag
    /// (step_10.3/10.4's untracked-root mechanism), which both keeps the
    /// box alive and re-bases the pointer if the GC moves it.
    fn unbox(&mut self, token: u32) -> CompileResult<()> {
        let (_resolved, class) =
            self.resolve_box_cast_class(token, ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Class)?;
        if !self.ee.is_value_class(class) {
            return Err(CompileError::BadIl("unbox of a non-value class"));
        }
        let payload = self.unbox_payload_call(class)?;
        self.push(Type::ByRef, payload)
    }

    /// `unbox.any` (0xA5): for a value class, `unbox` followed by the
    /// `ldobj` read of the payload (the struct copy semantics of
    /// step_10.9 apply through `StructVal`); for anything else it is
    /// exactly `castclass` (importer.cpp CEE_UNBOX_ANY).
    fn unbox_any(&mut self, token: u32) -> CompileResult<()> {
        let (resolved, class) =
            self.resolve_box_cast_class(token, ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Class)?;
        if !self.ee.is_value_class(class) {
            return self.cast_from_resolved(&resolved, class, true);
        }
        let raw = self.ee.as_cor_info_type(class);
        let ty = sig_elem_type(Some(raw), Some(class), self.ee, &mut self.struct_layouts)?;
        let payload = self.unbox_payload_call(class)?;
        if let Type::Struct(class) = ty {
            self.push(
                ty,
                hir::Expr::StructVal {
                    addr: Box::new(payload),
                    class,
                },
            )
        } else {
            // The boxed payload is a memory cell: a boxed bool/char/…
            // occupies its metadata width, exactly like a field.
            let (ty, access) = corinfo_mem_type(raw)?;
            self.push(
                ty,
                hir::Expr::Load {
                    addr: Box::new(payload),
                    offset: 0,
                    ty,
                    access,
                },
            )
        }
    }

    // --- step_10.8: arrays ---

    /// `newarr` (0x8D): allocate a 1-D zero-based array through the EE's
    /// `getNewArrHelper` — the two-argument `(MethodTable*, INT_PTR
    /// element_count) -> Object*` forms. The class handle embeds as a raw
    /// `NativeInt` constant (the newobj rule: a MethodTable* is never
    /// GC-rooted as a reference). The helper zero-initializes; a negative
    /// length zero-extends to a huge count and the helper throws
    /// `OverflowException` (RyuJIT's answer to the same shape).
    fn newarr(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let mut resolved = zeroed_out(|t: &mut ffi::CORINFO_RESOLVED_TOKEN| {
            t.tokenContext = self.info.ftn.as_raw() as ffi::CORINFO_CONTEXT_HANDLE;
            t.tokenScope = self.info.args.scope;
            t.token = token;
            t.tokenType = ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Newarr;
        });
        self.ee.resolve_token(&mut resolved);
        let Some(class) = ClassHandle::from_raw(resolved.hClass) else {
            return Err(CompileError::BadIl(
                "newarr token did not resolve to a class",
            ));
        };
        // `newarr` always builds an SZ (single-dimension, zero-based)
        // array; the gate is defensive.
        if !self.ee.is_sd_array(class) {
            return Err(CompileError::Unsupported("newarr of a non-SZ array"));
        }
        let helper = self.ee.get_new_arr_helper(class);
        if !matches!(
            helper,
            CorInfoHelpFunc::NEWARR_1_DIRECT
                | CorInfoHelpFunc::NEWARR_1_MAYBEFROZEN
                | CorInfoHelpFunc::NEWARR_1_PTR
                | CorInfoHelpFunc::NEWARR_1_VC
                | CorInfoHelpFunc::NEWARR_1_ALIGN8
        ) {
            return Err(CompileError::Unsupported(
                "newarr allocation helper outside the accepted set",
            ));
        }
        let (lt, len) = self.pop()?;
        if !matches!(lt, Type::Int32 | Type::NativeInt) {
            return Err(CompileError::BadIl(
                "newarr length must be int32 or native int",
            ));
        }
        let mt = self.embed_class_const(class)?;
        self.spill_stack(stmts, il_offset)?;
        let len = self.index_native(lt, len);
        self.push(
            Type::Ref,
            hir::Expr::Call {
                target: CallTarget::Helper(helper),
                sig: CallSig {
                    ret: Type::Ref,
                    args: vec![Type::NativeInt, Type::NativeInt],
                    has_this: false,
                },
                args: vec![mt, len],
            },
        )
    }

    /// `ldlen` (0x8E): the element count — an [`hir::Expr::ArrLen`] tree,
    /// Int32 on the stack. The load it lowers to doubles as the null
    /// check (step_10.4's trap model).
    fn ldlen(&mut self) -> CompileResult<()> {
        let array = self.pop_array()?;
        self.push(
            Type::Int32,
            hir::Expr::ArrLen {
                array: Box::new(array),
            },
        )
    }

    /// `ldelem.*` (0x90..=0x9A) / `ldelem` (0xA3): the bounds check
    /// statement, then the typed load through the computed element
    /// address. A struct element yields the address as a `StructVal`
    /// (step_10.9); a reference element is a natural-width `Ref` load
    /// whose temp is automatically a GC root.
    fn ldelem(
        &mut self,
        elem: ElemKind,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        match elem {
            ElemKind::Cell(ty, access, elem_size) => {
                let addr = self.bounds_checked_addr(ty, elem_size, stmts, il_offset)?;
                self.push(
                    ty,
                    hir::Expr::Load {
                        addr: Box::new(addr),
                        offset: 0,
                        ty,
                        access,
                    },
                )
            }
            ElemKind::Struct(class) => {
                let elem_size = self.struct_layouts[&class].size;
                let addr =
                    self.bounds_checked_addr(Type::Struct(class), elem_size, stmts, il_offset)?;
                self.push(
                    Type::Struct(class),
                    hir::Expr::StructVal {
                        addr: Box::new(addr),
                        class,
                    },
                )
            }
        }
    }

    /// `stelem.*` (0x9B..=0xA2) / `stelem` (0xA4): the bounds check
    /// statement, then the store through the computed element address.
    /// A reference store goes through the `CORINFO_HELP_ARRADDR_ST`
    /// helper instead (`CastHelpers.StelemRef`: the null check, bounds
    /// check, covariance type check, and write barrier are all the
    /// helper's — jithelpers.h:139), and a struct store is the block
    /// copy / bulk-write-barrier path of `stobj`.
    fn stelem(
        &mut self,
        elem: ElemKind,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        match elem {
            ElemKind::Cell(Type::Ref, _, _) => {
                let (vt, value) = self.pop()?;
                if vt != Type::Ref {
                    return Err(CompileError::BadIl("stelem value type mismatch"));
                }
                let (it, index) = self.pop_index()?;
                let array = self.pop_array()?;
                self.spill_stack(stmts, il_offset)?;
                let index = self.index_native(it, index);
                stmts.push(hir::Stmt {
                    il_offset,
                    kind: hir::StmtKind::Eval(hir::Expr::Call {
                        target: CallTarget::Helper(CorInfoHelpFunc::ARRADDR_ST),
                        sig: CallSig {
                            ret: Type::Void,
                            args: vec![Type::Ref, Type::NativeInt, Type::Ref],
                            has_this: false,
                        },
                        args: vec![array, index, value],
                    }),
                });
                Ok(())
            }
            ElemKind::Cell(ty, access, elem_size) => {
                let (vt, value) = self.pop()?;
                if vt != ty {
                    return Err(CompileError::BadIl("stelem value type mismatch"));
                }
                let addr = self.bounds_checked_addr(ty, elem_size, stmts, il_offset)?;
                stmts.push(hir::Stmt {
                    il_offset,
                    kind: hir::StmtKind::StoreInd {
                        addr,
                        offset: 0,
                        value,
                        access,
                    },
                });
                Ok(())
            }
            ElemKind::Struct(class) => {
                let (vt, value) = self.pop()?;
                if vt != Type::Struct(class) {
                    return Err(CompileError::BadIl("stelem value type mismatch"));
                }
                let elem_size = self.struct_layouts[&class].size;
                let addr =
                    self.bounds_checked_addr(Type::Struct(class), elem_size, stmts, il_offset)?;
                self.store_struct_through(addr, class, value, stmts, il_offset)
            }
        }
    }

    /// `ldelema` (0x8F): the element's address, an interior `ByRef` (the
    /// temp it lands in is an interior-pointer GC root, the `unbox`
    /// discipline). A reference element takes the
    /// `CORINFO_HELP_LDELEMA_REF` helper (`CastHelpers.LdelemaRef`): a
    /// byref into a covariant array would bypass `stelem.ref`'s
    /// covariance check, so the helper null-checks, bounds-checks, and
    /// exact-element-type-checks itself — no separate BoundsCheck.
    fn ldelema(
        &mut self,
        token: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<()> {
        let elem = self.elem_kind_of(token)?;
        match elem {
            ElemKind::Cell(Type::Ref, _, _) => {
                let (it, index) = self.pop_index()?;
                let array = self.pop_array()?;
                self.spill_stack(stmts, il_offset)?;
                let index = self.index_native(it, index);
                self.push(
                    Type::ByRef,
                    hir::Expr::Call {
                        target: CallTarget::Helper(CorInfoHelpFunc::LDELEMA_REF),
                        sig: CallSig {
                            ret: Type::ByRef,
                            args: vec![Type::Ref, Type::NativeInt],
                            has_this: false,
                        },
                        args: vec![array, index],
                    },
                )
            }
            ElemKind::Cell(ty, _, elem_size) => {
                let addr = self.bounds_checked_addr(ty, elem_size, stmts, il_offset)?;
                self.push(Type::ByRef, addr)
            }
            ElemKind::Struct(class) => {
                let elem_size = self.struct_layouts[&class].size;
                let addr =
                    self.bounds_checked_addr(Type::Struct(class), elem_size, stmts, il_offset)?;
                self.push(Type::ByRef, addr)
            }
        }
    }

    /// The element kind of a token form (`ldelema`/`ldelem`/`stelem`):
    /// the element class resolved with the Class hint (importer.cpp's
    /// `impResolveToken` for these opcodes), its storage type from
    /// `asCorInfoType`. A value class is a struct element (the layout
    /// registers into the side table); a reference type is exactly the
    /// `*.ref` fixed form (RyuJIT does the same); anything else maps
    /// through the stored-cell table.
    fn elem_kind_of(&mut self, token: u32) -> CompileResult<ElemKind> {
        let (_resolved, class) =
            self.resolve_box_cast_class(token, ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Class)?;
        let raw = self.ee.as_cor_info_type(class);
        if raw == CorInfoType::ValueClass {
            let ty = sig_elem_type(Some(raw), Some(class), self.ee, &mut self.struct_layouts)?;
            let Type::Struct(class) = ty else {
                return Err(CompileError::Internal(
                    "a value-class element is always a struct",
                ));
            };
            return Ok(ElemKind::Struct(class));
        }
        let (ty, access) = corinfo_mem_type(raw)
            .map_err(|_| CompileError::Unsupported("array element type outside the array pack"))?;
        let elem_size = match access.narrow_bytes() {
            Some(n) => u32::from(n),
            None => natural_cell_size(ty),
        };
        Ok(ElemKind::Cell(ty, access, elem_size))
    }

    /// The shared core of the bounds-checked element accesses: pops the
    /// index, then the array (IL pop order), spills the pending stack,
    /// and emits the [`hir::StmtKind::BoundsCheck`] statement. Both
    /// operands feed the check AND the returned `ArrElemAddr` tree, so a
    /// non-trivial tree (one containing a call) materializes into a temp
    /// first — it must evaluate exactly once, array before index (the IL
    /// push order).
    fn bounds_checked_addr(
        &mut self,
        elem: Type,
        elem_size: u32,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> CompileResult<hir::Expr> {
        let (it, index) = self.pop_index()?;
        let array = self.pop_array()?;
        self.spill_stack(stmts, il_offset)?;
        let (array, check_array) = self.duplicate(Type::Ref, array, stmts, il_offset);
        let (index, check_index) = self.duplicate(it, index, stmts, il_offset);
        stmts.push(hir::Stmt {
            il_offset,
            kind: hir::StmtKind::BoundsCheck {
                array: check_array,
                index: check_index,
            },
        });
        Ok(hir::Expr::ArrElemAddr {
            array: Box::new(array),
            index: Box::new(index),
            elem,
            elem_size,
        })
    }

    /// Pops the array operand of an element access: a reference (null
    /// included — the bounds check's length load faults on it).
    fn pop_array(&mut self) -> CompileResult<hir::Expr> {
        let (ty, array) = self.pop()?;
        if ty != Type::Ref {
            return Err(CompileError::BadIl("array operand must be a reference"));
        }
        Ok(array)
    }

    /// Pops the index operand of an element access (int32 or native int).
    fn pop_index(&mut self) -> CompileResult<(Type, hir::Expr)> {
        let (ty, index) = self.pop()?;
        if !matches!(ty, Type::Int32 | Type::NativeInt) {
            return Err(CompileError::BadIl(
                "array index must be int32 or native int",
            ));
        }
        Ok((ty, index))
    }

    /// An array index/count widened to native int for a helper signature:
    /// a zero-extending `conv.u` (the bounds check proves 0 <= index).
    fn index_native(&mut self, ty: Type, index: hir::Expr) -> hir::Expr {
        if ty == Type::NativeInt {
            index
        } else {
            hir::Expr::Conv {
                to: Type::NativeInt,
                overflow: false,
                unsigned: true,
                arg: Box::new(index),
            }
        }
    }

    /// Duplicates a popped operand tree for a two-consumer shape (the
    /// bounds-check statement and the element address both read the array
    /// and the index). A trivial tree (a constant, a local read/address)
    /// copies outright; anything else spills to a temp first so its
    /// effects — a call — evaluate exactly once.
    fn duplicate(
        &mut self,
        ty: Type,
        value: hir::Expr,
        stmts: &mut Vec<hir::Stmt>,
        il_offset: IlOffset,
    ) -> (hir::Expr, hir::Expr) {
        match value {
            hir::Expr::Const(k) => (hir::Expr::Const(k), hir::Expr::Const(k)),
            hir::Expr::Local(id) => (hir::Expr::Local(id), hir::Expr::Local(id)),
            hir::Expr::LocalAddr(id) => (hir::Expr::LocalAddr(id), hir::Expr::LocalAddr(id)),
            value => {
                let t = self.temp(ty);
                stmts.push(hir::Stmt {
                    il_offset,
                    kind: hir::StmtKind::Store { dst: t, value },
                });
                (hir::Expr::Local(t), hir::Expr::Local(t))
            }
        }
    }

    /// Imports the instructions of block `b` (leader `leaders[b]`). The
    /// stack starts empty — the entry check below rejects leaders a
    /// predecessor already entered non-empty, and a later pass over
    /// `expected_depth` proves the assumption against the remaining
    /// (backward-edge) predecessors, so any leftover from the previous
    /// block is discarded here.
    fn import_block(
        &mut self,
        b: usize,
        leaders: &[u32],
        insns: &[Insn],
    ) -> CompileResult<hir::Block> {
        self.stack.clear();
        let start = leaders[b];
        // A block an already-imported predecessor entered with a non-empty
        // evaluation stack (a join carrying values — csc's ternary shape,
        // e.g. `x + (c ? a : b)`) is outside the supported shape: reject
        // it here, as Unsupported, rather than underflowing mid-block and
        // misreporting valid IL as BadIl. (Backward edges are recorded
        // only after the header imported; the post-pass over
        // `expected_depth` in `import` is the backstop for those.)
        if !self.catch_entries.contains(&start)
            && self.expected_depth.get(&start).is_some_and(|&d| d != 0)
        {
            return Err(CompileError::Unsupported(
                "evaluation-stack values crossing a block boundary",
            ));
        }
        let end = leaders
            .get(b + 1)
            .copied()
            .unwrap_or(self.info.il.len() as u32);
        let first = insns.partition_point(|i| i.offset < start);
        let last = insns.partition_point(|i| i.offset < end);

        let mut stmts = Vec::new();
        let mut terminator = None;
        // A catch handler's entry block is entered by the VM with the
        // exception object on the eval stack (step_10.6): a synthesized
        // store of the funclet's incoming argument into a fresh Ref temp
        // — an ordinary always-live untracked root, zeroed by the main
        // prolog — and a read of that temp as the initial stack (depth 1,
        // which the expected-depth check enforces against any IL-level
        // predecessor).
        if self.catch_entries.contains(&start) {
            let exc = self.temp(Type::Ref);
            stmts.push(hir::Stmt {
                il_offset: IlOffset(start),
                kind: hir::StmtKind::Store {
                    dst: exc,
                    value: hir::Expr::CatchArg,
                },
            });
            self.push(Type::Ref, hir::Expr::Local(exc))?;
        }
        for insn in &insns[first..last] {
            let il_offset = IlOffset(insn.offset);
            match insn.op {
                Op::Nop => {}
                Op::LdArg(index) => {
                    let id = self.il_arg_id(u32::from(index))?;
                    let (ty, expr) = self.local_value_expr(id);
                    self.push(ty, expr)?;
                }
                Op::LdLoc(index) => {
                    let id = self.il_local_id(u32::from(index))?;
                    let (ty, expr) = self.local_value_expr(id);
                    self.push(ty, expr)?;
                }
                Op::StLoc(index) => {
                    let id = self.il_local_id(u32::from(index))?;
                    self.store_local(id, &mut stmts, il_offset)?;
                }
                Op::LdArgA(index) => {
                    let id = self.il_arg_id(u32::from(index))?;
                    self.push(Type::ByRef, hir::Expr::LocalAddr(id))?;
                }
                Op::StArg(index) => {
                    let id = self.il_arg_id(u32::from(index))?;
                    self.store_local(id, &mut stmts, il_offset)?;
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
                Op::LdToken(token) => self.ldtoken(token)?,
                Op::SizeOf(token) => self.sizeof_(token)?,
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
                Op::LdFld(token) => self.ldfld(token, &mut stmts, il_offset)?,
                Op::LdFldA(token) => self.ldflda(token, &mut stmts, il_offset)?,
                Op::StFld(token) => self.stfld(token, &mut stmts, il_offset)?,
                Op::LdSFld(token) => self.ldsfld(token, &mut stmts, il_offset)?,
                Op::LdSFldA(token) => self.ldsflda(token, &mut stmts, il_offset)?,
                Op::StSFld(token) => self.stsfld(token, &mut stmts, il_offset)?,
                Op::CpObj(token) => self.cpobj(token, &mut stmts, il_offset)?,
                Op::LdObj(token) => self.ldobj(token)?,
                Op::StObj(token) => self.stobj(token, &mut stmts, il_offset)?,
                Op::InitObj(token) => self.initobj(token, &mut stmts, il_offset)?,
                Op::CastClass(token) => self.cast(token, true)?,
                Op::IsInst(token) => self.cast(token, false)?,
                Op::Unbox(token) => self.unbox(token)?,
                Op::Box(token) => self.box_(token, &mut stmts, il_offset)?,
                Op::UnboxAny(token) => self.unbox_any(token)?,
                Op::NewArr(token) => self.newarr(token, &mut stmts, il_offset)?,
                Op::LdLen => self.ldlen()?,
                Op::LdElemA(token) => self.ldelema(token, &mut stmts, il_offset)?,
                Op::LdElemK(ty, access, size) => {
                    self.ldelem(ElemKind::Cell(ty, access, size), &mut stmts, il_offset)?
                }
                Op::LdElem(token) => {
                    let elem = self.elem_kind_of(token)?;
                    self.ldelem(elem, &mut stmts, il_offset)?
                }
                Op::StElemK(ty, access, size) => {
                    self.stelem(ElemKind::Cell(ty, access, size), &mut stmts, il_offset)?
                }
                Op::StElem(token) => {
                    let elem = self.elem_kind_of(token)?;
                    self.stelem(elem, &mut stmts, il_offset)?
                }
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
                    terminator = Some(self.ret(&mut stmts, il_offset)?);
                }
                Op::Throw => {
                    terminator = Some(self.throw(&mut stmts, il_offset)?);
                }
                Op::Leave { target } => {
                    terminator = Some(self.leave(b, target, &mut stmts, il_offset)?);
                }
                Op::EndFinally => {
                    terminator = Some(self.endfinally(&mut stmts, il_offset)?);
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
            ret_class: None,
            arg_classes: Vec::new(),
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
                ret_class: None,
                arg_classes: Vec::new(),
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
    fn values_crossing_a_join_are_a_clean_unsupported_not_bad_il() {
        // Regression for the post-10.6 triage: VerifyMagnitudePhase-
        // Properties (GitHub_18362) — csc's `phase += (phase < 0) ? PI :
        // -PI` evaluates `phase` before the ternary, so the conditional's
        // arm blocks and their join carry evaluation-stack values. The
        // importer doesn't support crossing values; the failure must be a
        // clean Unsupported (the IL is valid), never a BadIl stack
        // underflow mid-block.
        //
        // ldarg.0; ldarg.0; brfalse.s F; ldc.i4.1; br.s J; F: ldc.i4.2;
        // J: add; ret.
        let il = [0x02, 0x02, 0x2C, 0x03, 0x17, 0x2B, 0x01, 0x18, 0x58, 0x2A];
        let err = import_ii(&il).err().expect("crossing values are out");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("crossing a block boundary")),
            "{err:?}"
        );
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
            ret_class: None,
            arg_classes: Vec::new(),
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
        // cpobj's byte neighbor, an undefined single byte, an unsupported
        // 0xFE form.
        for il in [
            &[0x77, 0x2A][..],
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

    // --- step_10.6: EH (try/catch/finally) ---

    /// A canned typed-catch clause (flags NONE; raw class mdToken).
    fn catch_clause(
        try_start: u32,
        try_len: u32,
        handler_start: u32,
        handler_len: u32,
        class_token: u32,
    ) -> ffi::CORINFO_EH_CLAUSE {
        let mut clause: ffi::CORINFO_EH_CLAUSE = unsafe { std::mem::zeroed() };
        clause.Flags = ffi::CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_NONE;
        clause.TryOffset = try_start;
        clause.TryLength = try_len;
        clause.HandlerOffset = handler_start;
        clause.HandlerLength = handler_len;
        clause.__bindgen_anon_1.ClassToken = class_token;
        clause
    }

    /// A canned finally clause.
    fn finally_clause(
        try_start: u32,
        try_len: u32,
        handler_start: u32,
        handler_len: u32,
    ) -> ffi::CORINFO_EH_CLAUSE {
        let mut clause = catch_clause(try_start, try_len, handler_start, handler_len, 0);
        clause.Flags = ffi::CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_FINALLY;
        clause
    }

    /// A fixture with canned EH clauses installed on the mock (and
    /// `eh_count` matching), `void f()` shape with `locals`.
    fn eh_fixture(
        il: &[u8],
        locals: &[CorInfoType],
        clauses: &[ffi::CORINFO_EH_CLAUSE],
    ) -> (MockEe, MethodInfo) {
        let (mut ee, mut info) = fixture_full(
            il,
            &sig(CorInfoType::Void, &[]),
            locals,
            8,
            clauses.len() as u32,
        );
        ee.eh_clauses = clauses.to_vec();
        info.eh_count = clauses.len() as u32;
        (ee, info)
    }

    /// The try/finally shape of RyuJIT's B1 reference
    /// (target/eh-ref/B1o0.disasm.txt):
    /// `0: nop; 1: leave.s +1 (-> 4); 3: endfinally; 4: ret` with a
    /// finally clause try [0,3), handler [3,4).
    const TRY_FINALLY_IL: [u8; 5] = [0x00, 0xDE, 0x01, 0xDC, 0x2A];

    fn as_leave(t: &hir::Terminator) -> BlockId {
        match t {
            hir::Terminator::Leave { target } => *target,
            _ => panic!("expected Terminator::Leave"),
        }
    }

    fn as_jump(t: &hir::Terminator) -> BlockId {
        match t {
            hir::Terminator::Jump { target } => *target,
            _ => panic!("expected Terminator::Jump"),
        }
    }

    fn as_call_finally(t: &hir::Terminator) -> (BlockId, BlockId) {
        match t {
            hir::Terminator::CallFinally {
                funclet,
                continuation,
            } => (*funclet, *continuation),
            _ => panic!("expected Terminator::CallFinally"),
        }
    }

    #[test]
    fn throw_pops_a_reference_after_spilling() {
        // ldarg.0; call fib; ldnull; throw — the pending call tree
        // evaluates (into a spill temp) before the throw, and the throw
        // consumes the reference.
        let il = [0x02, 0x28, 0x01, 0x00, 0x00, 0x06, 0x14, 0x7A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Int]), &[]);
        let m = import(&info, &ee).expect("imports");
        assert_eq!(m.blocks.len(), 1);
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1, "just the spill store");
        let (dst, value) = store(&stmts[0]);
        assert_eq!(dst, LocalId(1), "the spill temp follows the one arg");
        assert!(matches!(value, hir::Expr::Call { .. }));
        match &m.blocks[0].terminator {
            hir::Terminator::Throw { exception } => {
                assert!(matches!(exception, hir::Expr::Const(Const::NullRef)));
            }
            _ => panic!("expected Terminator::Throw"),
        }
    }

    #[test]
    fn throw_of_a_non_reference_is_bad_il() {
        // ldc.i4.0; throw.
        assert!(matches!(
            import_ii(&[0x16, 0x7A]),
            Err(CompileError::BadIl(_))
        ));
        // throw on an empty stack underflows.
        assert!(matches!(import_ii(&[0x7A]), Err(CompileError::BadIl(_))));
    }

    #[test]
    fn leave_forms_decode_and_import() {
        // nop; leave.s -3 (-> 0) — the negative-delta short form loops to
        // the method start. No enclosing region: a plain Leave. (One
        // block: the nop falls through into the leave.)
        let (ee, info) = fixture(&[0x00, 0xDE, 0xFD], &sig(CorInfoType::Void, &[]), &[]);
        let m = import(&info, &ee).expect("imports");
        assert_eq!(m.blocks.len(), 1);
        assert_eq!(as_leave(&m.blocks[0].terminator), BlockId(0));

        // The long form (i32 delta): nop; leave -6 (-> 0).
        let il = [0x00, 0xDD, 0xFA, 0xFF, 0xFF, 0xFF];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Void, &[]), &[]);
        let m = import(&info, &ee).expect("imports");
        assert_eq!(m.blocks.len(), 1);
        assert_eq!(as_leave(&m.blocks[0].terminator), BlockId(0));
    }

    #[test]
    fn rethrow_and_endfilter_stay_unsupported() {
        let err = import_ii(&[0xFE, 0x1A]).err().expect("rethrow");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("rethrow")),
            "{err:?}"
        );
        let err = import_ii(&[0xFE, 0x11]).err().expect("endfilter");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("filter")),
            "{err:?}"
        );
    }

    #[test]
    fn try_finally_builds_a_call_finally_step_block() {
        let (ee, info) = eh_fixture(&TRY_FINALLY_IL, &[], &[finally_clause(0, 3, 3, 1)]);
        let m = import(&info, &ee).expect("imports");
        // Layout: [b0 (try body), S1 (step), L (leave), T (ret), H
        // (handler)]. The leave at offset 1 crosses the finally.
        assert_eq!(m.blocks.len(), 5);
        assert_eq!(as_jump(&m.blocks[0].terminator), BlockId(1));
        let (funclet, continuation) = as_call_finally(&m.blocks[1].terminator);
        assert_eq!(funclet, BlockId(4), "the handler group sits at the tail");
        assert_eq!(continuation, BlockId(2));
        assert!(
            m.blocks[1].stmts.is_empty() && m.blocks[2].stmts.is_empty(),
            "step blocks carry no statements"
        );
        assert_eq!(as_leave(&m.blocks[2].terminator), BlockId(3));
        assert!(matches!(
            m.blocks[3].terminator,
            hir::Terminator::Return { value: None }
        ));
        assert!(matches!(
            m.blocks[4].terminator,
            hir::Terminator::EndFinally
        ));
        assert_eq!(m.eh_regions.len(), 1);
        let region = &m.eh_regions[0];
        assert!(matches!(region.kind, hir::EhRegionKind::Finally));
        assert_eq!(region.try_start, BlockId(0));
        assert_eq!(region.try_end, BlockId(1), "the try is just b0");
        assert_eq!(region.handler_start, BlockId(4));
        assert_eq!(region.handler_end, BlockId(5));
    }

    #[test]
    fn try_catch_synthesizes_the_exception_store() {
        // 0: nop; 1: leave.s +3 (-> 6); 3: stloc.0; 4: leave.s +0 (-> 6);
        // 6: ret — a catch handler that pops the exception into local 0
        // and leaves.
        let il = [0x00, 0xDE, 0x03, 0x0A, 0xDE, 0x00, 0x2A];
        let (ee, info) = eh_fixture(
            &il,
            &[CorInfoType::Class],
            &[catch_clause(0, 3, 3, 3, 0x0200_0042)],
        );
        let m = import(&info, &ee).expect("imports");
        // Layout: [b0 (try), b2 (ret), H (catch)] — no finally, no steps.
        assert_eq!(m.blocks.len(), 3);
        assert_eq!(as_leave(&m.blocks[0].terminator), BlockId(1));
        // The catch handler's entry: Store(exc_temp, CatchArg) first,
        // then the IL stloc.0 reads the temp (the depth-1 entry stack).
        let handler = &m.blocks[2];
        let (dst, value) = store(&handler.stmts[0]);
        assert!(matches!(value, hir::Expr::CatchArg));
        assert_eq!(m.locals[dst.0 as usize].ty, Type::Ref);
        assert_eq!(m.locals[dst.0 as usize].kind, hir::LocalKind::Temp);
        let (dst0, value0) = store(&handler.stmts[1]);
        assert_eq!(dst0, LocalId(0), "the IL local");
        assert_eq!(as_local(value0), dst, "the caught exception read back");
        assert_eq!(handler.stmts[0].il_offset, IlOffset(3));
        // The handler's leave is the plain funclet-return form.
        assert_eq!(as_leave(&handler.terminator), BlockId(1));
        assert_eq!(m.eh_regions.len(), 1);
        let region = &m.eh_regions[0];
        match region.kind {
            hir::EhRegionKind::Catch { class_token } => {
                assert_eq!(class_token, 0x0200_0042, "the raw mdToken passes through")
            }
            _ => panic!("expected a Catch region"),
        }
        assert_eq!((region.try_start, region.try_end), (BlockId(0), BlockId(1)));
        assert_eq!(
            (region.handler_start, region.handler_end),
            (BlockId(2), BlockId(3))
        );
    }

    #[test]
    fn nested_finallys_build_an_innermost_first_step_chain() {
        // The B2 shape (target/eh-ref/B2o0.disasm.txt):
        // 0: nop; 1: nop; 2: leave.s +4 (-> 8); 4: endfinally;
        // 5: leave.s +1 (-> 8); 7: endfinally; 8: ret
        // inner finally: try [1,4), handler [4,5); outer: try [0,7),
        // handler [7,8). The leave at 2 crosses both; the leave at 5
        // crosses the outer only.
        let il = [0x00, 0x00, 0xDE, 0x04, 0xDC, 0xDE, 0x01, 0xDC, 0x2A];
        let (ee, info) = eh_fixture(
            &il,
            &[],
            &[finally_clause(1, 3, 4, 1), finally_clause(0, 7, 7, 1)],
        );
        let m = import(&info, &ee).expect("imports");
        // Original blocks: b0 [0,1), b1 [1,4) (leave), b2 [4,5) (inner
        // handler), b3 [5,7) (leave), b4 [7,8) (outer handler), b5 [8,9)
        // (ret). Layout:
        //   b0, b1, S1(inner hop of b1's chain), b3,
        //   S2(outer hop of b1's chain), L(b1's chain),
        //   S1'(b3's chain), L'(b3's chain), b5,
        //   then the handler groups: b2 (inner), b4 (outer).
        assert_eq!(m.blocks.len(), 11);
        // b1 -> S1 (id 2) -> S2 (id 4) -> L (id 5) -> b5 (id 8).
        assert_eq!(as_jump(&m.blocks[1].terminator), BlockId(2));
        assert_eq!(
            as_call_finally(&m.blocks[2].terminator),
            (BlockId(9), BlockId(4))
        );
        assert_eq!(
            as_call_finally(&m.blocks[4].terminator),
            (BlockId(10), BlockId(5))
        );
        assert_eq!(as_leave(&m.blocks[5].terminator), BlockId(8));
        // b3 -> S1' (id 6) -> L' (id 7) -> b5.
        assert_eq!(as_jump(&m.blocks[3].terminator), BlockId(6));
        assert_eq!(
            as_call_finally(&m.blocks[6].terminator),
            (BlockId(10), BlockId(7))
        );
        assert_eq!(as_leave(&m.blocks[7].terminator), BlockId(8));
        // The funclets.
        assert!(matches!(
            m.blocks[9].terminator,
            hir::Terminator::EndFinally
        ));
        assert!(matches!(
            m.blocks[10].terminator,
            hir::Terminator::EndFinally
        ));
        // Region table: the inner try is just b1; the outer try's main
        // run is b0..b3 — which the inner-exit step S1 (id 2) sits
        // inside, so its CallFinally's return address is attributed to
        // the outer try.
        assert_eq!(m.eh_regions.len(), 2);
        let inner = &m.eh_regions[0];
        assert_eq!((inner.try_start, inner.try_end), (BlockId(1), BlockId(2)));
        assert_eq!(
            (inner.handler_start, inner.handler_end),
            (BlockId(9), BlockId(10))
        );
        let outer = &m.eh_regions[1];
        assert_eq!(
            (outer.try_start, outer.try_end),
            (BlockId(0), BlockId(4)),
            "the outer try spans its main blocks plus the inner hop's step block"
        );
        assert_eq!(
            (outer.handler_start, outer.handler_end),
            (BlockId(10), BlockId(11))
        );
        // The outer-exit steps (ids 4, 6) lie outside the outer try.
        assert!(outer.try_end <= BlockId(4));
    }

    #[test]
    fn leave_from_a_catch_across_a_finally_aims_at_the_step_block() {
        // 0: nop; 1: nop; 2: leave.s +3 (-> 7); 4: stloc.0;
        // 5: leave.s +3 (-> 10); 7: leave.s +1 (-> 10); 9: endfinally;
        // 10: ret
        // inner catch: try [1,4), handler [4,7); outer finally:
        // try [0,9), handler [9,10). The catch handler's leave (offset
        // 5) crosses the outer finally: the VM runs no finallys on
        // resume (exceptionhandling.cpp ResumeAfterCatch), so the catch
        // funclet must return the STEP BLOCK's address — clr-abi.md's
        // ThreadAbort example shape.
        let il = [
            0x00, 0x00, 0xDE, 0x03, 0x0A, 0xDE, 0x03, 0xDE, 0x01, 0xDC, 0x2A,
        ];
        let (ee, info) = eh_fixture(
            &il,
            &[CorInfoType::Class],
            &[
                catch_clause(1, 3, 4, 3, 0x0200_0007),
                finally_clause(0, 9, 9, 1),
            ],
        );
        let m = import(&info, &ee).expect("imports");
        // Original blocks: b0 [0,1), b1 [1,4) (leave), b2 [4,7) (catch,
        // leave), b3 [7,9) (leave), b4 [9,10) (finally), b5 [10,11)
        // (ret). Only b2's and b3's leaves cross the outer finally.
        // Layout: b0, b1, b3, S(b2's chain), L, S'(b3's chain), L', b5,
        // then handlers: b2, b4.
        assert_eq!(m.blocks.len(), 10);
        // The catch handler is at the tail (id 8) and keeps a Leave —
        // but aimed at its chain's first step block (id 3), not at b5.
        assert_eq!(as_leave(&m.blocks[8].terminator), BlockId(3));
        assert_eq!(
            as_call_finally(&m.blocks[3].terminator),
            (BlockId(9), BlockId(4))
        );
        assert_eq!(as_leave(&m.blocks[4].terminator), BlockId(7));
        // b3's own chain: Jump to S' (id 5).
        assert_eq!(as_jump(&m.blocks[2].terminator), BlockId(5));
        assert_eq!(
            as_call_finally(&m.blocks[5].terminator),
            (BlockId(9), BlockId(6))
        );
        assert_eq!(as_leave(&m.blocks[6].terminator), BlockId(7));
        // The inner try body leave crosses nothing (its target is inside
        // the outer try): a plain Leave to b3 (id 2).
        assert_eq!(as_leave(&m.blocks[1].terminator), BlockId(2));
        // The outer try's main run: b0, b1, b3 — ids 0..3; no step of an
        // inner try trails it (the chains exit the outer try itself).
        let outer = &m.eh_regions[1];
        assert_eq!((outer.try_start, outer.try_end), (BlockId(0), BlockId(3)));
        // Handler groups in handler IL order: the catch (id 8), then the
        // finally (id 9).
        let catch = &m.eh_regions[0];
        assert_eq!(
            (catch.handler_start, catch.handler_end),
            (BlockId(8), BlockId(9))
        );
        assert_eq!(
            (outer.handler_start, outer.handler_end),
            (BlockId(9), BlockId(10))
        );
    }

    #[test]
    fn same_try_multiple_catches_keep_clause_order() {
        // 0: nop; 1: leave.s +4 (-> 7); 3: leave.s +2 (-> 7);
        // 5: leave.s +0 (-> 7); 7: ret — one try [0,3) with two catch
        // handlers [3,5) and [5,7).
        let il = [0x00, 0xDE, 0x04, 0xDE, 0x02, 0xDE, 0x00, 0x2A];
        let (ee, info) = eh_fixture(
            &il,
            &[],
            &[
                catch_clause(0, 3, 5, 2, 0x0200_00AA),
                catch_clause(0, 3, 3, 2, 0x0200_00BB),
            ],
        );
        let m = import(&info, &ee).expect("imports");
        // Handler groups at the tail follow the handler IL order (not
        // the clause registration order): [3,5) then [5,7).
        assert_eq!(m.blocks.len(), 4);
        assert_eq!(m.eh_regions.len(), 2);
        let (r0, r1) = (&m.eh_regions[0], &m.eh_regions[1]);
        // Both clauses share the try span.
        assert_eq!((r0.try_start, r0.try_end), (BlockId(0), BlockId(1)));
        assert_eq!((r1.try_start, r1.try_end), (BlockId(0), BlockId(1)));
        // Clause 0's handler ([5,7) in IL) is the second group.
        assert_eq!((r0.handler_start, r0.handler_end), (BlockId(3), BlockId(4)));
        assert_eq!((r1.handler_start, r1.handler_end), (BlockId(2), BlockId(3)));
        match r0.kind {
            hir::EhRegionKind::Catch { class_token } => assert_eq!(class_token, 0x0200_00AA),
            _ => panic!("expected Catch"),
        }
    }

    #[test]
    fn filter_and_fault_clauses_are_named_unsupported() {
        let mut filter = catch_clause(0, 3, 3, 1, 0);
        filter.Flags = ffi::CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_FILTER;
        let (ee, info) = eh_fixture(&TRY_FINALLY_IL, &[], &[filter]);
        let err = import(&info, &ee).err().expect("filter clause");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("EH filter clauses")),
            "{err:?}"
        );
        let mut fault = catch_clause(0, 3, 3, 1, 0);
        fault.Flags = ffi::CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_FAULT;
        let (ee, info) = eh_fixture(&TRY_FINALLY_IL, &[], &[fault]);
        let err = import(&info, &ee).err().expect("fault clause");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("EH fault clauses")),
            "{err:?}"
        );
    }

    #[test]
    fn malformed_eh_tables_are_bad_il() {
        // Partially overlapping try regions: neither nests.
        let (ee, info) = eh_fixture(
            &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A],
            &[],
            &[finally_clause(0, 5, 5, 1), finally_clause(3, 4, 7, 1)],
        );
        let err = import(&info, &ee).err().expect("overlapping tries");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("nested")),
            "{err:?}"
        );
        // A handler overlapping its own try.
        let (ee, info) = eh_fixture(&[0x00, 0xDC, 0x2A], &[], &[finally_clause(0, 2, 1, 1)]);
        let err = import(&info, &ee)
            .err()
            .expect("handler inside its own try");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("overlaps its own try")),
            "{err:?}"
        );
        // A region end past the IL stream.
        let (ee, info) = eh_fixture(&TRY_FINALLY_IL, &[], &[finally_clause(0, 3, 3, 9)]);
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));
        // A region boundary mid-instruction (the leave.s operand).
        let (ee, info) = eh_fixture(&TRY_FINALLY_IL, &[], &[finally_clause(0, 2, 3, 1)]);
        let err = import(&info, &ee).err().expect("mid-instruction boundary");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("instruction boundary")),
            "{err:?}"
        );
        // An empty region.
        let (ee, info) = eh_fixture(&TRY_FINALLY_IL, &[], &[finally_clause(0, 0, 3, 1)]);
        let err = import(&info, &ee).err().expect("empty region");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("empty EH region")),
            "{err:?}"
        );
        // A try covering only handler IL protects nothing.
        let il = [0x00, 0x00, 0xDC, 0xDC, 0x2A];
        let (ee, info) = eh_fixture(
            &il,
            &[],
            &[
                finally_clause(0, 2, 2, 1), // inner: try [0,2), handler [2,3)
                finally_clause(2, 1, 3, 1), // outer: try [2,3), handler [3,4)
            ],
        );
        let err = import(&info, &ee).err().expect("try of pure handler IL");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("protects no instructions")),
            "{err:?}"
        );
    }

    #[test]
    fn falling_into_a_catch_handler_is_bad_il() {
        // 0: nop; 1: stloc.0; 2: ret — the try body (try [0,1)) falls
        // through into the catch handler at 1 instead of leaving.
        let il = [0x00, 0x0A, 0x2A];
        let (ee, info) = eh_fixture(&il, &[CorInfoType::Class], &[catch_clause(0, 1, 1, 1, 1)]);
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));
    }

    #[test]
    fn endfinally_outside_a_finally_handler_is_bad_il() {
        // No clauses at all.
        assert!(matches!(import_ii(&[0xDC]), Err(CompileError::BadIl(_))));
        // Inside a CATCH handler.
        // 0: nop; 1: leave.s +2 (-> 5); 3: stloc.0; 4: endfinally; 5: ret.
        let il = [0x00, 0xDE, 0x02, 0x0A, 0xDC, 0x2A];
        let (ee, info) = eh_fixture(&il, &[CorInfoType::Class], &[catch_clause(0, 3, 3, 2, 1)]);
        let err = import(&info, &ee)
            .err()
            .expect("endfinally in a catch handler");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("endfinally")),
            "{err:?}"
        );
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
    fn conv_u1_u2_expand_to_zero_extending_shift_pairs() {
        // ldarg.0; conv.u1; ret — ((v << 24) >>> 24): the logical
        // back-shift zero-fills (conv.i1's arithmetic twin re-signs).
        let m = import_ii(&[0x02, 0xD2, 0x2A]).expect("imports");
        let (op, shl, count) = as_binary(return_value(&m, 0));
        assert_eq!(op, BinaryOp::UShr, "logical shift right zero-extends");
        assert_eq!(as_i32(count), 24);
        let (op, value, _) = as_binary(shl);
        assert_eq!(op, BinaryOp::Shl);
        assert_eq!(as_local(value), LocalId(0));

        // conv.u2: the same pair with shifts of 16.
        let m = import_ii(&[0x02, 0xD1, 0x2A]).expect("imports");
        let (op, shl, count) = as_binary(return_value(&m, 0));
        assert_eq!(op, BinaryOp::UShr);
        assert_eq!(as_i32(count), 16);
        let (op, _, count) = as_binary(shl);
        assert_eq!(op, BinaryOp::Shl);
        assert_eq!(as_i32(count), 16);
    }

    #[test]
    fn conv_u_zero_extends_to_native_int() {
        // conv.u of an i32 arg: an unsigned Conv node to NativeInt.
        let (ee, info) = fixture(
            &[0x02, 0xE0, 0x2A],
            &sig(CorInfoType::NativeInt, &[CorInfoType::Int]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (to, overflow, unsigned, arg) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::NativeInt);
        assert!(!overflow && unsigned);
        assert_eq!(as_local(arg), LocalId(0));

        // A 64-bit operand is already there: no node, re-typed NativeInt.
        let (ee, info) = fixture(
            &[0x02, 0xE0, 0x2A],
            &sig(CorInfoType::NativeInt, &[CorInfoType::Long]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        assert_eq!(as_local(return_value(&m, 0)), LocalId(0));

        // From a float it is the saturating unsigned conversion
        // (step_10.11): the Conv node keeps the unsigned flag; the
        // HIR→LIR lowering expands the sequence.
        let (ee, info) = fixture(
            &[0x02, 0xE0, 0x2A],
            &sig(CorInfoType::NativeInt, &[CorInfoType::Double]),
            &[],
        );
        let m = import(&info, &ee).expect("conv.u of a double imports");
        let (to, overflow, unsigned, arg) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::NativeInt);
        assert!(!overflow && unsigned);
        assert_eq!(as_local(arg), LocalId(0));

        // conv.u8 of a native-int operand: the identity, but the IL stack
        // type becomes Int64 (a NativeInt source re-types) — otherwise
        // `ret` sees a type mismatch (found by tests/convu.cs).
        let (ee, info) = fixture(
            &[0x02, 0x6E, 0x2A],
            &sig(CorInfoType::Long, &[CorInfoType::NativeInt]),
            &[],
        );
        let m = import(&info, &ee).expect("conv.u8 of native int imports");
        assert_eq!(as_local(return_value(&m, 0)), LocalId(0));
    }

    #[test]
    fn conv_u8_from_a_float_is_the_unsigned_conv_node() {
        // conv.u8 of a double: no longer rejected (step_10.11) — the
        // saturating expansion happens at the HIR→LIR lowering.
        let (ee, info) = fixture(
            &[0x02, 0x6E, 0x2A],
            &sig(CorInfoType::Long, &[CorInfoType::Double]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (to, overflow, unsigned, arg) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::Int64);
        assert!(!overflow && unsigned);
        assert_eq!(as_local(arg), LocalId(0));

        // Same from a float32.
        let (ee, info) = fixture(
            &[0x02, 0x6E, 0x2A],
            &sig(CorInfoType::Long, &[CorInfoType::Float]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (_, _, unsigned, _) = as_conv(return_value(&m, 0));
        assert!(unsigned);
    }

    #[test]
    fn conv_u1_u2_from_a_float_clamp_in_the_float_domain() {
        // conv.u1 of a double (step_10.11): the saturating small-type
        // semantics clamp the float source to [0, 255] BEFORE the
        // conversion — maxs(v, 0) then mins(…, 255) — then the usual
        // shift pair narrows the (in-range) Int32.
        let (ee, info) = fixture(
            &[0x02, 0xD2, 0x2A],
            &sig(CorInfoType::Int, &[CorInfoType::Double]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (op, shl, count) = as_binary(return_value(&m, 0));
        assert_eq!(op, BinaryOp::UShr);
        assert_eq!(as_i32(count), 24);
        let (op, value, _) = as_binary(shl);
        assert_eq!(op, BinaryOp::Shl);
        let (to, _, _, arg) = as_conv(value);
        assert_eq!(to, Type::Int32);
        // The Conv's operand is mins(maxs(d, 0.0), 255.0).
        let (op, clamped, limit) = as_binary(arg);
        assert_eq!(op, BinaryOp::MinF);
        assert!(matches!(limit, hir::Expr::Const(Const::Double(v)) if *v == 255.0));
        let (op, value, zero) = as_binary(clamped);
        assert_eq!(op, BinaryOp::MaxF);
        assert_eq!(as_local(value), LocalId(0));
        assert!(matches!(zero, hir::Expr::Const(Const::Double(v)) if *v == 0.0));

        // conv.u2 of a float32: the constants take the source's width.
        let (ee, info) = fixture(
            &[0x02, 0xD1, 0x2A],
            &sig(CorInfoType::Int, &[CorInfoType::Float]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (_, shl, count) = as_binary(return_value(&m, 0));
        assert_eq!(as_i32(count), 16);
        let (_, value, _) = as_binary(shl);
        let (_, _, _, arg) = as_conv(value);
        let (op, _, limit) = as_binary(arg);
        assert_eq!(op, BinaryOp::MinF);
        assert!(matches!(limit, hir::Expr::Const(Const::Float(v)) if *v == 65535.0));
    }

    #[test]
    fn conv_i_sign_extends_to_native_int() {
        // conv.i of an i32 arg (step_10.11): a SIGNED Conv node to
        // NativeInt — the conv.u twin zero-extends.
        let (ee, info) = fixture(
            &[0x02, 0xD3, 0x2A],
            &sig(CorInfoType::NativeInt, &[CorInfoType::Int]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (to, overflow, unsigned, arg) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::NativeInt);
        assert!(!overflow && !unsigned);
        assert_eq!(as_local(arg), LocalId(0));

        // A 64-bit operand is the identity, re-typed NativeInt.
        let (ee, info) = fixture(
            &[0x02, 0xD3, 0x2A],
            &sig(CorInfoType::NativeInt, &[CorInfoType::Long]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        assert_eq!(as_local(return_value(&m, 0)), LocalId(0));

        // From a float the saturating signed conversion is the still-open
        // 10.2 follow-up — stays unsupported.
        let (ee, info) = fixture(
            &[0x02, 0xD3, 0x2A],
            &sig(CorInfoType::NativeInt, &[CorInfoType::Double]),
            &[],
        );
        assert!(matches!(
            import(&info, &ee),
            Err(CompileError::Unsupported(_))
        ));
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

    // --- step_10.10: ldtoken / sizeof ---

    const LDTYPE_TOKEN: u32 = 0x0200_0007;
    const LDMETHOD_TOKEN: u32 = 0x0600_0009;
    const LDFIELD_TOKEN: u32 = 0x0400_0003;

    /// A MockEe with the canned handle-struct class (the
    /// RuntimeTypeHandle/RuntimeMethodHandle/RuntimeFieldHandle stand-in:
    /// 8 bytes, one reference cell, one integer eightbyte) registered as
    /// `get_token_type_as_handle`'s answer.
    fn ldtoken_fixture(il: &[u8]) -> (MockEe, MethodInfo) {
        let (mut ee, info) = fixture(il, &sig(CorInfoType::Int, &[]), &[]);
        let handle_class = ee.add_class(
            8,
            8,
            &[(0, false)],
            Some(rokajit_ee::mock::sysv_descriptor(&[(
                ffi::SystemVClassificationType_SystemVClassificationTypeInteger,
                8,
            )])),
        );
        ee.token_type_class = Some(handle_class);
        (ee, info)
    }

    /// `ldtoken <tok>; pop; ldc.i4.0; ret` — the pushed handle struct is
    /// discarded; the conversion helper call stays as an Eval (a call is
    /// observable).
    fn ldtoken_eval(m: &hir::Method) -> (&CallTarget<hir::Expr>, &CallSig, &[hir::Expr]) {
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::Eval(hir::Expr::Call { target, sig, args }) => (target, sig, args),
            _ => panic!("expected an Eval of the conversion helper call"),
        }
    }

    #[test]
    fn ldtoken_of_a_type_embeds_the_handle_and_calls_the_type_helper() {
        let il = [0xD0, 0x07, 0x00, 0x00, 0x02, 0x26, 0x16, 0x2A];
        let (mut ee, info) = ldtoken_fixture(&il);
        let cls = ee.add_class(16, 8, &[], None);
        ee.class_tokens.insert(LDTYPE_TOKEN, cls);
        let m = import(&info, &ee).expect("imports");
        let handle_class = ee.token_type_class.unwrap();

        let (target, sig, args) = ldtoken_eval(&m);
        assert!(
            matches!(target, CallTarget::Helper(h) if *h == CorInfoHelpFunc::TYPEHANDLE_TO_RUNTIMETYPEHANDLE)
        );
        assert_eq!(sig.ret, Type::Struct(handle_class));
        assert_eq!(sig.args, vec![Type::NativeInt]);
        // The embedded handle constant: the mock cans 0x7A7A_0000 + token.
        assert!(
            matches!(&args[0], hir::Expr::Const(Const::NativeInt(v)) if *v == (0x7A7A_0000usize + LDTYPE_TOKEN as usize) as isize)
        );
        // The handle struct's layout was queried into the side table.
        assert!(m.struct_layouts.contains_key(&handle_class));
    }

    #[test]
    fn ldtoken_of_a_method_uses_the_methoddesc_helper() {
        let il = [0xD0, 0x09, 0x00, 0x00, 0x06, 0x26, 0x16, 0x2A];
        let (mut ee, info) = ldtoken_fixture(&il);
        ee.add_method(LDMETHOD_TOKEN, sig(CorInfoType::Void, &[]));
        let m = import(&info, &ee).expect("imports");
        let (target, _, _) = ldtoken_eval(&m);
        assert!(
            matches!(target, CallTarget::Helper(h) if *h == CorInfoHelpFunc::METHODDESC_TO_STUBRUNTIMEMETHOD)
        );
    }

    #[test]
    fn ldtoken_of_a_field_uses_the_fielddesc_helper() {
        let il = [0xD0, 0x03, 0x00, 0x00, 0x04, 0x26, 0x16, 0x2A];
        let (mut ee, info) = ldtoken_fixture(&il);
        ee.add_field(LDFIELD_TOKEN, CorInfoType::Int, 8);
        let m = import(&info, &ee).expect("imports");
        let (target, _, _) = ldtoken_eval(&m);
        assert!(
            matches!(target, CallTarget::Helper(h) if *h == CorInfoHelpFunc::FIELDDESC_TO_STUBRUNTIMEFIELD)
        );
    }

    #[test]
    fn ldtoken_rejection_forms_are_named_unsupported() {
        let il = [0xD0, 0x07, 0x00, 0x00, 0x02, 0x26, 0x16, 0x2A];
        // A generic-context runtime lookup is the shared-generics step.
        let (mut ee, info) = ldtoken_fixture(&il);
        let cls = ee.add_class(16, 8, &[], None);
        ee.class_tokens.insert(LDTYPE_TOKEN, cls);
        ee.embed_runtime_lookup = true;
        let err = import(&info, &ee).err().expect("rejected");
        assert!(
            matches!(err, CompileError::Unsupported(m) if m.contains("generic-context runtime lookup"))
        );

        // An indirection cell needs load/reloc plumbing tier 0 lacks.
        let (mut ee, info) = ldtoken_fixture(&il);
        let cls = ee.add_class(16, 8, &[], None);
        ee.class_tokens.insert(LDTYPE_TOKEN, cls);
        ee.embed_indirection = true;
        let err = import(&info, &ee).err().expect("rejected");
        assert!(matches!(err, CompileError::Unsupported(m) if m.contains("indirection cell")));

        // No resolved handles at all is bad IL.
        let (mut ee, info) = ldtoken_fixture(&il);
        ee.token_type_class = None;
        let err = import(&info, &ee).err().expect("rejected");
        assert!(matches!(err, CompileError::BadIl(m) if m.contains("did not resolve")));
    }

    #[test]
    fn sizeof_folds_to_the_ee_class_size() {
        // sizeof <tok>; ret — an Int32 constant straight from getClassSize.
        let il = [0xFE, 0x1C, 0x07, 0x00, 0x00, 0x02, 0x2A];
        let (mut ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        let cls = ee.add_class(40, 8, &[], None);
        ee.class_tokens.insert(LDTYPE_TOKEN, cls);
        let m = import(&info, &ee).expect("imports");
        assert_eq!(as_i32(return_value(&m, 0)), 40);
    }

    #[test]
    fn sizeof_of_an_unresolvable_token_is_bad_il() {
        let il = [0xFE, 0x1C, 0x07, 0x00, 0x00, 0x02, 0x2A];
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        let err = import(&info, &ee).err().expect("rejected");
        assert!(matches!(err, CompileError::BadIl(m) if m.contains("did not resolve")));
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

        // conv.u8 from a float: the unsigned Conv node (step_10.11 — the
        // saturating expansion happens at the HIR→LIR lowering).
        let (ee, info) = fixture(
            &[0x02, 0x6E, 0x2A],
            &sig(CorInfoType::Long, &[CorInfoType::Double]),
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let (to, _, unsigned, _) = as_conv(return_value(&m, 0));
        assert_eq!(to, Type::Int64);
        assert!(unsigned);
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
            ret_class: None,
            arg_classes: Vec::new(),
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
            hir::Expr::Load {
                addr, offset, ty, ..
            } => {
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
            ret_class: None,
            arg_classes: Vec::new(),
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
                ..
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
                ret_class: None,
                arg_classes: Vec::new(),
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
                ret_class: None,
                arg_classes: Vec::new(),
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
                ret_class: None,
                arg_classes: Vec::new(),
            },
        );
        ee.new_helper = Some(CorInfoHelpFunc::NEWARR_1_PTR);
        assert!(matches!(
            import(&info, &ee),
            Err(CompileError::Unsupported(_))
        ));
    }

    #[test]
    fn newobj_of_a_value_class_constructs_in_place() {
        // csc's `new S(args)` for a struct: no allocation — a fresh
        // zeroed struct temp, the constructor on its address, the temp as
        // the pushed value (GitHub_18362's `new Complex(real, imaginary)`
        // inside System.Numerics.Complex.Conjugate).
        let (mut ee, c) = struct_ee(1, &[], None);
        ee.add_method(
            CTOR_TOKEN,
            MockSig {
                ret: CorInfoType::Void,
                args: vec![CorInfoType::Bool],
                has_this: true,
                ret_class: None,
                arg_classes: Vec::new(),
            },
        );
        ee.class_tokens.insert(0x0600_0004, c);
        let entry = sig(CorInfoType::Void, &[]);
        // ldc.i4.1; newobj CTOR; pop; ret
        let info = struct_info(
            &mut ee,
            &[0x17, 0x73, 0x04, 0x00, 0x00, 0x06, 0x26, 0x2A],
            &entry,
            &[],
            &[],
        );
        let m = import(&info, &ee).expect("newobj of a value class imports");
        // No allocation helper was consulted.
        assert!(ee.new_helper.is_none(), "a value class allocates nothing");
        let stmts = &m.blocks[0].stmts;
        // The zeroed temp (the only local — no args, no IL locals).
        match &stmts[0].kind {
            hir::StmtKind::BlockZero { addr, class } => {
                assert_eq!(*class, c);
                assert_eq!(as_local_addr(addr), LocalId(0));
            }
            _ => panic!("expected the BlockZero of the fresh temp"),
        }
        // The constructor on the temp's address, argument in push order.
        match &stmts[1].kind {
            hir::StmtKind::Eval(hir::Expr::Call { target, sig, args }) => {
                assert!(matches!(target, CallTarget::Direct(_)));
                assert!(sig.has_this);
                assert_eq!(as_local_addr(&args[0]), LocalId(0), "byref this");
                assert!(matches!(&args[1], hir::Expr::Const(Const::Int32(1))));
            }
            _ => panic!("expected the constructor call"),
        }
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

        // A byref-typed field (a `ref` field of a ref struct) stays out.
        let (mut ee, info) = object_fixture(&il);
        ee.fields.get_mut(&FIELD_TOKEN).unwrap().ty = CorInfoType::ByRef;
        let err = import(&info, &ee)
            .err()
            .expect("byref field is unsupported");
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

    // --- step_10.7: the statics pack (ldsfld/ldsflda/stsfld) ---

    #[test]
    fn ldsfld_loads_through_the_static_address() {
        // ldsfld int; ret — a Load at offset 0 through the EE-answered
        // final address, a raw NativeInt constant.
        let il = [0x7E, 0x01, 0x00, 0x00, 0x04, 0x2A];
        let (mut ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        ee.add_static_field(FIELD_TOKEN, CorInfoType::Int);
        let m = import(&info, &ee).expect("imports");
        match return_value(&m, 0) {
            hir::Expr::Load {
                addr,
                offset,
                ty,
                access,
            } => {
                assert_eq!(*offset, 0);
                assert_eq!(*ty, Type::Int32);
                assert_eq!(*access, MemAccess::Natural);
                assert!(
                    matches!(**addr, hir::Expr::Const(Const::NativeInt(_))),
                    "the field's final address is a pointer constant"
                );
            }
            _ => panic!("expected Expr::Load"),
        }
    }

    #[test]
    fn ldsflda_pushes_the_static_address_as_a_byref() {
        // ldsflda; stloc.0 — the address stores into a ByRef local.
        let il = [0x7F, 0x01, 0x00, 0x00, 0x04, 0x0A, 0x16, 0x2A];
        let (mut ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[CorInfoType::ByRef]);
        ee.add_static_field(FIELD_TOKEN, CorInfoType::Int);
        let m = import(&info, &ee).expect("imports");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(0));
        assert!(matches!(value, hir::Expr::Const(Const::NativeInt(_))));
    }

    #[test]
    fn stsfld_of_an_int_stores_through_the_static_address() {
        // ldc.i4.s 42; stsfld int; ldc.i4.0; ret.
        let il = [0x1F, 0x2A, 0x80, 0x01, 0x00, 0x00, 0x04, 0x16, 0x2A];
        let (mut ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        ee.add_static_field(FIELD_TOKEN, CorInfoType::Int);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1, "no cctor flag — no INITCLASS call");
        match &stmts[0].kind {
            hir::StmtKind::StoreInd {
                addr,
                offset,
                value,
                access,
            } => {
                assert_eq!(*offset, 0);
                assert_eq!(*access, MemAccess::Natural);
                assert!(matches!(addr, hir::Expr::Const(Const::NativeInt(_))));
                assert_eq!(as_i32(value), 42);
            }
            _ => panic!("expected StmtKind::StoreInd"),
        }
    }

    #[test]
    fn stsfld_of_a_reference_goes_through_the_write_barrier() {
        // stsfld <ref field> of null — the checked write barrier with the
        // static address as arg 0 (a raw NativeInt const, not a Ref).
        let il = [0x14, 0x80, 0x02, 0x00, 0x00, 0x04, 0x16, 0x2A];
        let (mut ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        ee.add_static_field(REF_FIELD_TOKEN, CorInfoType::Class);
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
                assert!(matches!(args[0], hir::Expr::Const(Const::NativeInt(_))));
                assert!(matches!(args[1], hir::Expr::Const(Const::NullRef)));
            }
            _ => panic!("expected the barrier helper Eval"),
        }
    }

    #[test]
    fn statics_emit_the_init_class_helper_before_the_access() {
        // ldsfld with the INITCLASS flag and a USE_HELPER verdict: the
        // INITCLASS Eval is the block's only statement; the load stays a
        // tree (it reads the field after the cctor runs).
        let il = [0x7E, 0x01, 0x00, 0x00, 0x04, 0x2A];
        let (mut ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        ee.add_static_field(FIELD_TOKEN, CorInfoType::Int);
        ee.fields.get_mut(&FIELD_TOKEN).unwrap().init_class = true;
        ee.init_class_result = CorInfoInitClassResult::USE_HELPER;
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1);
        match &stmts[0].kind {
            hir::StmtKind::Eval(hir::Expr::Call { target, sig, args }) => {
                assert!(
                    matches!(target, CallTarget::Helper(h) if *h == CorInfoHelpFunc::INITCLASS),
                    "the INITCLASS helper"
                );
                assert_eq!(sig.ret, Type::Void);
                assert_eq!(args.len(), 1, "the embedded owning-class handle");
            }
            _ => panic!("expected the INITCLASS helper call"),
        }

        // The same flag with a NOT_REQUIRED verdict (the default) emits
        // nothing.
        let (mut ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        ee.add_static_field(FIELD_TOKEN, CorInfoType::Int);
        ee.fields.get_mut(&FIELD_TOKEN).unwrap().init_class = true;
        let m = import(&info, &ee).expect("imports");
        assert!(m.blocks[0].stmts.is_empty());
    }

    #[test]
    fn stsfld_with_init_class_orders_value_then_cctor_then_store() {
        // ldc.i4.s 42; stsfld int; ldc.i4.0; ret with a cctor: the value
        // materializes into a temp, THEN the INITCLASS helper runs, THEN
        // the store (the uniform value → cctor → store order).
        let il = [0x1F, 0x2A, 0x80, 0x01, 0x00, 0x00, 0x04, 0x16, 0x2A];
        let (mut ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        ee.add_static_field(FIELD_TOKEN, CorInfoType::Int);
        ee.fields.get_mut(&FIELD_TOKEN).unwrap().init_class = true;
        ee.init_class_result = CorInfoInitClassResult::USE_HELPER;
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 3, "value temp store, INITCLASS, the store");
        let (dst, value) = store(&stmts[0]);
        assert_eq!(dst, LocalId(0), "the value temp is the first local");
        assert_eq!(as_i32(value), 42);
        assert!(matches!(
            &stmts[1].kind,
            hir::StmtKind::Eval(hir::Expr::Call { target: CallTarget::Helper(h), .. })
                if *h == CorInfoHelpFunc::INITCLASS
        ));
        match &stmts[2].kind {
            hir::StmtKind::StoreInd { value, .. } => {
                assert_eq!(as_local(value), LocalId(0), "the temp's value stores");
            }
            _ => panic!("expected StmtKind::StoreInd"),
        }
    }

    #[test]
    fn statics_gate_instance_fields_and_unsupported_accessors() {
        // stsfld of an instance field: BadIl (RyuJIT BADCODEs it too).
        let il = [0x16, 0x80, 0x01, 0x00, 0x00, 0x04, 0x16, 0x2A];
        let (mut ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        ee.add_field(FIELD_TOKEN, CorInfoType::Int, 16);
        assert!(matches!(import(&info, &ee), Err(CompileError::BadIl(_))));

        // A thread-static accessor is a named Unsupported.
        let il = [0x7E, 0x01, 0x00, 0x00, 0x04, 0x2A];
        let (mut ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        ee.add_static_field(FIELD_TOKEN, CorInfoType::Int);
        ee.fields.get_mut(&FIELD_TOKEN).unwrap().accessor =
            Some(ffi::CORINFO_FIELD_ACCESSOR_CORINFO_FIELD_STATIC_TLS_MANAGED);
        let err = import(&info, &ee).err().expect("TLS statics are out");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("ThreadStatic")),
            "{err:?}"
        );
    }

    #[test]
    fn boxed_static_struct_load_indirects_through_the_cell() {
        // STATIC_IN_HEAP: the cell holds the frozen box object; the field
        // address is box + TARGET_POINTER_SIZE (8). A struct-typed static
        // loads as the StructVal of that address.
        let (mut ee, class) = struct_ee(8, &[], None);
        ee.add_static_field(FIELD_TOKEN, CorInfoType::ValueClass);
        let field = &mut ee.fields.get_mut(&FIELD_TOKEN).unwrap();
        field.value_class = Some(class);
        field.in_heap = true;
        let entry = sig(CorInfoType::Void, &[]);
        // ldsfld S; pop; ret
        let info = struct_info(
            &mut ee,
            &[0x7E, 0x01, 0x00, 0x00, 0x04, 0x26, 0x2A],
            &entry,
            &[],
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1, "the popped struct value Evals");
        match &stmts[0].kind {
            hir::StmtKind::Eval(value) => {
                let (addr, c) = as_struct_val(value);
                assert_eq!(c, class);
                match addr {
                    hir::Expr::FieldAddr { obj, offset, .. } => {
                        assert_eq!(*offset, 8, "past the object header");
                        match &**obj {
                            hir::Expr::Load {
                                addr, offset, ty, ..
                            } => {
                                assert_eq!(*offset, 0);
                                assert_eq!(*ty, Type::Ref);
                                assert!(matches!(**addr, hir::Expr::Const(Const::NativeInt(_))));
                            }
                            _ => panic!("expected the cell Load"),
                        }
                    }
                    _ => panic!("expected the box-data FieldAddr"),
                }
            }
            _ => panic!("expected the popped-value Eval"),
        }
    }

    #[test]
    fn float_and_subint_fields_import_with_their_access_shape() {
        // Regression for the post-10.6 triage MISMATCHES
        // (JIT/Regression_3/GitHub_18362, JIT/jit64/regress/vsw/471729):
        // float and sub-Int32 fields used to be rejected by the field-type
        // gate, and the test's own try/catch swallowed the resulting
        // InvalidProgramException into a wrong exit code/stdout.
        //
        // A Double field loads as a Float64-typed value at the natural
        // width: `double get_d() { return this.d; }`.
        let il = [0x02, 0x7B, 0x01, 0x00, 0x00, 0x04, 0x2A];
        let entry = MockSig {
            ret: CorInfoType::Double,
            args: vec![],
            has_this: true,
            ret_class: None,
            arg_classes: Vec::new(),
        };
        let (mut ee, info) = fixture(&il, &entry, &[]);
        ee.add_field(FIELD_TOKEN, CorInfoType::Double, 16);
        let m = import(&info, &ee).expect("a double field imports");
        match return_value(&m, 0) {
            hir::Expr::Load {
                addr,
                offset,
                ty,
                access,
            } => {
                assert_eq!(*offset, 16);
                assert_eq!(*ty, Type::Double);
                assert_eq!(*access, MemAccess::Natural);
                assert_eq!(as_local(as_null_check(addr)), LocalId(0));
            }
            _ => panic!("expected Expr::Load"),
        }

        // A bool field: Int32 on the stack, a 1-byte zero-extending cell.
        let entry = MockSig {
            ret: CorInfoType::Int,
            args: vec![],
            has_this: true,
            ret_class: None,
            arg_classes: Vec::new(),
        };
        let (mut ee, info) = fixture(&il, &entry, &[]);
        ee.add_field(FIELD_TOKEN, CorInfoType::Bool, 16);
        let m = import(&info, &ee).expect("a bool field imports");
        match return_value(&m, 0) {
            hir::Expr::Load { ty, access, .. } => {
                assert_eq!(*ty, Type::Int32);
                assert_eq!(*access, MemAccess::U8);
            }
            _ => panic!("expected Expr::Load"),
        }

        // Every sub-Int32 metadata type maps to its width and ECMA-335
        // §III.1.1.1 extension (I1/I2 sign, BOOLEAN/CHAR/U1/U2 zero).
        for (cor, want) in [
            (CorInfoType::Byte, MemAccess::I8),
            (CorInfoType::UByte, MemAccess::U8),
            (CorInfoType::Short, MemAccess::I16),
            (CorInfoType::UShort, MemAccess::U16),
            (CorInfoType::Char, MemAccess::U16),
        ] {
            let (mut ee, info) = fixture(&il, &entry, &[]);
            ee.add_field(FIELD_TOKEN, cor, 16);
            let m = import(&info, &ee).expect("sub-Int32 field imports");
            match return_value(&m, 0) {
                hir::Expr::Load { ty, access, .. } => {
                    assert_eq!(*ty, Type::Int32);
                    assert_eq!(*access, want, "{cor:?}");
                }
                _ => panic!("expected Expr::Load"),
            }
        }

        // `stfld` of a bool: the Int32 stack value stores through the
        // narrow cell.
        let il = [0x02, 0x03, 0x7D, 0x01, 0x00, 0x00, 0x04, 0x2A];
        let entry = MockSig {
            ret: CorInfoType::Void,
            args: vec![CorInfoType::Int],
            has_this: true,
            ret_class: None,
            arg_classes: Vec::new(),
        };
        let (mut ee, info) = fixture(&il, &entry, &[]);
        ee.add_field(FIELD_TOKEN, CorInfoType::Bool, 16);
        let m = import(&info, &ee).expect("a bool stfld imports");
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::StoreInd {
                addr,
                offset,
                value,
                access,
            } => {
                assert_eq!(*offset, 16);
                assert_eq!(*access, MemAccess::U8);
                assert_eq!(as_local(as_null_check(addr)), LocalId(0));
                assert_eq!(as_local(value), LocalId(1));
            }
            _ => panic!("expected StmtKind::StoreInd"),
        }
    }

    #[test]
    fn shared_generic_methods_are_rejected_by_the_callconv_gate() {
        // Regression for the post-10.6 triage MISMATCH JIT/opt/Enum/shared:
        // a static method on a generic type carries
        // CORINFO_CALLCONV_PARAMTYPE (a hidden instantiation argument) —
        // above the 4-bit convention mask, so the generic gate missed it
        // and the method compiled with no generic context, silently
        // producing the wrong answer.
        let (ee, mut info) = fixture(&[0x2A], &sig(CorInfoType::Void, &[]), &[]);
        info.args.callConv |= ffi::CorInfoCallConv_CORINFO_CALLCONV_PARAMTYPE;
        let err = import(&info, &ee).err().expect("PARAMTYPE is generic");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("generic")),
            "{err:?}"
        );
    }

    // --- step_10.9: value types ---

    use rokajit_ee::handles::ClassHandle;
    use rokajit_ee::mock::sysv_descriptor;

    const CLASS_TOKEN: u32 = 0x0200_0001;
    const STRUCT_FIELD_TOKEN: u32 = 0x0400_0009;
    const STRUCT_FN_TOKEN: u32 = 0x0600_0010;

    const INT_EB: ffi::SystemVClassificationType =
        ffi::SystemVClassificationType_SystemVClassificationTypeInteger;

    /// A MockEe with one registered value class and a class token for it,
    /// plus the usual three canned methods.
    fn struct_ee(
        size: u32,
        gc_cells: &[(u32, bool)],
        sysv: Option<ffi::SYSTEMV_AMD64_CORINFO_STRUCT_REG_PASSING_DESCRIPTOR>,
    ) -> (MockEe, ClassHandle) {
        let mut ee = MockEe::default();
        let class = ee.add_class(size, 8, gc_cells, sysv);
        ee.class_tokens.insert(CLASS_TOKEN, class);
        (ee, class)
    }

    fn struct_info(
        ee: &mut MockEe,
        il: &[u8],
        entry: &MockSig,
        locals: &[CorInfoType],
        local_classes: &[Option<ClassHandle>],
    ) -> MethodInfo {
        MethodInfo {
            ftn: MethodHandle::from_raw(1usize as ffi::CORINFO_METHOD_HANDLE).unwrap(),
            il: il.to_vec(),
            max_stack: 8,
            eh_count: 0,
            init_locals: false,
            args: ee.make_method_sig(entry),
            locals: ee.make_locals_sig_with_classes(locals, local_classes),
        }
    }

    fn as_struct_val(e: &hir::Expr) -> (&hir::Expr, ClassHandle) {
        match e {
            hir::Expr::StructVal { addr, class } => (addr, *class),
            _ => panic!("expected Expr::StructVal"),
        }
    }

    #[test]
    fn initobj_imports_as_a_block_zero() {
        // ldloca.0; initobj C; ret — the local's slot zeroes.
        let (mut ee, c) = struct_ee(12, &[], None);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(
            &mut ee,
            &[0x12, 0x00, 0xFE, 0x15, 0x01, 0x00, 0x00, 0x02, 0x2A],
            &entry,
            &[CorInfoType::ValueClass],
            &[Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        assert_eq!(m.locals[0].ty, Type::Struct(c));
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::BlockZero { addr, class } => {
                assert_eq!(*class, c);
                assert_eq!(as_local_addr(addr), LocalId(0));
            }
            _ => panic!("expected StmtKind::BlockZero"),
        }
        // The layout was queried into the side table.
        assert_eq!(m.struct_layouts[&c].size, 12);
    }

    #[test]
    fn ldobj_stobj_cpobj_import_as_struct_copies() {
        let (mut ee, c) = struct_ee(8, &[], None);
        // ldloca.0; ldobj C; stloc.1; ret — ldobj yields a StructVal of
        // the address; the stloc stores it.
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(
            &mut ee,
            &[0x12, 0x00, 0x71, 0x01, 0x00, 0x00, 0x02, 0x0B, 0x2A],
            &entry,
            &[CorInfoType::ValueClass, CorInfoType::ValueClass],
            &[Some(c), Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(1));
        let (addr, class) = as_struct_val(value);
        assert_eq!(class, c);
        assert_eq!(as_local_addr(addr), LocalId(0));

        // ldloca.0; ldloc.1; stobj C; ret — a StoreInd of the StructVal.
        let (mut ee, c) = struct_ee(8, &[], None);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(
            &mut ee,
            &[0x12, 0x00, 0x07, 0x81, 0x01, 0x00, 0x00, 0x02, 0x2A],
            &entry,
            &[CorInfoType::ValueClass, CorInfoType::ValueClass],
            &[Some(c), Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::StoreInd {
                addr,
                offset,
                value,
                ..
            } => {
                assert_eq!(as_local_addr(addr), LocalId(0));
                assert_eq!(*offset, 0);
                let (src, _) = as_struct_val(value);
                assert_eq!(as_local_addr(src), LocalId(1));
            }
            _ => panic!("expected StmtKind::StoreInd"),
        }

        // ldloca.0; ldloca.1; cpobj C; ret — same shape, both addresses.
        let (mut ee, c) = struct_ee(8, &[], None);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(
            &mut ee,
            &[0x12, 0x00, 0x12, 0x01, 0x70, 0x01, 0x00, 0x00, 0x02, 0x2A],
            &entry,
            &[CorInfoType::ValueClass, CorInfoType::ValueClass],
            &[Some(c), Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::StoreInd {
                addr,
                offset,
                value,
                ..
            } => {
                assert_eq!(as_local_addr(addr), LocalId(0));
                assert_eq!(*offset, 0);
                let (src, _) = as_struct_val(value);
                assert_eq!(as_local_addr(src), LocalId(1));
            }
            _ => panic!("expected StmtKind::StoreInd"),
        }
    }

    #[test]
    fn struct_opcodes_reject_a_non_value_class() {
        // A class token the mock resolves to nothing value-class-like.
        let mut ee = MockEe::default();
        ee.class_tokens.insert(
            CLASS_TOKEN,
            ClassHandle::from_raw(0x999usize as ffi::CORINFO_CLASS_HANDLE).unwrap(),
        );
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(
            &mut ee,
            &[0x12, 0x00, 0xFE, 0x15, 0x01, 0x00, 0x00, 0x02, 0x2A],
            &entry,
            &[CorInfoType::Int],
            &[],
        );
        assert!(matches!(
            import(&info, &ee),
            Err(CompileError::Unsupported(_))
        ));
    }

    #[test]
    fn struct_instance_method_this_is_a_byref() {
        // The method's class is a value class: `this` is a ByRef arg and
        // ldarg.0 yields it (mutations flow back to the caller's memory).
        let (mut ee, c) = struct_ee(8, &[], None);
        ee.method_classes.insert(1, c);
        let entry = MockSig {
            ret: CorInfoType::Void,
            args: vec![],
            has_this: true,
            ret_class: None,
            arg_classes: Vec::new(),
        };
        // ldarg.0; pop; ret.
        let info = struct_info(&mut ee, &[0x02, 0x26, 0x2A], &entry, &[], &[]);
        let m = import(&info, &ee).expect("imports");
        assert_eq!(m.num_args, 1);
        assert_eq!(m.locals[0].ty, Type::ByRef);
    }

    #[test]
    fn callee_side_retbuf_is_an_implicit_arg_after_this() {
        // An instance method returning a 17-byte (memory-class) struct:
        // locals are [this: ByRef, retbuf: ByRef, arg], and `ret` of the
        // struct value block-copies through the retbuf and returns it.
        let (mut ee, c) = struct_ee(17, &[], None);
        ee.method_classes.insert(1, c);
        let entry = MockSig {
            ret: CorInfoType::ValueClass,
            args: vec![CorInfoType::Int],
            has_this: true,
            ret_class: Some(c),
            arg_classes: Vec::new(),
        };
        // ldloc.0; ret — IL local 0 (the struct) is LocalId(3).
        let info = struct_info(
            &mut ee,
            &[0x06, 0x2A],
            &entry,
            &[CorInfoType::ValueClass],
            &[Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        assert_eq!(m.num_args, 3);
        assert_eq!(m.locals[0].ty, Type::ByRef, "this");
        assert_eq!(m.locals[1].ty, Type::ByRef, "the hidden retbuf");
        assert_eq!(m.locals[2].ty, Type::Int32);
        assert_eq!(m.locals[3].ty, Type::Struct(c));
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::StoreInd { addr, value, .. } => {
                assert_eq!(as_local(addr), LocalId(1), "copy through the retbuf");
                let (src, _) = as_struct_val(value);
                assert_eq!(as_local_addr(src), LocalId(3));
            }
            _ => panic!("expected the retbuf block copy"),
        }
        // The method returns the retbuf pointer (rax on return).
        match &m.blocks[0].terminator {
            hir::Terminator::Return { value: Some(v) } => {
                assert_eq!(as_local(v), LocalId(1));
            }
            _ => panic!("expected Return of the retbuf pointer"),
        }
    }

    #[test]
    fn caller_side_retbuf_is_an_implicit_arg_after_this() {
        let (mut ee, c) = struct_ee(17, &[], None);
        // A static fn returning the 17-byte struct, one int arg.
        ee.add_method(
            STRUCT_FN_TOKEN,
            MockSig {
                ret: CorInfoType::ValueClass,
                args: vec![CorInfoType::Int],
                has_this: false,
                ret_class: Some(c),
                arg_classes: Vec::new(),
            },
        );
        let entry = sig(CorInfoType::Void, &[]);
        // ldc.i4.1; call F; pop; ret.
        let info = struct_info(
            &mut ee,
            &[0x17, 0x28, 0x10, 0x00, 0x00, 0x06, 0x26, 0x2A],
            &entry,
            &[],
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::Eval(hir::Expr::Call { sig, args, .. }) => {
                assert_eq!(sig.args, vec![Type::ByRef, Type::Int32]);
                assert_eq!(sig.ret, Type::Struct(c));
                assert!(!sig.has_this);
                // The retbuf temp is the first local; it heads the args.
                assert_eq!(as_local_addr(&args[0]), LocalId(0));
                assert_eq!(as_i32(&args[1]), 1);
                assert_eq!(m.locals[0].ty, Type::Struct(c));
                assert_eq!(m.locals[0].kind, hir::LocalKind::Temp);
            }
            _ => panic!("expected the retbuf call"),
        }
    }

    #[test]
    fn retbuf_shifts_user_arg_indices() {
        // Same retbuf shape as the callee-side test; IL `ldarg.1` must
        // read the user argument (LocalId 2), not the retbuf (LocalId 1).
        let (mut ee, c) = struct_ee(16, &[], None);
        ee.method_classes.insert(1, c);
        ee.add_struct_field(FIELD_TOKEN, c, 0);
        ee.fields.get_mut(&FIELD_TOKEN).unwrap().ty = CorInfoType::Int;
        let entry = MockSig {
            ret: CorInfoType::ValueClass,
            args: vec![CorInfoType::Int],
            has_this: true,
            ret_class: Some(c),
            arg_classes: Vec::new(),
        };
        // ldloca.s 0; ldarg.1; stfld F; ldloc.0; ret — the user arg stores
        // into the local struct's int field.
        let info = struct_info(
            &mut ee,
            &[0x12, 0x00, 0x03, 0x7D, 0x01, 0x00, 0x00, 0x04, 0x06, 0x2A],
            &entry,
            &[CorInfoType::ValueClass],
            &[Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::StoreInd { addr, value, .. } => {
                assert_eq!(as_local_addr(addr), LocalId(3), "the struct local");
                assert_eq!(as_local(value), LocalId(2), "the user arg, past the retbuf");
            }
            _ => panic!("expected the field store"),
        }
    }

    #[test]
    fn register_passed_struct_call_pushes_the_call_itself() {
        let (mut ee, c) = struct_ee(16, &[], Some(sysv_descriptor(&[(INT_EB, 8), (INT_EB, 8)])));
        ee.add_method(
            STRUCT_FN_TOKEN,
            MockSig {
                ret: CorInfoType::ValueClass,
                args: vec![],
                has_this: false,
                ret_class: Some(c),
                arg_classes: Vec::new(),
            },
        );
        let entry = sig(CorInfoType::Void, &[]);
        // call F; stloc.0; ret — no retbuf anywhere.
        let info = struct_info(
            &mut ee,
            &[0x28, 0x10, 0x00, 0x00, 0x06, 0x0A, 0x2A],
            &entry,
            &[CorInfoType::ValueClass],
            &[Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(0));
        match value {
            hir::Expr::Call { sig, args, .. } => {
                assert_eq!(sig.ret, Type::Struct(c));
                assert!(sig.args.is_empty());
                assert!(args.is_empty());
            }
            _ => panic!("expected the struct-returning call as the value"),
        }
    }

    #[test]
    fn struct_field_load_yields_the_field_address() {
        let (mut ee, c) = struct_ee(8, &[], None);
        ee.add_struct_field(STRUCT_FIELD_TOKEN, c, 8);
        let entry = sig(CorInfoType::Void, &[CorInfoType::Class]);
        // ldarg.0; ldfld F; stloc.0; ret — a StructVal of the
        // (null-checked) field address.
        let info = struct_info(
            &mut ee,
            &[0x02, 0x7B, 0x09, 0x00, 0x00, 0x04, 0x0A, 0x2A],
            &entry,
            &[CorInfoType::ValueClass],
            &[Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(1));
        let (addr, class) = as_struct_val(value);
        assert_eq!(class, c);
        match addr {
            hir::Expr::FieldAddr { obj, offset, .. } => {
                assert_eq!(*offset, 8);
                assert!(
                    matches!(**obj, hir::Expr::NullCheck { .. }),
                    "the reference receiver is null-checked"
                );
            }
            _ => panic!("expected Expr::FieldAddr"),
        }
    }

    #[test]
    fn struct_field_access_on_a_byref_receiver_skips_the_null_check() {
        // An int field declared in the value class (the mock ties the
        // declaring class to `value_class`; the type is overridden back
        // to Int): `this` is a byref, so no null check wraps it.
        let (mut ee, c) = struct_ee(8, &[], None);
        ee.method_classes.insert(1, c);
        ee.add_struct_field(FIELD_TOKEN, c, 4);
        ee.fields.get_mut(&FIELD_TOKEN).unwrap().ty = CorInfoType::Int;
        let entry = MockSig {
            ret: CorInfoType::Int,
            args: vec![],
            has_this: true,
            ret_class: None,
            arg_classes: Vec::new(),
        };
        // ldarg.0; ldfld F; ret.
        let info = struct_info(
            &mut ee,
            &[0x02, 0x7B, 0x01, 0x00, 0x00, 0x04, 0x2A],
            &entry,
            &[],
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        match return_value(&m, 0) {
            hir::Expr::Load {
                addr, offset, ty, ..
            } => {
                assert_eq!(*offset, 4);
                assert_eq!(*ty, Type::Int32);
                // No NullCheck wraps the byref receiver.
                assert_eq!(as_local(addr), LocalId(0));
            }
            _ => panic!("expected a plain Load through the byref this"),
        }
    }

    #[test]
    fn stfld_of_a_gc_struct_field_uses_the_bulk_write_barrier() {
        let (mut ee, c) = struct_ee(16, &[(8, false)], None);
        ee.add_struct_field(STRUCT_FIELD_TOKEN, c, 8);
        let entry = sig(CorInfoType::Void, &[CorInfoType::Class]);
        // ldarg.0; ldloc.0; stfld F; ret — the destination may be heap,
        // and the struct embeds a reference: the copy goes through
        // CORINFO_HELP_BULK_WRITEBARRIER.
        let info = struct_info(
            &mut ee,
            &[0x02, 0x06, 0x7D, 0x09, 0x00, 0x00, 0x04, 0x2A],
            &entry,
            &[CorInfoType::ValueClass],
            &[Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::Eval(hir::Expr::Call { target, sig, args }) => {
                assert!(matches!(
                    target,
                    CallTarget::Helper(h) if *h == CorInfoHelpFunc::BULK_WRITEBARRIER
                ));
                assert_eq!(sig.args, vec![Type::ByRef, Type::ByRef, Type::NativeInt]);
                assert_eq!(args.len(), 3);
                // dst: the (null-checked) field address; src: the local's
                // address; size: the layout's.
                assert!(matches!(&args[0], hir::Expr::FieldAddr { .. }));
                assert_eq!(as_local_addr(&args[1]), LocalId(1));
            }
            _ => panic!("expected the bulk-write-barrier call"),
        }
    }

    #[test]
    fn initobj_spills_pending_stack_trees_first() {
        // Runtime_62524's shape: `bool k = a.Value == 1; a = default;
        // if (k) return 1;` — csc keeps the compare on the evaluation
        // stack across the initobj. IL order evaluates the compare
        // BEFORE the zeroing, so the importer must spill the pending
        // compare tree to a temp ahead of the BlockZero.
        let (mut ee, c) = struct_ee(8, &[], None);
        ee.add_struct_field(FIELD_TOKEN, c, 0);
        ee.fields.get_mut(&FIELD_TOKEN).unwrap().ty = CorInfoType::Int;
        ee.add_struct_field(STRUCT_FIELD_TOKEN, c, 4);
        ee.fields.get_mut(&STRUCT_FIELD_TOKEN).unwrap().ty = CorInfoType::Int;
        let entry = MockSig {
            ret: CorInfoType::Int,
            args: vec![CorInfoType::ValueClass],
            has_this: false,
            ret_class: None,
            arg_classes: vec![Some(c)],
        };
        // ldarg.0; ldfld B; ldc.i4.1; ceq; ldarga.s 0; initobj C;
        // brfalse.s +2; ldc.i4.1; ret; ldarg.0; ldfld A; ret
        let info = struct_info(
            &mut ee,
            &[
                0x02, 0x7B, 0x09, 0x00, 0x00, 0x04, 0x17, 0xFE, 0x01, 0x0F, 0x00, 0xFE, 0x15, 0x01,
                0x00, 0x00, 0x02, 0x2C, 0x02, 0x17, 0x2A, 0x02, 0x7B, 0x01, 0x00, 0x00, 0x04, 0x2A,
            ],
            &entry,
            &[CorInfoType::Bool],
            &[],
        );
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        // First statement: the spill of the compare tree into a temp.
        let (dst, value) = store(&stmts[0]);
        let (op, _, _) = as_binary(value);
        assert_eq!(op, BinaryOp::Eq);
        // Second: the BlockZero — AFTER the spill.
        assert!(matches!(stmts[1].kind, hir::StmtKind::BlockZero { .. }));
        // The branch condition reads the spilled temp, not the tree
        // (brfalse compares it against zero).
        match &m.blocks[0].terminator {
            hir::Terminator::Branch { cond, .. } => {
                let (_, lhs, _) = as_binary(cond);
                assert_eq!(as_local(lhs), dst);
            }
            _ => panic!("expected Branch"),
        }
    }

    #[test]
    fn ldarg_of_a_struct_reads_the_slot_address() {
        // A struct argument reads as StructVal of its frame slot.
        let (mut ee, c) = struct_ee(8, &[], None);
        let entry = MockSig {
            ret: CorInfoType::Void,
            args: vec![CorInfoType::ValueClass],
            has_this: false,
            ret_class: None,
            arg_classes: vec![Some(c)],
        };
        // ldarg.0; stloc.0; ret.
        let info = struct_info(
            &mut ee,
            &[0x02, 0x0A, 0x2A],
            &entry,
            &[CorInfoType::ValueClass],
            &[Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        assert_eq!(m.locals[0].ty, Type::Struct(c));
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(1));
        let (addr, _) = as_struct_val(value);
        assert_eq!(as_local_addr(addr), LocalId(0));
    }

    // --- step_10.5: box / unbox / unbox.any / isinst / castclass ---

    const BOX_CLASS_TOKEN: u32 = 0x0200_0007;

    /// A MockEe with one registered class under BOX_CLASS_TOKEN.
    /// `value_class`: registered in `classes` (a value class) or not (a
    /// plain reference type stand-in). `cor_info_type` overrides
    /// `as_cor_info_type` (a *primitive* value class like System.Int32).
    fn box_ee(value_class: bool, cor_info_type: Option<CorInfoType>) -> (MockEe, ClassHandle) {
        let mut ee = MockEe::default();
        let class = if value_class {
            ee.add_class(8, 8, &[], None)
        } else {
            // A reference-type stand-in: a non-null handle the `classes`
            // map doesn't know.
            ClassHandle::from_raw(0x7000usize as ffi::CORINFO_CLASS_HANDLE).unwrap()
        };
        if let Some(ty) = cor_info_type {
            ee.class_cor_info_types.insert(class.as_raw() as usize, ty);
        }
        ee.class_tokens.insert(BOX_CLASS_TOKEN, class);
        (ee, class)
    }

    fn tok(token: u32) -> [u8; 4] {
        token.to_le_bytes()
    }

    /// Extracts (helper, sig, args) from a call expression.
    fn as_helper_call(e: &hir::Expr) -> (CorInfoHelpFunc, &CallSig, &[hir::Expr]) {
        match e {
            hir::Expr::Call {
                target: CallTarget::Helper(h),
                sig,
                args,
            } => (*h, sig, args),
            _ => panic!("expected a helper call"),
        }
    }

    fn assert_mt_first(args: &[hir::Expr]) {
        assert_eq!(args.len(), 2);
        assert!(
            matches!(args[0], hir::Expr::Const(Const::NativeInt(_))),
            "the class handle is a raw pointer constant, never a Ref"
        );
    }

    #[test]
    fn box_of_a_primitive_class_calls_the_box_helper() {
        // ldc.i4.7; box C(int-like); pop; ldc.i4.0; ret — the value spills
        // to a temp and the helper gets its address.
        let t = tok(BOX_CLASS_TOKEN);
        let il = [0x1D, 0x8C, t[0], t[1], t[2], t[3], 0x26, 0x16, 0x2A];
        let (mut ee, _c) = box_ee(true, Some(CorInfoType::Int));
        let entry = sig(CorInfoType::Int, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[], &[]);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        // The value store, then the Eval of the box call (pop must_eval).
        assert_eq!(stmts.len(), 2);
        let (dst, value) = store(&stmts[0]);
        assert_eq!(m.locals[dst.0 as usize].ty, Type::Int32);
        assert_eq!(as_i32(value), 7);
        match &stmts[1].kind {
            hir::StmtKind::Eval(call) => {
                let (helper, sig, args) = as_helper_call(call);
                assert_eq!(helper, CorInfoHelpFunc::BOX);
                assert_eq!(
                    sig,
                    &CallSig {
                        ret: Type::Ref,
                        args: vec![Type::NativeInt, Type::ByRef],
                        has_this: false,
                    }
                );
                assert_mt_first(args);
                assert!(
                    matches!(args[1], hir::Expr::LocalAddr(id) if id == dst),
                    "the helper gets the address of the spilled value"
                );
            }
            _ => panic!("expected the box helper Eval"),
        }
    }

    #[test]
    fn box_of_a_struct_passes_the_value_address() {
        // ldloca.0; ldobj C; box C; pop; ret — no extra copy: the struct
        // value's own address goes to the helper.
        let t = tok(BOX_CLASS_TOKEN);
        let il = [
            0x12, 0x00, 0x71, t[0], t[1], t[2], t[3], 0x8C, t[0], t[1], t[2], t[3], 0x26, 0x2A,
        ];
        let (mut ee, c) = box_ee(true, None);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[CorInfoType::ValueClass], &[Some(c)]);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1, "only the box Eval");
        match &stmts[0].kind {
            hir::StmtKind::Eval(call) => {
                let (helper, _, args) = as_helper_call(call);
                assert_eq!(helper, CorInfoHelpFunc::BOX);
                assert!(
                    matches!(args[1], hir::Expr::LocalAddr(id) if id == LocalId(0)),
                    "the struct local's address"
                );
            }
            _ => panic!("expected the box helper Eval"),
        }
    }

    #[test]
    fn box_of_a_reference_class_is_a_nop() {
        // ldnull; box C(ref); ret-int path: the reference passes through.
        let t = tok(BOX_CLASS_TOKEN);
        let il = [0x14, 0x8C, t[0], t[1], t[2], t[3], 0x26, 0x16, 0x2A];
        let (mut ee, _c) = box_ee(false, None);
        let entry = sig(CorInfoType::Int, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[], &[]);
        let m = import(&info, &ee).expect("imports");
        assert!(
            m.blocks[0].stmts.is_empty(),
            "box of a ref class emits nothing (pop of null is pure)"
        );
    }

    #[test]
    fn box_of_nullable_is_unsupported() {
        let t = tok(BOX_CLASS_TOKEN);
        let il = [
            0x12, 0x00, 0x71, t[0], t[1], t[2], t[3], 0x8C, t[0], t[1], t[2], t[3], 0x26, 0x2A,
        ];
        let (mut ee, c) = box_ee(true, None);
        ee.box_helper = Some(CorInfoHelpFunc::BOX_NULLABLE);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[CorInfoType::ValueClass], &[Some(c)]);
        let err = import(&info, &ee).err().expect("box of Nullable<T>");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("Nullable")),
            "{err:?}"
        );
    }

    #[test]
    fn unbox_pushes_the_payload_byref_call() {
        // ldnull; unbox C; pop; ret.
        let t = tok(BOX_CLASS_TOKEN);
        let il = [0x14, 0x79, t[0], t[1], t[2], t[3], 0x26, 0x2A];
        let (mut ee, _c) = box_ee(true, None);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[], &[]);
        let m = import(&info, &ee).expect("imports");
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::Eval(call) => {
                let (helper, sig, args) = as_helper_call(call);
                assert_eq!(helper, CorInfoHelpFunc::UNBOX);
                assert_eq!(
                    sig,
                    &CallSig {
                        ret: Type::ByRef,
                        args: vec![Type::NativeInt, Type::Ref],
                        has_this: false,
                    }
                );
                assert_mt_first(args);
                assert!(matches!(args[1], hir::Expr::Const(Const::NullRef)));
            }
            _ => panic!("expected the unbox helper Eval"),
        }
    }

    #[test]
    fn unbox_any_of_a_primitive_loads_through_the_payload_byref() {
        // ldnull; unbox.any C(int-like); pop; ret.
        let t = tok(BOX_CLASS_TOKEN);
        let il = [0x14, 0xA5, t[0], t[1], t[2], t[3], 0x26, 0x2A];
        let (mut ee, _c) = box_ee(true, Some(CorInfoType::Int));
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[], &[]);
        let m = import(&info, &ee).expect("imports");
        // pop of a Load must_eval: one Eval of a Load through the unbox
        // call's byref.
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::Eval(hir::Expr::Load {
                addr, offset, ty, ..
            }) => {
                assert_eq!(*offset, 0);
                assert_eq!(*ty, Type::Int32);
                let (helper, _, _) = as_helper_call(addr);
                assert_eq!(helper, CorInfoHelpFunc::UNBOX);
            }
            _ => panic!("expected Eval(Load through the unbox byref)"),
        }
    }

    #[test]
    fn unbox_any_of_a_struct_is_a_struct_val_of_the_payload() {
        // ldnull; unbox.any C; stloc.0; ret — the store block-copies out
        // of the box payload.
        let t = tok(BOX_CLASS_TOKEN);
        let il = [0x14, 0xA5, t[0], t[1], t[2], t[3], 0x0A, 0x2A];
        let (mut ee, c) = box_ee(true, None);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[CorInfoType::ValueClass], &[Some(c)]);
        let m = import(&info, &ee).expect("imports");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(m.locals[dst.0 as usize].ty, Type::Struct(c));
        let (addr, class) = as_struct_val(value);
        assert_eq!(class, c);
        let (helper, _, _) = as_helper_call(addr);
        assert_eq!(helper, CorInfoHelpFunc::UNBOX);
    }

    #[test]
    fn unbox_any_of_a_reference_class_is_castclass() {
        // ldnull; unbox.any C(ref); pop; ret.
        let t = tok(BOX_CLASS_TOKEN);
        let il = [0x14, 0xA5, t[0], t[1], t[2], t[3], 0x26, 0x2A];
        let (mut ee, _c) = box_ee(false, None);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[], &[]);
        let m = import(&info, &ee).expect("imports");
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::Eval(call) => {
                let (helper, sig, _) = as_helper_call(call);
                assert_eq!(helper, CorInfoHelpFunc::CHKCASTANY);
                assert_eq!(sig.ret, Type::Ref);
            }
            _ => panic!("expected the cast helper Eval"),
        }
    }

    #[test]
    fn unbox_of_nullable_is_unsupported() {
        let t = tok(BOX_CLASS_TOKEN);
        let il = [0x14, 0x79, t[0], t[1], t[2], t[3], 0x26, 0x2A];
        let (mut ee, _c) = box_ee(true, None);
        ee.unbox_helper = Some(CorInfoHelpFunc::UNBOX_NULLABLE);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[], &[]);
        let err = import(&info, &ee).err().expect("unbox of Nullable<T>");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("Nullable")),
            "{err:?}"
        );
    }

    #[test]
    fn isinst_uses_the_null_producing_helper() {
        // ldnull; isinst C; pop; ret.
        let t = tok(BOX_CLASS_TOKEN);
        let il = [0x14, 0x75, t[0], t[1], t[2], t[3], 0x26, 0x2A];
        let (mut ee, _c) = box_ee(false, None);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[], &[]);
        let m = import(&info, &ee).expect("imports");
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::Eval(call) => {
                let (helper, sig, args) = as_helper_call(call);
                assert_eq!(helper, CorInfoHelpFunc::ISINSTANCEOFANY);
                assert_eq!(
                    sig,
                    &CallSig {
                        ret: Type::Ref,
                        args: vec![Type::NativeInt, Type::Ref],
                        has_this: false,
                    }
                );
                assert_mt_first(args);
            }
            _ => panic!("expected the isinst helper Eval"),
        }
    }

    #[test]
    fn castclass_uses_the_throwing_helper() {
        // ldnull; castclass C; pop; ret.
        let t = tok(BOX_CLASS_TOKEN);
        let il = [0x14, 0x74, t[0], t[1], t[2], t[3], 0x26, 0x2A];
        let (mut ee, _c) = box_ee(false, None);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[], &[]);
        let m = import(&info, &ee).expect("imports");
        match &m.blocks[0].stmts[0].kind {
            hir::StmtKind::Eval(call) => {
                let (helper, _, _) = as_helper_call(call);
                assert_eq!(helper, CorInfoHelpFunc::CHKCASTANY);
            }
            _ => panic!("expected the castclass helper Eval"),
        }
    }

    #[test]
    fn an_out_of_set_casting_helper_is_unsupported() {
        let t = tok(BOX_CLASS_TOKEN);
        let il = [0x14, 0x74, t[0], t[1], t[2], t[3], 0x26, 0x2A];
        let (mut ee, _c) = box_ee(false, None);
        ee.casting_helper = Some(CorInfoHelpFunc::CHKCASTCLASS_SPECIAL);
        let entry = sig(CorInfoType::Void, &[]);
        let info = struct_info(&mut ee, &il, &entry, &[], &[]);
        let err = import(&info, &ee).err().expect("helper outside the set");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("casting helper")),
            "{err:?}"
        );
    }

    // --- step_10.8: arrays ---

    const ARR_TOKEN: u32 = 0x0200_0042;
    const ELEM_TOKEN: u32 = 0x0200_0043;

    /// A MockEe resolving ARR_TOKEN to a canned array class; the mock's
    /// defaults answer the happy path (an SZ array, the NEWARR_1_PTR
    /// helper).
    fn array_fixture(il: &[u8], entry: &MockSig, locals: &[CorInfoType]) -> (MockEe, MethodInfo) {
        let (mut ee, info) = fixture(il, entry, locals);
        let arr = ClassHandle::from_raw(0xA550usize as ffi::CORINFO_CLASS_HANDLE).unwrap();
        ee.class_tokens.insert(ARR_TOKEN, arr);
        (ee, info)
    }

    /// A canned element class resolved from ELEM_TOKEN, with
    /// `as_cor_info_type` answering `cor` (a registered value class for a
    /// struct element; a bare handle with a `class_cor_info_types`
    /// override otherwise).
    fn with_elem(ee: &mut MockEe, elem: ClassHandle, cor: CorInfoType) {
        ee.class_tokens.insert(ELEM_TOKEN, elem);
        ee.class_cor_info_types.insert(elem.as_raw() as usize, cor);
    }

    fn as_bounds_check(stmt: &hir::Stmt) -> (&hir::Expr, &hir::Expr) {
        match &stmt.kind {
            hir::StmtKind::BoundsCheck { array, index } => (array, index),
            _ => panic!("expected StmtKind::BoundsCheck"),
        }
    }

    fn as_arr_elem_addr(e: &hir::Expr) -> (&hir::Expr, &hir::Expr, Type, u32) {
        match e {
            hir::Expr::ArrElemAddr {
                array,
                index,
                elem,
                elem_size,
            } => (array, index, *elem, *elem_size),
            _ => panic!("expected Expr::ArrElemAddr"),
        }
    }

    #[test]
    fn newarr_imports_as_the_allocation_helper_call() {
        // ldc.i4.3; newarr T; stloc.0; ldc.i4.0; ret — a Ref local holds
        // the fresh array.
        let t = tok(ARR_TOKEN);
        let il = [0x19, 0x8D, t[0], t[1], t[2], t[3], 0x0A, 0x16, 0x2A];
        let (ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[]), &[CorInfoType::Class]);
        let m = import(&info, &ee).expect("imports");
        let (dst, value) = store(&m.blocks[0].stmts[0]);
        assert_eq!(dst, LocalId(0));
        assert_eq!(m.locals[0].ty, Type::Ref);
        let (helper, csig, args) = as_helper_call(value);
        assert_eq!(helper, CorInfoHelpFunc::NEWARR_1_PTR);
        assert_eq!(
            csig,
            &CallSig {
                ret: Type::Ref,
                args: vec![Type::NativeInt, Type::NativeInt],
                has_this: false,
            }
        );
        assert!(
            matches!(args[0], hir::Expr::Const(Const::NativeInt(_))),
            "the array class embeds as a raw pointer constant, never a Ref"
        );
        // The Int32 length zero-extends to native int.
        let (to, overflow, unsigned, arg) = as_conv(&args[1]);
        assert_eq!(to, Type::NativeInt);
        assert!(!overflow && unsigned);
        assert_eq!(as_i32(arg), 3);

        // A native-int length rides through unchanged.
        let il = [0x02, 0x8D, t[0], t[1], t[2], t[3], 0x0A, 0x16, 0x2A];
        let (ee, info) = array_fixture(
            &il,
            &sig(CorInfoType::Int, &[CorInfoType::NativeInt]),
            &[CorInfoType::Class],
        );
        let m = import(&info, &ee).expect("imports");
        let (_, value) = store(&m.blocks[0].stmts[0]);
        let (_, _, args) = as_helper_call(value);
        assert_eq!(as_local(&args[1]), LocalId(0), "no widening node");
    }

    #[test]
    fn newarr_gates() {
        let t = tok(ARR_TOKEN);
        let il = [0x17, 0x8D, t[0], t[1], t[2], t[3], 0x26, 0x16, 0x2A];
        // A non-SZ array class (the defensive gate) is Unsupported.
        let (mut ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        let arr = ee.class_tokens[&ARR_TOKEN];
        ee.non_sd_arrays.insert(arr.as_raw() as usize);
        let err = import(&info, &ee).err().expect("a non-SZ array is out");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("non-SZ")),
            "{err:?}"
        );
        // An allocation helper outside the accepted set is Unsupported.
        let (mut ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        ee.new_arr_helper = Some(CorInfoHelpFunc::NEWFAST);
        let err = import(&info, &ee).err().expect("helper outside the set");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("allocation helper")),
            "{err:?}"
        );
        // A float length is BadIl.
        let il = [
            0x23, 0, 0, 0, 0, 0, 0, 0, 0, 0x8D, t[0], t[1], t[2], t[3], 0x26, 0x16, 0x2A,
        ];
        let (ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        let err = import(&info, &ee).err().expect("a float length is bad IL");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("newarr length")),
            "{err:?}"
        );
        // An unresolvable token is BadIl.
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        let err = import(&info, &ee).err().expect("unresolvable token");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("did not resolve")),
            "{err:?}"
        );
    }

    #[test]
    fn ldlen_pushes_the_arr_len_tree() {
        // ldarg.0; ldlen; ret — the length read, Int32.
        let il = [0x02, 0x8E, 0x2A];
        let (ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
        let m = import(&info, &ee).expect("imports");
        match return_value(&m, 0) {
            hir::Expr::ArrLen { array } => assert_eq!(as_local(array), LocalId(0)),
            _ => panic!("expected Expr::ArrLen"),
        }

        // ldlen of a non-reference is BadIl.
        let (ee, info) = fixture(
            &[0x02, 0x8E, 0x2A],
            &sig(CorInfoType::Int, &[CorInfoType::Int]),
            &[],
        );
        let err = import(&info, &ee).err().expect("ldlen of an int");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("must be a reference")),
            "{err:?}"
        );
    }

    #[test]
    fn ldelem_fixed_kinds_import_as_bounds_check_plus_typed_load() {
        for (i, &(ty, access, size)) in LDELEM_FIXED_KINDS.iter().enumerate() {
            // ldarg.0; ldc.i4.1; ldelem.X; pop; ldc.i4.0; ret
            let il = [0x02, 0x17, 0x90 + i as u8, 0x26, 0x16, 0x2A];
            let (ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
            let m = import(&info, &ee).expect("imports");
            let stmts = &m.blocks[0].stmts;
            assert_eq!(
                stmts.len(),
                2,
                "opcode {:#04x}: the bounds check, then the popped load's Eval",
                0x90 + i as u8
            );
            let (array, index) = as_bounds_check(&stmts[0]);
            assert_eq!(as_local(array), LocalId(0));
            assert_eq!(as_i32(index), 1);
            match &stmts[1].kind {
                hir::StmtKind::Eval(hir::Expr::Load {
                    addr,
                    offset,
                    ty: lty,
                    access: lacc,
                }) => {
                    assert_eq!(*offset, 0);
                    assert_eq!(*lty, ty, "opcode {:#04x}", 0x90 + i as u8);
                    assert_eq!(*lacc, access);
                    let (a, idx, elem, elem_size) = as_arr_elem_addr(addr);
                    assert_eq!(as_local(a), LocalId(0));
                    assert_eq!(as_i32(idx), 1);
                    assert_eq!(elem, ty);
                    assert_eq!(elem_size, size);
                }
                _ => panic!("expected the Eval of the element load"),
            }
        }
    }

    #[test]
    fn stelem_fixed_kinds_import_as_bounds_check_plus_store_ind() {
        // The value IL matching each element kind's stack type (stelem.ref
        // — the ARRADDR_ST helper form — has its own test).
        let value_ils: [&[u8]; 7] = [
            &[0x16, 0xE0],                         // native int: ldc.i4.0; conv.u
            &[0x1F, 0x2A],                         // i1: ldc.i4.s 42
            &[0x1F, 0x2A],                         // i2
            &[0x1F, 0x2A],                         // i4
            &[0x21, 0, 0, 0, 0, 0, 0, 0, 0],       // i8: ldc.i8 0
            &[0x22, 0, 0, 0x80, 0x3F],             // r4: ldc.r4 1.0
            &[0x23, 0, 0, 0, 0, 0, 0, 0xF0, 0x3F], // r8: ldc.r8 1.0
        ];
        for (i, (&(ty, access, size), value_il)) in
            STELEM_FIXED_KINDS[..7].iter().zip(&value_ils).enumerate()
        {
            // ldarg.0; ldc.i4.1; <value>; stelem.X; ldc.i4.0; ret
            let mut il = vec![0x02, 0x17];
            il.extend_from_slice(value_il);
            il.extend_from_slice(&[0x9B + i as u8, 0x16, 0x2A]);
            let (ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
            let m = import(&info, &ee).expect("imports");
            let stmts = &m.blocks[0].stmts;
            assert_eq!(
                stmts.len(),
                2,
                "opcode {:#04x}: the bounds check, then the store",
                0x9B + i as u8
            );
            let (array, index) = as_bounds_check(&stmts[0]);
            assert_eq!(as_local(array), LocalId(0));
            assert_eq!(as_i32(index), 1);
            match &stmts[1].kind {
                hir::StmtKind::StoreInd {
                    addr,
                    offset,
                    access: sacc,
                    ..
                } => {
                    assert_eq!(*offset, 0);
                    assert_eq!(*sacc, access);
                    let (a, idx, elem, elem_size) = as_arr_elem_addr(addr);
                    assert_eq!(as_local(a), LocalId(0));
                    assert_eq!(as_i32(idx), 1);
                    assert_eq!(elem, ty);
                    assert_eq!(elem_size, size);
                }
                _ => panic!("expected StmtKind::StoreInd"),
            }
        }
    }

    #[test]
    fn stelem_ref_is_the_arraddr_st_helper_call() {
        // ldarg.0; ldc.i4.1; ldnull; stelem.ref; ldc.i4.0; ret — the
        // helper checks null, bounds, covariance, and applies the write
        // barrier; NO BoundsCheck statement.
        let il = [0x02, 0x17, 0x14, 0xA2, 0x16, 0x2A];
        let (ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1, "the helper call only — no bounds check");
        match &stmts[0].kind {
            hir::StmtKind::Eval(hir::Expr::Call { target, sig, args }) => {
                assert!(
                    matches!(target, CallTarget::Helper(h) if *h == CorInfoHelpFunc::ARRADDR_ST),
                    "CastHelpers.StelemRef"
                );
                assert_eq!(
                    sig,
                    &CallSig {
                        ret: Type::Void,
                        args: vec![Type::Ref, Type::NativeInt, Type::Ref],
                        has_this: false,
                    }
                );
                assert_eq!(as_local(&args[0]), LocalId(0));
                // The Int32 index zero-extends to native int.
                let (to, _, unsigned, arg) = as_conv(&args[1]);
                assert_eq!(to, Type::NativeInt);
                assert!(unsigned);
                assert_eq!(as_i32(arg), 1);
                assert!(matches!(args[2], hir::Expr::Const(Const::NullRef)));
            }
            _ => panic!("expected the ARRADDR_ST helper Eval"),
        }
    }

    #[test]
    fn ldelema_of_a_primitive_is_a_bounds_checked_byref() {
        // ldarg.0; ldc.i4.0; ldelema <int>; stloc.0 (ByRef); ldc.i4.0; ret.
        let t = tok(ELEM_TOKEN);
        let il = [0x02, 0x16, 0x8F, t[0], t[1], t[2], t[3], 0x0A, 0x16, 0x2A];
        let (mut ee, info) = array_fixture(
            &il,
            &sig(CorInfoType::Int, &[CorInfoType::Class]),
            &[CorInfoType::ByRef],
        );
        let elem = ClassHandle::from_raw(0xE1E4usize as ffi::CORINFO_CLASS_HANDLE).unwrap();
        with_elem(&mut ee, elem, CorInfoType::Int);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2, "the bounds check, then the stloc");
        let (array, _) = as_bounds_check(&stmts[0]);
        assert_eq!(as_local(array), LocalId(0));
        let (dst, value) = store(&stmts[1]);
        assert_eq!(dst, LocalId(1), "the ByRef IL local");
        let (_, _, elem_ty, elem_size) = as_arr_elem_addr(value);
        assert_eq!(elem_ty, Type::Int32);
        assert_eq!(elem_size, 4);
        assert_eq!(m.locals[1].ty, Type::ByRef);
    }

    #[test]
    fn ldelema_of_a_reference_element_is_the_ldelema_ref_helper() {
        // A reference element: the LDELEMA_REF helper (it null-checks,
        // bounds-checks, and exact-type-checks itself) — no BoundsCheck.
        let t = tok(ELEM_TOKEN);
        let il = [0x02, 0x16, 0x8F, t[0], t[1], t[2], t[3], 0x0A, 0x16, 0x2A];
        let (mut ee, info) = array_fixture(
            &il,
            &sig(CorInfoType::Int, &[CorInfoType::Class]),
            &[CorInfoType::ByRef],
        );
        let elem = ClassHandle::from_raw(0xE1EFusize as ffi::CORINFO_CLASS_HANDLE).unwrap();
        ee.class_tokens.insert(ELEM_TOKEN, elem); // a reference type (the default)
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1, "the stloc only — no bounds check");
        let (dst, value) = store(&stmts[0]);
        assert_eq!(dst, LocalId(1));
        let (helper, sig, args) = as_helper_call(value);
        assert_eq!(helper, CorInfoHelpFunc::LDELEMA_REF);
        assert_eq!(
            sig,
            &CallSig {
                ret: Type::ByRef,
                args: vec![Type::Ref, Type::NativeInt],
                has_this: false,
            }
        );
        assert_eq!(as_local(&args[0]), LocalId(0));
        let (to, _, unsigned, _) = as_conv(&args[1]);
        assert_eq!(to, Type::NativeInt);
        assert!(unsigned);
    }

    #[test]
    fn ldelem_stelem_token_forms_follow_the_element_type() {
        let t = tok(ELEM_TOKEN);
        // ldelem <int> ≡ ldelem.i4 (0x94): BoundsCheck + Int32 load.
        let il = [0x02, 0x17, 0xA3, t[0], t[1], t[2], t[3], 0x26, 0x16, 0x2A];
        let (mut ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
        let elem = ClassHandle::from_raw(0xE1E4usize as ffi::CORINFO_CLASS_HANDLE).unwrap();
        with_elem(&mut ee, elem, CorInfoType::Int);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2);
        let _ = as_bounds_check(&stmts[0]);
        match &stmts[1].kind {
            hir::StmtKind::Eval(hir::Expr::Load {
                ty, access, addr, ..
            }) => {
                assert_eq!(*ty, Type::Int32);
                assert_eq!(*access, MemAccess::Natural);
                assert_eq!(as_arr_elem_addr(addr).3, 4);
            }
            _ => panic!("expected the Eval of the element load"),
        }

        // ldelem <a reference type> ≡ ldelem.ref (0x9A).
        let (mut ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
        let elem = ClassHandle::from_raw(0xE1EFusize as ffi::CORINFO_CLASS_HANDLE).unwrap();
        ee.class_tokens.insert(ELEM_TOKEN, elem);
        let m = import(&info, &ee).expect("imports");
        match &m.blocks[0].stmts[1].kind {
            hir::StmtKind::Eval(hir::Expr::Load { ty, addr, .. }) => {
                assert_eq!(*ty, Type::Ref);
                assert_eq!(as_arr_elem_addr(addr).3, 8);
            }
            _ => panic!("expected the Eval of the element load"),
        }

        // stelem <a reference type> ≡ stelem.ref: the ARRADDR_ST helper.
        let il = [0x02, 0x17, 0x14, 0xA4, t[0], t[1], t[2], t[3], 0x16, 0x2A];
        let (mut ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
        ee.class_tokens.insert(ELEM_TOKEN, elem);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1, "the helper call only — no bounds check");
        match &stmts[0].kind {
            hir::StmtKind::Eval(hir::Expr::Call { target, .. }) => {
                assert!(
                    matches!(target, CallTarget::Helper(h) if *h == CorInfoHelpFunc::ARRADDR_ST)
                );
            }
            _ => panic!("expected the ARRADDR_ST helper Eval"),
        }
    }

    #[test]
    fn ldelem_of_a_struct_yields_a_struct_val_over_the_element_address() {
        // ldarg.0; ldc.i4.0; ldelem S; stloc.0 (a struct local); ret.
        let (mut ee, c) = struct_ee(12, &[], None);
        ee.class_tokens.insert(ELEM_TOKEN, c);
        let entry = sig(CorInfoType::Void, &[CorInfoType::Class]);
        let t = tok(ELEM_TOKEN);
        let info = struct_info(
            &mut ee,
            &[0x02, 0x16, 0xA3, t[0], t[1], t[2], t[3], 0x0A, 0x2A],
            &entry,
            &[CorInfoType::ValueClass],
            &[Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2, "the bounds check, then the stloc");
        let _ = as_bounds_check(&stmts[0]);
        let (dst, value) = store(&stmts[1]);
        assert_eq!(dst, LocalId(1));
        let (addr, class) = as_struct_val(value);
        assert_eq!(class, c);
        let (_, _, elem, elem_size) = as_arr_elem_addr(addr);
        assert_eq!(elem, Type::Struct(c));
        assert_eq!(elem_size, 12, "the layout size is the element size");
    }

    #[test]
    fn stelem_of_a_struct_is_the_block_copy_shape() {
        // ldarg.0; ldc.i4.0; ldloc.0; stelem S; ret — BoundsCheck, then
        // StoreInd of the StructVal (the flattener's BlockCopy path).
        let (mut ee, c) = struct_ee(12, &[], None);
        ee.class_tokens.insert(ELEM_TOKEN, c);
        let entry = sig(CorInfoType::Void, &[CorInfoType::Class]);
        let t = tok(ELEM_TOKEN);
        let info = struct_info(
            &mut ee,
            &[0x02, 0x16, 0x06, 0xA4, t[0], t[1], t[2], t[3], 0x2A],
            &entry,
            &[CorInfoType::ValueClass],
            &[Some(c)],
        );
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 2, "the bounds check, then the store");
        let _ = as_bounds_check(&stmts[0]);
        match &stmts[1].kind {
            hir::StmtKind::StoreInd {
                addr,
                offset,
                value,
                access,
            } => {
                assert_eq!(*offset, 0);
                assert_eq!(*access, MemAccess::Natural);
                assert_eq!(as_arr_elem_addr(addr).3, 12);
                let (src, class) = as_struct_val(value);
                assert_eq!(class, c);
                assert_eq!(as_local_addr(src), LocalId(1), "the struct local's address");
            }
            _ => panic!("expected StmtKind::StoreInd"),
        }
    }

    #[test]
    fn a_call_shaped_array_evaluates_exactly_once() {
        // ldc.i4.3; newarr T; ldc.i4.0; ldelem.i4; pop; ldc.i4.0; ret —
        // the array is a call tree: it materializes into a temp, and both
        // the bounds check and the element address read the temp.
        let t = tok(ARR_TOKEN);
        let il = [
            0x19, 0x8D, t[0], t[1], t[2], t[3], 0x16, 0x94, 0x26, 0x16, 0x2A,
        ];
        let (ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(
            stmts.len(),
            3,
            "the array temp store, the bounds check, the popped load's Eval"
        );
        let (tmp, value) = store(&stmts[0]);
        assert_eq!(m.locals[tmp.0 as usize].ty, Type::Ref);
        assert!(matches!(value, hir::Expr::Call { .. }), "the newarr call");
        let (array, _) = as_bounds_check(&stmts[1]);
        assert_eq!(as_local(array), tmp);
        match &stmts[2].kind {
            hir::StmtKind::Eval(hir::Expr::Load { addr, .. }) => {
                let (a, _, _, _) = as_arr_elem_addr(addr);
                assert_eq!(as_local(a), tmp, "the address reads the same temp");
            }
            _ => panic!("expected the Eval of the element load"),
        }
    }

    #[test]
    fn element_access_type_gates_are_bad_il() {
        // An index of a reference type.
        let il = [0x02, 0x02, 0x94, 0x26, 0x16, 0x2A];
        let (ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
        let err = import(&info, &ee).err().expect("a ref index is bad IL");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("array index")),
            "{err:?}"
        );
        // A non-reference array operand.
        let il = [0x16, 0x16, 0x94, 0x26, 0x16, 0x2A];
        let (ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        let err = import(&info, &ee)
            .err()
            .expect("an int array operand is bad IL");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("array operand")),
            "{err:?}"
        );
        // A stelem value of the wrong stack type.
        let il = [0x02, 0x16, 0x14, 0x9E, 0x16, 0x2A]; // stelem.i4 of null
        let (ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
        let err = import(&info, &ee).err().expect("a ref value for stelem.i4");
        assert!(
            matches!(&err, CompileError::BadIl(m) if m.contains("stelem value type mismatch")),
            "{err:?}"
        );
    }

    #[test]
    fn an_out_of_pack_element_type_is_unsupported() {
        // ldelem <a byref-typed class>: no such array storage exists.
        let t = tok(ELEM_TOKEN);
        let il = [0x02, 0x17, 0xA3, t[0], t[1], t[2], t[3], 0x26, 0x16, 0x2A];
        let (mut ee, info) = array_fixture(&il, &sig(CorInfoType::Int, &[CorInfoType::Class]), &[]);
        let elem = ClassHandle::from_raw(0xE1E5usize as ffi::CORINFO_CLASS_HANDLE).unwrap();
        with_elem(&mut ee, elem, CorInfoType::ByRef);
        let err = import(&info, &ee).err().expect("a byref element is out");
        assert!(
            matches!(&err, CompileError::Unsupported(m) if m.contains("array element type")),
            "{err:?}"
        );
    }

    #[test]
    fn method_prolog_class_init_trigger() {
        // RyuJIT's morph.cpp:50 fgMorphMainInit shape: a method whose own
        // class's precise cctor has not run starts with the INITCLASS
        // helper call (the 10.8 entry-cctor fix — loops_with_eh.cs).
        let il = [0x16, 0x2A]; // ldc.i4.0; ret
        let (mut ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        ee.prolog_init_class = Some(CorInfoInitClassResult::USE_HELPER);
        let m = import(&info, &ee).expect("imports");
        let stmts = &m.blocks[0].stmts;
        assert_eq!(stmts.len(), 1, "just the prolog trigger");
        match &stmts[0].kind {
            hir::StmtKind::Eval(hir::Expr::Call { target, sig, args }) => {
                assert!(matches!(
                    target,
                    CallTarget::Helper(h) if *h == CorInfoHelpFunc::INITCLASS
                ));
                assert_eq!(sig.ret, Type::Void);
                assert_eq!(args.len(), 1);
                assert!(
                    matches!(args[0], hir::Expr::Const(Const::NativeInt(_))),
                    "the method's own class embeds as a raw pointer"
                );
            }
            _ => panic!("expected the prolog INITCLASS call"),
        }
        assert_eq!(stmts[0].il_offset, IlOffset(0));

        // The default verdict (NOT_REQUIRED — a beforefieldinit class)
        // emits nothing.
        let (ee, info) = fixture(&il, &sig(CorInfoType::Int, &[]), &[]);
        let m = import(&info, &ee).expect("imports");
        assert!(m.blocks[0].stmts.is_empty());
    }
}
