//! The RokaJIT IR (frozen; prose in `RokaJIT-internal/docs/ir-design.md`,
//! decision in `decisions/2026-09-11-ir-design.md`).
//!
//! One tree IR, two type levels: [`hir`] is what the importer builds —
//! expression trees, calls nest inside expressions; [`lir`] is what the
//! lowering pass produces — one operation per statement, operands
//! restricted to [`lir::Operand`]. Lowering is a function `hir::Method ->
//! lir::Method`; no pass mutates an IR in place, and no level gains or
//! loses node kinds halfway through (the GenTree "one mega-IR mutated by
//! every phase" model is explicitly rejected).
//!
//! Hard rules for both levels:
//!
//! - **CIL types only** ([`Type`]). Machine registers never appear in the
//!   IR; register binding is the backend's business entirely.
//! - **Containers are statements inside basic blocks**; control flow lives
//!   on block terminators, never inside expressions. EH regions are
//!   explicit ranges over blocks.
//! - **IL offsets live on statements** ([`IlOffset`]), not on expression
//!   nodes: debug boundaries and GC safepoints are statement-level
//!   concepts. `IL_OFFSET_NONE` marks synthesized statements.

use rokajit_ee::enums::CorInfoHelpFunc;
use rokajit_ee::handles::{ClassHandle, FieldHandle, MethodHandle};

use crate::structs::StructLayouts;

/// The IR's entire type vocabulary: the ECMA-335 evaluation-stack types.
/// `Bool`/`Char`/`Short`/`Byte` are normalized to `Int32` at import
/// (ECMA-335 §III.1.1.1); `Void` appears only in signatures, never as the
/// type of an expression.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Type {
    Void,
    Int32,
    Int64,
    NativeInt,
    Float,
    Double,
    /// A managed object reference (`CORINFO_TYPE_CLASS`).
    Ref,
    /// A managed byref pointer (`CORINFO_TYPE_BYREF`).
    ByRef,
    /// A struct value; layout questions are answered by the EE through the
    /// class handle.
    Struct(ClassHandle),
}

/// The shape of the memory cell a `Load`/`StoreInd` (HIR) or
/// `Load`/`Store` (LIR) touches, when it is narrower than the value's
/// stack type. Sub-Int32 fields (`ldfld`/`stfld` of `bool`/`char`/
/// `sbyte`/…) are the source: ECMA-335 §III.1.1.1 normalizes the value
/// to `Int32` on the evaluation stack, but the field keeps its metadata
/// size — a 4-byte access would read or clobber the neighboring bytes.
/// Loads extend to `Int32`, signed per the field's metadata type
/// (ELEMENT_TYPE_I1/I2 sign-extend; BOOLEAN/CHAR/U1/U2 zero-extend);
/// stores write only the low bytes (extension is a store-time no-op, so
/// the store forms carry the width only).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MemAccess {
    /// The type's natural width — every access except a sub-Int32 field.
    Natural,
    /// 1 byte, sign-extended on load (ELEMENT_TYPE_I1).
    I8,
    /// 1 byte, zero-extended on load (ELEMENT_TYPE_BOOLEAN/U1).
    U8,
    /// 2 bytes, sign-extended on load (ELEMENT_TYPE_I2).
    I16,
    /// 2 bytes, zero-extended on load (ELEMENT_TYPE_CHAR/U2).
    U16,
}

impl MemAccess {
    /// `true` when the access is the value type's natural width.
    pub fn is_natural(self) -> bool {
        matches!(self, MemAccess::Natural)
    }

    /// The cell size in bytes for a narrow access, `None` for `Natural`.
    pub fn narrow_bytes(self) -> Option<u8> {
        match self {
            MemAccess::Natural => None,
            MemAccess::I8 | MemAccess::U8 => Some(1),
            MemAccess::I16 | MemAccess::U16 => Some(2),
        }
    }

    /// `true` when a load of this shape sign-extends (the I1/I2 forms).
    pub fn sign_extends(self) -> bool {
        matches!(self, MemAccess::I8 | MemAccess::I16)
    }
}

/// Identity of a local slot. Indices `0..num_args` are the IL arguments,
/// `num_args..num_args+num_il_locals` the IL locals, everything above is a
/// compiler temp. One flat namespace, like RyuJIT's `lvaTable`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct LocalId(pub u32);

/// Identity of a basic block within one method.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// The IL offset a statement was imported from, for debug boundaries and
/// GC info. `IL_OFFSET_NONE` for statements no IL instruction maps to.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct IlOffset(pub u32);

pub const IL_OFFSET_NONE: IlOffset = IlOffset(u32::MAX);

/// The array object layout (corinfo.h:2071-2076's
/// `OFFSETOF__CORINFO_Array__length`/`data`, x64): the element count is a
/// 32-bit field at offset 8, the data starts at offset 16. Compile-time
/// constants — no EE query exists for them.
pub const ARRAY_LENGTH_OFFSET: u32 = 8;
pub const ARRAY_DATA_OFFSET: u32 = 16;

/// An IL literal.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Const {
    Int32(i32),
    Int64(i64),
    NativeInt(isize),
    Float(f32),
    Double(f64),
    /// `ldnull`.
    NullRef,
    /// A reference to an EE-frozen (immovable, process-lifetime) object,
    /// embedded as a 64-bit immediate — what `ldstr` resolves to
    /// (`decisions/2026-09-12-ldstr-and-gc-roots.md`).
    FrozenRef(u64),
}

/// A call's signature in IR terms (not the EE's `CORINFO_SIG_INFO`): the
/// stack-relevant shape only.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CallSig {
    pub ret: Type,
    pub args: Vec<Type>,
    pub has_this: bool,
}

/// Integer/Float arithmetic and comparison operators. Comparison results
/// are `Int32` 0/1 (ECMA-335 `ceq`/`clt`/…). `U*` are the unsigned forms
/// (`un.` prefix in IL). `MinF`/`MaxF` are the float-domain minimum/
/// maximum (SSE `mins*`/`maxs*` semantics: the *second* operand wins on
/// NaN) — no IL opcode maps to them; the importer and the HIR→LIR
/// lowering build them for the saturating float→unsigned-int conversion
/// sequences (step_10.11).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    UDiv,
    Rem,
    URem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    UShr,
    Eq,
    Ne,
    Lt,
    ULt,
    Le,
    ULe,
    Gt,
    UGt,
    Ge,
    UGe,
    MinF,
    MaxF,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    Neg,
    Not,
    /// The IEEE square root (floats only — the Sse/Avx `Sqrt` leaf
    /// expansion; MXCSR rounding, correctly rounded).
    Sqrt,
}

/// How a call reaches its target. Shared by both IR levels; only the
/// argument/operand rules differ.
#[derive(Clone, PartialEq, Debug)]
pub enum CallTarget<A> {
    /// A direct call to a known method (address from the EE at emit time).
    Direct(MethodHandle),
    /// A virtual dispatch through the receiver's vtable.
    Virtual { method: MethodHandle },
    /// `calli`: indirect through a computed function pointer.
    Indirect(Box<A>),
    /// A call to an EE runtime helper (not a managed method).
    Helper(CorInfoHelpFunc),
}

/// How a shared-generic method receives its instantiation context
/// (corinfo.h:709-715's CORINFO_GENERICS_CTXT_* options bits): through
/// `this` (instance methods on shared generic types), or through the
/// hidden context argument as an InstantiatedMethodDesc* (generic
/// methods) or a MethodTable* (statics on shared generic types).
/// Step_11.3B.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum GenericsContext {
    This,
    MethodDesc,
    MethodTable,
}

/// The frame slot holding a method's generics context, reported in the
/// GC info's fat header (gcinfoencoder.cpp:936-1046): the hidden context
/// argument (a PARAMTYPE entry signature), or `this` when the EE asks
/// for it (FROM_THIS + CORINFO_GENERICS_CTXT_KEEP_ALIVE).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct GenericsContextSlot {
    pub local: LocalId,
    pub kind: GenericsContext,
}

/// HIR: the importer's output. Expression trees rooted at statements.
pub mod hir {
    use super::*;

    /// One method in HIR. Blocks are stored in layout order; EH regions
    /// refer to contiguous ranges of that order.
    pub struct Method {
        pub blocks: Vec<Block>,
        pub locals: Vec<Local>,
        pub eh_regions: Vec<EhRegion>,
        /// Number of leading `locals` that are IL arguments.
        pub num_args: u32,
        /// Number of IL locals after the arguments (temps start at
        /// `num_args + num_il_locals`).
        pub num_il_locals: u32,
        /// Layout facts of every value class the method mentions
        /// (step_10.9; populated at import, one EE query set per class).
        pub struct_layouts: StructLayouts,
        /// The generics context to report in the GC info (step_11.3B);
        /// `None` for ordinary (non-shared) methods.
        pub generics_context: Option<GenericsContextSlot>,
    }

    /// One local/arg/temp slot.
    pub struct Local {
        pub ty: Type,
        pub kind: LocalKind,
        /// GC-relevant: byrefs into the heap must be reported even when the
        /// slot never escapes.
        pub pinned: bool,
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub enum LocalKind {
        IlArg(u32),
        IlLocal(u32),
        Temp,
    }

    pub struct Block {
        pub id: BlockId,
        pub stmts: Vec<Stmt>,
        pub terminator: Terminator,
    }

    /// One HIR statement: a tree with its IL offset.
    pub struct Stmt {
        pub il_offset: IlOffset,
        pub kind: StmtKind,
    }

    /// HIR statements — side effects live here and in nested
    /// [`Expr::Call`]s only. There is no comma/sequence node: anything with
    /// more than one effect is multiple statements.
    pub enum StmtKind {
        /// Assignment to a local/temp (`stloc`, `starg`, temp defs).
        Store { dst: LocalId, value: Expr },
        /// Store through a byref (`stind.*`, `stfld`, `stelem`, `stobj`).
        StoreInd {
            addr: Expr,
            offset: u32,
            value: Expr,
            /// The memory cell's shape (sub-Int32 fields; [`MemAccess::Natural`]
            /// everywhere else).
            access: MemAccess,
        },
        /// Zero a block of memory (`initobj`): `size_of(class)` bytes at
        /// `addr` (step_10.9).
        BlockZero { addr: Expr, class: ClassHandle },
        /// Copy `size` bytes from `src` to `dst` (`cpblk`, 0xFE 17): the
        /// runtime-sized sibling of the struct block copy — a plain byte
        /// copy, no null checks (importer.cpp:11110). Fields evaluate in
        /// IL push order: destination, source, size.
        BlockCopyDyn { dst: Expr, src: Expr, size: Expr },
        /// Fill `size` bytes at `dst` with the low byte of `fill`
        /// (`initblk`, 0xFE 18): the runtime-sized sibling of
        /// [`StmtKind::BlockZero`].
        BlockFillDyn { dst: Expr, fill: Expr, size: Expr },
        /// The `X86Base.X64.DivRem(ulong, ulong, ulong)` expansion
        /// (step_11.14 phase 3): the unsigned 128-by-64 hardware divide —
        /// `dst_q`/`dst_r` receive quotient and remainder of
        /// `hi:lo / divisor` (x64 `div`: rdx:rax / operand). Like the
        /// `div.un` lowering, #DE on a zero divisor or a quotient
        /// overflow is the hardware trap (the managed API documents the
        /// same fault). Fields evaluate in signature order: lo, hi,
        /// divisor.
        DivRem {
            dst_q: LocalId,
            dst_r: LocalId,
            lo: Expr,
            hi: Expr,
            divisor: Expr,
        },
        /// The `X86Base.CpuId(int, int)` expansion (step_11.15): the
        /// `cpuid` instruction with `function` in eax and `sub_id` in ecx;
        /// the four output registers land in `dst_eax`/`dst_ebx`/
        /// `dst_ecx`/`dst_edx` (the tuple's Item1..Item4 order). Pure —
        /// no memory effect, same inputs → same outputs. Fields evaluate
        /// in signature order: function, sub_id.
        CpuId {
            dst_eax: LocalId,
            dst_ebx: LocalId,
            dst_ecx: LocalId,
            dst_edx: LocalId,
            function: Expr,
            sub_id: Expr,
        },
        /// The array bounds check (step_10.8: `ldelem`/`stelem`/`ldelema`):
        /// throws `IndexOutOfRangeException` unless `0 <= index < len`
        /// (unsigned — a negative index is huge); a null array faults on
        /// the length load, the 10.4 trap model's NRE.
        BoundsCheck { array: Expr, index: Expr },
        /// Evaluate and discard (expression statements: `pop` of a call
        /// result, etc.).
        Eval(Expr),
    }

    pub enum Terminator {
        /// Conditional branch; operands are HIR expressions (the compare is
        /// still a tree at this level).
        Branch {
            cond: Expr,
            then: BlockId,
            else_: BlockId,
        },
        Jump {
            target: BlockId,
        },
        Switch {
            value: Expr,
            targets: Vec<BlockId>,
            default: BlockId,
        },
        Return {
            value: Option<Expr>,
        },
        Throw {
            exception: Expr,
        },
        /// `rethrow` (0xFE 1A) — re-raise the in-flight exception: the
        /// never-returning `CORINFO_HELP_RETHROW` call. Valid only inside
        /// a catch handler; ends the block like `throw` (IL after it is
        /// dead — the handler's region-end fallthrough would otherwise
        /// fabricate an edge into the next region).
        Rethrow,
        /// `leave` out of a protected region.
        Leave {
            target: BlockId,
        },
        /// Call a finally funclet, then continue at `continuation` — one
        /// hop of a `leave` chain (step_10.6). Only ever terminates a
        /// synthetic statement-less step block sitting immediately after
        /// the try region being exited.
        CallFinally {
            funclet: BlockId,
            continuation: BlockId,
        },
        /// `endfinally` at the end of a finally/fault funclet.
        EndFinally,
        /// `endfilter` (0xFE 11) at the end of a filter funclet
        /// (step_11.11): the filter's verdict — an Int32, materialized
        /// into rax by the backend; the VM's `CallFilterFunclet` executes
        /// the handler iff rax == 1 (anything else continues the search).
        EndFilter {
            value: Expr,
        },
    }

    /// HIR expression: a typed tree. Children evaluate depth-first in
    /// operand (field) order — i.e. IL push order — and every
    /// exception-point/call in the tree observes that order.
    #[derive(Clone)]
    pub enum Expr {
        Const(Const),
        /// Read a local/arg/temp (`ldloc`, `ldarg`).
        Local(LocalId),
        /// Address of a local (`ldloca`, `ldarga`) — a `ByRef`.
        LocalAddr(LocalId),
        /// Load through a byref (`ldind.*`, `ldfld`, `ldelem`, `ldobj`).
        Load {
            addr: Box<Expr>,
            offset: u32,
            ty: Type,
            /// The memory cell's shape (sub-Int32 fields; [`MemAccess::Natural`]
            /// everywhere else). Narrow accesses always produce `Int32`.
            access: MemAccess,
        },
        /// Instance field access: object plus the EE-supplied offset.
        /// `offset` is baked at import (`getFieldOffset`); lowering turns
        /// the node into address arithmetic — `obj + offset` as a `ByRef`
        /// (step_10.4; the `field` handle stays for EE lookups that need
        /// it, e.g. the write barrier).
        FieldAddr {
            obj: Box<Expr>,
            field: FieldHandle,
            offset: u32,
        },
        /// Static field address, as resolved by the EE.
        StaticFieldAddr {
            field: FieldHandle,
        },
        Unary {
            op: UnaryOp,
            arg: Box<Expr>,
        },
        Binary {
            op: BinaryOp,
            lhs: Box<Expr>,
            rhs: Box<Expr>,
        },
        /// Checked integer arithmetic (`add.ovf`/`sub.ovf`/`mul.ovf` and
        /// their `.un` forms): `op` is `Add`/`Sub`/`Mul` only, and
        /// `unsigned` is the IL `.un` suffix — it selects the overflow
        /// condition (carry vs. signed overflow), not a different
        /// operator. Throws `OverflowException` on overflow; the result
        /// is the promoted operand type. Effectful: a discarded tree
        /// must still evaluate (it can throw).
        BinaryOvf {
            op: BinaryOp,
            unsigned: bool,
            lhs: Box<Expr>,
            rhs: Box<Expr>,
        },
        /// Numeric conversion (`conv.*`); `overflow`/`unsigned` from the IL
        /// opcode suffixes.
        Conv {
            to: Type,
            overflow: bool,
            unsigned: bool,
            arg: Box<Expr>,
        },
        /// Checked conversion (`conv.ovf.*` and their `.un` forms) from an
        /// integer source: `arg`, read with `unsigned_src` signedness,
        /// must fit the target range — `dst_bits` (8/16/32/64) with
        /// `signed_dst` — or the conversion throws `OverflowException`.
        /// The eval-stack types normalize the sub-Int32 targets away, so
        /// the width and target signedness ride the node; `to` is the
        /// pushed stack type (Int32 for ≤32-bit targets, Int64/NativeInt
        /// for 64-bit). Float sources never appear here — the importer
        /// emits the `DBL2*_OVF` helper calls instead (RyuJIT's
        /// fgCastRequiresHelper split, morph.cpp:413-436). Effectful: a
        /// discarded tree can still throw.
        ConvOvf {
            to: Type,
            dst_bits: u32,
            signed_dst: bool,
            unsigned_src: bool,
            arg: Box<Expr>,
        },
        /// `ckfinite`: `arg` (Float or Double) passes through unchanged,
        /// but a non-finite value (`±Inf`, NaN — the exponent all ones)
        /// throws `OverflowException` (RyuJIT's SCK_ARITH_EXCPN helper,
        /// CORINFO_HELP_OVERFLOW — flowgraph.cpp:3494). Effectful like
        /// [`Expr::ConvOvf`].
        CkFinite {
            arg: Box<Expr>,
        },
        /// Round-to-nearest-even float→integer conversion — the
        /// `cvtss2si`/`cvtsd2si` semantics (MXCSR's default rounding),
        /// NOT the truncating `conv.*` of [`Expr::Conv`]. Only the
        /// hardware-intrinsic expansions build it (Sse.ConvertToInt32
        /// & co., step_11.14 phase 3); no IL opcode maps to it.
        /// Out-of-range/NaN yields the hardware's integer-indefinite
        /// value (`0x8000…`). `to` is Int32 or Int64.
        ConvRne {
            to: Type,
            arg: Box<Expr>,
        },
        /// `Interlocked.CompareExchange` — the importer's expansion of the
        /// deliberately self-recursive [Intrinsic] bodies
        /// (Interlocked.cs:322's "Must expand intrinsic"; RyuJIT's
        /// GT_CMPXCHG, importercalls.cpp:4506): atomically compare the
        /// `bits`-wide (8/16/32/64) cell at `addr` with `comparand` and
        /// store `value` when equal; the result is the cell's OLD value.
        /// An opaque global store (GTF_ASG — can fault on a null byref).
        /// Refs are NOT in scope (the write-barrier-on-success story is a
        /// later step; the overloads stay gated). `signed` is the cell
        /// type's signedness (sbyte/short): the narrow old-value result
        /// sign-extends to the IL stack answer when set, zero-extends
        /// otherwise (RyuJIT's `varTypeIsSigned → INS_movsx`,
        /// codegenxarch.cpp:4388-4392).
        AtomicCmpXchg {
            addr: Box<Expr>,
            value: Box<Expr>,
            comparand: Box<Expr>,
            bits: u8,
            signed: bool,
        },
        /// `Interlocked.Exchange` — the same family (RyuJIT's GT_XCHG):
        /// atomically swap `value` into the `bits`-wide cell at `addr`;
        /// the result is the OLD value. Same effects/gates as
        /// [`Expr::AtomicCmpXchg`].
        AtomicXchg {
            addr: Box<Expr>,
            value: Box<Expr>,
            bits: u8,
            signed: bool,
        },
        /// `Interlocked.ExchangeAdd` (RyuJIT's GT_XADD): atomically add
        /// `value` to the `bits`-wide cell at `addr`; the result is the
        /// OLD value. Same effects/gates as [`Expr::AtomicCmpXchg`].
        /// (No narrow ExchangeAdd overloads exist — `signed` is always
        /// false here; kept for one shared atomic shape.)
        AtomicXadd {
            addr: Box<Expr>,
            value: Box<Expr>,
            bits: u8,
            signed: bool,
        },
        /// `Interlocked.MemoryBarrier` (RyuJIT's GT_MEMORYBARRIER,
        /// BARRIER_FULL): a full hardware fence — x64 `lock or dword
        /// [rsp], 0`. Effectful by definition.
        MemoryFence,
        /// `X86Serialize.Serialize()` (RyuJIT's NI_X86Serialize_Serialize,
        /// INS_serialize): the `serialize` instruction (0F 01 E8), a
        /// genuine serializing instruction — NOT a no-op. Effectful by
        /// definition.
        Serialize,
        /// **A call in HIR is an expression node** (`Expr::Call`) that may
        /// nest anywhere a value is legal. Compare `lir::StmtKind::Call`.
        Call {
            target: CallTarget<Expr>,
            sig: CallSig,
            args: Vec<Expr>,
        },
        /// Explicit null check (`ldfld` receiver rules).
        NullCheck {
            arg: Box<Expr>,
        },
        ArrLen {
            array: Box<Expr>,
        },
        /// `ldelema`-style element address (step_10.8): `array + data
        /// offset + index * elem_size`, a managed byref. `elem_size` is
        /// baked at import (the element's cell width; `elem` can't
        /// express the sub-Int32 sizes). The bounds check is the separate
        /// [`StmtKind::BoundsCheck`] statement.
        ArrElemAddr {
            array: Box<Expr>,
            index: Box<Expr>,
            elem: Type,
            elem_size: u32,
        },
        /// `isinst`/`castclass`.
        Cast {
            arg: Box<Expr>,
            class: ClassHandle,
            throwing: bool,
        },
        /// `box`.
        Box {
            arg: Box<Expr>,
            class: ClassHandle,
        },
        /// A struct-typed value produced by copying `size` bytes from
        /// `addr` (value of `ldobj`).
        StructVal {
            addr: Box<Expr>,
            class: ClassHandle,
        },
        /// The exception object a catch handler is entered with (type
        /// `Ref`; step_10.6). Legal only as the value of the synthesized
        /// first store of a catch handler's entry block — the funclet's
        /// incoming argument register is not a value anywhere else.
        CatchArg,
        /// `localloc` (0xFE 0F): `size` bytes of dynamic stack space,
        /// zero-initialized, its address a `native int` value (RyuJIT's
        /// TYP_I_IMPL — a ByRef would become a GC-reported interior-
        /// pointer root; the space holds no tracked roots). Effectful:
        /// the allocation moves rsp for the rest of the method.
        LocAlloc {
            size: Box<Expr>,
        },
        /// The `System.StubHelpers.StubHelpers.NextCallReturnAddress`
        /// intrinsic (step_11.15; RyuJIT's
        /// NI_System_StubHelpers_NextCallReturnAddress,
        /// importercalls.cpp:3541-3549): the address immediately
        /// following the NEXT call instruction emitted for this method —
        /// a NativeInt. RyuJIT lowers it to a `GT_LABEL` whose temp label
        /// codegen defines after the next call
        /// (genDefinePendingCallLabel, codegencommon.cpp:6236); RokaJIT's
        /// codegen does the same (the pending-call-label mechanism). Its
        /// CoreLib body is `throw new UnreachableException()` — an
        /// "Unconditionally expanded intrinsic" (StubHelpers.cs:2566) —
        /// so compiling it literally crashes the reflection invoke stubs
        /// (InvokerEmitUtil.cs:217 emits `call NextCallReturnAddress;
        /// pop` ahead of the target call). The intrinsic's other effect —
        /// `compHasNextCallRetAddr` barring inlining and fast tail calls
        /// — is vacuous here: RokaJIT does neither. Pure (a `lea` of a
        /// code address): a discarded value emits nothing.
        NextCallReturnAddress,
        /// A function pointer with its method (step_11.8): the value form
        /// of `ldftn`/`ldvirtftn` — a NativeInt. The wrapper is the
        /// provenance a delegate `newobj` needs for the EE's
        /// `GetDelegateCtor` substitution (RyuJIT's GT_FTN_ADDR carries
        /// the MethodDesc for the same reason); lowering is the inner
        /// entry expression. Never reaches LIR.
        FtnAddr {
            entry: Box<Expr>,
            method: MethodHandle,
        },
    }

    /// An EH region over a contiguous block range (half-open). In the
    /// importer's layout (step_10.6/11.11) the main-area blocks come first
    /// in IL order — with synthetic `CallFinally` step blocks spliced in —
    /// then each clause's blocks form one contiguous group at the tail
    /// (a filter clause's filter group immediately before its handler
    /// group). A try range IL-containing a nested handler maps to just
    /// its own blocks: `try_start..try_end` is that run, the nested
    /// handler having moved to the tail. A try nested INSIDE a handler
    /// (step_11.11) maps to its run inside the enclosing handler's group.
    pub struct EhRegion {
        pub kind: EhRegionKind,
        pub try_start: BlockId,
        pub try_end: BlockId,
        pub handler_start: BlockId,
        pub handler_end: BlockId,
        /// The try region's IL span (step_11.11) — pure identity cargo
        /// for the SAMETRY marking: nested tries can collapse onto the
        /// same NATIVE range (a try whose only content is a nested
        /// construct reports the inner try's bytes), and the VM's
        /// collided-unwind/rethrow skip keys on the flag, not the
        /// offsets (genReportEHClauses, codegencommon.cpp:2826-2840 —
        /// SAMETRY means same IL try, not same offsets).
        pub il_try_start: u32,
        pub il_try_end: u32,
    }

    pub enum EhRegionKind {
        /// A typed catch; `class_token` is the raw mdToken from the EE's
        /// `getEHinfo`, passed through to the artifact untouched (the VM
        /// resolves and type-tests it). Only context-free clause types
        /// reach this form: the importer probes `embed_generic_handle`
        /// for every catch clause, and a runtime-lookup answer (shared
        /// generic code) converts the clause to a synthesized
        /// [`EhRegionKind::Filter`] instead (RyuJIT's
        /// fgCreateFiltersForGenericExceptions, jiteh.cpp:2596).
        /// step_10.6.
        Catch {
            class_token: u32,
        },
        Finally,
        Fault,
        /// The filter itself is a separate block range.
        Filter {
            filter_start: BlockId,
            filter_end: BlockId,
        },
    }
}

/// LIR: the lowering pass's output and the backend's input. One operation
/// per statement; every value use is an [`Operand`].
pub mod lir {
    use super::*;

    /// Same container shape as HIR (`hir::Method` is consumed whole; the
    /// locals table and EH regions carry over).
    pub struct Method {
        pub blocks: Vec<Block>,
        pub locals: Vec<hir::Local>,
        pub eh_regions: Vec<hir::EhRegion>,
        pub num_args: u32,
        pub num_il_locals: u32,
        /// Carried over from HIR (step_10.9): frame layout, call
        /// classification, and GC roots consult it.
        pub struct_layouts: StructLayouts,
        /// Carried over from HIR (step_11.3B): the generics-context slot
        /// to report in the GC info.
        pub generics_context: Option<GenericsContextSlot>,
    }

    pub struct Block {
        pub id: BlockId,
        /// The terminator is the last statement in the list (`Jump`,
        /// `Branch`, `Return`, …), not a separate field: in LIR everything
        /// is a statement in one explicit order.
        pub stmts: Vec<Stmt>,
    }

    pub struct Stmt {
        pub il_offset: IlOffset,
        pub kind: StmtKind,
    }

    /// A value use in LIR: temp, local, constant, or address-of-local.
    /// Nothing else — no nesting, no loads, no calls.
    #[derive(Copy, Clone, PartialEq, Debug)]
    pub enum Operand {
        /// A temp (`LocalKind::Temp`) defined by exactly one earlier
        /// statement in the same block (or a predecessor, for values live
        /// across blocks).
        Temp(LocalId),
        /// An IL arg/local.
        Local(LocalId),
        Const(Const),
        /// Address of a local (`ByRef`).
        AddrOf(LocalId),
    }

    pub enum StmtKind {
        Copy {
            dst: LocalId,
            src: Operand,
        },
        Unary {
            dst: LocalId,
            op: UnaryOp,
            src: Operand,
        },
        Binary {
            dst: LocalId,
            op: BinaryOp,
            lhs: Operand,
            rhs: Operand,
        },
        /// Checked integer arithmetic (`add.ovf`/`sub.ovf`/`mul.ovf`,
        /// `.un` forms): like [`StmtKind::Binary`], but `op` is restricted
        /// to `Add`/`Sub`/`Mul` and an overflowing result throws
        /// `OverflowException` (the EE's `CORINFO_HELP_OVERFLOW`) instead
        /// of wrapping. `unsigned` selects the unsigned (carry) overflow
        /// condition.
        BinaryOvf {
            dst: LocalId,
            op: BinaryOp,
            unsigned: bool,
            lhs: Operand,
            rhs: Operand,
        },
        Conv {
            dst: LocalId,
            to: Type,
            overflow: bool,
            unsigned: bool,
            src: Operand,
        },
        /// Checked conversion (HIR [`Expr::ConvOvf`]): `src`, read with
        /// `unsigned_src` signedness, must fit the `dst_bits` (8/16/32/64)
        /// target range (`signed_dst`) or the statement throws
        /// `OverflowException` — codegen owns the conditional-throw
        /// sequence (the [`StmtKind::BinaryOvf`] shape). `dst` is defined
        /// only on the no-throw edge.
        ConvOvf {
            dst: LocalId,
            dst_bits: u32,
            signed_dst: bool,
            unsigned_src: bool,
            src: Operand,
        },
        /// `ckfinite` (HIR [`Expr::CkFinite`]): the float `src` passes to
        /// `dst` unchanged; a non-finite value throws `OverflowException`.
        CkFinite {
            dst: LocalId,
            src: Operand,
        },
        /// Round-to-nearest-even float→integer (HIR [`Expr::ConvRne`]):
        /// `cvtss2si`/`cvtsd2si` — the Sse.ConvertToInt32 expansion's
        /// rounding, not `conv.*`'s truncation.
        ConvRne {
            dst: LocalId,
            to: Type,
            src: Operand,
        },
        /// The unsigned 128-by-64 hardware divide (HIR
        /// [`StmtKind::DivRem`]): `dst_q`/`dst_r` = quotient/remainder of
        /// `hi:lo / divisor`.
        DivRem {
            dst_q: LocalId,
            dst_r: LocalId,
            lo: Operand,
            hi: Operand,
            divisor: Operand,
        },
        /// The `cpuid` instruction (HIR [`StmtKind::CpuId`]): eax/ecx in
        /// (`function`/`sub_id`), the four output registers into
        /// `dst_eax`/`dst_ebx`/`dst_ecx`/`dst_edx`.
        CpuId {
            dst_eax: LocalId,
            dst_ebx: LocalId,
            dst_ecx: LocalId,
            dst_edx: LocalId,
            function: Operand,
            sub_id: Operand,
        },
        /// Load through a byref operand at a constant offset.
        Load {
            dst: LocalId,
            addr: Operand,
            offset: u32,
            ty: Type,
            access: MemAccess,
        },
        /// Store through a byref operand at a constant offset.
        Store {
            addr: Operand,
            offset: u32,
            src: Operand,
            access: MemAccess,
        },
        /// **A call in LIR is always a top-level statement** whose result,
        /// if any, lands in a fresh temp. Arguments are operands — any
        /// computation that fed an argument is an earlier statement.
        Call {
            dst: Option<LocalId>,
            target: CallTarget<Operand>,
            sig: CallSig,
            args: Vec<Operand>,
        },
        /// Seeded but never produced: the flattener expands HIR `ArrLen`
        /// directly into a [`StmtKind::Load`] at
        /// [`ARRAY_LENGTH_OFFSET`](crate::ir::ARRAY_LENGTH_OFFSET)
        /// (step_10.8; the x64 ruleset's missing rule is the guard).
        ArrLen {
            dst: LocalId,
            array: Operand,
        },
        /// Seeded but never produced (step_10.8): the flattener expands
        /// HIR `ArrElemAddr` into the `mul`/`add` chain over plain
        /// [`StmtKind::Binary`] statements.
        ArrElemAddr {
            dst: LocalId,
            array: Operand,
            index: Operand,
            elem: Type,
        },
        /// `localloc`: allocate `size` bytes of dynamic stack space
        /// (16-rounded, so call sites keep their alignment), zero it,
        /// and define `dst` (a NativeInt temp) as its base address. A
        /// statement — like `Call`, it never nests: it moves rsp and its
        /// zero-init is a helper call.
        LocAlloc {
            dst: LocalId,
            size: Operand,
        },
        /// The array bounds check (step_10.8): the length load doubles as
        /// the null check; on failure the RNGCHKFAIL helper throws
        /// `IndexOutOfRangeException`. A statement (no result value).
        BoundsCheck {
            array: Operand,
            index: Operand,
        },
        /// The `Interlocked.CompareExchange` expansion (HIR
        /// [`Expr::AtomicCmpXchg`]): `dst` = the OLD value of the
        /// `bits`-wide cell at `addr` after the atomic
        /// compare-and-swap with `comparand`/`value` — extended per
        /// `signed` on the narrow widths.
        AtomicCmpXchg {
            dst: LocalId,
            addr: Operand,
            value: Operand,
            comparand: Operand,
            bits: u8,
            signed: bool,
        },
        /// The `Interlocked.Exchange` expansion (HIR
        /// [`Expr::AtomicXchg`]): `dst` = the OLD value of the
        /// `bits`-wide cell at `addr` after the atomic swap with `value`.
        AtomicXchg {
            dst: LocalId,
            addr: Operand,
            value: Operand,
            bits: u8,
            signed: bool,
        },
        /// The `Interlocked.ExchangeAdd` expansion (HIR
        /// [`Expr::AtomicXadd`]): `dst` = the OLD value of the
        /// `bits`-wide cell at `addr` after the atomic add of `value`.
        AtomicXadd {
            dst: LocalId,
            addr: Operand,
            value: Operand,
            bits: u8,
            signed: bool,
        },
        /// The `Interlocked.MemoryBarrier` expansion (HIR
        /// [`Expr::MemoryFence`]): a full hardware fence. A statement (no
        /// result value).
        MemoryFence,
        /// The `X86Serialize.Serialize()` expansion (HIR
        /// [`Expr::Serialize`]): the `serialize` instruction. A statement
        /// (no result value).
        Serialize,
        /// The `StubHelpers.NextCallReturnAddress` expansion (HIR
        /// [`Expr::NextCallReturnAddress`]): `dst` receives the address
        /// immediately following the next call instruction emitted for
        /// this method (RyuJIT's GT_LABEL + genDefinePendingCallLabel).
        /// Codegen owns the pending-label discipline.
        NextCallReturnAddress {
            dst: LocalId,
        },
        Cast {
            dst: LocalId,
            src: Operand,
            class: ClassHandle,
            throwing: bool,
        },
        Box {
            dst: LocalId,
            src: Operand,
            class: ClassHandle,
        },
        NullCheck {
            arg: Operand,
        },
        /// Copy a struct value between two memory locations (struct
        /// `stloc`/`starg`/`stfld`, `stobj`, `cpobj`, the hidden-retbuf
        /// copy; step_10.9). Both operands are ByRef addresses; the copy
        /// is `size_of(class)` bytes.
        BlockCopy {
            dst_addr: Operand,
            dst_offset: u32,
            src_addr: Operand,
            class: ClassHandle,
        },
        /// Zero `size_of(class)` bytes at `dst_addr` (`initobj`).
        BlockZero {
            dst_addr: Operand,
            class: ClassHandle,
        },
        /// Copy `size` bytes from `src_addr` to `dst_addr` (`cpblk`) —
        /// the runtime-sized sibling of [`StmtKind::BlockCopy`]. Codegen
        /// always emits the `CORINFO_HELP_MEMCPY` call (tier 0: no
        /// inline-threshold decision on a dynamic size).
        BlockCopyDyn {
            dst_addr: Operand,
            src_addr: Operand,
            size: Operand,
        },
        /// Fill `size` bytes at `dst_addr` with the low byte of `fill`
        /// (`initblk`) — the runtime-sized sibling of
        /// [`StmtKind::BlockZero`], always the `CORINFO_HELP_MEMSET`
        /// call.
        BlockFillDyn {
            dst_addr: Operand,
            fill: Operand,
            size: Operand,
        },
        /// `ret` of a register-passed struct value: `addr` is the value's
        /// address (a ByRef operand). The non-register-passed form never
        /// reaches LIR as a struct return — the importer rewrites it to a
        /// block copy through the hidden retbuf pointer plus a plain
        /// `Return` of that pointer.
        ReturnStruct {
            addr: Operand,
            class: ClassHandle,
        },
        /// Conditional branch to `target`; fallthrough is the next
        /// statement (the compare is folded into the branch — the one
        /// permitted two-input operation besides `Binary`).
        Branch {
            cond: BranchCond,
            target: BlockId,
        },
        Jump {
            target: BlockId,
        },
        Switch {
            value: Operand,
            targets: Vec<BlockId>,
            default: BlockId,
        },
        Return {
            value: Option<Operand>,
        },
        Throw {
            exception: Operand,
        },
        /// `rethrow` — the never-returning `CORINFO_HELP_RETHROW` helper
        /// call (no argument: the VM finds the in-flight exception via
        /// the stack walk). Always the block's last statement, like
        /// `Throw`.
        Rethrow,
        Leave {
            target: BlockId,
        },
        /// Catch-handler entry (step_10.6): the throwable (the funclet's
        /// incoming argument-register value) lands in `dst`. Always the
        /// first statement of a catch handler's entry block.
        CatchArg {
            dst: LocalId,
        },
        /// Call a finally funclet, then continue at `continuation` — one
        /// hop of a `leave` chain (step_10.6). Always the block's last
        /// statement.
        CallFinally {
            funclet: BlockId,
            continuation: BlockId,
        },
        EndFinally,
        /// `endfilter` (step_11.11): the filter funclet's verdict — an
        /// Int32 that codegen moves to rax before the funclet epilog.
        EndFilter {
            value: Operand,
        },
    }

    impl StmtKind {
        /// The variant name, for error messages (the LIR types
        /// deliberately have no `Debug`; a rule-miss error that names the
        /// statement kind is what the triage buckets key on).
        pub fn kind_name(&self) -> &'static str {
            match self {
                StmtKind::Copy { .. } => "Copy",
                StmtKind::Unary { .. } => "Unary",
                StmtKind::Binary { .. } => "Binary",
                StmtKind::BinaryOvf { .. } => "BinaryOvf",
                StmtKind::Conv { .. } => "Conv",
                StmtKind::ConvOvf { .. } => "ConvOvf",
                StmtKind::CkFinite { .. } => "CkFinite",
                StmtKind::ConvRne { .. } => "ConvRne",
                StmtKind::DivRem { .. } => "DivRem",
                StmtKind::CpuId { .. } => "CpuId",
                StmtKind::Load { .. } => "Load",
                StmtKind::Store { .. } => "Store",
                StmtKind::Call { .. } => "Call",
                StmtKind::ArrLen { .. } => "ArrLen",
                StmtKind::ArrElemAddr { .. } => "ArrElemAddr",
                StmtKind::LocAlloc { .. } => "LocAlloc",
                StmtKind::BoundsCheck { .. } => "BoundsCheck",
                StmtKind::Cast { .. } => "Cast",
                StmtKind::Box { .. } => "Box",
                StmtKind::NullCheck { .. } => "NullCheck",
                StmtKind::AtomicCmpXchg { .. } => "AtomicCmpXchg",
                StmtKind::AtomicXchg { .. } => "AtomicXchg",
                StmtKind::AtomicXadd { .. } => "AtomicXadd",
                StmtKind::MemoryFence => "MemoryFence",
                StmtKind::Serialize => "Serialize",
                StmtKind::NextCallReturnAddress { .. } => "NextCallReturnAddress",
                StmtKind::BlockCopy { .. } => "BlockCopy",
                StmtKind::BlockZero { .. } => "BlockZero",
                StmtKind::BlockCopyDyn { .. } => "BlockCopyDyn",
                StmtKind::BlockFillDyn { .. } => "BlockFillDyn",
                StmtKind::ReturnStruct { .. } => "ReturnStruct",
                StmtKind::Branch { .. } => "Branch",
                StmtKind::Jump { .. } => "Jump",
                StmtKind::Switch { .. } => "Switch",
                StmtKind::Return { .. } => "Return",
                StmtKind::Throw { .. } => "Throw",
                StmtKind::Rethrow => "Rethrow",
                StmtKind::Leave { .. } => "Leave",
                StmtKind::CatchArg { .. } => "CatchArg",
                StmtKind::CallFinally { .. } => "CallFinally",
                StmtKind::EndFinally => "EndFinally",
                StmtKind::EndFilter { .. } => "EndFilter",
            }
        }
    }

    #[derive(Copy, Clone, PartialEq, Debug)]
    pub enum BranchCond {
        True(Operand),
        False(Operand),
        Cmp {
            op: BinaryOp,
            lhs: Operand,
            rhs: Operand,
        },
    }
}
