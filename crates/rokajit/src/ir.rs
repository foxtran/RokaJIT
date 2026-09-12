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
/// (`un.` prefix in IL).
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
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    Neg,
    Not,
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
        },
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
        /// `leave` out of a protected region.
        Leave {
            target: BlockId,
        },
        /// `endfinally` / `endfilter` at the end of a funclet.
        EndFinally,
    }

    /// HIR expression: a typed tree. Children evaluate depth-first in
    /// operand (field) order — i.e. IL push order — and every
    /// exception-point/call in the tree observes that order.
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
        },
        /// Instance field access: object plus the EE-supplied offset.
        FieldAddr {
            obj: Box<Expr>,
            field: FieldHandle,
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
        /// Numeric conversion (`conv.*`); `overflow`/`unsigned` from the IL
        /// opcode suffixes.
        Conv {
            to: Type,
            overflow: bool,
            unsigned: bool,
            arg: Box<Expr>,
        },
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
        /// `ldelema`-style element address.
        ArrElemAddr {
            array: Box<Expr>,
            index: Box<Expr>,
            elem: Type,
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
    }

    /// An EH region over a contiguous block range (half-open).
    pub struct EhRegion {
        pub kind: EhRegionKind,
        pub try_start: BlockId,
        pub try_end: BlockId,
        pub handler_start: BlockId,
        pub handler_end: BlockId,
    }

    pub enum EhRegionKind {
        Catch {
            class: ClassHandle,
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
        Conv {
            dst: LocalId,
            to: Type,
            overflow: bool,
            unsigned: bool,
            src: Operand,
        },
        /// Load through a byref operand at a constant offset.
        Load {
            dst: LocalId,
            addr: Operand,
            offset: u32,
            ty: Type,
        },
        /// Store through a byref operand at a constant offset.
        Store {
            addr: Operand,
            offset: u32,
            src: Operand,
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
        ArrLen {
            dst: LocalId,
            array: Operand,
        },
        ArrElemAddr {
            dst: LocalId,
            array: Operand,
            index: Operand,
            elem: Type,
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
        Leave {
            target: BlockId,
        },
        EndFinally,
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
