//! Idiomatic Rust enums wrapping the bindgen C++ enum bindings.
//!
//! bindgen emits C++ enums as `pub type X = c_int/c_uint` aliases plus
//! prefixed constants (`CorJitResult_CORJIT_OK`). This module freezes the
//! naming convention (see `decisions/2026-09-11-ee-info-trait-surface.md`):
//!
//! - **Small, closed enums become real Rust enums** with `#[repr]` matching
//!   the ABI and bindgen prefixes stripped (`CorJitResult_CORJIT_OK` →
//!   [`CorJitResult::Ok`]). Conversion from the raw ABI value is checked
//!   ([`CorJitResult::from_raw`] returns `Option`); conversion to raw is
//!   free.
//! - **Large or open-ended enums become transparent newtypes with associated
//!   constants** ([`CorInfoHelpFunc`], [`RelocType`], the `*Attribs` flag
//!   bags). The headers add values to these regularly; a checked Rust enum
//!   would turn every `runtime/` update into an exhaustiveness fire drill,
//!   and none of them are ever matched exhaustively — they are passed
//!   through to the EE.
//! - **Flag sets are `pub struct` newtypes over the raw integer** with
//!   `const` bit constants and `contains`-style helpers, never
//!   `bitflags!` — keeping the dependency tree at zero.

use rokajit_ffi as ffi;

// Kept in scope so the intra-doc links on [`InstructionSet`] and
// [`GetTailCallHelpersFlags`] resolve (same pattern as `ee_info/mod.rs`).
#[allow(unused_imports)]
use crate::ee_info::{Helpers, InliningAndTailCall};

/// Defines a checked Rust enum over a bindgen C++ enum: `#[repr($raw)]`,
/// `from_raw` (checked) and `to_raw`.
macro_rules! ffi_enum {
    ($name:ident, $raw:ty, $doc:literal, $( $variant:ident => $ffi:ident ),* $(,)?) => {
        #[doc = $doc]
        ///
        /// `#[repr]` matches the C++ ABI; values are the bindgen constants
        /// with the enum-name prefix stripped.
        #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
        #[repr($raw)]
        pub enum $name {
            $( $variant = ffi::$ffi as $raw, )*
        }

        impl $name {
            /// Converts a raw ABI value, returning `None` for values not in
            /// the enum (the C++ side can only produce these if the headers
            /// grew new variants).
            #[inline]
            pub fn from_raw(raw: $raw) -> Option<Self> {
                $( if raw == ffi::$ffi as $raw { return Some(Self::$variant); } )*
                None
            }

            /// The raw ABI value, for calls into the gasket forwarders.
            #[inline]
            pub const fn to_raw(self) -> $raw {
                self as $raw
            }
        }
    };
}

ffi_enum!(
    CorJitResult,
    i32,
    "Result of a `compileMethod` call (C++ `CorJitResult`, corjit.h:28).",
    Ok => CorJitResult_CORJIT_OK,
    BadCode => CorJitResult_CORJIT_BADCODE,
    OutOfMem => CorJitResult_CORJIT_OUTOFMEM,
    InternalError => CorJitResult_CORJIT_INTERNALERROR,
    Skipped => CorJitResult_CORJIT_SKIPPED,
    RecoverableError => CorJitResult_CORJIT_RECOVERABLEERROR,
    ImplLimitation => CorJitResult_CORJIT_IMPLLIMITATION,
    R2RUnsupported => CorJitResult_CORJIT_R2R_UNSUPPORTED,
);

ffi_enum!(
    CorInfoType,
    u32,
    "The EE's type enumeration (C++ `CorInfoType`, corinfo.h:593). Bool/char/\
     short and friends are *metadata* types; on the evaluation stack they are \
     `Int` (ECMA-335 §III.1.1.1) — the IR deals only in stack types.",
    Undef => CorInfoType_CORINFO_TYPE_UNDEF,
    Void => CorInfoType_CORINFO_TYPE_VOID,
    Bool => CorInfoType_CORINFO_TYPE_BOOL,
    Char => CorInfoType_CORINFO_TYPE_CHAR,
    Byte => CorInfoType_CORINFO_TYPE_BYTE,
    UByte => CorInfoType_CORINFO_TYPE_UBYTE,
    Short => CorInfoType_CORINFO_TYPE_SHORT,
    UShort => CorInfoType_CORINFO_TYPE_USHORT,
    Int => CorInfoType_CORINFO_TYPE_INT,
    UInt => CorInfoType_CORINFO_TYPE_UINT,
    Long => CorInfoType_CORINFO_TYPE_LONG,
    ULong => CorInfoType_CORINFO_TYPE_ULONG,
    NativeInt => CorInfoType_CORINFO_TYPE_NATIVEINT,
    NativeUInt => CorInfoType_CORINFO_TYPE_NATIVEUINT,
    Float => CorInfoType_CORINFO_TYPE_FLOAT,
    Double => CorInfoType_CORINFO_TYPE_DOUBLE,
    Ptr => CorInfoType_CORINFO_TYPE_PTR,
    ByRef => CorInfoType_CORINFO_TYPE_BYREF,
    ValueClass => CorInfoType_CORINFO_TYPE_VALUECLASS,
    Class => CorInfoType_CORINFO_TYPE_CLASS,
);

ffi_enum!(
    CorInfoOs,
    u32,
    "Target OS passed to `setTargetOS` (C++ `CORINFO_OS`, corinfo.h:1667).",
    WinNt => CORINFO_OS_CORINFO_WINNT,
    Unix => CORINFO_OS_CORINFO_UNIX,
    Apple => CORINFO_OS_CORINFO_APPLE,
);

ffi_enum!(
    CorInfoArch,
    u32,
    "Target architecture (C++ `CorInfoArch`, corinfo.h:1659).",
    X86 => CorInfoArch_CORINFO_ARCH_X86,
    X64 => CorInfoArch_CORINFO_ARCH_X64,
    Arm => CorInfoArch_CORINFO_ARCH_ARM,
    Arm64 => CorInfoArch_CORINFO_ARCH_ARM64,
    LoongArch64 => CorInfoArch_CORINFO_ARCH_LOONGARCH64,
    RiscV64 => CorInfoArch_CORINFO_ARCH_RISCV64,
    Wasm32 => CorInfoArch_CORINFO_ARCH_WASM32,
);

ffi_enum!(
    CorJitFuncKind,
    u32,
    "Which piece of a method an unwind blob describes (C++ `CorJitFuncKind`, \
     corjit.h:61).",
    Root => CorJitFuncKind_CORJIT_FUNC_ROOT,
    Handler => CorJitFuncKind_CORJIT_FUNC_HANDLER,
    Filter => CorJitFuncKind_CORJIT_FUNC_FILTER,
);

ffi_enum!(
    CorInfoInline,
    i32,
    "Inlineability verdict (C++ `CorInfoInline`, corinfo.h:688). Negative \
     values are failures.",
    Pass => CorInfoInline_INLINE_PASS,
    PrejitSuccess => CorInfoInline_INLINE_PREJIT_SUCCESS,
    CheckCanInlineSuccess => CorInfoInline_INLINE_CHECK_CAN_INLINE_SUCCESS,
    CheckCanInlineVmFail => CorInfoInline_INLINE_CHECK_CAN_INLINE_VMFAIL,
    Fail => CorInfoInline_INLINE_FAIL,
    Never => CorInfoInline_INLINE_NEVER,
);

ffi_enum!(
    TypeCompareState,
    i32,
    "Result of a type-comparison query (C++ `TypeCompareState`, corinfo.h:2117).",
    MustNot => TypeCompareState_MustNot,
    May => TypeCompareState_May,
    Must => TypeCompareState_Must,
);

ffi_enum!(
    CorInfoClassId,
    u32,
    "The EE's well-known class ids for `getBuiltinClass` (C++ `CorInfoClassId`, \
     corinfo.h:929).",
    SystemObject => CorInfoClassId_CLASSID_SYSTEM_OBJECT,
    TypedByref => CorInfoClassId_CLASSID_TYPED_BYREF,
    TypeHandle => CorInfoClassId_CLASSID_TYPE_HANDLE,
    FieldHandle => CorInfoClassId_CLASSID_FIELD_HANDLE,
    MethodHandle => CorInfoClassId_CLASSID_METHOD_HANDLE,
    String => CorInfoClassId_CLASSID_STRING,
    ArgumentHandle => CorInfoClassId_CLASSID_ARGUMENT_HANDLE,
    RuntimeType => CorInfoClassId_CLASSID_RUNTIME_TYPE,
    NumericsVectorT => CorInfoClassId_CLASSID_NUMERICS_VECTORT,
);

ffi_enum!(
    CorInfoArrayIntrinsic,
    i32,
    "Which runtime-provided array method a handle denotes (C++ \
     `CorInfoArrayIntrinsic`, corinfo.h:829).",
    Get => CorInfoArrayIntrinsic_GET,
    Set => CorInfoArrayIntrinsic_SET,
    Address => CorInfoArrayIntrinsic_ADDRESS,
    Illegal => CorInfoArrayIntrinsic_ILLEGAL,
);

ffi_enum!(
    CorInfoIsAccessAllowedResult,
    u32,
    "Access-check verdict (C++ `CorInfoIsAccessAllowedResult`, corinfo.h:1437).",
    Allowed => CorInfoIsAccessAllowedResult_CORINFO_ACCESS_ALLOWED,
    Illegal => CorInfoIsAccessAllowedResult_CORINFO_ACCESS_ILLEGAL,
);

ffi_enum!(
    CorInfoWasmType,
    u32,
    "Wasm primitive type for by-value struct passing (C++ `CorInfoWasmType`, \
     corinfo.h:621). `Void` means \"pass/return by reference\".",
    Void => CorInfoWasmType_CORINFO_WASM_TYPE_VOID,
    V128 => CorInfoWasmType_CORINFO_WASM_TYPE_V128,
    F64 => CorInfoWasmType_CORINFO_WASM_TYPE_F64,
    F32 => CorInfoWasmType_CORINFO_WASM_TYPE_F32,
    I64 => CorInfoWasmType_CORINFO_WASM_TYPE_I64,
    I32 => CorInfoWasmType_CORINFO_WASM_TYPE_I32,
);

ffi_enum!(
    GetTypeLayoutResult,
    i32,
    "Outcome of a `getTypeLayout` call (C++ `GetTypeLayoutResult`, \
     corinfo.h:2032). `Partial` means the tree was truncated to fit the \
     caller's buffer.",
    Success => GetTypeLayoutResult_Success,
    Partial => GetTypeLayoutResult_Partial,
    Failure => GetTypeLayoutResult_Failure,
);

ffi_enum!(
    CorInfoCallConvExtension,
    i32,
    "Unmanaged entry-point calling convention (C++ `CorInfoCallConvExtension`, \
     corinfo.h:673). `enum class`, ABI type `c_int`; small closed set, so a \
     checked Rust enum per the frozen enum policy.",
    Managed => CorInfoCallConvExtension_Managed,
    C => CorInfoCallConvExtension_C,
    Stdcall => CorInfoCallConvExtension_Stdcall,
    Thiscall => CorInfoCallConvExtension_Thiscall,
    Fastcall => CorInfoCallConvExtension_Fastcall,
    CMemberFunction => CorInfoCallConvExtension_CMemberFunction,
    StdcallMemberFunction => CorInfoCallConvExtension_StdcallMemberFunction,
    FastcallMemberFunction => CorInfoCallConvExtension_FastcallMemberFunction,
    Swift => CorInfoCallConvExtension_Swift,
);

ffi_enum!(
    CorInfoHFAElemType,
    u32,
    "HFA element kind of a valuetype (C++ `CorInfoHFAElemType`, corhdr.h:1762). \
     `None` = not an HFA (a regular value, not a failure sentinel).",
    None => CorInfoHFAElemType_CORINFO_HFA_ELEM_NONE,
    Float => CorInfoHFAElemType_CORINFO_HFA_ELEM_FLOAT,
    Double => CorInfoHFAElemType_CORINFO_HFA_ELEM_DOUBLE,
    Vector64 => CorInfoHFAElemType_CORINFO_HFA_ELEM_VECTOR64,
    Vector128 => CorInfoHFAElemType_CORINFO_HFA_ELEM_VECTOR128,
);

ffi_enum!(
    InfoAccessType,
    u32,
    "How an embedded value is reached at runtime (C++ `InfoAccessType`, \
     corinfo.h:839): directly, through one indirection, through two, or \
     through a relative indirection.",
    Value => InfoAccessType_IAT_VALUE,
    PValue => InfoAccessType_IAT_PVALUE,
    PPValue => InfoAccessType_IAT_PPVALUE,
    RelPValue => InfoAccessType_IAT_RELPVALUE,
);

/// Which EE helper to call (C++ `CorInfoHelpFunc`, corinfo.h:305).
///
/// A transparent newtype, not a checked enum: the header adds helpers
/// regularly (194 values at runtime commit ac550b6), the JIT passes most of
/// them straight through to `getHelperFtn`, and nothing ever matches on the
/// full set. Constants keep the bindgen names with the
/// `CorInfoHelpFunc_CORINFO_HELP_` prefix stripped.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct CorInfoHelpFunc(pub u32);

impl CorInfoHelpFunc {
    /// Wraps a raw ABI value (all `u32`s are representable; unknown values
    /// pass through).
    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
    /// The raw ABI value.
    #[inline]
    pub const fn to_raw(self) -> u32 {
        self.0
    }

    pub const UNDEF: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_UNDEF);
    pub const DIV: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DIV);
    pub const MOD: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_MOD);
    pub const UDIV: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_UDIV);
    pub const UMOD: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_UMOD);
    pub const LLSH: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_LLSH);
    pub const LRSH: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_LRSH);
    pub const LRSZ: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_LRSZ);
    pub const LMUL: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_LMUL);
    pub const LMUL_OVF: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_LMUL_OVF);
    pub const ULMUL_OVF: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ULMUL_OVF);
    pub const LDIV: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_LDIV);
    pub const LMOD: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_LMOD);
    pub const ULDIV: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ULDIV);
    pub const ULMOD: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ULMOD);
    pub const LNG2FLT: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_LNG2FLT);
    pub const LNG2DBL: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_LNG2DBL);
    pub const ULNG2FLT: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ULNG2FLT);
    pub const ULNG2DBL: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ULNG2DBL);
    pub const DBL2INT_OVF: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DBL2INT_OVF);
    pub const DBL2LNG: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DBL2LNG);
    pub const DBL2LNG_OVF: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DBL2LNG_OVF);
    pub const DBL2UINT_OVF: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DBL2UINT_OVF);
    pub const DBL2ULNG: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DBL2ULNG);
    pub const DBL2ULNG_OVF: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DBL2ULNG_OVF);
    pub const FLTREM: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_FLTREM);
    pub const DBLREM: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DBLREM);
    pub const NEWFAST: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWFAST);
    pub const NEWFAST_MAYBEFROZEN: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWFAST_MAYBEFROZEN);
    pub const NEWSFAST: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWSFAST);
    pub const NEWSFAST_FINALIZE: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWSFAST_FINALIZE);
    pub const NEWSFAST_ALIGN8: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWSFAST_ALIGN8);
    pub const NEWSFAST_ALIGN8_VC: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWSFAST_ALIGN8_VC);
    pub const NEWSFAST_ALIGN8_FINALIZE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWSFAST_ALIGN8_FINALIZE);
    pub const NEW_MDARR: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEW_MDARR);
    pub const NEW_MDARR_RARE: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEW_MDARR_RARE);
    pub const NEWARR_1_DIRECT: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWARR_1_DIRECT);
    pub const NEWARR_1_MAYBEFROZEN: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWARR_1_MAYBEFROZEN);
    pub const NEWARR_1_PTR: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWARR_1_PTR);
    pub const NEWARR_1_VC: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWARR_1_VC);
    pub const NEWARR_1_ALIGN8: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NEWARR_1_ALIGN8);
    pub const INITCLASS: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_INITCLASS);
    pub const INITINSTCLASS: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_INITINSTCLASS);
    pub const ISINSTANCEOFINTERFACE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ISINSTANCEOFINTERFACE);
    pub const ISINSTANCEOFARRAY: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ISINSTANCEOFARRAY);
    pub const ISINSTANCEOFCLASS: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ISINSTANCEOFCLASS);
    pub const ISINSTANCEOFANY: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ISINSTANCEOFANY);
    pub const CHKCASTINTERFACE: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHKCASTINTERFACE);
    pub const CHKCASTARRAY: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHKCASTARRAY);
    pub const CHKCASTCLASS: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHKCASTCLASS);
    pub const CHKCASTANY: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHKCASTANY);
    pub const CHKCASTCLASS_SPECIAL: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHKCASTCLASS_SPECIAL);
    pub const ISINSTANCEOF_EXCEPTION: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ISINSTANCEOF_EXCEPTION);
    pub const BOX: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_BOX);
    pub const BOX_NULLABLE: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_BOX_NULLABLE);
    pub const UNBOX: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_UNBOX);
    pub const UNBOX_TYPETEST: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_UNBOX_TYPETEST);
    pub const UNBOX_NULLABLE: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_UNBOX_NULLABLE);
    pub const GETREFANY: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETREFANY);
    pub const ARRADDR_ST: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ARRADDR_ST);
    pub const LDELEMA_REF: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_LDELEMA_REF);
    pub const THROW: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_THROW);
    pub const RETHROW: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_RETHROW);
    pub const THROWEXACT: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_THROWEXACT);
    pub const USER_BREAKPOINT: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_USER_BREAKPOINT);
    pub const RNGCHKFAIL: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_RNGCHKFAIL);
    pub const OVERFLOW: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_OVERFLOW);
    pub const THROWDIVZERO: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_THROWDIVZERO);
    pub const THROWNULLREF: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_THROWNULLREF);
    pub const VERIFICATION: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_VERIFICATION);
    pub const FAIL_FAST: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_FAIL_FAST);
    pub const METHOD_ACCESS_EXCEPTION: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_METHOD_ACCESS_EXCEPTION);
    pub const FIELD_ACCESS_EXCEPTION: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_FIELD_ACCESS_EXCEPTION);
    pub const CLASS_ACCESS_EXCEPTION: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CLASS_ACCESS_EXCEPTION);
    pub const MON_ENTER: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_MON_ENTER);
    pub const MON_EXIT: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_MON_EXIT);
    pub const GETCLASSFROMMETHODPARAM: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETCLASSFROMMETHODPARAM);
    pub const GETSYNCFROMCLASSHANDLE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETSYNCFROMCLASSHANDLE);
    pub const STOP_FOR_GC: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_STOP_FOR_GC);
    pub const POLL_GC: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_POLL_GC);
    pub const CHECK_OBJ: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHECK_OBJ);
    pub const ASSIGN_REF: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ASSIGN_REF);
    pub const CHECKED_ASSIGN_REF: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHECKED_ASSIGN_REF);
    pub const BULK_WRITEBARRIER: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_BULK_WRITEBARRIER);
    pub const GETFIELDADDR: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETFIELDADDR);
    pub const GETSTATICFIELDADDR: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETSTATICFIELDADDR);
    pub const GETSTATICFIELDADDR_TLS: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETSTATICFIELDADDR_TLS);
    pub const GET_GCSTATIC_BASE: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GET_GCSTATIC_BASE);
    pub const GET_NONGCSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GET_NONGCSTATIC_BASE);
    pub const GETDYNAMIC_GCSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_GCSTATIC_BASE);
    pub const GETDYNAMIC_NONGCSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_NONGCSTATIC_BASE);
    pub const GETPINNED_GCSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETPINNED_GCSTATIC_BASE);
    pub const GETPINNED_NONGCSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETPINNED_NONGCSTATIC_BASE);
    pub const GET_GCSTATIC_BASE_NOCTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GET_GCSTATIC_BASE_NOCTOR);
    pub const GET_NONGCSTATIC_BASE_NOCTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GET_NONGCSTATIC_BASE_NOCTOR);
    pub const GETDYNAMIC_GCSTATIC_BASE_NOCTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_GCSTATIC_BASE_NOCTOR);
    pub const GETDYNAMIC_NONGCSTATIC_BASE_NOCTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_NONGCSTATIC_BASE_NOCTOR);
    pub const GETPINNED_GCSTATIC_BASE_NOCTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETPINNED_GCSTATIC_BASE_NOCTOR);
    pub const GETPINNED_NONGCSTATIC_BASE_NOCTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETPINNED_NONGCSTATIC_BASE_NOCTOR);
    pub const GET_GCTHREADSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GET_GCTHREADSTATIC_BASE);
    pub const GET_NONGCTHREADSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GET_NONGCTHREADSTATIC_BASE);
    pub const GETDYNAMIC_GCTHREADSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_GCTHREADSTATIC_BASE);
    pub const GETDYNAMIC_NONGCTHREADSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE);
    pub const GET_GCTHREADSTATIC_BASE_NOCTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GET_GCTHREADSTATIC_BASE_NOCTOR);
    pub const GET_NONGCTHREADSTATIC_BASE_NOCTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GET_NONGCTHREADSTATIC_BASE_NOCTOR);
    pub const GETDYNAMIC_GCTHREADSTATIC_BASE_NOCTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_GCTHREADSTATIC_BASE_NOCTOR);
    pub const GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR);
    pub const GETDYNAMIC_GCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_GCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED);
    pub const GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED);
    pub const GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED2: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED2);
    pub const GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED2_NOJITOPT: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED2_NOJITOPT);
    pub const GETDIRECTONTHREADLOCALDATA_NONGCTHREADSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETDIRECTONTHREADLOCALDATA_NONGCTHREADSTATIC_BASE);
    pub const DBG_IS_JUST_MY_CODE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DBG_IS_JUST_MY_CODE);
    pub const PROF_FCN_ENTER: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_PROF_FCN_ENTER);
    pub const PROF_FCN_LEAVE: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_PROF_FCN_LEAVE);
    pub const PROF_FCN_TAILCALL: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_PROF_FCN_TAILCALL);
    pub const TAILCALL: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_TAILCALL);
    pub const GETCURRENTMANAGEDTHREADID: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GETCURRENTMANAGEDTHREADID);
    pub const INIT_PINVOKE_FRAME: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_INIT_PINVOKE_FRAME);
    pub const MEMSET: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_MEMSET);
    pub const MEMZERO: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_MEMZERO);
    pub const MEMCPY: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_MEMCPY);
    pub const NATIVE_MEMSET: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_NATIVE_MEMSET);
    pub const RUNTIMEHANDLE_METHOD: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_RUNTIMEHANDLE_METHOD);
    pub const RUNTIMEHANDLE_CLASS: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_RUNTIMEHANDLE_CLASS);
    pub const TYPEHANDLE_TO_RUNTIMETYPE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_TYPEHANDLE_TO_RUNTIMETYPE);
    pub const METHODDESC_TO_STUBRUNTIMEMETHOD: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_METHODDESC_TO_STUBRUNTIMEMETHOD);
    pub const FIELDDESC_TO_STUBRUNTIMEFIELD: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_FIELDDESC_TO_STUBRUNTIMEFIELD);
    pub const TYPEHANDLE_TO_RUNTIMETYPEHANDLE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_TYPEHANDLE_TO_RUNTIMETYPEHANDLE);
    pub const VIRTUAL_FUNC_PTR: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_VIRTUAL_FUNC_PTR);
    pub const READYTORUN_NEW: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_NEW);
    pub const READYTORUN_NEWARR_1: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_NEWARR_1);
    pub const READYTORUN_ISINSTANCEOF: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_ISINSTANCEOF);
    pub const READYTORUN_CHKCAST: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_CHKCAST);
    pub const READYTORUN_GCSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_GCSTATIC_BASE);
    pub const READYTORUN_NONGCSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_NONGCSTATIC_BASE);
    pub const READYTORUN_THREADSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_THREADSTATIC_BASE);
    pub const READYTORUN_THREADSTATIC_BASE_NOCTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_THREADSTATIC_BASE_NOCTOR);
    pub const READYTORUN_NONGCTHREADSTATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_NONGCTHREADSTATIC_BASE);
    pub const READYTORUN_VIRTUAL_FUNC_PTR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_VIRTUAL_FUNC_PTR);
    pub const READYTORUN_GENERIC_HANDLE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_GENERIC_HANDLE);
    pub const READYTORUN_DELEGATE_CTOR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_DELEGATE_CTOR);
    pub const READYTORUN_GENERIC_STATIC_BASE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_READYTORUN_GENERIC_STATIC_BASE);
    pub const EE_PERSONALITY_ROUTINE: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_EE_PERSONALITY_ROUTINE);
    pub const EE_PERSONALITY_ROUTINE_FILTER_FUNCLET: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_EE_PERSONALITY_ROUTINE_FILTER_FUNCLET);
    pub const ASSIGN_REF_EAX: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ASSIGN_REF_EAX);
    pub const ASSIGN_REF_EBX: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ASSIGN_REF_EBX);
    pub const ASSIGN_REF_ECX: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ASSIGN_REF_ECX);
    pub const ASSIGN_REF_ESI: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ASSIGN_REF_ESI);
    pub const ASSIGN_REF_EDI: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ASSIGN_REF_EDI);
    pub const ASSIGN_REF_EBP: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ASSIGN_REF_EBP);
    pub const CHECKED_ASSIGN_REF_EAX: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHECKED_ASSIGN_REF_EAX);
    pub const CHECKED_ASSIGN_REF_EBX: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHECKED_ASSIGN_REF_EBX);
    pub const CHECKED_ASSIGN_REF_ECX: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHECKED_ASSIGN_REF_ECX);
    pub const CHECKED_ASSIGN_REF_ESI: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHECKED_ASSIGN_REF_ESI);
    pub const CHECKED_ASSIGN_REF_EDI: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHECKED_ASSIGN_REF_EDI);
    pub const CHECKED_ASSIGN_REF_EBP: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CHECKED_ASSIGN_REF_EBP);
    pub const LOOP_CLONE_CHOICE_ADDR: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_LOOP_CLONE_CHOICE_ADDR);
    pub const DEBUG_LOG_LOOP_CLONING: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DEBUG_LOG_LOOP_CLONING);
    pub const THROW_ARGUMENTEXCEPTION: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_THROW_ARGUMENTEXCEPTION);
    pub const THROW_ARGUMENTOUTOFRANGEEXCEPTION: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_THROW_ARGUMENTOUTOFRANGEEXCEPTION);
    pub const THROW_NOT_IMPLEMENTED: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_THROW_NOT_IMPLEMENTED);
    pub const THROW_PLATFORM_NOT_SUPPORTED: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_THROW_PLATFORM_NOT_SUPPORTED);
    pub const THROW_TYPE_NOT_SUPPORTED: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_THROW_TYPE_NOT_SUPPORTED);
    pub const THROW_AMBIGUOUS_RESOLUTION_EXCEPTION: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_THROW_AMBIGUOUS_RESOLUTION_EXCEPTION);
    pub const THROW_ENTRYPOINT_NOT_FOUND_EXCEPTION: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_THROW_ENTRYPOINT_NOT_FOUND_EXCEPTION);
    pub const JIT_PINVOKE_BEGIN: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_JIT_PINVOKE_BEGIN);
    pub const JIT_PINVOKE_END: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_JIT_PINVOKE_END);
    pub const JIT_REVERSE_PINVOKE_ENTER: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_JIT_REVERSE_PINVOKE_ENTER);
    pub const JIT_REVERSE_PINVOKE_ENTER_TRACK_TRANSITIONS: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_JIT_REVERSE_PINVOKE_ENTER_TRACK_TRANSITIONS);
    pub const JIT_REVERSE_PINVOKE_EXIT: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_JIT_REVERSE_PINVOKE_EXIT);
    pub const JIT_REVERSE_PINVOKE_EXIT_TRACK_TRANSITIONS: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_JIT_REVERSE_PINVOKE_EXIT_TRACK_TRANSITIONS);
    pub const GVMLOOKUP_FOR_SLOT: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_GVMLOOKUP_FOR_SLOT);
    pub const INTERFACEDISPATCH_FOR_SLOT: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_INTERFACEDISPATCH_FOR_SLOT);
    pub const INTERFACELOOKUP_FOR_SLOT: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_INTERFACELOOKUP_FOR_SLOT);
    pub const STACK_PROBE: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_STACK_PROBE);
    pub const PATCHPOINT: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_PATCHPOINT);
    pub const PATCHPOINT_FORCED: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_PATCHPOINT_FORCED);
    pub const CLASSPROFILE32: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CLASSPROFILE32);
    pub const CLASSPROFILE64: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_CLASSPROFILE64);
    pub const DELEGATEPROFILE32: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DELEGATEPROFILE32);
    pub const DELEGATEPROFILE64: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DELEGATEPROFILE64);
    pub const VTABLEPROFILE32: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_VTABLEPROFILE32);
    pub const VTABLEPROFILE64: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_VTABLEPROFILE64);
    pub const COUNTPROFILE32: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_COUNTPROFILE32);
    pub const COUNTPROFILE64: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_COUNTPROFILE64);
    pub const VALUEPROFILE32: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_VALUEPROFILE32);
    pub const VALUEPROFILE64: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_VALUEPROFILE64);
    pub const VALIDATE_INDIRECT_CALL: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_VALIDATE_INDIRECT_CALL);
    pub const DISPATCH_INDIRECT_CALL: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_DISPATCH_INDIRECT_CALL);
    pub const ALLOC_CONTINUATION: Self = Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ALLOC_CONTINUATION);
    pub const ALLOC_CONTINUATION_METHOD: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ALLOC_CONTINUATION_METHOD);
    pub const ALLOC_CONTINUATION_CLASS: Self =
        Self(ffi::CorInfoHelpFunc_CORINFO_HELP_ALLOC_CONTINUATION_CLASS);
}

/// Relocation kind for [`crate::ee_info::Relocations::record_relocation`]
/// (C++ `CorInfoReloc`, corinfo.h:639).
///
/// Newtype for the same reason as [`CorInfoHelpFunc`]: the set is
/// architecture-specific and grows per-target. The generic kinds are named
/// constants; target-specific ones are constructed with
/// [`RelocType::from_raw`] from the value `getRelocTypeHint` returned.
/// `CorInfoReloc` is an `enum class` (bindgen type `c_int`), hence `i32`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct RelocType(pub i32);

impl RelocType {
    pub const NONE: Self = Self(ffi::CorInfoReloc_NONE);
    pub const DIRECT: Self = Self(ffi::CorInfoReloc_DIRECT);
    pub const RELATIVE32: Self = Self(ffi::CorInfoReloc_RELATIVE32);

    #[inline]
    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }
    #[inline]
    pub const fn to_raw(self) -> i32 {
        self.0
    }
}

/// Target instruction set for [`Helpers::notify_instruction_set_usage`] (C++
/// `CORINFO_InstructionSet`).
///
/// A transparent newtype, not a checked enum: the set is large,
/// architecture-specific, and grows with the hardware (49 values at runtime
/// commit ac550b6); nothing matches on it exhaustively. Only the boundary
/// sentinels are named; everything else is constructed with
/// [`InstructionSet::from_raw`].
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct InstructionSet(pub u32);

impl InstructionSet {
    pub const ILLEGAL: Self = Self(ffi::CORINFO_InstructionSet_InstructionSet_ILLEGAL);
    pub const NONE: Self = Self(ffi::CORINFO_InstructionSet_InstructionSet_NONE);

    /// Wraps a raw ABI value (all `u32`s are representable).
    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
    /// The raw ABI value.
    #[inline]
    pub const fn to_raw(self) -> u32 {
        self.0
    }
}

/// Call-site modifiers for [`InliningAndTailCall::get_tail_call_helpers`]
/// (C++ `CORINFO_GET_TAILCALL_HELPERS_FLAGS`, corinfo.h:1868).
///
/// Hand-rolled rather than `flag_set!` because the per-bit doc on
/// [`GetTailCallHelpersFlags::IS_CALLVIRT`] does not fit the macro shape;
/// the API is otherwise identical to the macro expansion.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct GetTailCallHelpersFlags(pub u32);

impl GetTailCallHelpersFlags {
    /// The callsite is a callvirt instruction.
    pub const IS_CALLVIRT: Self =
        Self(ffi::CORINFO_GET_TAILCALL_HELPERS_FLAGS_CORINFO_TAILCALL_IS_CALLVIRT);
    pub const THIS_ARG_IS_BYREF: Self =
        Self(ffi::CORINFO_GET_TAILCALL_HELPERS_FLAGS_CORINFO_TAILCALL_THIS_ARG_IS_BYREF);

    pub const EMPTY: Self = Self(0);

    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
    #[inline]
    pub const fn to_raw(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn contains(self, bit: Self) -> bool {
        self.0 & bit.0 == bit.0
    }
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOr for GetTailCallHelpersFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

/// Defines a bitmask newtype over a bindgen flag field: `const` bits,
/// `const` `|` composition, `contains`.
macro_rules! flag_set {
    ($name:ident, $raw:ty, $doc:literal, $( $bit:ident => $ffi:ident ),* $(,)?) => {
        #[doc = $doc]
        #[repr(transparent)]
        #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
        pub struct $name(pub $raw);

        impl $name {
            $( pub const $bit: Self = Self(ffi::$ffi); )*

            pub const EMPTY: Self = Self(0);

            #[inline]
            pub const fn from_raw(raw: $raw) -> Self {
                Self(raw)
            }
            #[inline]
            pub const fn to_raw(self) -> $raw {
                self.0
            }
            #[inline]
            pub const fn contains(self, bit: Self) -> bool {
                self.0 & bit.0 == bit.0
            }
            #[inline]
            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }
        }

        impl ::std::ops::BitOr for $name {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }
    };
}

flag_set!(
    AllocMemFlags,
    u32,
    "Per-chunk allocation kind for `allocMem` (C++ `CorJitAllocMemFlag`, \
     corjit.h:48).",
    HOT_CODE => CorJitAllocMemFlag_CORJIT_ALLOCMEM_HOT_CODE,
    COLD_CODE => CorJitAllocMemFlag_CORJIT_ALLOCMEM_COLD_CODE,
    READONLY_DATA => CorJitAllocMemFlag_CORJIT_ALLOCMEM_READONLY_DATA,
    HAS_POINTERS_TO_CODE => CorJitAllocMemFlag_CORJIT_ALLOCMEM_HAS_POINTERS_TO_CODE,
);

flag_set!(
    CallInfoFlags,
    u32,
    "Call modifiers passed to `getCallInfo` (C++ `CORINFO_CALLINFO_FLAGS`, \
     corinfo.h:1260).",
    ALLOWINSTPARAM => CORINFO_CALLINFO_FLAGS_CORINFO_CALLINFO_ALLOWINSTPARAM,
    CALLVIRT => CORINFO_CALLINFO_FLAGS_CORINFO_CALLINFO_CALLVIRT,
    DISALLOW_STUB => CORINFO_CALLINFO_FLAGS_CORINFO_CALLINFO_DISALLOW_STUB,
    SECURITYCHECKS => CORINFO_CALLINFO_FLAGS_CORINFO_CALLINFO_SECURITYCHECKS,
    LDFTN => CORINFO_CALLINFO_FLAGS_CORINFO_CALLINFO_LDFTN,
);

flag_set!(
    MethodAttribs,
    u32,
    "Method attribute bits returned by `getMethodAttribs` (C++ `CorInfoFlag`, \
     corinfo.h:732). Only the bits a consumer has needed so far are named; \
     the rest are reachable via [`MethodAttribs::from_raw`] bits.",
    STATIC => CorInfoFlag_CORINFO_FLG_STATIC,
    FINAL => CorInfoFlag_CORINFO_FLG_FINAL,
    SYNCH => CorInfoFlag_CORINFO_FLG_SYNCH,
    VIRTUAL => CorInfoFlag_CORINFO_FLG_VIRTUAL,
    INTRINSIC_TYPE => CorInfoFlag_CORINFO_FLG_INTRINSIC_TYPE,
);

flag_set!(
    ClassAttribs,
    u32,
    "Class attribute bits returned by `getClassAttribs` (C++ `CorInfoFlag`, \
     corinfo.h:732). Bits are named on demand, like [`MethodAttribs`].",
    // Shared prefix with MethodAttribs (both are CorInfoFlag); value classes
    // are the one bit every class consumer tests.
    VALUECLASS => CorInfoFlag_CORINFO_FLG_VALUECLASS,
    // The class is a generic type parameter (!!T/!T — corinfo.h:780);
    // answers about it describe the canonical representative.
    GENERIC_TYPE_VARIABLE => CorInfoFlag_CORINFO_FLG_GENERIC_TYPE_VARIABLE,
    // A delegate class (corinfo.h:775): delegate construction/invocation
    // is special-cased by the EE — handled in the importer (step_11.8).
    DELEGATE => CorInfoFlag_CORINFO_FLG_DELEGATE,
    // A variable-sized class (only String today — corinfo.h:773): the JIT
    // never allocates it; `newobj` calls the internalcall .ctor with no
    // `this`, and the runtime redirects to the allocating static `Ctor`
    // whose return is the object (importer.cpp:9056).
    VAROBJSIZE => CorInfoFlag_CORINFO_FLG_VAROBJSIZE,
);

flag_set!(
    EhClauseFlags,
    u32,
    "EH clause kind bits (C++ `CORINFO_EH_CLAUSE_FLAGS`, corinfo.h:623).",
    FILTER => CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_FILTER,
    FINALLY => CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_FINALLY,
    FAULT => CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_FAULT,
    SAMETRY => CORINFO_EH_CLAUSE_FLAGS_CORINFO_EH_CLAUSE_SAMETRY,
);

flag_set!(
    CorInfoInitClassResult,
    u32,
    "Class-initialization verdict returned by `initClass` (C++ \
     `CorInfoInitClassResult`, corinfo.h:966). The EE may combine \
     `USE_HELPER | DONT_INLINE`, so this is a flag set, not a plain enum.",
    NOT_REQUIRED => CorInfoInitClassResult_CORINFO_INITCLASS_NOT_REQUIRED,
    INITIALIZED => CorInfoInitClassResult_CORINFO_INITCLASS_INITIALIZED,
    USE_HELPER => CorInfoInitClassResult_CORINFO_INITCLASS_USE_HELPER,
    DONT_INLINE => CorInfoInitClassResult_CORINFO_INITCLASS_DONT_INLINE,
);

flag_set!(
    AccessFlags,
    u32,
    "Access modifiers for `getFunctionEntryPoint` (C++ `CORINFO_ACCESS_FLAGS`, \
     corinfo.h:795). `CORINFO_ACCESS_ANY` is zero, i.e. [`AccessFlags::EMPTY`]. \
     Only the method-access bits are named; the field-access bits (GET/SET/\
     ADDRESS/INIT_ARRAY/INLINECHECK) are reachable via [`AccessFlags::from_raw`].",
    THIS => CORINFO_ACCESS_FLAGS_CORINFO_ACCESS_THIS,
    PREFER_SLOT_OVER_TEMPORARY_ENTRYPOINT => CORINFO_ACCESS_FLAGS_CORINFO_ACCESS_PREFER_SLOT_OVER_TEMPORARY_ENTRYPOINT,
    NONNULL => CORINFO_ACCESS_FLAGS_CORINFO_ACCESS_NONNULL,
    LDFTN => CORINFO_ACCESS_FLAGS_CORINFO_ACCESS_LDFTN,
    UNMANAGED_CALLER_MAYBE => CORINFO_ACCESS_FLAGS_CORINFO_ACCESS_UNMANAGED_CALLER_MAYBE,
);

flag_set!(
    MethodRuntimeFlags,
    u32,
    "Flags the JIT reports back about a compiled method (C++ \
     `CorInfoMethodRuntimeFlags`, corinfo.h:785).",
    BAD_INLINEE => CorInfoMethodRuntimeFlags_CORINFO_FLG_BAD_INLINEE,
    SWITCHED_TO_MIN_OPT => CorInfoMethodRuntimeFlags_CORINFO_FLG_SWITCHED_TO_MIN_OPT,
    SWITCHED_TO_OPTIMIZED => CorInfoMethodRuntimeFlags_CORINFO_FLG_SWITCHED_TO_OPTIMIZED,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cor_jit_result_round_trip() {
        assert_eq!(CorJitResult::Ok.to_raw(), 0);
        assert_eq!(CorJitResult::from_raw(0), Some(CorJitResult::Ok));
        assert_eq!(
            CorJitResult::from_raw(ffi::CorJitResult_CORJIT_INTERNALERROR),
            Some(CorJitResult::InternalError)
        );
        assert_eq!(CorJitResult::from_raw(12345), None);
    }

    #[test]
    fn cor_info_type_values_match_cpp() {
        // Spot-check against the values bindgen read from the headers.
        assert_eq!(CorInfoType::Int.to_raw(), 8);
        assert_eq!(CorInfoType::ByRef.to_raw(), 17);
        assert_eq!(CorInfoType::Class.to_raw(), 19);
    }

    #[test]
    fn flags_compose() {
        let f = AllocMemFlags::HOT_CODE | AllocMemFlags::HAS_POINTERS_TO_CODE;
        assert!(f.contains(AllocMemFlags::HOT_CODE));
        assert!(!f.contains(AllocMemFlags::COLD_CODE));
    }
}
