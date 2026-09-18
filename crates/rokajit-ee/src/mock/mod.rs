//! `MockEe` — an `EeInfo` implementation with canned answers, proving the
//! trait is implementable without a live EE (and serving as the test double
//! for compiler-core unit tests). Test-only.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;

use rokajit_ffi as ffi;

// Only the in-module tests use the trait surface (`EeInfo`, `ChunkRequest`,
// `AllocMemFlags`); the mock bodies reach them through `super::MockEe`.
#[cfg(test)]
use crate::ee_info::*;
use crate::enums::*;
use crate::handles::*;

mod class_queries;
mod debug_info;
mod field_queries;
mod helpers;
mod host;
mod inlining_and_tail_call;
mod method_queries;
mod output_sinks;
mod pgo;
mod relocations;
mod tokens_and_signatures;

/// A canned method signature — the stack-relevant shape only (importer
/// fixtures; step_07.2). `ret_class`/`arg_classes` carry the value-class
/// handles of struct-typed elements (step_10.9); both are `None`/empty
/// for struct-free signatures.
#[derive(Clone)]
pub struct MockSig {
    pub ret: CorInfoType,
    pub args: Vec<CorInfoType>,
    pub has_this: bool,
    /// The value-class handle for a `CorInfoType::ValueClass` return.
    pub ret_class: Option<ClassHandle>,
    /// Per-argument value-class handles (`None` entries for non-struct
    /// args); shorter-than-`args` is padded with `None`.
    pub arg_classes: Vec<Option<ClassHandle>>,
}

/// One signature argument as the mock's cursor table stores it: the EE
/// type plus the value-class handle `getArgType` reports for structs.
#[derive(Copy, Clone)]
pub struct MockArg {
    pub ty: CorInfoType,
    pub class: Option<ClassHandle>,
    /// The `CORINFO_TYPE_MOD_PINNED` bit (a `fixed` local).
    pub pinned: bool,
}

/// A canned value class (step_10.9): the layout and SysV descriptor facts
/// the class queries answer with.
pub struct MockClass {
    /// The fake handle the class queries key on.
    pub handle: ClassHandle,
    pub size: u32,
    pub align: u32,
    /// GC-pointer cells as `(offset, is_byref)`; drives `get_class_gc_layout`.
    pub gc_cells: Vec<(u32, bool)>,
    /// The canned SysV descriptor; `None` answers "not register-passed".
    pub sysv: Option<ffi::SYSTEMV_AMD64_CORINFO_STRUCT_REG_PASSING_DESCRIPTOR>,
}

/// Builds a canned SysV descriptor from `(classification, size)` pairs
/// (one per eightbyte, offsets 0/8). An empty slice builds the
/// not-passed-in-registers answer.
pub fn sysv_descriptor(
    eightbytes: &[(ffi::SystemVClassificationType, u8)],
) -> ffi::SYSTEMV_AMD64_CORINFO_STRUCT_REG_PASSING_DESCRIPTOR {
    let mut desc: ffi::SYSTEMV_AMD64_CORINFO_STRUCT_REG_PASSING_DESCRIPTOR =
        unsafe { std::mem::zeroed() };
    desc.passedInRegisters = !eightbytes.is_empty();
    desc.eightByteCount = eightbytes.len() as u8;
    for (i, &(class, size)) in eightbytes.iter().enumerate() {
        desc.eightByteClassifications[i] = class;
        desc.eightByteSizes[i] = size;
        desc.eightByteOffsets[i] = (i * 8) as u8;
    }
    desc
}

/// A canned method the mock resolves metadata tokens to.
pub struct MockMethod {
    /// The fake handle `resolve_token`/`get_call_info` hand back.
    pub handle: MethodHandle,
    pub sig: MockSig,
    /// Extra callconv bits OR'd into the sig mirrors (step_11.3B:
    /// `CORINFO_CALLCONV_GENERIC`/`CORINFO_CALLCONV_PARAMTYPE`) — the
    /// MockSig surface itself has no callconv flags.
    pub call_conv_flags: ffi::CorInfoCallConv,
    /// Index into the mock's arg-list table (drives sig-cursor walking).
    arg_list: usize,
}

/// A canned instance field the mock resolves metadata tokens to and the
/// field queries (`get_field_offset`/`get_field_type`/`is_field_static`)
/// answer from (step_10.4).
pub struct MockField {
    /// The fake handle `resolve_token` hands back.
    pub handle: FieldHandle,
    pub offset: u32,
    pub ty: CorInfoType,
    pub is_static: bool,
    /// The value-class handle for a struct-typed field (step_10.9).
    pub value_class: Option<ClassHandle>,
    /// Statics-pack knobs (step_10.7): whether `get_field_info` sets
    /// `CORINFO_FLG_FIELD_INITCLASS`, whether it sets
    /// `CORINFO_FLG_FIELD_STATIC_IN_HEAP` (the boxed-static indirection),
    /// and a forced `fieldAccessor` override (to can an unsupported
    /// accessor family).
    pub init_class: bool,
    pub in_heap: bool,
    pub accessor: Option<ffi::CORINFO_FIELD_ACCESSOR>,
    /// The canned `CORINFO_FIELD_INFO.helper` for the
    /// `GENERICS_STATIC_HELPER` accessor (step_11.3D) — e.g.
    /// `GET_GCSTATIC_BASE`; absent cans a zeroed (rejected) helper.
    pub statics_helper: Option<CorInfoHelpFunc>,
    /// Step_11.7 knobs: the canned `get_thread_local_field_info` index
    /// (the TLS_MANAGED accessor's helper argument), a forced
    /// `CORINFO_ACCESS_ILLEGAL` access verdict (the access-callout path —
    /// the callout desc mirrors the real EE's FIELD_ACCESS_EXCEPTION
    /// shape), and `fieldLookup` as IAT_PVALUE (the
    /// address-through-a-cell shape) instead of the default IAT_VALUE
    /// final address.
    pub tls_index: u32,
    pub access_illegal: bool,
    pub address_via_cell: bool,
}

/// Canned EE. Every query returns the stored/default value; output sinks
/// record what they were handed so tests can assert on the flow.
#[derive(Default)]
pub struct MockEe {
    pub method_attribs: MethodAttribs,
    pub class_attribs: ClassAttribs,
    /// Per-class `get_class_attribs` overrides (step_11.14: one fixture
    /// mixing a GC-pointer-carrying struct with a plain one), keyed by
    /// the class handle's raw value; absent handles answer
    /// `class_attribs`.
    pub class_attribs_overrides: HashMap<usize, ClassAttribs>,
    pub method_name: Option<String>,
    /// Canned methods for `resolve_token`/`get_call_info`/`get_method_sig`,
    /// keyed by metadata token.
    pub methods: HashMap<u32, MockMethod>,
    /// Canned fields for `resolve_token` and the field queries, keyed by
    /// metadata token (step_10.4).
    pub fields: HashMap<u32, MockField>,
    /// Method tokens for which `get_call_info` cans a non-`CORINFO_CALL`
    /// kind (the importer's "non-direct call kind" path; step_10.4).
    pub non_direct_calls: std::collections::HashSet<u32>,
    /// Per-token `get_call_info` kind overrides (step_10.12: can STUB or
    /// LDVIRTFTN verdicts for the interface/generic-virtual helper
    /// fallback). Consulted before `non_direct_calls`.
    pub call_kinds: HashMap<u32, ffi::CORINFO_CALL_KIND>,
    /// Canned `get_method_vtable_offset` verdicts (step_10.12), keyed by
    /// the method handle's raw value; absent handles answer the default —
    /// no chunk indirection, slot offset 0x28 (an ordinary mid-table
    /// slot).
    pub vtable_offsets: HashMap<usize, (u32, u32, bool)>,
    /// Canned `find_sig` answers for `calli`'s StandAloneSig tokens
    /// (step_10.12), registered via [`MockEe::add_calli_sig`]: the
    /// signature body and its arg-list table index.
    calli_sigs: HashMap<u32, (MockSig, usize)>,
    /// Low-nibble callConv overrides for `calli` callsite sigs (step_11.12:
    /// the unmanaged flavors), keyed like `calli_sigs`; absent tokens keep
    /// the DEFAULT/HASTHIS shape derived from the sig.
    pub calli_sig_convs: HashMap<u32, ffi::CorInfoCallConv>,
    /// The canned `get_unmanaged_call_conv` answer (step_11.12); `None`
    /// cans `(Managed, false)` — the EE's "this sig is not unmanaged".
    pub unmanaged_call_conv: Option<(CorInfoCallConvExtension, bool)>,
    /// The canned `get_ee_info` answer (step_11.16: the unmanaged calli's
    /// InlinedCallFrame slot sizes from
    /// `inlinedCallFrameInfo.size`); `None` cans zeros — a zero frame
    /// size is the named `Internal` test-setup error, so a fixture
    /// exercising the GC transition must can the real EE's answer.
    pub ee_info: Option<ffi::CORINFO_EE_INFO>,
    /// The canned `get_jit_flags` answer (step_11.16: the
    /// CORJIT_FLAG_PUBLISH_SECRET_PARAM gate); `None` cans zero flags.
    pub jit_flags: Option<ffi::CORJIT_FLAGS>,
    /// The flags each `get_call_info` call arrived with, in order
    /// (step_10.4 tests: `call` passes EMPTY, `callvirt` CALLVIRT).
    pub call_info_flags: RefCell<Vec<CallInfoFlags>>,
    /// The canned `get_new_helper` verdict's helper; `None` cans
    /// `CORINFO_HELP_NEWFAST` (step_10.4).
    pub new_helper: Option<CorInfoHelpFunc>,
    /// The canned `get_new_arr_helper` verdict (step_10.8); `None` cans
    /// `CORINFO_HELP_NEWARR_1_PTR`.
    pub new_arr_helper: Option<CorInfoHelpFunc>,
    /// Class handles (raw values) for which `is_sd_array` cans `false`
    /// (step_10.8's non-SZ-array gate); the default answers the happy
    /// path — every class is an SZ array.
    pub non_sd_arrays: std::collections::HashSet<usize>,
    /// The canned `get_array_rank` answer (the newobj MD-array branch
    /// picks `NEW_MDARR_RARE` on rank 1, `NEW_MDARR` otherwise); the
    /// default 0 answers an ordinary multi-dimensional array.
    pub array_rank: u32,
    /// The canned `init_class` verdict (step_10.4). `EMPTY` is
    /// `CORINFO_INITCLASS_NOT_REQUIRED` (bit value 0) — the default.
    pub init_class_result: CorInfoInitClassResult,
    /// The canned verdict for the method-prolog query
    /// (`init_class(None, None, context)` — the 10.8 entry-cctor fix);
    /// `None` answers NOT_REQUIRED, keeping the prolog inert by default.
    pub prolog_init_class: Option<CorInfoInitClassResult>,
    /// Canned `get_box_helper`/`get_un_box_helper` verdicts (step_10.5);
    /// `None` cans `CORINFO_HELP_BOX`/`CORINFO_HELP_UNBOX`.
    pub box_helper: Option<CorInfoHelpFunc>,
    pub unbox_helper: Option<CorInfoHelpFunc>,
    /// Per-class overrides consulted BEFORE `box_helper`/`unbox_helper`
    /// (step_11.4: one fixture mixing a `Nullable<T>` class with a plain
    /// one), keyed by the class handle's raw value.
    pub box_helpers: HashMap<usize, CorInfoHelpFunc>,
    pub unbox_helpers: HashMap<usize, CorInfoHelpFunc>,
    /// Canned `get_casting_helper` override (step_10.5); `None` cans
    /// CHKCASTANY (throwing) / ISINSTANCEOFANY.
    pub casting_helper: Option<CorInfoHelpFunc>,
    /// `as_cor_info_type` overrides keyed by the class handle's raw value
    /// (step_10.5: canning a *primitive* class like System.Int32, which is
    /// a value class whose CorInfoType is not VALUECLASS). Absent handles
    /// keep the default (ValueClass if registered in `classes`, Class
    /// otherwise).
    pub class_cor_info_types: HashMap<usize, CorInfoType>,
    /// Canned `get_type_instantiation_argument` answers (step_11.14
    /// phase 3: the vector element type of a `Vector128<T>` handle),
    /// keyed by (class handle raw value, index); absent keys answer
    /// `None`.
    pub type_inst_args: HashMap<(usize, u32), ClassHandle>,
    /// Canned directly-callable entry points for `get_function_entry_point`
    /// (step_07.5 codegen tests), keyed by the method handle's raw value.
    /// Absent handles get a zeroed lookup (`IAT_VALUE`, null address).
    pub entry_points: HashMap<usize, usize>,
    /// Canned entry-point *slots* for `get_function_entry_point` (step_07.7
    /// codegen tests of the IAT_PVALUE indirect-call form), keyed like
    /// [`Self::entry_points`] and consulted after it.
    pub entry_point_slots: HashMap<usize, usize>,
    /// Registered signature argument lists. Fake `ArgListHandle` cursors
    /// encode `(list, index)` — the mock never dereferences handles.
    arg_lists: Vec<Vec<MockArg>>,
    /// Canned value classes (step_10.9), keyed by the handle's raw value.
    pub classes: HashMap<usize, MockClass>,
    /// Metadata tokens `resolve_token` answers with a class handle
    /// (the `ldobj`/`stobj`/`cpobj`/`initobj` operand tokens; step_10.9).
    pub class_tokens: HashMap<u32, ClassHandle>,
    /// Per-method declaring-class overrides for `get_method_class`, keyed
    /// by the method handle's raw value (struct instance methods;
    /// step_10.9). Absent handles keep the default (the method handle's
    /// own address).
    pub method_classes: HashMap<usize, ClassHandle>,
    /// Canned EH clauses returned by `get_eh_info` by index (10.6). Empty
    /// keeps the old behavior (a zeroed clause).
    pub eh_clauses: Vec<ffi::CORINFO_EH_CLAUSE>,
    /// The canned `get_token_type_as_handle` answer (step_10.10 `ldtoken`):
    /// the RuntimeTypeHandle/RuntimeMethodHandle/RuntimeFieldHandle
    /// stand-in class.
    pub token_type_class: Option<ClassHandle>,
    /// Canned `get_builtin_class` answers (the TypedReference ops'
    /// CLASSID_TYPED_BYREF/CLASSID_TYPE_HANDLE), keyed by the
    /// `CorInfoClassId`'s raw value; absent ids answer `None`.
    pub builtin_classes: HashMap<u32, ClassHandle>,
    /// `embed_generic_handle` rejection forms (step_10.10): can a
    /// generic-context runtime lookup or an indirection-cell answer. The
    /// default is a direct, token-deterministic IAT_VALUE handle.
    pub embed_runtime_lookup: bool,
    pub embed_indirection: bool,
    /// A canned `CORINFO_LOOKUP` answer for `embed_generic_handle`
    /// (step_11.3B: the runtime-lookup emitter's fixture) — returned
    /// verbatim, ahead of the `embed_*` rejection flags.
    pub embed_lookup: Option<ffi::CORINFO_LOOKUP>,
    /// The canned `get_location_of_this_type` answer (step_11.4: the
    /// prolog class-init trigger in shared generic code); `None` cans the
    /// zeroed `!needsRuntimeLookup` verdict (unshared code).
    pub this_type_lookup: Option<ffi::CORINFO_LOOKUP_KIND>,
    /// Canned `get_call_info` generics-context answers (step_11.3B),
    /// keyed by metadata token: the (tagged) `contextHandle` and
    /// `exactContextNeedsRuntimeLookup`.
    pub call_contexts: HashMap<u32, (usize, bool)>,
    /// Canned `entryPointLookup` answers for a
    /// `CORINFO_CALL_CODE_POINTER` verdict (step_11.3B), keyed by
    /// metadata token — copied into the call-info union verbatim.
    pub call_code_pointer_lookups: HashMap<u32, ffi::CORINFO_LOOKUP>,
    /// Canned `thisTransform` answers for a `constrained.` callvirt
    /// (step_11.3C), keyed by the call's method metadata token; absent
    /// tokens answer CORINFO_NO_THIS_TRANSFORM (the zeroed default).
    pub this_transforms: HashMap<u32, ffi::CORINFO_THIS_TRANSFORM>,
    /// A canned one-class method instantiation on the call sig
    /// (sigInst.methInstCount = 1 — the `GetArrayDataReference<T>`
    /// intrinsic's element class), keyed by the call's method metadata
    /// token; the boxed cell keeps the `methInst` pointer stable.
    pub call_meth_inst: HashMap<u32, Box<ffi::CORINFO_CLASS_HANDLE>>,
    /// Canned [Intrinsic] methods (the GetMethodTable fixtures), keyed by
    /// the method handle's raw value; `is_intrinsic` answers true for them.
    pub intrinsic_methods: std::collections::HashSet<usize>,
    /// Canned `get_delegate_ctor` answer (step_11.8): the alternate ctor
    /// plus the pArg3/4/5 values the EE writes back. `None` = no usable
    /// delegate ctor (the C++ null return).
    pub delegate_ctor: Option<(MethodHandle, [usize; 3])>,
    /// Canned `get_class_name_from_metadata` answers — (name, namespace) —
    /// keyed by the class handle's raw value; absent handles answer `None`.
    pub class_names: HashMap<usize, (String, Option<String>)>,
    /// Canned `get_method_declaring_namespace` answers (step_11.8),
    /// keyed by the method handle's raw value; absent handles answer
    /// `None`.
    pub method_namespaces: HashMap<usize, String>,
    /// Canned `get_method_declaring_enclosing_class_name` answers
    /// (step_11.14 — the nested X64/Wide fixtures), keyed by the method
    /// handle's raw value; absent handles answer `None` (non-nested).
    pub method_enclosing_classes: HashMap<usize, String>,
    /// The constrained-token operand each `get_call_info` call arrived
    /// with (its metadata token; 0 when the call had no `constrained.`
    /// prefix), in order (step_11.3C).
    pub constrained_seen: RefCell<Vec<u32>>,
    /// Canned `get_class_static_dynamic_info` /
    /// `get_class_thread_static_dynamic_info` pointers (step_11.7's
    /// shared-static-helper argument); `None` = the C++ null.
    pub static_dynamic_info: Option<usize>,
    pub thread_static_dynamic_info: Option<usize>,
    /// Can the string-literal queries' indirection-cell answer
    /// (IAT_PVALUE — step_11.7); the default is a direct IAT_VALUE
    /// object reference.
    pub string_literal_cell: bool,
    /// Sink calls observed, newest last, as "(kind, detail)" strings.
    pub sink_log: RefCell<Vec<String>>,
    /// Buffers handed out by the fake `alloc_mem`/`alloc_gc_info`, kept
    /// alive for the mock's lifetime.
    buffers: RefCell<Vec<Box<[u8]>>>,
}

// Cursors pack a (list, index) pair into the opaque handle value; the +1
// keeps the first cursor of each list non-null.
const CURSOR_INDEX_BITS: usize = 20;

impl MockEe {
    fn cursor_raw(list: usize, index: usize) -> ffi::CORINFO_ARG_LIST_HANDLE {
        (((list + 1) << CURSOR_INDEX_BITS) | index) as ffi::CORINFO_ARG_LIST_HANDLE
    }

    fn cursor(list: usize, index: usize) -> Option<ArgListHandle> {
        ArgListHandle::from_raw(Self::cursor_raw(list, index))
    }

    fn decode_cursor(cursor: ArgListHandle) -> (usize, usize) {
        let raw = cursor.as_raw() as usize;
        (
            (raw >> CURSOR_INDEX_BITS) - 1,
            raw & ((1 << CURSOR_INDEX_BITS) - 1),
        )
    }

    /// Builds the `CORINFO_SIG_INFO` mirror over an already-registered
    /// argument list.
    fn build_sig_info(
        &self,
        call_conv: ffi::CorInfoCallConv,
        ret: CorInfoType,
        ret_class: Option<ClassHandle>,
        arg_list: usize,
    ) -> ffi::CORINFO_SIG_INFO {
        let mut sig: ffi::CORINFO_SIG_INFO = unsafe { std::mem::zeroed() };
        sig.callConv = call_conv;
        // A real EE always names the signature's module; consumers
        // (resolve_token's tokenScope, ldstr's constructStringLiteral)
        // pass it back to the EE.
        sig.scope = 0xC0DEusize as ffi::CORINFO_MODULE_HANDLE;
        sig.set_retType(ret.to_raw());
        sig.retTypeClass = ret_class.map_or(std::ptr::null_mut(), |c| c.as_raw());
        sig.set_numArgs(self.arg_lists[arg_list].len() as u32);
        sig.args = Self::cursor_raw(arg_list, 0);
        sig
    }

    /// The `CORINFO_SIG_INFO` for a registered method.
    fn method_sig_info(&self, method: &MockMethod) -> ffi::CORINFO_SIG_INFO {
        let call_conv = if method.sig.has_this {
            ffi::CorInfoCallConv_CORINFO_CALLCONV_HASTHIS
        } else {
            ffi::CorInfoCallConv_CORINFO_CALLCONV_DEFAULT
        } | method.call_conv_flags;
        self.build_sig_info(
            call_conv,
            method.sig.ret,
            method.sig.ret_class,
            method.arg_list,
        )
    }

    /// Registers one argument list, pairing each type with its value-class
    /// handle (padded with `None`).
    fn push_arg_list(&mut self, args: &[CorInfoType], classes: &[Option<ClassHandle>]) -> usize {
        self.push_arg_list_pinned(args, classes, &[])
    }

    /// [`MockEe::push_arg_list`] with per-argument pinned bits (empty =
    /// none pinned).
    fn push_arg_list_pinned(
        &mut self,
        args: &[CorInfoType],
        classes: &[Option<ClassHandle>],
        pins: &[bool],
    ) -> usize {
        let list = args
            .iter()
            .enumerate()
            .map(|(i, &ty)| MockArg {
                ty,
                class: classes.get(i).copied().flatten(),
                pinned: pins.get(i).copied().unwrap_or(false),
            })
            .collect();
        self.arg_lists.push(list);
        self.arg_lists.len() - 1
    }

    /// Registers a canned method under `token`; returns its fake handle.
    pub fn add_method(&mut self, token: u32, sig: MockSig) -> MethodHandle {
        let arg_list = self.push_arg_list(&sig.args, &sig.arg_classes);
        // Non-null stand-in; the mock never dereferences handles.
        let raw = (0x1000 + 0x10 * self.methods.len()) as ffi::CORINFO_METHOD_HANDLE;
        let handle = MethodHandle::from_raw(raw).expect("fake handle is non-null");
        self.methods.insert(
            token,
            MockMethod {
                handle,
                sig,
                call_conv_flags: ffi::CorInfoCallConv_CORINFO_CALLCONV_DEFAULT,
                arg_list,
            },
        );
        handle
    }

    /// Registers a canned `calli` callsite signature under a
    /// StandAloneSig `token` (step_10.12); `find_sig` answers it.
    pub fn add_calli_sig(&mut self, token: u32, sig: MockSig) {
        let arg_list = self.push_arg_list(&sig.args, &sig.arg_classes);
        self.calli_sigs.insert(token, (sig, arg_list));
    }

    pub fn add_class(
        &mut self,
        size: u32,
        align: u32,
        gc_cells: &[(u32, bool)],
        sysv: Option<ffi::SYSTEMV_AMD64_CORINFO_STRUCT_REG_PASSING_DESCRIPTOR>,
    ) -> ClassHandle {
        // Non-null stand-in in a separate address band; the mock never
        // dereferences handles.
        let raw = (0x8000 + 0x10 * self.classes.len()) as ffi::CORINFO_CLASS_HANDLE;
        let handle = ClassHandle::from_raw(raw).expect("fake handle is non-null");
        self.classes.insert(
            raw as usize,
            MockClass {
                handle,
                size,
                align,
                gc_cells: gc_cells.to_vec(),
                sysv,
            },
        );
        handle
    }

    /// Registers a canned instance field under `token` (step_10.4);
    /// returns its fake handle.
    pub fn add_field(&mut self, token: u32, ty: CorInfoType, offset: u32) -> FieldHandle {
        // Non-null stand-in in a separate address band from method handles;
        // the mock never dereferences handles.
        let raw = (0x4000 + 0x10 * self.fields.len()) as ffi::CORINFO_FIELD_HANDLE;
        let handle = FieldHandle::from_raw(raw).expect("fake handle is non-null");
        self.fields.insert(
            token,
            MockField {
                handle,
                offset,
                ty,
                is_static: false,
                value_class: None,
                init_class: false,
                in_heap: false,
                accessor: None,
                statics_helper: None,
                tls_index: 0,
                access_illegal: false,
                address_via_cell: false,
            },
        );
        handle
    }

    /// Registers a canned static field under `token` (step_10.7):
    /// `get_field_info` answers `STATIC_ADDRESS`/`IAT_VALUE` with a
    /// distinct canned address per field; the returned field's
    /// `init_class`/`in_heap`/`accessor` knobs tune the answer.
    pub fn add_static_field(&mut self, token: u32, ty: CorInfoType) -> FieldHandle {
        let handle = self.add_field(token, ty, 0);
        self.fields.get_mut(&token).unwrap().is_static = true;
        handle
    }

    /// Registers a canned struct-typed instance field under `token`
    /// (step_10.9); returns its fake handle.
    pub fn add_struct_field(&mut self, token: u32, class: ClassHandle, offset: u32) -> FieldHandle {
        let handle = self.add_field(token, CorInfoType::ValueClass, offset);
        self.fields.get_mut(&token).unwrap().value_class = Some(class);
        handle
    }

    /// Builds the argument-signature mirror for a method the mock doesn't
    /// resolve tokens to — e.g. the entry method's `MethodInfo::args`.
    pub fn make_method_sig(&mut self, sig: &MockSig) -> ffi::CORINFO_SIG_INFO {
        let arg_list = self.push_arg_list(&sig.args, &sig.arg_classes);
        let method = MockMethod {
            handle: MethodHandle::from_raw(1usize as ffi::CORINFO_METHOD_HANDLE)
                .expect("fake handle is non-null"),
            sig: sig.clone(),
            call_conv_flags: ffi::CorInfoCallConv_CORINFO_CALLCONV_DEFAULT,
            arg_list,
        };
        self.method_sig_info(&method)
    }

    /// Builds the locals-signature mirror (`CORINFO_CALLCONV_LOCAL_SIG`)
    /// for `MethodInfo::locals` — the only source of IL local types.
    /// Struct-typed locals pass their class handles in `classes`
    /// (step_10.9; shorter-than-`locals` is padded with `None`).
    pub fn make_locals_sig(&mut self, locals: &[CorInfoType]) -> ffi::CORINFO_SIG_INFO {
        self.make_locals_sig_with_classes(locals, &[])
    }

    /// The pinned-local form of [`MockEe::make_locals_sig`]: `pins[i]`
    /// marks local `i` with `CORINFO_TYPE_MOD_PINNED` (a `fixed` local).
    pub fn make_locals_sig_pinned(
        &mut self,
        locals: &[CorInfoType],
        pins: &[bool],
    ) -> ffi::CORINFO_SIG_INFO {
        let arg_list = self.push_arg_list_pinned(locals, &[], pins);
        self.build_sig_info(
            ffi::CorInfoCallConv_CORINFO_CALLCONV_LOCAL_SIG,
            CorInfoType::Void,
            None,
            arg_list,
        )
    }

    /// The class-carrying form of [`MockEe::make_locals_sig`].
    pub fn make_locals_sig_with_classes(
        &mut self,
        locals: &[CorInfoType],
        classes: &[Option<ClassHandle>],
    ) -> ffi::CORINFO_SIG_INFO {
        let arg_list = self.push_arg_list(locals, classes);
        self.build_sig_info(
            ffi::CorInfoCallConv_CORINFO_CALLCONV_LOCAL_SIG,
            CorInfoType::Void,
            None,
            arg_list,
        )
    }
}

impl MockEe {
    fn fake_alloc(&self, size: usize) -> NonNull<u8> {
        let mut buf = vec![0u8; size.max(1)].into_boxed_slice();
        let ptr = NonNull::new(buf.as_mut_ptr()).unwrap();
        self.buffers.borrow_mut().push(buf);
        ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_method_handle() -> MethodHandle {
        // Non-null stand-in; the mock never dereferences handles.
        MethodHandle::from_raw(1usize as ffi::CORINFO_METHOD_HANDLE).unwrap()
    }

    /// The acceptance test: `MockEe` instantiates and the trait is usable
    /// through the `EeInfo` supertrait with no EE anywhere.
    #[test]
    fn mock_ee_implements_ee_info() {
        let ee = MockEe {
            method_name: Some("Fib".into()),
            ..MockEe::default()
        };
        // Exercise it as the core would: through the composed supertrait.
        fn core_view(ee: &dyn EeInfo, ftn: MethodHandle) -> (MethodAttribs, Option<String>) {
            (
                ee.get_method_attribs(ftn),
                ee.get_method_name_from_metadata(ftn),
            )
        }
        let (attribs, name) = core_view(&ee, fake_method_handle());
        assert_eq!(attribs, MethodAttribs::EMPTY);
        assert_eq!(name.as_deref(), Some("Fib"));
    }

    #[test]
    fn sinks_record_in_order() {
        let ee = MockEe::default();
        ee.reserve_unwind_info(false, false, 32);
        let chunks = ee.alloc_mem(
            &[ChunkRequest {
                alignment: 16,
                size: 64,
                flags: AllocMemFlags::HOT_CODE,
            }],
            0,
        );
        assert_eq!(chunks.len(), 1);
        ee.set_eh_count(0);
        let log = ee.sink_log.borrow();
        assert_eq!(
            log.as_slice(),
            [
                "reserve_unwind_info(false, false, 32)",
                "alloc_mem(1, xcptns=0)",
                "set_eh_count(0)",
            ]
        );
    }
}
