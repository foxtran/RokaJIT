//! Step_05 live-EE spot check (diagnostic logging only).
//!
//! Compile-time success proves nothing about vtable slots: a gasket
//! forwarder can compile while calling the wrong virtual. This module
//! actually CALLS a representative sample of forwarders — one or two per
//! group — through the safe `EeInfo` surface against the live EE and logs
//! the returned values, so they can be hand-verified against the compiled
//! method's metadata. It runs for the first few distinct methods only (the
//! compile spine is re-entered for every method the EE jits) and changes no
//! behavior: since step_07.7, `rokajit_compile_method` proceeds to really
//! compile after logging.
//!
//! Contract findings baked into this code (step_05, verified against the
//! live EE; see `docs/step_05-completion.md`):
//!
//! - The sig arg walk must be bounded by `numArgs`: `getArgNext` never
//!   returns null (CEEInfo::getArgNext, jitinterface.cpp:9724 — it
//!   advances one element unconditionally), so walking until `None` runs
//!   off the end of the signature blob and hard-faults inside the EE.
//! - `CORINFO_RESOLVED_TOKEN.tokenType` is an IN hint, not an out-param
//!   (impResolveToken sets it, importer.cpp:70); passing 0 hard-faults
//!   the EE's `resolveToken` instead of producing a catchable exception.

use std::sync::atomic::{AtomicUsize, Ordering};

use rokajit_ee::ee_info::{
    ClassQueries, DebugInfo, FieldQueries, GasketEeInfo, Helpers, InliningAndTailCall,
    MethodQueries, OutputSinks, Pgo, Relocations, TokensAndSignatures,
};
use rokajit_ee::enums::{CallInfoFlags, CorInfoClassId};
use rokajit_ee::handles::MethodHandle;
use rokajit_ee::host::EeHost;
use rokajit_ffi::{self as ffi, CORINFO_METHOD_INFO, CORINFO_RESOLVED_TOKEN, CORINFO_SIG_INFO};

/// Spot-check the first N distinct methods the EE hands us.
const SPOT_CHECK_METHODS: usize = 4;
static SEEN: [AtomicUsize; SPOT_CHECK_METHODS] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];

/// Runs the sample for the first few distinct methods. Everything here is
/// a query the EE answers during a normal compilation; sinks and
/// notifications (`alloc_mem`, `set_eh_info`, `report_*`, …) are
/// deliberately excluded — they mutate EE state and carry ordering
/// contracts.
pub fn run(info: &CORINFO_METHOD_INFO, ee: &GasketEeInfo) {
    let ftn_addr = info.ftn as usize;
    if ftn_addr == 0 {
        return;
    }
    // Claim a slot for this ftn (first-come, single-threaded in practice:
    // the EE drives compileMethod on one thread here).
    for slot in &SEEN {
        let seen = slot.load(Ordering::Relaxed);
        if seen == ftn_addr {
            return; // already spot-checked this method
        }
        if seen == 0
            && slot
                .compare_exchange(0, ftn_addr, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            break; // claimed: run the sample below
        }
        if std::ptr::eq(slot, &SEEN[SPOT_CHECK_METHODS - 1]) {
            return; // all slots taken by other methods
        }
    }

    let Some(ftn) = MethodHandle::from_raw(info.ftn) else {
        return;
    };

    // method queries
    let attribs = ee.get_method_attribs(ftn);
    let name = ee.get_method_name_from_metadata(ftn);
    let printed = ee.print_method_name(ftn);
    let cls = ee.get_method_class(ftn);
    eprintln!("rokajit: spot: attribs={attribs:?} name={name:?} printed={printed:?} class={cls:?}");

    // tokens/signatures: method sig + arg walk
    let sig = ee.get_method_sig(ftn, None);
    log_sig("method sig", &sig, ee);

    // class queries
    eprintln!(
        "rokajit: spot: class: size={} attribs={:?} is_value_class={} name={:?} instance_fields={}",
        ee.get_class_size(cls),
        ee.get_class_attribs(cls),
        ee.is_value_class(cls),
        ee.print_class_name(cls),
        ee.get_class_num_instance_fields(cls),
    );

    // field queries, against System.String's instance fields (the sampled
    // methods' own classes have none)
    if let Some(string_cls) = ee.get_builtin_class(CorInfoClassId::String) {
        let n = ee.get_class_num_instance_fields(string_cls);
        if n > 0 {
            let field = ee.get_field_in_class(string_cls, 0);
            eprintln!(
                "rokajit: spot: string fields={n} field[0]: name={:?} static={} offset={}",
                ee.print_field_name(field),
                ee.is_field_static(field),
                ee.get_field_offset(field),
            );
        } else {
            eprintln!("rokajit: spot: string fields=0 (unexpected)");
        }
    }

    // helpers
    let ee_info = ee.get_ee_info();
    let entry = ee.get_function_entry_point(ftn);
    eprintln!(
        "rokajit: spot: ee_info: osType={} osPageSize={} frameSize={}; entry_point: accessType={} addr={:?}",
        ee_info.osType,
        ee_info.osPageSize,
        ee_info.inlinedCallFrameInfo.size,
        entry.accessType,
        unsafe { entry.__bindgen_anon_1.addr },
    );

    // debug info
    eprintln!(
        "rokajit: spot: debug: boundaries={} vars={}",
        ee.get_boundaries(ftn).len(),
        ee.get_vars(ftn).len(),
    );

    // output sinks (the one pure query in the group)
    eprintln!(
        "rokajit: spot: jit_flags={:#x}",
        ee.get_jit_flags().corJitFlags
    );

    // PGO
    match ee.get_pgo_instrumentation_results(ftn) {
        Ok(results) => eprintln!(
            "rokajit: spot: pgo: Ok schema={} data={} source={:?} dynamic={}",
            results.schema.len(),
            results.data.len(),
            results.source,
            results.dynamic_pgo,
        ),
        Err(hr) => eprintln!("rokajit: spot: pgo: Err({hr:#x})"),
    }

    // relocations
    eprintln!(
        "rokajit: spot: reloc: hint(ftn)={:?} arch={:?}",
        ee.get_reloc_type_hint(ftn.as_raw() as usize),
        ee.get_expected_target_architecture(),
    );

    // host
    eprintln!(
        "rokajit: spot: host: int(unset)={} string(unset)={:?}",
        ee.get_int_config_value("DOTNET_RokaJitSpotCheckUnset", -12345),
        ee.get_string_config_value("DOTNET_RokaJitSpotCheckUnset"),
    );

    // tokens/signatures: resolve the first call in the IL and ask for its
    // call info; inlining: probe the verdict for a self-call
    if let Some(il) = il_bytes(info) {
        match scan_call_token(il) {
            Some(token) => log_call(info, ee, ftn, token),
            None => eprintln!("rokajit: spot: no decodable call token in IL"),
        }
    }
    eprintln!(
        "rokajit: spot: can_inline(self, self)={:?}",
        ee.can_inline(ftn, ftn)
    );
}

fn log_sig(label: &str, sig: &CORINFO_SIG_INFO, ee: &GasketEeInfo) {
    let mut args = String::new();
    let mut cursor = rokajit_ee::handles::ArgListHandle::from_raw(sig.args);
    // Bounded by numArgs — see the module docs (getArgNext never returns
    // null).
    for _ in 0..sig.numArgs() {
        let Some(arg) = cursor else { break };
        let (ty, cls, _pinned) = ee.get_arg_type(sig, arg);
        if !args.is_empty() {
            args.push_str(", ");
        }
        args.push_str(&format!("{ty:?}"));
        if let Some(cls) = cls {
            args.push_str(&format!("{cls:?}"));
        }
        cursor = ee.get_arg_next(arg);
    }
    eprintln!(
        "rokajit: spot: {label}: callConv={:#x} numArgs={} ret={:#x} args=[{args}]",
        sig.callConv,
        sig.numArgs(),
        sig.retType(),
    );
}

fn log_call(info: &CORINFO_METHOD_INFO, ee: &GasketEeInfo, ftn: MethodHandle, token: u32) {
    let mut resolved: CORINFO_RESOLVED_TOKEN = unsafe { std::mem::zeroed() };
    // Built as RyuJIT's impResolveToken builds it (importer.cpp:70): the
    // method context (MAKE_METHODCONTEXT — CORINFO_CONTEXTFLAGS_METHOD is
    // 0x00, corinfo.h:1024, so the handle is the context unchanged), the
    // compilation scope, and the token kind as an IN hint.
    resolved.tokenContext = ftn.as_raw() as ffi::CORINFO_CONTEXT_HANDLE;
    resolved.tokenScope = info.scope;
    resolved.token = token;
    resolved.tokenType = ffi::CorInfoTokenKind_CORINFO_TOKENKIND_Method;
    ee.resolve_token(&mut resolved);
    eprintln!(
        "rokajit: spot: resolve_token({token:#x}): hClass={:p} hMethod={:p} (self={})",
        resolved.hClass,
        resolved.hMethod,
        resolved.hMethod == ftn.as_raw(),
    );

    let call = ee.get_call_info(&mut resolved, None, ftn, CallInfoFlags::EMPTY);
    eprintln!(
        "rokajit: spot: get_call_info: kind={} hMethod={:p} numArgs={} ret={:#x}",
        call.kind,
        call.hMethod,
        call.sig.numArgs(),
        call.sig.retType(),
    );
    log_sig("call sig", &call.sig, ee);
}

fn il_bytes(info: &CORINFO_METHOD_INFO) -> Option<&[u8]> {
    if info.ILCode.is_null() || info.ILCodeSize == 0 {
        return None;
    }
    // SAFETY: the EE guarantees ILCode points at ILCodeSize readable bytes
    // for the duration of compileMethod.
    Some(unsafe { std::slice::from_raw_parts(info.ILCode, info.ILCodeSize as usize) })
}

/// Minimal IL walk: finds the first `call`/`callvirt`/`newobj` and returns
/// its metadata token. Only operand *lengths* matter; any opcode outside
/// the table (including the 0xFE prefix) stops the scan — we never resolve
/// a token we are not sure is a call operand.
fn scan_call_token(il: &[u8]) -> Option<u32> {
    let mut i = 0usize;
    while i < il.len() {
        let op = il[i];
        let operand_len: usize = match op {
            // InlineNone
            0x00..=0x0D
            | 0x14..=0x1E
            | 0x25..=0x26
            | 0x2A
            | 0x58..=0x6E
            | 0x76
            | 0x7A
            | 0x8E
            | 0x90..=0xA2
            | 0xB3..=0xC1
            | 0xC3
            | 0xD1..=0xDC
            | 0xDF..=0xE0 => 0,
            // ShortInlineVar / ShortInlineI / ShortInlineBrTarget
            0x0E..=0x13 | 0x1F | 0x2B..=0x37 | 0xDE => 1,
            // InlineI / InlineR4
            0x20 | 0x22 => 4,
            // InlineI8 / InlineR
            0x21 | 0x23 => 8,
            // InlineMethod/Tok/Type/Field/String/Sig / InlineBrTarget
            0x27..=0x29
            | 0x38..=0x44
            | 0x6F..=0x75
            | 0x79
            | 0x7B..=0x81
            | 0x8C..=0x8D
            | 0x8F
            | 0xA3..=0xA5
            | 0xC2
            | 0xC6
            | 0xD0
            | 0xDD => 4,
            // InlineSwitch: u32 count + count u32 targets
            0x45 => {
                let bytes: [u8; 4] = il.get(i + 1..i + 5)?.try_into().ok()?;
                4 + 4 * u32::from_le_bytes(bytes) as usize
            }
            _ => return None,
        };
        if matches!(op, 0x28 | 0x6F | 0x73) {
            let bytes: [u8; 4] = il.get(i + 1..i + 5)?.try_into().ok()?;
            return Some(u32::from_le_bytes(bytes));
        }
        i = i.checked_add(1 + operand_len)?;
    }
    None
}
