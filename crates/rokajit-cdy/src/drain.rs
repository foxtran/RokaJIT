//! The artifact drain (step_07.7): hand a finished [`CompilationArtifact`]
//! to the EE through its output sinks, in the sinks' ordering contract
//! (ee-surface.md, "Output sinks"):
//!
//! 1. `reserve_unwind_info` per fragment — strictly before `alloc_mem`
//!    (the unwind bytes are carved out of the same allocation).
//! 2. `alloc_mem`: hot code chunk, then cold (never today), then RO data;
//!    `xcptns_count` is the EH clause count (RyuJIT's `numExceptions`,
//!    ee_il_dll.cpp:1241).
//! 3. Code/RO bytes are copied into the chunks' *writable* aliases
//!    (`blockRW` — distinct from the executable view on W^X targets).
//! 4. `record_relocation` per relocation — the EE patches the slot through
//!    `location_rw` (e.g. RELATIVE32: `target - (location + 4)`,
//!    jitinterface.cpp:12345), so this runs after the byte copy.
//! 5. `alloc_unwind_info` per fragment (after `alloc_mem`).
//! 6. `alloc_gc_info`, then the blob is copied into the returned block.
//! 7. `set_eh_count` + `set_eh_info` — only when non-empty (the EE asserts
//!    `cEH != 0`, jitinterface.cpp `setEHcountWorker`).
//! 8. `set_boundaries` — only when the IL-offset map is non-empty (never
//!    today; the channel is drained, the contents arrive with debug info).
//! 9. `record_call_site` per managed call site. The EE ignores it ("only
//!    testing tools use this method", jitinterface.cpp:12316); the sig is
//!    passed as `None` — the artifact's `CallSig` is the IR's, not a
//!    `CORINFO_SIG_INFO` the sink could forward.
//!
//! Everything the EE returns is EE-owned and method-lifetime: the JIT
//! writes through it and never frees it.

use std::ptr::NonNull;

use rokajit::artifact::{ChunkRef, CompilationArtifact, EhClause};
use rokajit::error::{CompileError, CompileResult};
use rokajit_ee::ee_info::{AllocatedChunk, BoundaryMap, ChunkRequest, EeInfo};
use rokajit_ee::enums::{AllocMemFlags, CorJitFuncKind};
use rokajit_ee::handles::MethodHandle;

/// Drain `artifact` into the EE's sinks. Returns the entry pointer (the
/// hot chunk's executable address) and the total code size — the
/// `compileMethod` out-params.
pub fn drain(
    artifact: &CompilationArtifact,
    ftn: MethodHandle,
    ee: &dyn EeInfo,
) -> CompileResult<(NonNull<u8>, u32)> {
    if !artifact.eh_clauses.is_empty() {
        // Unreachable today (the importer rejects EH methods and the
        // metadata stage rejects EH regions); the class-token resolution
        // the sink encoding needs arrives with EH support.
        return Err(CompileError::Unsupported("draining EH clauses"));
    }

    // 1. Reserve unwind space before alloc_mem.
    for blob in &artifact.unwind {
        ee.reserve_unwind_info(
            blob.func_kind != CorJitFuncKind::Root,
            blob.is_cold_code,
            blob.bytes.len() as u32,
        );
    }

    // 2. Allocate the chunks: hot, cold, then RO data — result chunks line
    // up 1:1 with the request.
    let mut requests = Vec::with_capacity(2 + artifact.ro_data.len());
    requests.push(ChunkRequest {
        alignment: artifact.code.hot.alignment,
        size: artifact.code.hot.bytes.len() as u32,
        flags: AllocMemFlags::HOT_CODE,
    });
    if let Some(cold) = &artifact.code.cold {
        requests.push(ChunkRequest {
            alignment: cold.alignment,
            size: cold.bytes.len() as u32,
            flags: AllocMemFlags::COLD_CODE,
        });
    }
    for data in &artifact.ro_data {
        requests.push(ChunkRequest {
            alignment: data.alignment,
            size: data.bytes.len() as u32,
            flags: data.flags,
        });
    }
    let chunks = ee.alloc_mem(&requests, artifact.eh_clauses.len() as u32);
    let chunk = |r: ChunkRef| -> AllocatedChunk {
        match r {
            ChunkRef::HotCode => chunks[0],
            ChunkRef::ColdCode => chunks[1],
            ChunkRef::RoData(i) => {
                chunks[1 + usize::from(artifact.code.cold.is_some()) + i as usize]
            }
        }
    };

    // 3. Copy the bytes into the writable aliases.
    copy_into(chunk(ChunkRef::HotCode), &artifact.code.hot.bytes);
    if let Some(cold) = &artifact.code.cold {
        copy_into(chunk(ChunkRef::ColdCode), &cold.bytes);
    }
    for (i, data) in artifact.ro_data.iter().enumerate() {
        copy_into(chunk(ChunkRef::RoData(i as u32)), &data.bytes);
    }

    // 4. Relocations: the EE computes the final field value from the
    // chunk's real (executable) address and writes it through the writable
    // alias.
    for reloc in &artifact.relocations {
        let c = chunk(reloc.chunk);
        // SAFETY: `offset` is within the chunk by construction (codegen
        // recorded it against the chunk's own bytes).
        let location = unsafe { c.executable.add(reloc.offset as usize) };
        let location_rw = unsafe { c.writable.add(reloc.offset as usize) };
        ee.record_relocation(
            location,
            Some(location_rw),
            reloc.target,
            reloc.reloc_type,
            reloc.addl_delta,
        );
    }

    // 5. Unwind info, after alloc_mem.
    let hot = chunk(ChunkRef::HotCode);
    let cold = artifact
        .code
        .cold
        .as_ref()
        .map(|_| chunk(ChunkRef::ColdCode));
    for blob in &artifact.unwind {
        ee.alloc_unwind_info(
            hot.executable,
            cold.map(|c| c.executable),
            blob.start_offset,
            blob.end_offset,
            &blob.bytes,
            blob.func_kind,
        );
    }

    // 6. GC info.
    let gc_block = ee.alloc_gc_info(artifact.gc_info.len());
    // SAFETY: the EE handed us a block of exactly this size.
    unsafe {
        std::ptr::copy_nonoverlapping(
            artifact.gc_info.as_ptr(),
            gc_block.as_ptr(),
            artifact.gc_info.len(),
        );
    }

    // 7. EH clauses (empty today; see the guard above).
    if !artifact.eh_clauses.is_empty() {
        ee.set_eh_count(artifact.eh_clauses.len() as u32);
        for (index, clause) in artifact.eh_clauses.iter().enumerate() {
            ee.set_eh_info(index as u32, &to_corinfo(clause));
        }
    }

    // 8. The IL-offset map, when there is one.
    if !artifact.il_map.is_empty() {
        let map: Vec<BoundaryMap> = artifact
            .il_map
            .iter()
            .map(|e| BoundaryMap {
                il_offset: e.il_offset,
                native_offset: e.native_offset,
                source: 0,
            })
            .collect();
        ee.set_boundaries(ftn, &map);
    }

    // 9. Managed call sites.
    for site in &artifact.call_sites {
        ee.record_call_site(site.offset, None, site.method);
    }

    let size = artifact.code.hot.bytes.len() as u32
        + artifact
            .code
            .cold
            .as_ref()
            .map_or(0, |c| c.bytes.len() as u32);
    Ok((hot.executable, size))
}

/// Copy into an EE chunk's writable alias.
fn copy_into(chunk: AllocatedChunk, bytes: &[u8]) {
    // SAFETY: the EE allocated `bytes.len()` bytes for this chunk in
    // alloc_mem (the request sizes are the byte lengths).
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), chunk.writable.as_ptr(), bytes.len());
    }
}

/// Render an artifact EH clause as the sink's bindgen struct. Unreachable
/// today (the guard at the top of [`drain`]); written so the channel is
/// complete when EH lands.
fn to_corinfo(clause: &EhClause) -> rokajit_ffi::CORINFO_EH_CLAUSE {
    use rokajit::artifact::ClassTokenOrFilter;
    let mut result: rokajit_ffi::CORINFO_EH_CLAUSE = unsafe { std::mem::zeroed() };
    result.Flags = clause.flags.to_raw();
    result.TryOffset = clause.try_offset;
    result.TryLength = clause.try_length;
    result.HandlerOffset = clause.handler_offset;
    result.HandlerLength = clause.handler_length;
    match clause.class_or_filter {
        // The typed-catch class token needs an EE query that arrives with
        // EH support; the handle's address is a placeholder that no current
        // pipeline can produce.
        ClassTokenOrFilter::Class(handle) => {
            result.__bindgen_anon_1.ClassToken = handle.as_raw() as usize as u32
        }
        ClassTokenOrFilter::FilterOffset(offset) => result.__bindgen_anon_1.FilterOffset = offset,
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rokajit::artifact::{
        CallSite, ClassTokenOrFilter, CodeChunk, CodeChunks, IlMapEntry, Relocation, UnwindBlob,
    };
    use rokajit_ee::enums::RelocType;
    use rokajit_ee::mock::MockEe;

    /// A fib-shaped artifact: 73 code bytes, two call sites/relocations,
    /// a 10-byte unwind blob, a 4-byte GC blob.
    fn fib_artifact() -> CompilationArtifact {
        CompilationArtifact {
            code: CodeChunks {
                hot: CodeChunk {
                    bytes: vec![0xAA; 73],
                    alignment: 16,
                },
                cold: None,
            },
            ro_data: Vec::new(),
            unwind: vec![UnwindBlob {
                func_kind: CorJitFuncKind::Root,
                is_cold_code: false,
                start_offset: 0,
                end_offset: 73,
                bytes: vec![0xBB; 10],
            }],
            gc_info: vec![0xCC; 4],
            eh_clauses: Vec::new(),
            il_map: Vec::new(),
            relocations: vec![
                Relocation {
                    chunk: ChunkRef::HotCode,
                    offset: 33,
                    target: 0x5000,
                    reloc_type: RelocType::RELATIVE32,
                    addl_delta: 0,
                },
                Relocation {
                    chunk: ChunkRef::HotCode,
                    offset: 52,
                    target: 0x5000,
                    reloc_type: RelocType::RELATIVE32,
                    addl_delta: 0,
                },
            ],
            call_sites: vec![
                CallSite {
                    chunk: ChunkRef::HotCode,
                    offset: 32,
                    size: 5,
                    sig: None,
                    method: None,
                },
                CallSite {
                    chunk: ChunkRef::HotCode,
                    offset: 51,
                    size: 5,
                    sig: None,
                    method: None,
                },
            ],
        }
    }

    fn handle(raw: usize) -> MethodHandle {
        MethodHandle::from_raw(raw as *mut u8 as _).unwrap()
    }

    #[test]
    fn drains_in_sink_order_and_returns_the_entry() {
        let ee = MockEe::default();
        let (entry, size) = drain(&fib_artifact(), handle(1), &ee).expect("drains");
        assert_eq!(size, 73);

        // The ordering contract, as the mock observed it.
        let log = ee.sink_log.borrow();
        let find = |prefix: &str| {
            log.iter()
                .position(|l| l.starts_with(prefix))
                .unwrap_or_else(|| panic!("{prefix} not called: {log:?}"))
        };
        let reserve = find("reserve_unwind_info(false, false, 10)");
        let alloc = find("alloc_mem(1, xcptns=0)");
        let unwind = find("alloc_unwind_info(10, Root)");
        let gc = find("alloc_gc_info(4)");
        let reloc = find("record_relocation");
        let site = find("record_call_site(32)");
        assert!(reserve < alloc, "reserve before alloc_mem: {log:?}");
        assert!(alloc < reloc, "alloc before relocations: {log:?}");
        assert!(alloc < unwind, "alloc before unwind: {log:?}");
        assert!(alloc < gc, "alloc before gc info: {log:?}");
        // No EH clauses → no setEHcount at all (the EE asserts cEH != 0).
        assert!(!log.iter().any(|l| l.starts_with("set_eh")), "{log:?}");
        // No IL map → no setBoundaries.
        assert!(
            !log.iter().any(|l| l.starts_with("set_boundaries")),
            "{log:?}"
        );
        assert!(find("record_call_site(51)") > site, "both call sites");

        // The code bytes landed in the chunk the entry points at.
        // SAFETY: the mock's chunk is a live boxed slice of 73 bytes.
        let copied = unsafe { std::slice::from_raw_parts(entry.as_ptr(), 73) };
        assert!(copied.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn relocation_addresses_are_chunk_relative() {
        let ee = MockEe::default();
        let (entry, _) = drain(&fib_artifact(), handle(1), &ee).expect("drains");
        let log = ee.sink_log.borrow();
        let base = entry.as_ptr() as usize;
        assert!(
            log.iter()
                .any(|l| l.contains(&format!("loc={:#x}", base + 33))),
            "{log:?}"
        );
        assert!(
            log.iter()
                .any(|l| l.contains(&format!("loc={:#x}", base + 52))),
            "{log:?}"
        );
    }

    #[test]
    fn eh_clauses_are_a_named_unsupported() {
        let mut artifact = fib_artifact();
        artifact.eh_clauses.push(EhClause {
            flags: rokajit_ee::enums::EhClauseFlags::FINALLY,
            try_offset: 0,
            try_length: 8,
            handler_offset: 8,
            handler_length: 8,
            class_or_filter: ClassTokenOrFilter::FilterOffset(0),
        });
        assert!(matches!(
            drain(&artifact, handle(1), &MockEe::default()),
            Err(CompileError::Unsupported(_))
        ));
    }

    #[test]
    fn il_map_drains_to_set_boundaries() {
        let mut artifact = fib_artifact();
        artifact.il_map.push(IlMapEntry {
            il_offset: 0,
            native_offset: 8,
        });
        let ee = MockEe::default();
        drain(&artifact, handle(1), &ee).expect("drains");
        assert!(ee
            .sink_log
            .borrow()
            .iter()
            .any(|l| l.starts_with("set_boundaries")));
    }
}
