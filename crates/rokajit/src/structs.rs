//! Value-type (struct) layout facts, queried from the EE once per class
//! per compilation and cached in a side table (step_10.9;
//! `decisions/2026-09-13-value-types.md`).
//!
//! Struct *values* are memory-backed everywhere in the compiler (a byte
//! range at an address — `ir::Expr::StructVal`), so layout questions —
//! how big a frame slot is, which cells are GC pointers, how the SysV
//! ABI passes the struct — live here, off the type itself. The map is
//! populated by the importer (one EE query set per class, cached), stored
//! on [`crate::ir::hir::Method`], carried into [`crate::ir::lir::Method`],
//! and consulted by frame layout, call classification, lowering, and
//! GC-root computation.
//!
//! The EE is the single source of truth: sizes and alignments come from
//! `getClassSize`/`getClassAlignmentRequirement`, GC cells from
//! `getClassGClayout`, and the SysV eightbyte classification from
//! `getSystemVAmd64PassStructInRegisterDescriptor` — field-walk
//! classification is deliberately NOT reimplemented in Rust.

use std::collections::HashMap;

use rokajit_ee::ee_info::EeInfo;
use rokajit_ee::handles::ClassHandle;
use rokajit_ffi as ffi;

use crate::error::{CompileError, CompileResult};

/// One GC-pointer cell inside a struct, from `getClassGClayout` (one entry
/// per pointer-sized cell of the layout).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct GcCell {
    /// Byte offset of the cell within the struct (a multiple of 8).
    pub offset: u32,
    /// `true` for a byref cell (`TYPE_GC_BYREF`): GC info reports it with
    /// the interior-pointer flag.
    pub is_byref: bool,
}

/// One eightbyte's register class, per the EE's SysV classification.
/// `IntegerReference`/`IntegerByRef` count as Integer for register-file
/// purposes; the distinction is kept because those eightbytes hold GC
/// pointers.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SysVClass {
    /// INTEGER — travels in a GPR.
    Integer,
    /// INTEGER_REFERENCE (a managed object reference eightbyte) — a GPR.
    IntegerRef,
    /// INTEGER_BYREF — a GPR.
    IntegerByRef,
    /// SSE — travels in an XMM register.
    Sse,
}

impl SysVClass {
    /// The register file this class draws from: `true` for the XMM pool.
    pub fn is_sse(self) -> bool {
        matches!(self, SysVClass::Sse)
    }

    fn from_raw(raw: ffi::SystemVClassificationType) -> CompileResult<Self> {
        Ok(match raw {
            ffi::SystemVClassificationType_SystemVClassificationTypeInteger => SysVClass::Integer,
            ffi::SystemVClassificationType_SystemVClassificationTypeIntegerReference => {
                SysVClass::IntegerRef
            }
            ffi::SystemVClassificationType_SystemVClassificationTypeIntegerByRef => {
                SysVClass::IntegerByRef
            }
            ffi::SystemVClassificationType_SystemVClassificationTypeSSE => SysVClass::Sse,
            _ => {
                return Err(CompileError::Unsupported(
                    "SysV descriptor class outside the register set",
                ));
            }
        })
    }
}

/// The Rust-normalized copy of the EE's
/// `SYSTEMV_AMD64_CORINFO_STRUCT_REG_PASSING_DESCRIPTOR`: whether the
/// struct passes in registers at all, and per-eightbyte class/size/offset.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct SysVPass {
    /// `false` (>16 bytes, misaligned fields, SIMD intrinsics — the EE
    /// decides): the struct passes and returns by reference (stack copy /
    /// hidden return buffer).
    pub passed_in_registers: bool,
    /// The number of eightbytes (1 or 2 when `passed_in_registers`).
    pub count: u8,
    /// Per-eightbyte register class (`count` entries meaningful).
    pub classes: [SysVClass; 2],
    /// Per-eightbyte byte size — moves must respect it exactly (a 3-byte
    /// eightbyte moves 3 bytes, never 8).
    pub sizes: [u8; 2],
    /// Per-eightbyte byte offset within the struct.
    pub offsets: [u8; 2],
}

impl SysVPass {
    /// The descriptor for a struct the EE does not pass in registers.
    pub fn memory() -> Self {
        SysVPass {
            passed_in_registers: false,
            count: 0,
            classes: [SysVClass::Integer; 2],
            sizes: [0; 2],
            offsets: [0; 2],
        }
    }
}

/// Everything the compiler needs to know about one value class.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StructLayout {
    /// Byte size (`getClassSize`). Frame slots are exactly this many bytes.
    pub size: u32,
    /// Byte alignment (`getClassAlignmentRequirement`).
    pub align: u32,
    /// The GC-pointer cells, one per ref/byref the layout embeds.
    pub gc_cells: Vec<GcCell>,
    /// The SysV register-passing classification.
    pub sysv: SysVPass,
}

/// The side table: one layout per value class the compilation mentions.
pub type StructLayouts = HashMap<ClassHandle, StructLayout>;

/// The largest alignment the tier-0 frame contract supports. 64 covers
/// every SIMD vector type (Vector512 is 64 bytes at 64-alignment); the
/// frame's 16-byte base alignment suffices for any of them because no
/// emitted instruction is alignment-sensitive — block copies decompose
/// into GPR moves (8/4/2/1) and the float/vector cell moves are scalar
/// `movss`/`movsd` (step_11.10's measurement: 25 of the 42 gated tests
/// needed layout/passing only and drained with this lift; the rest need
/// real SIMD semantics, deferred — see decisions/2026-09-15-simd-*).
const MAX_STRUCT_ALIGN: u32 = 64;

/// Queries the EE for one class's layout facts (the one query set per
/// class per compilation — callers cache the result in a
/// [`StructLayouts`]).
pub fn query_layout(ee: &dyn EeInfo, class: ClassHandle) -> CompileResult<StructLayout> {
    let size = ee.get_class_size(class);
    let align = ee.get_class_alignment_requirement(class, false);
    if align > MAX_STRUCT_ALIGN {
        return Err(CompileError::Unsupported(
            "struct alignment above 16 (SIMD vector types)",
        ));
    }
    // The GC layout buffer is one byte per pointer-sized cell, ROUNDED
    // UP (corinfo.h getClassGClayout's contract): the EE memsets
    // `(size + 7) / 8` bytes unconditionally (jitinterface.cpp:2370+),
    // so a sub-8-byte struct still needs a 1-byte buffer — an empty
    // Vec's dangling pointer is a segfault inside the EE.
    let mut gc_ptrs = vec![0u8; (size.div_ceil(8)) as usize];
    ee.get_class_gc_layout(class, &mut gc_ptrs);
    let mut gc_cells = Vec::new();
    for (i, &ty) in gc_ptrs.iter().enumerate() {
        let offset = (i * 8) as u32;
        match u32::from(ty) {
            ffi::CorInfoGCType_TYPE_GC_REF => gc_cells.push(GcCell {
                offset,
                is_byref: false,
            }),
            ffi::CorInfoGCType_TYPE_GC_BYREF => gc_cells.push(GcCell {
                offset,
                is_byref: true,
            }),
            _ => {}
        }
    }
    let sysv = match ee.get_system_v_amd64_pass_struct_in_register_descriptor(class) {
        Some(desc) if desc.passedInRegisters => {
            if desc.eightByteCount == 0 || desc.eightByteCount > 2 {
                return Err(CompileError::Unsupported(
                    "SysV descriptor with an eightbyte count outside 1..=2",
                ));
            }
            // Only the first `eightByteCount` entries are meaningful: the
            // real EE leaves NoClass/Unknown in the unused slots.
            let mut pass = SysVPass::memory();
            pass.passed_in_registers = true;
            pass.count = desc.eightByteCount;
            for k in 0..desc.eightByteCount as usize {
                pass.classes[k] = SysVClass::from_raw(desc.eightByteClassifications[k])?;
                pass.sizes[k] = desc.eightByteSizes[k];
                pass.offsets[k] = desc.eightByteOffsets[k];
            }
            pass
        }
        _ => SysVPass::memory(),
    };
    Ok(StructLayout {
        size,
        align,
        gc_cells,
        sysv,
    })
}

/// The cached form of [`query_layout`]: returns the layout for `class`,
/// querying the EE on first mention.
pub fn layout_of<'a>(
    layouts: &'a mut StructLayouts,
    ee: &dyn EeInfo,
    class: ClassHandle,
) -> CompileResult<&'a StructLayout> {
    if let std::collections::hash_map::Entry::Vacant(e) = layouts.entry(class) {
        let layout = query_layout(ee, class)?;
        e.insert(layout);
    }
    Ok(&layouts[&class])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rokajit_ee::mock::MockEe;

    fn class(raw: usize) -> ClassHandle {
        ClassHandle::from_raw(raw as ffi::CORINFO_CLASS_HANDLE).unwrap()
    }

    #[test]
    fn layout_queries_the_ee_once_per_class() {
        let mut ee = MockEe::default();
        let c = ee.add_class(
            16,
            8,
            &[(8, false)],
            Some(rokajit_ee::mock::sysv_descriptor(&[
                (
                    ffi::SystemVClassificationType_SystemVClassificationTypeInteger,
                    8,
                ),
                (
                    ffi::SystemVClassificationType_SystemVClassificationTypeSSE,
                    4,
                ),
            ])),
        );
        let mut layouts = StructLayouts::new();
        let layout = layout_of(&mut layouts, &ee, c).expect("queries");
        assert_eq!(layout.size, 16);
        assert_eq!(layout.align, 8);
        assert_eq!(
            layout.gc_cells,
            vec![GcCell {
                offset: 8,
                is_byref: false
            }]
        );
        assert!(layout.sysv.passed_in_registers);
        assert_eq!(layout.sysv.count, 2);
        assert_eq!(layout.sysv.classes[0], SysVClass::Integer);
        assert_eq!(layout.sysv.classes[1], SysVClass::Sse);
        assert_eq!(layout.sysv.sizes, [8, 4]);
        // Cached: a second query against a mock that doesn't know the
        // class keeps answering the canned layout.
        assert_eq!(
            layout_of(&mut layouts, &MockEe::default(), c)
                .expect("cached")
                .size,
            16
        );
        assert_eq!(layouts.len(), 1);
    }

    #[test]
    fn memory_class_when_the_ee_has_no_descriptor() {
        let mut ee = MockEe::default();
        let c = ee.add_class(24, 8, &[], None);
        let layout = query_layout(&ee, c).expect("queries");
        assert!(!layout.sysv.passed_in_registers);
        assert_eq!(layout.sysv.count, 0);
    }

    #[test]
    fn sub_pointer_size_structs_query_safely() {
        // A 4-byte struct: the GC-layout buffer is still one byte (the EE
        // memsets ceil(size/8) unconditionally), with no cells in it.
        let mut ee = MockEe::default();
        let c = ee.add_class(4, 4, &[], None);
        let layout = query_layout(&ee, c).expect("queries");
        assert_eq!(layout.size, 4);
        assert!(layout.gc_cells.is_empty());
    }

    #[test]
    fn byref_cells_are_flagged() {
        let mut ee = MockEe::default();
        let c = ee.add_class(16, 8, &[(0, false), (8, true)], None);
        let layout = query_layout(&ee, c).expect("queries");
        assert_eq!(
            layout.gc_cells,
            vec![
                GcCell {
                    offset: 0,
                    is_byref: false
                },
                GcCell {
                    offset: 8,
                    is_byref: true
                },
            ]
        );
    }

    #[test]
    fn over_aligned_structs_are_unsupported() {
        let mut ee = MockEe::default();
        let c = ee.add_class(128, 128, &[], None);
        assert!(matches!(
            query_layout(&ee, c),
            Err(CompileError::Unsupported(_))
        ));
        let _ = class(1); // keep the helper used
    }

    #[test]
    fn simd_aligned_structs_query_fine() {
        // step_11.10: Vector256/Vector512 alignments (32/64) are inside
        // the tier-0 frame contract — no emitted instruction is
        // alignment-sensitive.
        let mut ee = MockEe::default();
        let c32 = ee.add_class(32, 32, &[], None);
        let layout = query_layout(&ee, c32).expect("32-aligned queries");
        assert_eq!(layout.align, 32);
        assert_eq!(layout.size, 32);
        let c64 = ee.add_class(64, 64, &[], None);
        let layout = query_layout(&ee, c64).expect("64-aligned queries");
        assert_eq!(layout.align, 64);
    }

    #[test]
    fn unused_descriptor_slots_may_hold_no_class() {
        // The real EE fills only the first `eightByteCount` entries; the
        // unused slot keeps NoClass/Unknown. A one-eightbyte struct must
        // still classify.
        let mut ee = MockEe::default();
        let mut desc = rokajit_ee::mock::sysv_descriptor(&[(
            ffi::SystemVClassificationType_SystemVClassificationTypeInteger,
            8,
        )]);
        desc.eightByteClassifications[1] =
            ffi::SystemVClassificationType_SystemVClassificationTypeNoClass;
        let c = ee.add_class(8, 8, &[], Some(desc));
        let layout = query_layout(&ee, c).expect("classifies");
        assert!(layout.sysv.passed_in_registers);
        assert_eq!(layout.sysv.count, 1);
        assert_eq!(layout.sysv.classes[0], SysVClass::Integer);
    }
}
