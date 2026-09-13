//! Pipeline stage 4 (step_07.5): tier-0 codegen — the Winch discipline
//! (docs/JITs/cranelift.md §2, §5 lesson 6; invariants recorded in
//! `decisions/2026-09-11-tier0-winch-codegen.md`).
//!
//! This module is the target-generic half: the pass skeleton ([`codegen`],
//! the tier gate the pipeline calls) and the value-stack machinery
//! ([`ValueState`]) a backend's emitter drives. The machine half —
//! instruction-descriptor binding, frame layout, prolog/epilog and call
//! emission — lives in the backend crate (`rokajit-x64::codegen`) behind
//! the [`Target::emit_tier0`](crate::target::Target::emit_tier0)
//! extension.
//!
//! # The Winch invariants (tier 0)
//!
//! - **No liveness, no regalloc data structures.** [`ValueState`] is the
//!   whole allocator: a tag per value and a round-robin cursor over the
//!   target's scratch pool. Tier 1 replaces it; tier 0 stays boring.
//! - **Every local and every GC reference is frame-resident.** IL locals
//!   and args live in their frame slots permanently; a GC-typed temp is
//!   spilled to its slot at definition. Root reporting (07.7) is then a
//!   static per-offset fact — [`gc_roots`] computes it from the locals
//!   table alone.
//! - **Spill-everything at control-flow joins.** Before an edge
//!   (`spill_all`), every tagged value is materialized into its frame
//!   slot, so merge-point state is deterministic: a block starts with
//!   every value in its slot, and a read of a value the current block
//!   never defined falls back to the slot — which a predecessor's join
//!   spill filled.
//!
//! # Value locations
//!
//! Every LIR value has a frame slot; the tags say where the *current*
//! value actually is:
//!
//! - [`Loc::Local`] — reads alias another (IL) local's slot, from a
//!   copy-of-local. IL locals are mutable, so a write to the aliased local
//!   must first materialize the aliases ([`ValueState::before_local_write`]).
//! - [`Loc::Reg`] — in a scratch register (tracked in the occupant map).
//! - [`Loc::Const`] — a known constant, rematerialized on use rather than
//!   kept live.
//! - [`Loc::Mem`] — in its own frame slot (also the meaning of "no tag").

use rokajit_ee::ee_info::EeInfo;

use crate::error::{CompileError, CompileResult};
use crate::ir::{hir, lir, LocalId, Type};
use crate::pipeline::{CodegenOutput, GcRootSlot, Tier};
use crate::structs::StructLayouts;
use crate::target::{PhysReg, Target};

/// Stage 4 body (the target-generic skeleton behind
/// [`crate::pipeline::codegen`]): the tier gate, then the target's emitter.
/// Tier 0 selects the Winch-style path; any other tier is
/// `Unsupported` until it exists.
pub fn codegen(
    method: &lir::Method,
    ee: &dyn EeInfo,
    target: &dyn Target,
    tier: Tier,
) -> CompileResult<CodegenOutput> {
    match tier {
        Tier::Tier0 => target.emit_tier0(method, ee),
        // `Tier` is non_exhaustive: tiers that arrive later fail here
        // until their own codegen path lands.
        #[allow(unreachable_patterns)]
        _ => Err(CompileError::Unsupported("codegen: only tier 0 exists")),
    }
}

/// Where a value currently lives (see the module docs). `Mem` offsets are
/// byte offsets below the target's frame base (x64 tier 0: `[rbp - off]`),
/// matching [`GcRootSlot::offset`].
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Loc {
    /// The value is in the named IL local/arg's frame slot (copy-of-local,
    /// zero instructions). Invalidate via [`ValueState::before_local_write`]
    /// before the local is overwritten.
    Local(LocalId),
    /// In a scratch register.
    Reg(PhysReg),
    /// A known constant; rematerialized on use or at a join spill.
    Const(i64),
    /// In a frame slot (its own, or — for a copy of a spilled temp — the
    /// source temp's slot).
    Mem(u32),
}

/// Where a read resolves, for the backend's operand construction. The
/// backend turns these into its machine operand types (`Rm`/`Rmi`).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ReadSrc {
    Reg(PhysReg),
    /// Frame slot at this offset below the frame base; the value's type
    /// comes from the locals table.
    Slot(u32),
    Imm(i64),
}

/// One move the state machine decided on; the backend emits it (in order)
/// with its own instruction selection. `ty` selects the move width.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Move {
    /// Store a scratch register into a frame slot (spill).
    Spill { reg: PhysReg, slot: u32, ty: Type },
    /// Load a frame slot into a scratch register.
    Reload { slot: u32, ty: Type, reg: PhysReg },
    /// Materialize a constant into a scratch register.
    Remat { imm: i64, reg: PhysReg, ty: Type },
}

/// The tier-0 value state: location tags plus round-robin scratch
/// allocation. Pure decisions, no emission — every state change that
/// needs code returns the [`Move`]s for the backend to emit, so the
/// machine stays testable without a target.
///
/// The scratch pool is the target's choice (x64 tier 0: the caller-saved
/// GPRs, so calls are handled by spilling registers and no callee-saved
/// traffic exists). Slot offsets and types come from the target's frame
/// layout and the locals table; both are indexed by [`LocalId`].
pub struct ValueState {
    /// Tag per local, indexed by `LocalId`. `None` = frame-resident in its
    /// own slot (the state of IL locals/args, and of everything at a join).
    locs: Vec<Option<Loc>>,
    /// Which temp each scratch register currently holds.
    occupants: std::collections::HashMap<PhysReg, LocalId>,
    /// The round-robin pool, in allocation order.
    pool: Vec<PhysReg>,
    cursor: usize,
    /// `LocalId` → frame slot offset (the target's frame layout result).
    slots: Vec<u32>,
    /// `LocalId` → IR type (from the locals table).
    tys: Vec<Type>,
}

impl ValueState {
    pub fn new(scratch_pool: &[PhysReg], slots: &[u32], tys: &[Type]) -> Self {
        debug_assert_eq!(slots.len(), tys.len());
        ValueState {
            locs: vec![None; slots.len()],
            occupants: std::collections::HashMap::new(),
            pool: scratch_pool.to_vec(),
            cursor: 0,
            slots: slots.to_vec(),
            tys: tys.to_vec(),
        }
    }

    /// The frame slot offset of a local (the target's frame layout).
    pub fn slot_of(&self, id: LocalId) -> u32 {
        self.slots[id.0 as usize]
    }

    fn ty_of(&self, id: LocalId) -> Type {
        self.tys[id.0 as usize]
    }

    /// Where a read of `id` resolves right now. An untagged value reads
    /// from its frame slot — correct across blocks because a join spill
    /// left every value frame-resident.
    pub fn read(&self, id: LocalId) -> ReadSrc {
        match self.locs[id.0 as usize] {
            None => ReadSrc::Slot(self.slot_of(id)),
            Some(Loc::Mem(off)) => ReadSrc::Slot(off),
            Some(Loc::Local(l)) => ReadSrc::Slot(self.slot_of(l)),
            Some(Loc::Reg(r)) => ReadSrc::Reg(r),
            Some(Loc::Const(i)) => ReadSrc::Imm(i),
        }
    }

    /// The next round-robin scratch register not in `exclude`, spilling
    /// its current occupant (if any) to the occupant's frame slot.
    pub fn take_scratch(&mut self, exclude: &[PhysReg]) -> (PhysReg, Vec<Move>) {
        debug_assert!(exclude.len() < self.pool.len(), "scratch pool exhausted");
        let reg = loop {
            let r = self.pool[self.cursor % self.pool.len()];
            self.cursor += 1;
            if !exclude.contains(&r) {
                break r;
            }
        };
        let mut moves = Vec::new();
        if let Some(prev) = self.occupants.remove(&reg) {
            moves.push(Move::Spill {
                reg,
                slot: self.slot_of(prev),
                ty: self.ty_of(prev),
            });
            self.locs[prev.0 as usize] = Some(Loc::Mem(self.slot_of(prev)));
        }
        (reg, moves)
    }

    /// Ensure `id`'s value is in a scratch register; no moves when it
    /// already is. Otherwise allocates (round-robin) and returns the
    /// reload/rematerialization move.
    pub fn ensure_reg(&mut self, id: LocalId, exclude: &[PhysReg]) -> (PhysReg, Vec<Move>) {
        match self.read(id) {
            ReadSrc::Reg(r) => (r, Vec::new()),
            ReadSrc::Slot(off) => {
                let (reg, mut moves) = self.take_scratch(exclude);
                moves.push(Move::Reload {
                    slot: off,
                    ty: self.ty_of(id),
                    reg,
                });
                moves.extend(self.define(id, Loc::Reg(reg)));
                (reg, moves)
            }
            ReadSrc::Imm(imm) => {
                let (reg, mut moves) = self.take_scratch(exclude);
                moves.push(Move::Remat {
                    imm,
                    reg,
                    ty: self.ty_of(id),
                });
                moves.extend(self.define(id, Loc::Reg(reg)));
                (reg, moves)
            }
        }
    }

    /// Record that `id` now lives at `loc`. Defining into an occupied
    /// register spills the previous occupant first (a caller bug — fixed
    /// registers are clobbered before adoption — handled safely anyway).
    pub fn define(&mut self, id: LocalId, loc: Loc) -> Vec<Move> {
        let mut moves = Vec::new();
        if let Loc::Reg(reg) = loc {
            if let Some(prev) = self.occupants.insert(reg, id) {
                if prev != id {
                    moves.push(Move::Spill {
                        reg,
                        slot: self.slot_of(prev),
                        ty: self.ty_of(prev),
                    });
                    self.locs[prev.0 as usize] = Some(Loc::Mem(self.slot_of(prev)));
                }
            }
        }
        self.locs[id.0 as usize] = Some(loc);
        moves
    }

    /// A fixed (ABI/architecture-pinned) register is about to be written:
    /// spill the temp it holds, if any.
    pub fn clobber(&mut self, reg: PhysReg) -> Vec<Move> {
        match self.occupants.remove(&reg) {
            Some(prev) => {
                self.locs[prev.0 as usize] = Some(Loc::Mem(self.slot_of(prev)));
                vec![Move::Spill {
                    reg,
                    slot: self.slot_of(prev),
                    ty: self.ty_of(prev),
                }]
            }
            None => Vec::new(),
        }
    }

    /// Call discipline: every scratch register dies (the pool is
    /// caller-saved by construction), so every register-resident temp is
    /// spilled. Constants and local aliases survive a call — they are not
    /// register-resident — so they keep their tags.
    pub fn spill_registers(&mut self) -> Vec<Move> {
        let mut moves = Vec::new();
        // Pool order, so the emitted spill sequence is deterministic.
        for i in 0..self.pool.len() {
            let reg = self.pool[i];
            if let Some(prev) = self.occupants.remove(&reg) {
                moves.push(Move::Spill {
                    reg,
                    slot: self.slot_of(prev),
                    ty: self.ty_of(prev),
                });
                self.locs[prev.0 as usize] = Some(Loc::Mem(self.slot_of(prev)));
            }
        }
        moves
    }

    /// Join discipline: spill *everything* so merge-point state is
    /// deterministic and frame-resident. Register values spill; constants
    /// rematerialize into their slots; local aliases copy into their own
    /// slots. After this, every value reads from its own slot — the state
    /// a successor block assumes.
    pub fn spill_all(&mut self) -> Vec<Move> {
        let mut moves = Vec::new();
        for i in 0..self.locs.len() {
            let id = LocalId(i as u32);
            match self.locs[i] {
                Some(Loc::Reg(reg)) => {
                    self.occupants.remove(&reg);
                    moves.push(Move::Spill {
                        reg,
                        slot: self.slot_of(id),
                        ty: self.ty_of(id),
                    });
                    self.locs[i] = Some(Loc::Mem(self.slot_of(id)));
                }
                Some(Loc::Const(imm)) => {
                    let (reg, mut alloc) = self.take_scratch(&[]);
                    moves.append(&mut alloc);
                    moves.push(Move::Remat {
                        imm,
                        reg,
                        ty: self.ty_of(id),
                    });
                    moves.push(Move::Spill {
                        reg,
                        slot: self.slot_of(id),
                        ty: self.ty_of(id),
                    });
                    self.locs[i] = Some(Loc::Mem(self.slot_of(id)));
                }
                Some(Loc::Local(l)) => {
                    let (reg, mut alloc) = self.take_scratch(&[]);
                    moves.append(&mut alloc);
                    moves.push(Move::Reload {
                        slot: self.slot_of(l),
                        ty: self.ty_of(id),
                        reg,
                    });
                    moves.push(Move::Spill {
                        reg,
                        slot: self.slot_of(id),
                        ty: self.ty_of(id),
                    });
                    self.locs[i] = Some(Loc::Mem(self.slot_of(id)));
                }
                None | Some(Loc::Mem(_)) => {}
            }
        }
        moves
    }

    /// An IL local/arg slot is about to be overwritten: materialize every
    /// temp aliasing it (the [`Loc::Local`] tag) into a scratch register
    /// first, so the alias doesn't observe the new value.
    pub fn before_local_write(&mut self, local: LocalId) -> Vec<Move> {
        let aliased: Vec<LocalId> = self
            .locs
            .iter()
            .enumerate()
            .filter(|(_, loc)| **loc == Some(Loc::Local(local)))
            .map(|(i, _)| LocalId(i as u32))
            .collect();
        let mut moves = Vec::new();
        for id in aliased {
            let (_, mut m) = self.ensure_reg(id, &[]);
            moves.append(&mut m);
        }
        moves
    }

    /// Block entry: every value is frame-resident (the join discipline
    /// guarantees it), so all tags and the round-robin cursor reset.
    pub fn reset(&mut self) {
        self.locs.fill(None);
        self.occupants.clear();
        self.cursor = 0;
    }
}

/// The GC root set, as a static per-offset fact (the frame-resident
/// invariant): every local of `Ref`/`ByRef` type reports its frame slot,
/// valid at every safepoint, and a struct local reports one slot per
/// embedded GC pointer (slot offset + cell offset; byref cells set the
/// interior flag; step_10.9). `slots` is the target's frame layout,
/// indexed by [`LocalId`]; `layouts` answers the struct cell questions.
///
/// Struct cell addresses are `rbp - (slot_offset - cell_offset)`: a slot's
/// bytes are `[rbp - slot_offset, rbp - slot_offset + size)`, so a cell at
/// struct-relative offset `c` sits `slot_offset - c` bytes below `rbp`.
/// Frame layout guarantees GC-cell struct slots are 8-aligned and
/// 8-rounded, so every reported offset is 8-aligned as the slot-table
/// encoder requires.
pub fn gc_roots(locals: &[hir::Local], slots: &[u32], layouts: &StructLayouts) -> Vec<GcRootSlot> {
    let mut roots = Vec::new();
    for (local, &offset) in locals.iter().zip(slots) {
        match local.ty {
            Type::Ref => roots.push(GcRootSlot {
                offset,
                is_byref: false,
                pinned: local.pinned,
            }),
            Type::ByRef => roots.push(GcRootSlot {
                offset,
                is_byref: true,
                pinned: local.pinned,
            }),
            Type::Struct(class) => {
                let layout = &layouts[&class];
                for cell in &layout.gc_cells {
                    debug_assert_eq!((offset - cell.offset) % 8, 0);
                    roots.push(GcRootSlot {
                        offset: offset - cell.offset,
                        is_byref: cell.is_byref,
                        pinned: local.pinned,
                    });
                }
            }
            _ => {}
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::BlockId;
    use crate::target::RegClassId;

    const R0: PhysReg = PhysReg(0);
    const R1: PhysReg = PhysReg(1);
    const R2: PhysReg = PhysReg(2);

    /// Three scratch registers, four Int32 slots at offsets 4/8/12/16.
    fn state() -> ValueState {
        ValueState::new(
            &[R0, R1, R2],
            &[4, 8, 12, 16],
            &[Type::Int32, Type::Int32, Type::Int32, Type::Int32],
        )
    }

    fn spill(reg: PhysReg, slot: u32) -> Move {
        Move::Spill {
            reg,
            slot,
            ty: Type::Int32,
        }
    }

    #[test]
    fn untagged_values_read_from_their_frame_slots() {
        let vs = state();
        assert_eq!(vs.read(LocalId(0)), ReadSrc::Slot(4));
        assert_eq!(vs.read(LocalId(3)), ReadSrc::Slot(16));
    }

    #[test]
    fn scratch_allocation_is_round_robin_and_evicts() {
        let mut vs = state();
        assert_eq!(vs.take_scratch(&[]), (R0, vec![]));
        vs.define(LocalId(1), Loc::Reg(R0));
        assert_eq!(vs.take_scratch(&[]), (R1, vec![]));
        assert_eq!(vs.take_scratch(&[]), (R2, vec![]));
        // The pool wraps: R0's occupant spills to its own slot and is
        // retagged, so a later read hits the slot.
        assert_eq!(vs.take_scratch(&[]), (R0, vec![spill(R0, 8)]));
        assert_eq!(vs.read(LocalId(1)), ReadSrc::Slot(8));
        // Exclusion skips pool entries without consuming them permanently.
        let mut vs = state();
        assert_eq!(vs.take_scratch(&[R0, R1]), (R2, vec![]));
        assert_eq!(vs.take_scratch(&[]), (R0, vec![]));
    }

    #[test]
    fn ensure_reg_reloads_or_rematerializes_once() {
        let mut vs = state();
        // Slot-resident value: reload into the first scratch register.
        let (reg, moves) = vs.ensure_reg(LocalId(2), &[]);
        assert_eq!(reg, R0);
        assert_eq!(
            moves,
            vec![Move::Reload {
                slot: 12,
                ty: Type::Int32,
                reg: R0
            }]
        );
        assert_eq!(vs.read(LocalId(2)), ReadSrc::Reg(R0));
        // Already in a register: no moves.
        assert_eq!(vs.ensure_reg(LocalId(2), &[]), (R0, vec![]));
        // Constant-tagged: rematerialize.
        vs.define(LocalId(3), Loc::Const(42));
        let (reg, moves) = vs.ensure_reg(LocalId(3), &[]);
        assert_eq!(reg, R1);
        assert_eq!(
            moves,
            vec![Move::Remat {
                imm: 42,
                reg: R1,
                ty: Type::Int32
            }]
        );
    }

    #[test]
    fn define_into_an_occupied_register_spills_the_occupant() {
        let mut vs = state();
        vs.define(LocalId(0), Loc::Reg(R0));
        let moves = vs.define(LocalId(1), Loc::Reg(R0));
        assert_eq!(moves, vec![spill(R0, 4)]);
        assert_eq!(vs.read(LocalId(0)), ReadSrc::Slot(4));
        assert_eq!(vs.read(LocalId(1)), ReadSrc::Reg(R0));
    }

    #[test]
    fn clobber_spills_only_the_affected_register() {
        let mut vs = state();
        vs.define(LocalId(0), Loc::Reg(R1));
        vs.define(LocalId(1), Loc::Reg(R2));
        assert_eq!(vs.clobber(R0), vec![]);
        assert_eq!(vs.clobber(R1), vec![spill(R1, 4)]);
        assert_eq!(vs.read(LocalId(0)), ReadSrc::Slot(4));
        assert_eq!(vs.read(LocalId(1)), ReadSrc::Reg(R2));
    }

    #[test]
    fn call_spill_empties_registers_but_keeps_constants_and_aliases() {
        let mut vs = state();
        vs.define(LocalId(0), Loc::Reg(R1));
        vs.define(LocalId(1), Loc::Const(7));
        vs.define(LocalId(2), Loc::Local(LocalId(3)));
        let moves = vs.spill_registers();
        // Deterministic pool order (R1 is pool index 1).
        assert_eq!(moves, vec![spill(R1, 4)]);
        assert_eq!(vs.read(LocalId(0)), ReadSrc::Slot(4));
        assert_eq!(vs.read(LocalId(1)), ReadSrc::Imm(7));
        assert_eq!(vs.read(LocalId(2)), ReadSrc::Slot(16));
    }

    #[test]
    fn join_spill_makes_everything_frame_resident_deterministically() {
        let mut vs = state();
        vs.define(LocalId(0), Loc::Reg(R1));
        vs.define(LocalId(1), Loc::Const(7));
        vs.define(LocalId(2), Loc::Local(LocalId(3)));
        let moves = vs.spill_all();
        assert_eq!(
            moves,
            vec![
                spill(R1, 4),
                Move::Remat {
                    imm: 7,
                    reg: R0,
                    ty: Type::Int32
                },
                spill(R0, 8),
                Move::Reload {
                    slot: 16,
                    ty: Type::Int32,
                    reg: R1
                },
                spill(R1, 12),
            ]
        );
        for i in 0..4 {
            let off = 4 * (i + 1);
            assert_eq!(vs.read(LocalId(i)), ReadSrc::Slot(off));
        }
    }

    #[test]
    fn local_write_materializes_aliases_first() {
        let mut vs = state();
        vs.define(LocalId(2), Loc::Local(LocalId(0)));
        vs.define(LocalId(3), Loc::Local(LocalId(0)));
        let moves = vs.before_local_write(LocalId(0));
        assert_eq!(
            moves,
            vec![
                Move::Reload {
                    slot: 4,
                    ty: Type::Int32,
                    reg: R0
                },
                Move::Reload {
                    slot: 4,
                    ty: Type::Int32,
                    reg: R1
                },
            ]
        );
        assert_eq!(vs.read(LocalId(2)), ReadSrc::Reg(R0));
        assert_eq!(vs.read(LocalId(3)), ReadSrc::Reg(R1));
        // Unrelated aliases are untouched.
        assert_eq!(vs.before_local_write(LocalId(1)), vec![]);
    }

    #[test]
    fn block_entry_resets_to_frame_resident() {
        let mut vs = state();
        vs.define(LocalId(0), Loc::Reg(R0));
        vs.take_scratch(&[]);
        vs.reset();
        assert_eq!(vs.read(LocalId(0)), ReadSrc::Slot(4));
        assert_eq!(vs.take_scratch(&[]), (R0, vec![]), "cursor resets too");
    }

    #[test]
    fn gc_roots_are_the_ref_and_byref_slots() {
        let local = |ty, pinned| hir::Local {
            ty,
            kind: hir::LocalKind::Temp,
            pinned,
        };
        let locals = vec![
            local(Type::Int32, false),
            local(Type::Ref, false),
            local(Type::ByRef, true),
            local(Type::Int64, false),
        ];
        let roots = gc_roots(&locals, &[4, 8, 16, 24], &StructLayouts::new());
        assert_eq!(
            roots,
            vec![
                GcRootSlot {
                    offset: 8,
                    is_byref: false,
                    pinned: false
                },
                GcRootSlot {
                    offset: 16,
                    is_byref: true,
                    pinned: true
                },
            ]
        );
    }

    #[test]
    fn struct_locals_report_one_slot_per_embedded_pointer() {
        // A struct with a ref cell at offset 0 and a byref cell at offset
        // 8 (step_10.9): two untracked slots, at frame offsets
        // slot − cell_offset; the byref cell sets the interior flag.
        let class = rokajit_ee::handles::ClassHandle::from_raw(0xAAusize as *mut u8 as _).unwrap();
        let mut layouts = StructLayouts::new();
        layouts.insert(
            class,
            crate::structs::StructLayout {
                size: 16,
                align: 8,
                gc_cells: vec![
                    crate::structs::GcCell {
                        offset: 0,
                        is_byref: false,
                    },
                    crate::structs::GcCell {
                        offset: 8,
                        is_byref: true,
                    },
                ],
                sysv: crate::structs::SysVPass::memory(),
            },
        );
        let local = |ty| hir::Local {
            ty,
            kind: hir::LocalKind::Temp,
            pinned: false,
        };
        let locals = vec![local(Type::Int32), local(Type::Struct(class))];
        // The struct slot's low byte is 24 below rbp.
        let roots = gc_roots(&locals, &[4, 24], &layouts);
        assert_eq!(
            roots,
            vec![
                GcRootSlot {
                    offset: 24,
                    is_byref: false,
                    pinned: false
                },
                GcRootSlot {
                    offset: 16,
                    is_byref: true,
                    pinned: false
                },
            ]
        );
    }

    // --- the pipeline stage: tier gate + delegation ---

    struct CannedTarget;

    impl Target for CannedTarget {
        fn pointer_size(&self) -> u8 {
            8
        }
        fn register_classes(&self) -> &'static [crate::target::RegisterClass] {
            &[]
        }
        fn class_of(&self, _ty: Type, _layouts: &StructLayouts) -> Option<RegClassId> {
            Some(RegClassId(0))
        }
        fn classify_call(
            &self,
            _sig: &crate::ir::CallSig,
            _layouts: &StructLayouts,
        ) -> CompileResult<crate::target::CallAbi> {
            Err(CompileError::Unsupported("canned target"))
        }
        fn call_site_stack_alignment(&self) -> u32 {
            16
        }
        fn emit_tier0(
            &self,
            _method: &lir::Method,
            _ee: &dyn EeInfo,
        ) -> CompileResult<CodegenOutput> {
            Ok(CodegenOutput {
                code: crate::artifact::CodeChunks {
                    hot: crate::artifact::CodeChunk {
                        bytes: vec![0xC3],
                        alignment: 16,
                    },
                    cold: None,
                },
                ro_data: Vec::new(),
                relocations: Vec::new(),
                call_sites: Vec::new(),
                frame: crate::pipeline::FrameInfo {
                    frame_size: 16,
                    gc_roots: Vec::new(),
                },
            })
        }
    }

    fn empty_method() -> lir::Method {
        lir::Method {
            blocks: vec![lir::Block {
                id: BlockId(0),
                stmts: Vec::new(),
            }],
            locals: Vec::new(),
            eh_regions: Vec::new(),
            num_args: 0,
            num_il_locals: 0,
            struct_layouts: StructLayouts::new(),
        }
    }

    #[test]
    fn tier0_delegates_to_the_target_emitter() {
        let ee = rokajit_ee::mock::MockEe::default();
        let out = codegen(&empty_method(), &ee, &CannedTarget, Tier::Tier0).expect("tier 0 emits");
        assert_eq!(out.code.hot.bytes, [0xC3]);
        assert_eq!(out.frame.frame_size, 16);
    }

    #[test]
    fn a_target_without_an_emitter_reports_unsupported() {
        // The default `emit_tier0` body: targets that haven't landed tier-0
        // emission fail cleanly rather than panicking.
        struct BareTarget;
        impl Target for BareTarget {
            fn pointer_size(&self) -> u8 {
                8
            }
            fn register_classes(&self) -> &'static [crate::target::RegisterClass] {
                &[]
            }
            fn class_of(&self, _ty: Type, _layouts: &StructLayouts) -> Option<RegClassId> {
                None
            }
            fn classify_call(
                &self,
                _sig: &crate::ir::CallSig,
                _layouts: &StructLayouts,
            ) -> CompileResult<crate::target::CallAbi> {
                Err(CompileError::Unsupported("bare target"))
            }
            fn call_site_stack_alignment(&self) -> u32 {
                16
            }
        }
        let ee = rokajit_ee::mock::MockEe::default();
        assert!(matches!(
            codegen(&empty_method(), &ee, &BareTarget, Tier::Tier0),
            Err(CompileError::Unsupported(_))
        ));
    }
}
