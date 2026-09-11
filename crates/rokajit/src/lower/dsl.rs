//! The lowering-rule DSL — ISLE's model (docs/JITs/README.md verdict 4,
//! docs/JITs/cranelift.md §2) as a Rust macro.
//!
//! A ruleset maps one typed subject (an LIR statement for statement
//! lowering; a small facts struct for frame lowering) to a sequence of
//! target-defined instruction descriptors. Each rule is one declarative
//! entry:
//!
//! ```text
//! rule name: <rust pattern> [if <bool guard> | if let <pat> = <extractor call>]
//!     => |cx| <expression producing Vec<Inst>>;
//! ```
//!
//! The ISLE correspondences:
//!
//! - **LHS terms** are ordinary Rust patterns over the IR's own enums
//!   (the DSL owns no IR — "external extractors" are plain functions
//!   called from guards, e.g. `if let Some(imm) = const_imm(*k)`).
//! - **RHS constructors** are plain Rust expressions building the
//!   backend's descriptor types; the descriptor types themselves carry
//!   the term typing (unallocated [`crate::lower::Val`] vs fixed physical
//!   register vs addressing mode), so invalid sequences are
//!   unrepresentable — checked by rustc, not by the DSL.
//! - **Rules are data**: order is declaration order, first match wins,
//!   and a hand-written `match` over opcodes with inline emission logic
//!   never appears in a rules file. (Like ISLE's compiler, the macro
//!   *generates* the decision structure; no author writes one.)
//! - **Guards** carry what patterns can't: literal predicates and
//!   fallible extraction. Compose several fallible lookups in one
//!   let-guard with a tuple pattern:
//!   `if let (Some(a), Some(b)) = (f(x), g(y))`.
//!
//! Deliberately absent versus ISLE (recorded in
//! `decisions/2026-09-11-lowering-rule-dsl.md`): the term-language type
//! checker (rustc fills that role here), implicit conversions, and the
//! overlap checker — rule *tests* pin each rule's firing instead.
//!
//! Expansion contract a rule author relies on:
//!
//! - Rules are tried in declaration order; the first whose pattern and
//!   guard both match produces the result. Remaining rules are skipped.
//! - A body may use `?` (the generated matcher returns `Option`), which
//!   aborts the *whole* match with `None` — use it only for
//!   internal-invariant failures, never for "try the next rule" (that is
//!   what guards are for).
//! - `None` from the matcher means "no rule matched"; the caller maps it
//!   to `CompileError::Unsupported`.

/// Defines one ruleset matcher function. See the module docs.
///
/// Header: `fn name(subject: Ty, cx: CxTy) -> Option<Vec<Inst>> matching
/// <scrutinee>;` — the scrutinee is the expression each rule's pattern
/// matches against (e.g. `&stmt.kind`). Then any number of `rule` items.
#[macro_export]
macro_rules! lower_rules {
    (
        $(#[$fmeta:meta])*
        $vis:vis fn $name:ident($subject:ident : $subject_ty:ty, $cx:ident : $cx_ty:ty)
            -> Option<Vec<$inst:ty>>
        matching $scrutinee:expr;
        $($rules:tt)*
    ) => {
        $(#[$fmeta])*
        // Rules may legitimately be irrefutable (a catch-all frame rule);
        // scoped to the generated matcher so real if-let mistakes
        // elsewhere still warn.
        #[allow(irrefutable_let_patterns)]
        $vis fn $name($subject: $subject_ty, $cx: $cx_ty)
            -> ::core::option::Option<::std::vec::Vec<$inst>>
        {
            // `cx` may be unused when every rule binds `|_|`.
            let _ = $cx;
            $crate::lower_rules!(@rules $cx, $scrutinee, $($rules)*);
            ::core::option::Option::None
        }
    };

    (@rules $cx:ident, $scrutinee:expr,) => {};

    // Rule with an if-let guard (external extractor). Before the
    // bool-guard arm: `if let` would not parse as `$guard:expr`, but
    // keeping the more specific arm first is clearer.
    (@rules $cx:ident, $scrutinee:expr,
        $(#[$rmeta:meta])*
        rule $rname:ident : $pat:pat if let $gpat:pat = $gexpr:expr => |$ca:tt| $body:expr;
        $($rest:tt)*
    ) => {
        $(#[$rmeta])*
        #[allow(non_upper_case_globals, dead_code)]
        const $rname: () = ();
        if let $pat = $scrutinee {
            if let $gpat = $gexpr {
                let $ca = $cx;
                return ::core::option::Option::Some($body);
            }
        }
        $crate::lower_rules!(@rules $cx, $scrutinee, $($rest)*);
    };

    // Rule with a boolean guard.
    (@rules $cx:ident, $scrutinee:expr,
        $(#[$rmeta:meta])*
        rule $rname:ident : $pat:pat if $guard:expr => |$ca:tt| $body:expr;
        $($rest:tt)*
    ) => {
        $(#[$rmeta])*
        #[allow(non_upper_case_globals, dead_code)]
        const $rname: () = ();
        if let $pat = $scrutinee {
            if $guard {
                let $ca = $cx;
                return ::core::option::Option::Some($body);
            }
        }
        $crate::lower_rules!(@rules $cx, $scrutinee, $($rest)*);
    };

    // Unguarded rule.
    (@rules $cx:ident, $scrutinee:expr,
        $(#[$rmeta:meta])*
        rule $rname:ident : $pat:pat => |$ca:tt| $body:expr;
        $($rest:tt)*
    ) => {
        $(#[$rmeta])*
        #[allow(non_upper_case_globals, dead_code)]
        const $rname: () = ();
        if let $pat = $scrutinee {
            let $ca = $cx;
            return ::core::option::Option::Some($body);
        }
        $crate::lower_rules!(@rules $cx, $scrutinee, $($rest)*);
    };
}

#[cfg(test)]
mod tests {
    use crate::ir::lir::{BranchCond, Operand, Stmt, StmtKind};
    use crate::ir::{BinaryOp, BlockId, Const, IlOffset, LocalId};

    /// A toy ruleset over LIR statements producing strings, pinning the
    /// macro's semantics without any backend: first match wins in
    /// declaration order, guards gate, extractors bind, no-match is None.
    mod toy {
        use super::*;

        pub(crate) fn imm32(op: &Operand) -> Option<i32> {
            match op {
                Operand::Const(Const::Int32(v)) => Some(*v),
                _ => None,
            }
        }

        lower_rules! {
            /// Toy ruleset for macro-semantics tests.
            pub(crate) fn toy_lower(stmt: &Stmt, cx: &str) -> Option<Vec<String>>
            matching &stmt.kind;

            /// Specific before general: a copy of a small constant.
            rule copy_imm: StmtKind::Copy { dst, src }
                if let (Some(v), true) = (imm32(src), cx.is_empty())
                => |_| vec![format!("mov {:?}, {v}", dst.0)];

            /// The general copy — shadowed for constants by copy_imm.
            rule copy_any: StmtKind::Copy { dst, src }
                => |_| vec![format!("mov {:?}, {:?}", dst.0, src)];

            /// A boolean guard over pattern bindings.
            rule add: StmtKind::Binary { dst, op, lhs, rhs }
                if matches!(op, BinaryOp::Add)
                => |cx| vec![format!("add {:?}, {:?}, {:?} [{cx}]", dst.0, lhs, rhs)];
            // Everything else: no rule.
        }
    }

    fn stmt(kind: StmtKind) -> Stmt {
        Stmt {
            il_offset: IlOffset(0),
            kind,
        }
    }

    fn copy_stmt(dst: u32, src: Operand) -> Stmt {
        Stmt {
            il_offset: IlOffset(0),
            kind: StmtKind::Copy {
                dst: LocalId(dst),
                src,
            },
        }
    }

    #[test]
    fn first_matching_rule_wins() {
        // copy_imm shadows copy_any for constant sources.
        let s = copy_stmt(0, Operand::Const(Const::Int32(7)));
        assert_eq!(toy::toy_lower(&s, ""), Some(vec!["mov 0, 7".to_string()]));
        let s = copy_stmt(0, Operand::Local(LocalId(1)));
        assert_eq!(
            toy::toy_lower(&s, ""),
            Some(vec!["mov 0, Local(LocalId(1))".to_string()])
        );
    }

    #[test]
    fn guard_rejection_falls_through_to_later_rules() {
        // cx non-empty defeats copy_imm's guard; copy_any still matches.
        let s = copy_stmt(0, Operand::Const(Const::Int32(7)));
        assert_eq!(
            toy::toy_lower(&s, "x"),
            Some(vec!["mov 0, Const(Int32(7))".to_string()])
        );
    }

    #[test]
    fn bool_guard_and_cx_binding() {
        let s = stmt(StmtKind::Binary {
            dst: LocalId(2),
            op: BinaryOp::Add,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Local(LocalId(1)),
        });
        assert_eq!(
            toy::toy_lower(&s, "toy"),
            Some(vec![
                "add 2, Local(LocalId(0)), Local(LocalId(1)) [toy]".to_string()
            ])
        );
        // Sub matches no rule.
        let s = stmt(StmtKind::Binary {
            dst: LocalId(2),
            op: BinaryOp::Sub,
            lhs: Operand::Local(LocalId(0)),
            rhs: Operand::Local(LocalId(1)),
        });
        assert_eq!(toy::toy_lower(&s, "toy"), None);
    }

    #[test]
    fn unmatched_subject_is_none() {
        let s = stmt(StmtKind::Branch {
            cond: BranchCond::True(Operand::Local(LocalId(0))),
            target: BlockId(1),
        });
        assert_eq!(toy::toy_lower(&s, ""), None);
    }

    #[test]
    fn no_cx_rule_compiles() {
        // Exercises the `|_|` no-cx body path on a non-matching statement.
        let s = stmt(StmtKind::Return { value: None });
        assert_eq!(toy::toy_lower(&s, ""), None);
    }
}
