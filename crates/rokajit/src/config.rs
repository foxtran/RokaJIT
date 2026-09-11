//! `JitConfig` — the config-knob access layer (step_06).
//!
//! Every knob in `runtime/src/coreclr/jit/jitconfigvalues.h` exists here as
//! typed, tiered, host-backed config: the generated [`crate::config_table`]
//! describes the knobs mechanically (name, key, kind, tier, default,
//! support flag); this module resolves them through the [`EeHost`] trait —
//! never `std::env`, because the EE aggregates env vars, runtimeconfig.json,
//! and knob files, and env-only reads would miss non-env sources.
//!
//! Resolution policy (recorded in `decisions/2026-09-11-step-06-config.md`):
//! **eager snapshot at `jitStartup`**, mirroring RyuJIT's own
//! `JitConfigValues::initialize(host)` (jitconfig.cpp:196), which resolves
//! every knob into a member up front. The startup warning scan needs every
//! value anyway, and a snapshot keeps per-compile queries host-free.

use std::sync::OnceLock;

use rokajit_ee::host::EeHost;

use crate::config_table::Knob;

/// Which build tier of RyuJIT a knob exists in (the header's macro family).
/// `Debug` and `Opt` knobs compile out of release RokaJIT builds via
/// `#[cfg(debug_assertions)]` in the generated table — in RyuJIT both
/// families expand to nothing without DEBUG (jitconfigvalues.h:8-30), and
/// RokaJIT mirrors that through the encoded tier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    /// `RELEASE_CONFIG_*`: present in every RyuJIT/RokaJIT build.
    Retail,
    /// `CONFIG_*`: debug-only.
    Debug,
    /// `OPT_CONFIG_*`: optimization toggles; also debug-only in RyuJIT
    /// (OPT_CONFIG is only defined under DEBUG), tracked as its own tier
    /// for fidelity with the header.
    Opt,
}

/// The value type of a knob (the header's macro suffix).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KnobKind {
    /// `*_CONFIG_INTEGER`: resolved via `EeHost::get_int_config_value`.
    Int,
    /// `*_CONFIG_STRING`: resolved via `EeHost::get_string_config_value`.
    String,
    /// `*_CONFIG_METHODSET`: a string knob holding a method-name list
    /// (`DOTNET_JitDisasm=Foo:Bar` style), resolved as a [`MethodSet`].
    MethodSet,
}

/// The static description of one knob, as generated from the header.
#[derive(Clone, Copy, Debug)]
pub struct KnobInfo {
    /// The C++ member name (first macro argument), e.g. `JitOrder`.
    pub name: &'static str,
    /// The config key (second macro argument), e.g. `JitOrder`. This is
    /// what `EeHost` is queried with; the environment-variable form is
    /// `DOTNET_<key>` (the EE does the prefix mapping).
    pub key: &'static str,
    pub kind: KnobKind,
    pub tier: Tier,
    /// The declared default for `Int` knobs (0 for other kinds). "Set" for
    /// the unsupported-knob warning means "differs from this default"; an
    /// explicit set-to-default is indistinguishable from unset (accepted,
    /// documented in the step_06 decision record).
    pub default_int: i32,
    /// The support registry (step_06 task 3): does RokaJIT honor this knob?
    /// Baked into the generated table from `config_supported.txt`; later
    /// steps flip knobs as features land.
    pub supported: bool,
}

/// A parsed `MethodSet` knob value: a space-separated list of glob patterns
/// (`DOTNET_JitDisasm=Foo:Bar` style).
///
/// Ported semantics (jitconfig.h MethodSet, jitconfig.cpp:21-192), minimal
/// scope: list parsing, the per-pattern name-composition flags, and the
/// `*`/`?` case-sensitive glob matcher. The EE-facing
/// `contains(methodHnd, classHnd, sig)` is NOT ported — it needs
/// `eePrintMethod` (compiler-side method printing, step_07+); until then,
/// [`MethodSet::matches`] matches a caller-formatted name.
#[derive(Debug, Default)]
pub struct MethodSet {
    /// The raw config string, `None` when the knob is unset (the C++
    /// `nullptr` default). Set-detection uses this, not pattern emptiness:
    /// an all-whitespace value is still "set".
    raw: Option<String>,
    patterns: Vec<MethodPattern>,
}

/// One pattern in the list, with the flags RyuJIT derives from its shape
/// (jitconfig.cpp:34-63). The flags tell the (future) compiler-side caller
/// how to format the method name before glob-matching; nothing reads them
/// until the EE-facing `contains` lands with method printing (step_07+).
#[derive(Debug)]
#[allow(dead_code)]
struct MethodPattern {
    glob: String,
    /// `assembly!class:method` — the pattern has an assembly part.
    contains_assembly_name: bool,
    /// `class:method` — the pattern has a class part.
    contains_class_name: bool,
    class_name_contains_instantiation: bool,
    method_name_contains_instantiation: bool,
    /// `method(sig)` — the pattern has a signature part.
    contains_signature: bool,
}

impl MethodSet {
    /// Parse a config value into a `MethodSet` (jitconfig.cpp:21-76).
    /// Patterns split on ASCII space only; empty segments are dropped.
    pub fn parse(raw: Option<String>) -> Self {
        let patterns = raw
            .as_deref()
            .map(|list| {
                list.split(' ')
                    .filter(|p| !p.is_empty())
                    .map(MethodPattern::parse)
                    .collect()
            })
            .unwrap_or_default();
        Self { raw, patterns }
    }

    /// RyuJIT's `isEmpty` (jitconfig.h:48): no patterns.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Set-detection for the unsupported-knob warning: the knob has a value
    /// at all (the C++ default is `nullptr`).
    pub fn is_set(&self) -> bool {
        self.raw.is_some()
    }

    /// The raw config string as set, if set.
    pub fn raw(&self) -> Option<&str> {
        self.raw.as_deref()
    }

    /// Match a caller-formatted method name against the pattern list
    /// (jitconfig.cpp:152-192, minus the EE printing). The caller is
    /// responsible for formatting the name with the parts the pattern
    /// demands (assembly/class/instantiation/signature flags).
    pub fn matches(&self, printed_name: &str) -> bool {
        self.patterns
            .iter()
            .any(|p| glob_match(&p.glob, printed_name))
    }
}

impl MethodPattern {
    /// The flag derivation from jitconfig.cpp:34-63, ported one-for-one.
    fn parse(pattern: &str) -> Self {
        let bytes = pattern.as_bytes();
        let exclamation = bytes.iter().position(|&c| c == b'!');
        let class_start = exclamation.map(|i| i + 1).unwrap_or(0);
        let colon = bytes[class_start..]
            .iter()
            .position(|&c| c == b':')
            .map(|i| class_start + i);
        let method_start = colon.map(|i| i + 1).unwrap_or(class_start);
        let parens = bytes[method_start..]
            .iter()
            .position(|&c| c == b'(')
            .map(|i| method_start + i);
        let method_name_end = parens.unwrap_or(bytes.len());
        Self {
            glob: pattern.to_owned(),
            contains_assembly_name: exclamation.is_some(),
            contains_class_name: colon.is_some(),
            class_name_contains_instantiation: colon
                .is_some_and(|colon| bytes[..colon].contains(&b'[')),
            method_name_contains_instantiation: bytes[method_start..method_name_end]
                .contains(&b'['),
            contains_signature: parens.is_some(),
        }
    }
}

/// RyuJIT's `matchGlob` (jitconfig.cpp:120-149): quadratic glob with `*`
/// (any run) and `?` (any single char), **case-sensitive**. Ported
/// one-for-one, operating on bytes like the original.
fn glob_match(pattern: &str, s: &str) -> bool {
    let pattern = pattern.as_bytes();
    let s = s.as_bytes();
    let (mut p, mut i) = (0usize, 0usize);
    let mut backtrack: Option<(usize, usize)> = None;
    loop {
        if p == pattern.len() {
            if i == s.len() {
                return true;
            }
        } else if pattern[p] == b'*' {
            p += 1;
            backtrack = Some((p, i));
            continue;
        } else if i == s.len() {
            // No match: the pattern needs at least one char in remaining cases.
        } else if pattern[p] == b'?' || pattern[p] == s[i] {
            p += 1;
            i += 1;
            continue;
        }
        // No match here; backtrack to the last '*' and consume one more char.
        match backtrack {
            Some((bp, bi)) if bi < s.len() => {
                p = bp;
                i = bi + 1;
                backtrack = Some((bp, bi + 1));
            }
            _ => return false,
        }
    }
}

/// One resolved knob value, parallel to [`Knob::ALL`].
#[derive(Debug)]
enum Value {
    Int(i32),
    Str(Option<String>),
    MethodSet(MethodSet),
}

/// The resolved config: one value per knob in [`Knob::ALL`], queried from
/// the host once at `jitStartup`.
#[derive(Debug)]
pub struct JitConfig {
    values: Vec<Value>,
}

impl JitConfig {
    /// Resolve every knob in the table through the host. One host query per
    /// knob; the EE aggregates all config sources (env vars,
    /// runtimeconfig.json, knob files) behind these two calls.
    pub fn resolve(host: &dyn EeHost) -> Self {
        let values = Knob::ALL
            .iter()
            .map(|knob| {
                let info = knob.info();
                match info.kind {
                    KnobKind::Int => {
                        Value::Int(host.get_int_config_value(info.key, info.default_int))
                    }
                    KnobKind::String => Value::Str(host.get_string_config_value(info.key)),
                    KnobKind::MethodSet => {
                        Value::MethodSet(MethodSet::parse(host.get_string_config_value(info.key)))
                    }
                }
            })
            .collect();
        Self { values }
    }

    /// The value of an `Int` knob.
    pub fn get_int(&self, knob: Knob) -> i32 {
        debug_assert_eq!(knob.info().kind, KnobKind::Int);
        match &self.values[knob.index()] {
            Value::Int(value) => *value,
            // Kind misuse is an internal bug; degrade to the declared default.
            _ => knob.info().default_int,
        }
    }

    /// The value of a `String` knob (`None` = unset).
    pub fn get_string(&self, knob: Knob) -> Option<&str> {
        debug_assert_eq!(knob.info().kind, KnobKind::String);
        match &self.values[knob.index()] {
            Value::Str(value) => value.as_deref(),
            _ => None,
        }
    }

    /// The value of a `MethodSet` knob.
    pub fn get_method_set(&self, knob: Knob) -> &MethodSet {
        debug_assert_eq!(knob.info().kind, KnobKind::MethodSet);
        match &self.values[knob.index()] {
            Value::MethodSet(value) => value,
            _ => {
                static EMPTY: MethodSet = MethodSet {
                    raw: None,
                    patterns: Vec::new(),
                };
                &EMPTY
            }
        }
    }
}

/// The process-wide snapshot installed by `rokajit_on_startup`.
static JIT_CONFIG: OnceLock<JitConfig> = OnceLock::new();

/// The startup snapshot, if `jitStartup` has run.
pub fn global() -> Option<&'static JitConfig> {
    JIT_CONFIG.get()
}

/// Install the startup snapshot. The EE calls `jitStartup` once; a second
/// install keeps the first (documented invariant, not an error).
pub(crate) fn install(config: JitConfig) {
    let _ = JIT_CONFIG.set(config);
}

/// The `jitStartup` config ritual (step_06), called once by the FFI edge's
/// `rokajit_on_startup`: resolve the snapshot through the host, print the
/// unsupported-knob warnings, install it process-wide.
pub fn init_from_host(host: &dyn EeHost) {
    let config = JitConfig::resolve(host);
    warn_unsupported_set(&config);
    install(config);
}

/// The startup warning scan (step_06 task 4): one line per knob that is set
/// (differs from its declared default) but not honored by RokaJIT. Returns
/// the lines so tests can assert without capturing stderr;
/// [`warn_unsupported_set`] prints them.
pub fn unsupported_set_warnings(config: &JitConfig) -> Vec<String> {
    let mut warnings = Vec::new();
    for knob in Knob::ALL {
        let info = knob.info();
        if info.supported {
            continue;
        }
        let is_set = match &config.values[knob.index()] {
            Value::Int(value) => *value != info.default_int,
            Value::Str(value) => value.is_some(),
            Value::MethodSet(value) => value.is_set(),
        };
        if is_set {
            warnings.push(format!(
                "rokajit: warning: DOTNET_{} is set but not supported",
                info.key
            ));
        }
    }
    warnings
}

/// Print the unsupported-set warnings to stderr. Runs at `jitStartup`,
/// before any `compileMethod`, so users learn immediately what RokaJIT
/// ignores. Greppable prefix: `rokajit: warning:`.
pub fn warn_unsupported_set(config: &JitConfig) {
    for warning in unsupported_set_warnings(config) {
        eprintln!("{warning}");
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_char;
    use std::path::PathBuf;
    use std::ptr::NonNull;

    use rokajit_ee::host::EeHost;

    use super::*;
    use crate::config_table::Knob;

    // -- glob_match (matchGlob port) ----------------------------------------

    #[test]
    fn glob_exact_and_case_sensitive() {
        assert!(glob_match("Foo", "Foo"));
        assert!(!glob_match("Foo", "foo"));
        assert!(!glob_match("foo", "Foo"));
        assert!(!glob_match("Foo", "FooBar"));
        assert!(!glob_match("FooBar", "Foo"));
    }

    #[test]
    fn glob_star_and_question() {
        assert!(glob_match("*", ""));
        assert!(glob_match("*", "anything at all"));
        assert!(glob_match("Foo*", "Foo"));
        assert!(glob_match("Foo*", "FooBar"));
        assert!(glob_match("*Bar", "FooBar"));
        assert!(glob_match("F*o*B*r", "FooBar"));
        assert!(glob_match("?", "x"));
        assert!(!glob_match("?", ""));
        assert!(!glob_match("?", "xy"));
        assert!(glob_match("F?o", "Foo"));
        assert!(!glob_match("F?o", "Fo"));
        // Backtracking: the '*' must give back characters to reach the tail.
        assert!(glob_match("*a*b", "aaab"));
        assert!(!glob_match("*a*b", "aaa"));
    }

    // -- MethodSet parsing ---------------------------------------------------

    #[test]
    fn method_set_unset_is_empty_and_not_set() {
        let set = MethodSet::parse(None);
        assert!(set.is_empty());
        assert!(!set.is_set());
        assert!(!set.matches("Anything"));
    }

    #[test]
    fn method_set_splits_on_space_only() {
        let set = MethodSet::parse(Some("Foo:Bar  Baz\tQux:Quux".to_owned()));
        assert!(set.is_set());
        assert!(!set.is_empty());
        assert!(set.matches("Foo:Bar"));
        // The tab is not a separator: "Baz\tQux:Quux" is one pattern.
        assert!(set.matches("Baz\tQux:Quux"));
        assert!(!set.matches("Baz"));
    }

    #[test]
    fn method_set_all_whitespace_is_set_but_empty() {
        let set = MethodSet::parse(Some("   ".to_owned()));
        assert!(set.is_set());
        assert!(set.is_empty());
    }

    #[test]
    fn method_set_pattern_flags() {
        let set = MethodSet::parse(Some(
            "mscorlib!System.Span[int]:Copy[T](void*) Plain".to_owned(),
        ));
        assert_eq!(set.patterns.len(), 2);
        let full = &set.patterns[0];
        assert!(full.contains_assembly_name);
        assert!(full.contains_class_name);
        assert!(full.class_name_contains_instantiation);
        assert!(full.method_name_contains_instantiation);
        assert!(full.contains_signature);
        let plain = &set.patterns[1];
        assert!(!plain.contains_assembly_name);
        assert!(!plain.contains_class_name);
        assert!(!plain.class_name_contains_instantiation);
        assert!(!plain.method_name_contains_instantiation);
        assert!(!plain.contains_signature);
    }

    #[test]
    fn method_set_matches_globs_case_sensitively() {
        let set = MethodSet::parse(Some("System.*:get_*".to_owned()));
        assert!(set.matches("System.Foo:get_Item"));
        assert!(!set.matches("System.Foo:Get_Item"));
        assert!(!set.matches("Other.Foo:get_Item"));
    }

    // -- JitConfig against a stub host ---------------------------------------

    #[derive(Default)]
    struct StubHost {
        ints: Vec<(&'static str, i32)>,
        strings: Vec<(&'static str, String)>,
    }

    impl EeHost for StubHost {
        fn allocate_memory(&self, _size: usize) -> Option<NonNull<u8>> {
            None
        }
        fn free_memory(&self, _block: NonNull<u8>) {}
        fn get_int_config_value(&self, name: &str, default: i32) -> i32 {
            self.ints
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| *value)
                .unwrap_or(default)
        }
        fn get_string_config_value(&self, name: &str) -> Option<String> {
            self.strings
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.clone())
        }
        fn free_string_config_value(&self, _value: NonNull<c_char>) {}
        fn allocate_slab(&self, _size: usize) -> Option<(NonNull<u8>, usize)> {
            None
        }
        fn free_slab(&self, _slab: NonNull<u8>, _actual_size: usize) {}
    }

    #[test]
    fn nothing_set_produces_no_warnings() {
        let config = JitConfig::resolve(&StubHost::default());
        assert_eq!(unsupported_set_warnings(&config), Vec::<String>::new());
    }

    #[cfg(debug_assertions)] // JitOrder is a debug-tier knob
    #[test]
    fn set_but_unsupported_int_warns() {
        let host = StubHost {
            ints: vec![("JitOrder", 1)],
            ..Default::default()
        };
        let config = JitConfig::resolve(&host);
        assert_eq!(
            unsupported_set_warnings(&config),
            vec!["rokajit: warning: DOTNET_JitOrder is set but not supported".to_owned()]
        );
        assert_eq!(config.get_int(Knob::JitOrder), 1);
    }

    #[cfg(debug_assertions)] // JitOrder is a debug-tier knob
    #[test]
    fn explicit_set_to_default_is_indistinguishable_from_unset() {
        // JitOrder's default is 0: an explicit "set to 0" reads exactly like
        // unset through ICorJitHost. Accepted set-detection semantics.
        let host = StubHost {
            ints: vec![("JitOrder", 0)],
            ..Default::default()
        };
        let config = JitConfig::resolve(&host);
        assert_eq!(unsupported_set_warnings(&config), Vec::<String>::new());
    }

    #[test]
    fn set_string_and_methodset_warn() {
        let host = StubHost {
            strings: vec![
                ("JitStdOutFile", "/tmp/jit.log".to_owned()),
                ("JitDisasm", "Main".to_owned()),
            ],
            ..Default::default()
        };
        let config = JitConfig::resolve(&host);
        let warnings = unsupported_set_warnings(&config);
        assert!(warnings.contains(
            &"rokajit: warning: DOTNET_JitStdOutFile is set but not supported".to_owned()
        ));
        assert!(warnings
            .contains(&"rokajit: warning: DOTNET_JitDisasm is set but not supported".to_owned()));
        assert_eq!(warnings.len(), 2);
        assert_eq!(config.get_string(Knob::JitStdOutFile), Some("/tmp/jit.log"));
        assert!(config.get_method_set(Knob::JitDisasm).matches("Main"));
    }

    #[test]
    fn unset_queries_return_defaults() {
        let config = JitConfig::resolve(&StubHost::default());
        assert_eq!(config.get_int(Knob::EnableAVX2), 1);
        assert_eq!(config.get_string(Knob::JitStdOutFile), None);
        assert!(config.get_method_set(Knob::JitDisasm).is_empty());
    }

    // -- table integrity against the header ----------------------------------

    /// Spot-check a dozen knobs end to end: kind, tier, key, and default,
    /// including symbolic, nested-symbolic, hex-wrapping, branch-dependent,
    /// and cfg-dependent defaults. Expectations hand-read from
    /// jitconfigvalues.h (and compiler.h / valuenum.h for the symbols).
    /// Debug-only: the debug-tier knobs it references compile out of
    /// release builds (release coverage is the cfg-dependent default below
    /// plus the parity test's release define set).
    #[cfg(debug_assertions)]
    #[test]
    fn spot_check_a_dozen_knobs() {
        let no_way_assert_default = 1; // debug RyuJIT branch (jitconfigvalues.h:559)

        let cases: &[(Knob, &str, KnobKind, Tier, i32)] = &[
            (Knob::JitOrder, "JitOrder", KnobKind::Int, Tier::Debug, 0),
            (
                Knob::JitHashBreak,
                "JitHashBreak",
                KnobKind::Int,
                Tier::Debug,
                -1,
            ),
            // 0xffffffff wraps to -1 in the 32-bit int the JIT stores.
            (
                Knob::BreakOnDumpToken,
                "BreakOnDumpToken",
                KnobKind::Int,
                Tier::Debug,
                -1,
            ),
            (
                Knob::AltJitLimit,
                "AltJitLimit",
                KnobKind::Int,
                Tier::Debug,
                0,
            ),
            // Name != key: the lookup string is the second macro argument.
            (
                Knob::DisplayLoopHoistStats,
                "JitLoopHoistStats",
                KnobKind::Int,
                Tier::Debug,
                0,
            ),
            (
                Knob::EnableAVX2,
                "EnableAVX2",
                KnobKind::Int,
                Tier::Retail,
                1,
            ),
            // The non-LOONGARCH64 branch of the #if.
            (
                Knob::EnableHWIntrinsic,
                "EnableHWIntrinsic",
                KnobKind::Int,
                Tier::Retail,
                1,
            ),
            // DEFAULT_INLINE_BUDGET (compiler.h:12616).
            (
                Knob::JitInlineBudget,
                "JitInlineBudget",
                KnobKind::Int,
                Tier::Retail,
                22,
            ),
            // DEFAULT_MAX_LOOPSIZE_FOR_ALIGN = DEFAULT_ALIGN_LOOP_BOUNDARY * 3 = 0x20*3.
            (
                Knob::JitAlignLoopMaxCodeSize,
                "JitAlignLoopMaxCodeSize",
                KnobKind::Int,
                Tier::Debug,
                96,
            ),
            // DEFAULT_MIN_OPTS_CODE_SIZE (compiler.h:11266).
            (
                Knob::JitMinOptsCodeSize,
                "JITMinOptsCodeSize",
                KnobKind::Int,
                Tier::Debug,
                60000,
            ),
            // The FEATURE_ON_STACK_REPLACEMENT branch.
            (
                Knob::TC_OnStackReplacement,
                "TC_OnStackReplacement",
                KnobKind::Int,
                Tier::Retail,
                1,
            ),
            (
                Knob::JitEnableNoWayAssert,
                "JitEnableNoWayAssert",
                KnobKind::Int,
                Tier::Retail,
                no_way_assert_default,
            ),
        ];
        assert_eq!(cases.len(), 12);
        for (knob, key, kind, tier, default) in cases {
            let info = knob.info();
            assert_eq!(info.key, *key, "{knob:?} key");
            assert_eq!(info.kind, *kind, "{knob:?} kind");
            assert_eq!(info.tier, *tier, "{knob:?} tier");
            assert_eq!(info.default_int, *default, "{knob:?} default");
            assert!(
                !info.supported,
                "{knob:?} supported (step_06 supports nothing)"
            );
        }
        // The two non-int kinds.
        let disasm = Knob::JitDisasm.info();
        assert_eq!(
            (disasm.kind, disasm.tier),
            (KnobKind::MethodSet, Tier::Retail)
        );
        let stdout_file = Knob::JitStdOutFile.info();
        assert_eq!(
            (stdout_file.kind, stdout_file.tier),
            (KnobKind::String, Tier::Retail)
        );
    }

    /// The cfg-dependent default's release branch (debug branch covered by
    /// the dozen above).
    #[cfg(not(debug_assertions))]
    #[test]
    fn cfg_dependent_default_release_branch() {
        // jitconfigvalues.h:557 — the !DEBUG && !_DEBUG branch.
        assert_eq!(Knob::JitEnableNoWayAssert.info().default_int, 0);
    }

    // -- header parity: re-derive the per-tier counts mechanically -----------

    /// The generator's define set (gen_config_table.py), duplicated here so
    /// the test derives counts from the header independently. DEBUG is in
    /// the set exactly when this build keeps the debug tier — release
    /// builds compile Debug/Opt knobs out, so their parity expectation is
    /// derived with the release define set.
    const TEST_DEFINES: &[&str] = &[
        #[cfg(debug_assertions)]
        "DEBUG",
        "TARGET_AMD64",
        "TARGET_XARCH",
        "TARGET_64BIT",
        "TARGET_UNIX",
        "HOST_UNIX",
        "FEATURE_SIMD",
        "FEATURE_LOOP_ALIGN",
        "FEATURE_ON_STACK_REPLACEMENT",
        "FEATURE_TIERED_COMPILATION",
    ];

    const TIER_MACROS: &[(&str, Tier)] = &[
        ("RELEASE_CONFIG_INTEGER", Tier::Retail),
        ("RELEASE_CONFIG_STRING", Tier::Retail),
        ("RELEASE_CONFIG_METHODSET", Tier::Retail),
        ("CONFIG_INTEGER", Tier::Debug),
        ("CONFIG_STRING", Tier::Debug),
        ("CONFIG_METHODSET", Tier::Debug),
        ("OPT_CONFIG_INTEGER", Tier::Opt),
        ("OPT_CONFIG_STRING", Tier::Opt),
        ("OPT_CONFIG_METHODSET", Tier::Opt),
    ];

    fn header_path() -> PathBuf {
        let runtime = std::env::var("ROKAJIT_RUNTIME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../runtime"));
        runtime.join("src/coreclr/jit/jitconfigvalues.h")
    }

    /// Strip `//` and `/* */` comments without touching string literals.
    fn strip_comments(text: &str) -> String {
        let bytes = text.as_bytes();
        let mut out = String::with_capacity(text.len());
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'"' => {
                    out.push('"');
                    i += 1;
                    while i < bytes.len() && bytes[i] != b'"' {
                        out.push(bytes[i] as char);
                        i += if bytes[i] == b'\\' { 2 } else { 1 };
                    }
                    out.push('"');
                    i += 1;
                }
                b'/' if bytes.get(i + 1) == Some(&b'/') => {
                    i += bytes[i..]
                        .iter()
                        .position(|&b| b == b'\n')
                        .unwrap_or(bytes.len() - i);
                }
                b'/' if bytes.get(i + 1) == Some(&b'*') => {
                    i += 2;
                    while bytes.get(i) != Some(&b'*') || bytes.get(i + 1) != Some(&b'/') {
                        i += 1;
                    }
                    i += 2;
                }
                b => {
                    out.push(b as char);
                    i += 1;
                }
            }
        }
        out
    }

    /// The small #if dialect the header uses: bare-symbol value tests and
    /// && / || chains of defined(X) / !defined(X).
    fn eval_if(expr: &str) -> bool {
        fn operand(term: &str) -> bool {
            let term = term.trim();
            if let Some(name) = term
                .strip_prefix("defined(")
                .and_then(|t| t.strip_suffix(')'))
            {
                return TEST_DEFINES.contains(&name);
            }
            if let Some(name) = term
                .strip_prefix("!defined(")
                .and_then(|t| t.strip_suffix(')'))
            {
                return !TEST_DEFINES.contains(&name);
            }
            assert!(
                term.chars().all(|c| c.is_alphanumeric() || c == '_'),
                "unsupported #if operand: {term}"
            );
            TEST_DEFINES.contains(&term)
        }
        expr.split("||").any(|group| group.split("&&").all(operand))
    }

    /// Count knob-macro invocations per tier exactly as the generator does:
    /// the DEBUG define set, conditional evaluation, macro at line start.
    fn header_tier_counts() -> (usize, usize, usize) {
        let text = strip_comments(&std::fs::read_to_string(header_path()).unwrap());
        let mut counts = [0usize; 3];
        // Stack frames: (parent_active, this_branch_active, already_taken).
        let mut stack: Vec<(bool, bool, bool)> = Vec::new();
        let mut active = true;
        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line
                .strip_prefix("#if ")
                .map(str::to_string)
                .or_else(|| {
                    line.strip_prefix("#ifdef ")
                        .map(|name| format!("defined({})", name.trim()))
                })
                .or_else(|| {
                    line.strip_prefix("#ifndef ")
                        .map(|name| format!("!defined({})", name.trim()))
                })
            {
                let cond = eval_if(&rest);
                stack.push((active, active && cond, active && cond));
                active = active && cond;
            } else if let Some(expr) = line.strip_prefix("#elif") {
                let frame = stack.last_mut().unwrap();
                let cond = eval_if(expr);
                frame.1 = frame.0 && !frame.2 && cond;
                frame.2 |= frame.1;
                active = frame.1;
            } else if line.starts_with("#else") {
                let frame = stack.last_mut().unwrap();
                frame.1 = frame.0 && !frame.2;
                frame.2 = true;
                active = frame.1;
            } else if line.starts_with("#endif") {
                stack.pop();
                active = stack.last().map(|f| f.1).unwrap_or(true);
            } else if active {
                for (mac, tier) in TIER_MACROS {
                    if line.starts_with(mac) && line[mac.len()..].trim_start().starts_with('(') {
                        counts[*tier as usize] += 1;
                        break;
                    }
                }
            }
        }
        assert!(stack.is_empty(), "unterminated #if in header");
        (counts[0], counts[1], counts[2])
    }

    #[test]
    fn table_parity_by_count_per_tier() {
        let (retail, debug, opt) = header_tier_counts();
        let mut table = [0usize; 3];
        for knob in Knob::ALL {
            table[knob.info().tier as usize] += 1;
        }
        assert_eq!(
            table[0], retail,
            "retail tier count disagrees with the header — regenerate config_table.rs \
             (python3 RokaJIT-internal/harness/gen_config_table.py)"
        );
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                (table[1], table[2]),
                (debug, opt),
                "debug/opt tier counts disagree with the header — regenerate config_table.rs"
            );
            assert_eq!(Knob::ALL.len(), retail + debug + opt);
        }
        #[cfg(not(debug_assertions))]
        {
            // The header still textually declares the debug/opt knobs in a
            // release-derivative count (only the 2 knobs inside the header's
            // own `#if defined(DEBUG)` region drop out); their absence from
            // the table is the deliberate compile-out, so the expectation
            // here is zero by construction.
            let _ = (debug, opt);
            assert_eq!(
                (table[1], table[2]),
                (0, 0),
                "debug/opt knobs must compile out in release"
            );
            assert_eq!(Knob::ALL.len(), retail);
        }
    }

    #[test]
    fn tier_discrimination_and_header_order() {
        // Tier values double as the counts array index; pin the mapping.
        assert_eq!(Tier::Retail as usize, 0);
        assert_eq!(Tier::Debug as usize, 1);
        assert_eq!(Tier::Opt as usize, 2);
        // Names are unique (the enum guarantees it, the lookup relies on it).
        let mut names: Vec<_> = Knob::ALL.iter().map(|k| k.info().name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total);
        // index() matches ALL position (the snapshot's addressing scheme).
        for (i, knob) in Knob::ALL.iter().enumerate() {
            assert_eq!(knob.index(), i);
        }
    }

    // -- tier gating per profile ----------------------------------------------

    #[cfg(debug_assertions)]
    #[test]
    fn debug_build_keeps_all_three_tiers() {
        for tier in [Tier::Retail, Tier::Debug, Tier::Opt] {
            assert!(
                Knob::ALL.iter().any(|k| k.info().tier == tier),
                "debug build must keep {tier:?} knobs"
            );
        }
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn release_build_compiles_out_debug_and_opt_knobs() {
        assert!(!Knob::ALL.is_empty());
        assert!(Knob::ALL.iter().all(|k| k.info().tier == Tier::Retail));
    }
}
