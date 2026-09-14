#!/usr/bin/env python3
"""Triage RyuJIT's own test tree (runtime/src/tests/JIT) against RokaJIT.

step_08.0 deliverable: enumerate standalone candidates, compile each, run
it under BOTH the reference RyuJIT and RokaJIT, and bucket every RokaJIT
failure by the missing feature its stderr ``CompileError`` marker names
(``rokajit: compilation failed: <method>: Unsupported("...")`` — the
step_07.x unsupported-opcode discipline paying off). Aggregates into
``RokaJIT/docs/runtime-test-triage.md``, the ordered backlog for
step_08.1+.

Candidate = a single .cs file the harness can run standalone:

- a real ``static int Main`` (compiled as-is), or
- ``[Fact]`` no-arg ``int``/``void`` methods (dominantly
  ``public static int TestEntryPoint()``), compiled with a synthesized
  entry point and Xunit *attribute* stubs — mirroring the runtime's own
  XUnitWrapperGenerator "legacy standalone entry point" semantics
  (int return: 100 = pass; exceptions propagate). ``[Theory]`` files and
  tests leaning on helper libraries (TestLibrary, InlineIL, Xunit.Assert)
  are still attempted and land in COMPILE_FAIL with their csc error class.

Self-contained: needs only the sibling ``runtime/`` checkout and the
built ``target/debug/librokajit.so`` — no RokaJIT-internal dependency
(the harness machinery step_08.0 reused from
``RokaJIT-internal/mcp-server/server.py`` is inlined below). Coreroots
are staged ONCE per run, and the staged ``libclrjit.so`` is refreshed
when ``target/debug/librokajit.so`` is newer. Compiled dlls are cached
by mtime under ``target/triage/bin``.

RokaJIT aborts (SIGABRT) on methods it can't compile — CoreCLR treats the
JIT's CORJIT_IMPLLIMITATION as fatal — so child runs disable minidumps
(``DOTNET_DbgEnableMiniDump=0``) and coredumps (``RLIMIT_CORE=0``).

Resumable: per-test records append to ``target/triage/results.jsonl``
(last record per test wins); a test is re-run only when its source mtime
changed or ``--rerun`` is given.

Usage (from the workspace root):
    python3 RokaJIT/scripts/triage_runtime_tests.py
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
import resource
import shutil
import signal
import sys
import time
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_REPO = SCRIPT_DIR.parent  # RokaJIT/ — the JIT workspace
WORKSPACE = MAIN_REPO.parent

# --- harness machinery ------------------------------------------------------
# Inlined from RokaJIT-internal/mcp-server/server.py (step_08.0 originally
# imported it) so this script runs without the internal repo. JIT
# comparison works by staging a hardlinked copy of the CoreCLR artifacts
# directory and swapping libclrjit.so — necessary because the runtime is a
# Release build, where the INTERNAL DOTNET_JitPath knob is ignored.

RUNTIME_REPO = Path(
    os.environ.get("ROKAJIT_RUNTIME_REPO", WORKSPACE / "runtime")
).resolve()
RUNTIME_BIN = RUNTIME_REPO / "artifacts" / "bin" / "coreclr" / "linux.x64.Release"
RUNTIME_TESTS = RUNTIME_REPO / "src" / "tests"
ROKAJIT_WS = Path(os.environ.get("ROKAJIT_WS", MAIN_REPO)).resolve()

STATE_DIR = MAIN_REPO / "target" / "triage"  # target/ is gitignored
RESULT_VERSION = 3
DEFAULT_RESULTS = STATE_DIR / "results.jsonl"
DEFAULT_REPORT = MAIN_REPO / "docs" / "runtime-test-triage.md"
BIN_DIR = STATE_DIR / "bin"

MAX_CMD_OUTPUT_CHARS = 6000


def _tail(text: str, limit: int = MAX_CMD_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"... (first {len(text) - limit} chars dropped)\n{text[-limit:]}"


async def _run_capture(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> dict:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
        timed_out = False
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass  # exited between timeout and kill — harmless
        stdout, stderr = await proc.communicate()
        timed_out = True
    return {
        "exit_code": proc.returncode,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "timed_out": timed_out,
    }


def _rokajit_lib(profile: str = "debug") -> Path:
    return ROKAJIT_WS / "target" / profile / "librokajit.so"


def _find_csc() -> list[str]:
    candidates = sorted(
        glob.glob(str(RUNTIME_REPO / ".dotnet" / "sdk" / "*" / "Roslyn" / "bincore" / "csc.dll"))
    )
    if not candidates:
        raise ValueError(f"csc.dll not found under {RUNTIME_REPO}/.dotnet/sdk")
    return [str(RUNTIME_REPO / ".dotnet" / "dotnet"), candidates[-1]]


def _runtime_pack() -> Path | None:
    """The managed runtime pack (impl assemblies) once `build.sh libs` has run."""
    cands = sorted(RUNTIME_REPO.glob("artifacts/bin/runtime/net*-linux-Release-x64"))
    return cands[-1] if cands else None


def _stage_coreroot(jit: str) -> Path:
    """Hardlink-copy the runtime artifacts and, for 'rokajit', swap the JIT."""
    if not (RUNTIME_BIN / "corerun").is_file():
        raise ValueError(f"{RUNTIME_BIN} has no corerun — build the runtime first")
    root = STATE_DIR / f"coreroot-{jit}"
    root.mkdir(parents=True, exist_ok=True)
    for entry in RUNTIME_BIN.iterdir():
        dest = root / entry.name
        if dest.exists() or dest.is_symlink():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if entry.is_dir():
            shutil.copytree(entry, dest, symlinks=True)
        else:
            os.link(entry, dest)
    if jit == "rokajit":
        lib = _rokajit_lib("debug")
        if not lib.is_file():
            raise ValueError(f"{lib} missing — run `cargo build --workspace` first")
        target = root / "libclrjit.so"
        target.unlink()
        shutil.copy(lib, target)
    return root

# --- candidate enumeration --------------------------------------------------

MAIN_RE = re.compile(r"\bstatic\s+int\s+Main\s*\(")
XUNIT_RE = re.compile(r"xunit", re.IGNORECASE)
THEORY_RE = re.compile(r"\[\s*(Xunit\.)?Theory\b")
FACT_RE = re.compile(r"\[\s*(Xunit\.)?Fact(?:\s*\([^\]]*\))?\s*\]")
FACT_METHOD_RE = re.compile(
    r"^\s*((?:(?:public|private|internal|protected|static|unsafe|sealed|new)\s+)*)"
    r"(int|void)\s+(\w+)\s*\(\s*\)"
)
CLASS_DECL_RE = re.compile(r"\b(?:class|struct|record)\s+(\w+)[^{;]*\{")
NAMESPACE_BLOCK_RE = re.compile(r"\bnamespace\s+([\w.]+)\s*\{")
NAMESPACE_SCOPED_RE = re.compile(r"\bnamespace\s+([\w.]+)\s*;")

# Synthesized entry point + Xunit attribute stubs, mirroring
# XUnitWrapperGenerator's LegacyStandaloneEntryPointTestMethod: an int
# return of 100 is success, anything else propagates; exceptions escape.
# The attribute stubs mirror the metadata-only attributes tests get from
# Microsoft.DotNet.XUnitExtensions; TestLibrary.PlatformDetection is
# mirrored with TRUE semantics for our target (linux-x64 CoreCLR) since
# tests branch on it at RUNTIME (step_12.0).
WRAPPER_TEMPLATE = """\
// Auto-generated by triage_runtime_tests.py — Xunit attribute stubs and a
// synthesized entry point for a [Fact]-style standalone test.
namespace Xunit
{
    [System.Flags]
    public enum TestPlatforms
    {
        Windows = 1 << 0, Linux = 1 << 1, OSX = 1 << 2, FreeBSD = 1 << 3,
        NetBSD = 1 << 4, illumos = 1 << 5, Solaris = 1 << 6, iOS = 1 << 7,
        tvOS = 1 << 8, Android = 1 << 9, Browser = 1 << 10,
        MacCatalyst = 1 << 11, LinuxBionic = 1 << 12, Wasi = 1 << 13,
        Haiku = 1 << 14, OpenBSD = 1 << 15,
        AnyApple = OSX | iOS | tvOS | MacCatalyst,
        AnyUnix = AnyApple | Linux | FreeBSD | NetBSD | OpenBSD | illumos | Solaris | Android | Browser | LinuxBionic | Wasi | Haiku,
        Any = ~0
    }
    [System.Flags]
    public enum TestFrameworks { None = 0, CoreCLR = 1 << 0, Mono = 1 << 1, NativeAOT = 1 << 2, Any = ~0 }
    [System.Flags]
    public enum RuntimeTestModes
    {
        None = 0, RegularRun = 1 << 0, JitStress = 1 << 1,
        JitStressRegs = 1 << 2, TieredCompilation = 1 << 3,
        DisableTieredCompilation = 1 << 4, Any = ~0
    }
    public enum TargetFrameworkMonikers { Net462 = 1, NetCore = 2, Uap = 4, UapAot = 8, NetFramework = 16, Netcoreapp = 32 }

    public class FactAttribute : System.Attribute
    {
        public string Skip { get; set; }
        public string DisplayName { get; set; }
        public int Timeout { get; set; }
    }
    public class TheoryAttribute : FactAttribute { }
    public class InlineDataAttribute : System.Attribute
    {
        public InlineDataAttribute(params object[] data) { }
    }
    public class MemberDataAttribute : System.Attribute
    {
        public MemberDataAttribute(string memberName) { }
        public MemberDataAttribute(string memberName, params object[] parameters) { }
    }
    public class ConditionalFactAttribute : FactAttribute
    {
        public ConditionalFactAttribute() { }
        public ConditionalFactAttribute(params System.Type[] types) { }
        public ConditionalFactAttribute(System.Type type, string member) { }
        public ConditionalFactAttribute(System.Type type, string member, params object[] args) { }
    }
    public class ConditionalTheoryAttribute : TheoryAttribute
    {
        public ConditionalTheoryAttribute() { }
        public ConditionalTheoryAttribute(params System.Type[] types) { }
        public ConditionalTheoryAttribute(System.Type type, string member) { }
    }
    public class OuterLoopAttribute : System.Attribute
    {
        public OuterLoopAttribute() { }
        public OuterLoopAttribute(string reason) { }
        public OuterLoopAttribute(string reason, TestPlatforms platforms) { }
    }
    public class ActiveIssueAttribute : System.Attribute
    {
        public ActiveIssueAttribute(int issueNumber) { }
        public ActiveIssueAttribute(string url) { }
        public ActiveIssueAttribute(int issueNumber, TestPlatforms platforms) { }
        public ActiveIssueAttribute(string url, TestPlatforms platforms) { }
        public ActiveIssueAttribute(int issueNumber, TestFrameworks frameworks) { }
        public ActiveIssueAttribute(string url, TestFrameworks frameworks) { }
        public ActiveIssueAttribute(string url, TestPlatforms platforms, TestFrameworks frameworks) { }
        public ActiveIssueAttribute(int issueNumber, TestPlatforms platforms, TestFrameworks frameworks) { }
    }
    public class SkipOnCoreClrAttribute : System.Attribute
    {
        public SkipOnCoreClrAttribute(string reason) { }
        public SkipOnCoreClrAttribute(string reason, RuntimeTestModes modes) { }
        public SkipOnCoreClrAttribute(RuntimeTestModes modes) { }
    }
    public class SkipOnMonoAttribute : System.Attribute
    {
        public SkipOnMonoAttribute(string reason) { }
    }
    public class SkipOnNativeAotAttribute : System.Attribute
    {
        public SkipOnNativeAotAttribute(string reason) { }
    }
    public class SkipOnTargetFrameworkAttribute : System.Attribute
    {
        public SkipOnTargetFrameworkAttribute(string reason, TargetFrameworkMonikers frameworks) { }
    }
    public class TraitAttribute : System.Attribute
    {
        public TraitAttribute(string name, string value) { }
    }
    public class CollectionAttribute : System.Attribute
    {
        public CollectionAttribute(string name) { }
    }
    public class CollectionDefinitionAttribute : System.Attribute
    {
        public CollectionDefinitionAttribute(string name) { }
    }
}

namespace TestLibrary
{
    // Mirror of runtime/src/tests/Common/CoreCLRTestLibrary/PlatformDetection.cs
    // with values hardcoded for the triage target: linux-x64, CoreCLR,
    // no interpreter, no AOT. Tests BRANCH on these at runtime — wrong
    // values silently invalidate results (step_12.0 rule).
    public static class PlatformDetection
    {
        public static bool Is32BitProcess => false;
        public static bool Is64BitProcess => true;
        public static bool IsX86Process => false;
        public static bool IsX64Process => true;
        public static bool IsNotX86Process => true;
        public static bool IsArmProcess => false;
        public static bool IsArm64Process => false;
        public static bool IsRiscv64Process => false;
        public static bool IsWindows => false;
        public static bool IsNotWindows => true;
        public static bool IsLinux => true;
        public static bool IsOSX => false;
        public static bool IsAndroid => false;
        public static bool IsAppleMobile => false;
        public static bool IsBrowser => false;
        public static bool IsWasi => false;
        public static bool IsWasm => false;
        public static bool IsMonoRuntime => false;
        public static bool IsMonoAnyAOT => false;
        public static bool IsMonoInterpreter => false;
        public static bool IsCoreCLR => true;
        public static bool IsNotCoreCLR => false;
        public static bool IsNativeAot => false;
        public static bool IsNotNativeAot => true;
        public static bool IsCoreClrInterpreter => false;
        public static bool IsMultithreadingSupported => true;
        public static bool IsNotMultithreadingSupported => false;
        public static bool IsPreciseGcSupported => true;
        public static bool IsReadyToRunCompiled => false;
        public static bool IsBuiltInComEnabled => false;
        public static bool IsICorProfilerEnabled => true;
        public static bool IsRareEnumsSupported => true;
        public static bool IsCollectibleAssembliesSupported => true;
        public static bool IsVarArgSupported => false;
        public static bool IsExceptionInteropSupported => false;
        public static bool IsTypeEquivalenceSupported => false;
    }
}

internal static class RokaJitTriageEntryPoint
{
%s
}
"""

WRAPPER_BODY_TEMPLATE = """\
    private static int Main()
    {
%s
        return 100;
    }
"""


def enclosing_type(text: str, pos: int) -> str | None:
    """The dotted name (namespace + nested types) of the innermost type
    declaration containing `pos`, via a brace-matching scan from the start
    of the file. Handles block and file-scoped namespaces."""
    prefix: list[str] = []
    m = NAMESPACE_SCOPED_RE.search(text, 0, pos)
    if m:
        prefix = m.group(1).split(".")
    stack: list[tuple[str, int]] = []  # (name, depth after its '{')
    depth = 0
    i = 0
    while i < pos:
        m = CLASS_DECL_RE.match(text, i) or NAMESPACE_BLOCK_RE.match(text, i)
        if m:
            is_namespace = NAMESPACE_BLOCK_RE.match(text, i) is not None
            name = m.group(1)
            i = m.end()
            depth += 1
            if not (is_namespace and prefix):
                stack.append((name, depth))
            continue
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            while stack and stack[-1][1] > depth:
                stack.pop()
        i += 1
    names = prefix + [name for name, _ in stack]
    return ".".join(names) if names else None


def find_fact_methods(text: str) -> list[tuple[str, bool, str, str]]:
    """(type, is_static, method, return_type) for every [Fact] no-arg
    int/void method whose enclosing type we can determine."""
    methods = []
    for fact in FACT_RE.finditer(text):
        # The method declaration follows the attribute (possibly after more
        # attribute lines) — search the next few hundred chars line by line.
        for line in text[fact.end() : fact.end() + 600].splitlines():
            m = FACT_METHOD_RE.match(line)
            if not m:
                if line.strip() and not line.strip().startswith("["):
                    break  # first real code line isn't a matching method
                continue
            modifiers, ret, name = m.groups()
            ty = enclosing_type(text, fact.start() + m.start())
            if ty is not None:
                methods.append((ty, "static" in modifiers.split(), name, ret))
            break
    return methods


def synthesize_wrapper(methods: list[tuple[str, bool, str, str]]) -> str:
    lines = []
    for i, (ty, is_static, name, ret) in enumerate(methods):
        target = f"{ty}.{name}()" if is_static else f"new {ty}().{name}()"
        if ret == "int":
            # Legacy standalone semantics: non-100 propagates as the exit code.
            lines.append(f"        int rc{i} = {target};")
            lines.append(f"        if (rc{i} != 100) return rc{i};")
        else:
            lines.append(f"        {target};")
    return WRAPPER_TEMPLATE % WRAPPER_BODY_TEMPLATE % "\n".join(lines)


def enumerate_candidates() -> tuple[int, list[dict]]:
    """All .cs under runtime/src/tests/JIT that the standalone harness can
    attempt: real `static int Main` (entry='main') or [Fact] no-arg
    int/void methods (entry='fact', synthesized wrapper)."""
    tests_dir = RUNTIME_TESTS / "JIT"
    scanned = 0
    candidates = []
    for src in sorted(tests_dir.rglob("*.cs")):
        scanned += 1
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = src.relative_to(RUNTIME_TESTS).as_posix()
        if MAIN_RE.search(text) and not XUNIT_RE.search(text):
            candidates.append({"test": rel, "entry": "main"})
            continue
        if THEORY_RE.search(text):
            continue
        methods = find_fact_methods(text)
        if methods:
            candidates.append(
                {"test": rel, "entry": "fact", "methods": methods}
            )
    return scanned, candidates


# --- compilation ------------------------------------------------------------


def _sanitize(rel_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", rel_path)


def _write_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


async def compile_candidate(candidate: dict) -> Path:
    """Compile a candidate (test source + siblings + synthesized wrapper for
    [Fact] tests) against CoreLib + the runtime pack. Multi-file first (all
    .cs in the test's directory — many tests keep helper types in sibling
    files), falling back to single-file when siblings collide (duplicate
    types/entry points); the winning mode is cached in a sidecar file.
    Compiled dlls are cached by source mtimes."""
    rel_path = candidate["test"]
    src = (RUNTIME_TESTS / rel_path).resolve()
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    base = _sanitize(rel_path)
    dll = BIN_DIR / f"{base}.dll"
    mode_file = BIN_DIR / f"{base}.mode"
    wrapper_sources: list[str] = []
    if candidate["entry"] == "fact":
        wrapper = BIN_DIR / f"{base}.wrapper.cs"
        _write_if_changed(wrapper, synthesize_wrapper(candidate["methods"]))
        wrapper_sources.append(str(wrapper))

    def source_sets() -> list[list[str]]:
        siblings = [str(p) for p in sorted(src.parent.glob("*.cs"))]
        sets = []
        if mode == "single" or siblings == [str(src)]:
            sets = [[str(src)]]
        elif mode == "multi":
            sets = [siblings]
        else:  # unknown: try multi, fall back to single
            sets = [siblings, [str(src)]]
        return [s + wrapper_sources for s in sets]

    mode = mode_file.read_text().strip() if mode_file.is_file() else ""
    corelib = RUNTIME_BIN / "System.Private.CoreLib.dll"
    pack = _runtime_pack()
    refs = [f"-r:{corelib}"]
    if pack is not None:
        refs += [f"-r:{p}" for p in sorted(pack.glob("*.dll"))]
    last_error = ""
    for sources in source_sets():
        newest_src = max(Path(s).stat().st_mtime for s in sources)
        if dll.exists() and dll.stat().st_mtime >= newest_src:
            return dll
        cmd = [
            *_find_csc(),
            "-nologo", "-nostdlib", "-noconfig", "-optimize+", "-unsafe+",
            f"-out:{dll}", *refs, *sources,
        ]
        result = await _run_capture(cmd, cwd=BIN_DIR, timeout_seconds=120)
        if result["exit_code"] == 0:
            _write_if_changed(
                mode_file,
                "multi" if len(sources) - len(wrapper_sources) > 1 else "single",
            )
            return dll
        last_error = result["stdout"] + result["stderr"]
    raise ValueError(f"csc failed:\n{_tail(last_error)}")


def prelink_runtime_pack() -> None:
    """Link the runtime-pack dlls next to the compiled test dlls (app-local
    resolution), once, serially."""
    pack = _runtime_pack()
    if pack is None:
        return
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    for dll in pack.glob("*.dll"):
        dest = BIN_DIR / dll.name
        if not dest.exists():
            try:
                os.link(dll, dest)
            except FileExistsError:
                pass


# --- subprocess plumbing ----------------------------------------------------


def _no_core_dump() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


async def run_test_binary(coreroot: Path, dll: Path, timeout: int) -> dict:
    """Run corerun on dll with minidumps/coredumps suppressed."""
    env = dict(os.environ, DOTNET_DbgEnableMiniDump="0")
    proc = await asyncio.create_subprocess_exec(
        str(coreroot / "corerun"),
        str(dll),
        cwd=str(coreroot),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=_no_core_dump,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        timed_out = False
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass  # exited between timeout and kill — harmless
        stdout, stderr = await proc.communicate()
        timed_out = True
    return {
        "exit_code": proc.returncode,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "timed_out": timed_out,
    }


# --- classification ---------------------------------------------------------

# First CompileError marker wins: it is the construct that killed the run.
FAILED_RE = re.compile(
    r'rokajit: compilation failed: \S+: (Unsupported|BadIl|Internal|Skipped)\("([^"]*)"\)'
)
DRAIN_RE = re.compile(r'rokajit: drain failed: (\w+)\("([^"]*)"\)')
PANIC_RE = re.compile(r"rokajit: panic in compileMethod")

# Unsupported payload -> feature bucket, first match wins. Order matters:
# specific payloads before the importer's catch-all opcode messages.
# (step_10.1 landed compare-as-value, unary/conv and the div/shift/logic
# ops, so their rules are gone; step_10.2 landed floats, so the
# float-argument rule is gone; step_10.3 landed ldstr and the GC slot
# table, so the slot-table rule is gone; step_10.4 landed the object
# pack, so the byref-load/store and null-check lowering rules are gone;
# messages evolve with the importer. step_10.9 landed value types: the
# "value types in signatures" gate, the lowering "structs:" reject and the
# tier-0 struct/stack-arg messages are gone; the bucket keeps the residual
# struct rejects. step_10.6 landed EH: the "EH regions"/"EH control flow"
# gate messages are gone; the residual EH rejects are the out-of-scope
# filter/fault clauses, endfilter and rethrow. step_10.8 landed arrays:
# the "arrays:" reject is gone; the bucket keeps the residual array-pack
# gates.)
BUCKET_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"EH filter clauses|EH fault clauses"), "EH filters/faults (out of 10.6 scope)"),
    (re.compile(r"^rethrow$"), "rethrow (out of 10.6 scope)"),
    (re.compile(r"generic methods"), "generics"),
    (re.compile(r"non-class receiver \(value types\)|initobj/ldobj/stobj/cpobj of a non-value class|struct alignment above 16|SysV descriptor"), "structs & value types"),
    (re.compile(r"newobj of a value class"), "newobj of a value class"),
    (re.compile(r"newarr of a non-SZ array|newarr allocation helper|array element type outside"), "array pack gates (10.8: non-SZ/helper/element)"),
    (re.compile(r"thread-local statics|static field of a shared-generic|static field through an address helper|static field accessor outside|static field needing an access callout|static field address through an indirection cell"), "statics pack gates (10.7: TLS/generic/R2R/accessors)"),
    (re.compile(r"static fields"), "static fields"),
    (re.compile(r"field type outside the 10\.4 object pack"), "field types outside the object pack"),
    (re.compile(r"allocation helper outside the newobj set"), "allocation helpers outside the newobj set"),
    (re.compile(r"class handle through an indirection cell"), "class handle indirection (IAT_PVALUE/PPVALUE)"),
    (re.compile(r"ldtoken.*(indirection cell|runtime lookup)"), "ldtoken handle embedding (10.10 gates)"),
    (re.compile(r"cast/box|box of Nullable|unbox of Nullable|helper outside the (box|unbox|isinst/castclass) set"), "boxing & casts"),
    (re.compile(r"switch:"), "switch"),
    (re.compile(r"ldstr through a handle-cell"), "ldstr indirection (IAT_PVALUE/PPVALUE)"),
    (re.compile(r"non-direct call kind"), "non-direct calls (callvirt/calli)"),
    (re.compile(r"non-default calling convention"), "non-default calling conventions"),
    (re.compile(r"evaluation-stack values crossing"), "eval-stack values across block boundaries"),
    (re.compile(r"local's type has no register class"), "locals without a register class"),
    (re.compile(r"opcode outside the supported set|0xFE-prefixed opcode"), "unsupported IL opcode (importer)"),
]


def extract_reason(stderr: str) -> tuple[str, str]:
    """Map a RokaJIT failure's stderr to (bucket, detail). Failures with no
    CompileError marker land in 'needs investigation' with a signature."""
    m = FAILED_RE.search(stderr)
    if m:
        kind, payload = m.group(1), m.group(2)
        if kind == "Unsupported":
            for pattern, bucket in BUCKET_RULES:
                if pattern.search(payload):
                    return bucket, f'{kind}("{payload}")'
            return f"unsupported (unmapped): {payload}", f'{kind}("{payload}")'
        label = {"BadIl": "bad IL rejected by importer", "Internal": "internal error (ICE)"}.get(
            kind, f"rokajit {kind}"
        )
        if kind == "BadIl" and payload == "operand must be an integer":
            # Importer over-restriction, not bad IL: brtrue/brfalse on a
            # reference (the `if (obj != null)` pattern) is legal per
            # ECMA-335 but rejected by the integer-only branch popper.
            label = "branches on references (brtrue/brfalse null checks)"
        return label, f'{kind}("{payload}")'
    m = DRAIN_RE.search(stderr)
    if m:
        return "drain failure (EE output sinks)", f'{m.group(1)}("{m.group(2)}")'
    if PANIC_RE.search(stderr):
        return "panic in compileMethod (ICE)", "panic"
    signature = ""
    for line in stderr.splitlines():
        # rokajit's own diagnostics were checked above; the signature is the
        # first line of anything else (CoreCLR's output, managed exceptions).
        if line.strip() and not line.startswith("rokajit: "):
            signature = line.strip()[:160]
            break
    return "needs investigation", signature or "(no stderr output)"


def signal_name(returncode: int) -> str:
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"SIG{-returncode}"


CS_ERROR_RE = re.compile(r"error ([A-Z]+\d+)")


def classify_compile_error(message: str) -> str:
    classes = CS_ERROR_RE.findall(message)
    if not classes:
        return "unknown"
    counts = Counter(classes)
    # Most frequent class; ties broken by lowest CS number for determinism.
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# --- per-test driver --------------------------------------------------------


async def triage_one(candidate: dict, coreroots: dict[str, Path], timeout: int) -> dict:
    rel_path = candidate["test"]
    record: dict = {"test": rel_path, "version": RESULT_VERSION, "entry": candidate["entry"]}
    src = RUNTIME_TESTS / rel_path
    record["src_mtime"] = src.stat().st_mtime
    try:
        dll = await compile_candidate(candidate)
    except ValueError as exc:
        record["category"] = "COMPILE_FAIL"
        record["detail"] = classify_compile_error(str(exc))
        return record
    ref, ours = await asyncio.gather(
        run_test_binary(coreroots["ryujit"], dll, timeout),
        run_test_binary(coreroots["rokajit"], dll, timeout),
    )
    record["ref"] = {
        "exit_code": ref["exit_code"],
        "timed_out": ref["timed_out"],
        "stdout": ref["stdout"][-500:],
    }
    record["ours"] = {
        "exit_code": ours["exit_code"],
        "timed_out": ours["timed_out"],
        "stdout": ours["stdout"][-500:],
    }
    if ref["timed_out"] or ours["timed_out"]:
        record["category"] = "TIMEOUT"
        record["detail"] = ",".join(
            name
            for name, run in (("ryujit", ref), ("rokajit", ours))
            if run["timed_out"]
        )
    elif ref["exit_code"] == ours["exit_code"] and ref["stdout"] == ours["stdout"]:
        record["category"] = "MATCH"
        record["detail"] = str(ref["exit_code"])
    else:
        bucket, detail = extract_reason(ours["stderr"])
        record["bucket"] = bucket
        record["bucket_detail"] = detail
        if ours["exit_code"] is not None and ours["exit_code"] < 0:
            record["category"] = "CRASH"
            record["detail"] = signal_name(ours["exit_code"])
        else:
            record["category"] = "MISMATCH"
            record["detail"] = f"ref={ref['exit_code']} ours={ours['exit_code']}"
        record["stderr_tail"] = ours["stderr"][-1000:]
    return record


# --- result cache -----------------------------------------------------------


def load_results(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[rec["test"]] = rec  # last record per test wins
    return records


# --- report -----------------------------------------------------------------


def fmt_examples(tests: list[str], limit: int = 5) -> str:
    shown = sorted(tests)[:limit]
    rest = len(tests) - len(shown)
    text = "<br>".join(f"`{t}`" for t in shown)
    if rest:
        text += f"<br>… and {rest} more"
    return text


def render_report(records: dict[str, dict], scanned: int, candidates: list[dict]) -> str:
    recs = [records[t] for t in sorted(records)]
    cats = Counter(r["category"] for r in recs)
    entries = Counter(c["entry"] for c in candidates)
    buckets: dict[str, list[str]] = {}
    for r in recs:
        if "bucket" in r:
            buckets.setdefault(r["bucket"], []).append(r["test"])
    compile_fails = Counter(
        r["detail"] for r in recs if r["category"] == "COMPILE_FAIL"
    )
    investigations = sorted(
        (r for r in recs if r.get("bucket") == "needs investigation"),
        key=lambda r: r["test"],
    )
    timeouts = Counter(
        r["detail"] for r in recs if r["category"] == "TIMEOUT"
    )
    convention_pass = sum(
        1 for r in recs if r["category"] == "MATCH" and r["detail"] == "100"
    )

    lines = [
        "# Runtime-test triage (step_08.0)",
        "",
        "Generated by `scripts/triage_runtime_tests.py` — do not edit by hand.",
        "Reproduce (from the workspace root):",
        "",
        "```sh",
        "python3 RokaJIT/scripts/triage_runtime_tests.py",
        "```",
        "",
        "Per-test results cache in",
        "`RokaJIT/target/triage/results.jsonl` (resumable;",
        "`--rerun` forces a fresh run, `--limit N` / `--only REGEX` select a",
        "subset, `--concurrency N` / `--timeout S` tune execution). Output is",
        "deterministic: same results file → identical document.",
        "",
        "Candidates are .cs tests the harness can run standalone: either a",
        "real `static int Main` or `[Fact]` no-arg `int`/`void` methods",
        "(synthesized entry point + attribute stubs mirroring",
        "Microsoft.DotNet.XUnitExtensions, legacy standalone semantics:",
        "100 = pass), compiled with all sibling .cs files from the test's",
        "directory (single-file fallback on collisions). Stubbed for our",
        "target with true semantics: `TestLibrary.PlatformDetection`",
        "(linux-x64 CoreCLR). Tests needing behavior-bearing helpers",
        "(Xunit.Assert, TestLibrary utilities, InlineIL) fail compile and",
        "are counted under COMPILE_FAIL.",
        "",
        "## Totals",
        "",
        f"- .cs files under `runtime/src/tests/JIT`: {scanned}",
        f"- Candidates: {len(candidates)} "
        f"(real Main: {entries.get('main', 0)}, [Fact] wrapper: {entries.get('fact', 0)})",
        f"- Triaged: {len(recs)}",
        "",
        "## Outcome categories",
        "",
        "| Category | Tests |",
        "| --- | ---: |",
    ]
    for cat in ("MATCH", "MISMATCH", "CRASH", "TIMEOUT", "COMPILE_FAIL"):
        if cats.get(cat):
            lines.append(f"| {cat} | {cats[cat]} |")
    lines += [
        "",
        f"Of the MATCHes, {convention_pass} exit 100 (the CoreCLR pass",
        "convention). Categories: COMPILE_FAIL = csc can't build it",
        "standalone; MATCH = same exit code and stdout under both JITs;",
        "MISMATCH = both ran, results differ; CRASH = RokaJIT-side run died",
        "on a signal (SIGABRT = the EE rejecting RokaJIT's",
        "CORJIT_IMPLLIMITATION); TIMEOUT = 10s per-test limit hit.",
        "",
        "## Feature buckets — the ordered backlog for step_08.1+",
        "",
        "Every CRASH/MISMATCH whose RokaJIT stderr carries a `CompileError`",
        "marker, bucketed by the missing feature the marker names. Ordered by",
        "test count, descending.",
        "",
        "| Bucket | Tests | Example tests |",
        "| --- | ---: | --- |",
    ]
    for bucket, tests in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if bucket == "needs investigation":
            continue
        lines.append(f"| {bucket} | {len(tests)} | {fmt_examples(tests)} |")
    lines += [
        "",
        "## COMPILE_FAIL by csc error class",
        "",
        "Tests the standalone harness cannot build (helper-library deps:",
        "TestLibrary, InlineIL, Xunit.Assert; multi-file projects; exotic",
        "entry shapes). Out of scope for triage; listed for the record.",
        "",
        "| csc error class | Tests |",
        "| --- | ---: |",
    ]
    for cls, count in sorted(compile_fails.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {cls} | {count} |")
    if timeouts:
        lines += [
            "",
            "## TIMEOUT detail",
            "",
            "| Which JIT timed out | Tests |",
            "| --- | ---: |",
        ]
        for which, count in sorted(timeouts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| {which} | {count} |")
    lines += [
        "",
        "## Needs investigation",
        "",
        "RokaJIT failures with no `CompileError` marker in stderr — either",
        "silent-wrong-result bugs (MISMATCH with a clean run) or crashes the",
        "error model didn't classify. Each carries its stderr signature.",
        "",
    ]
    if investigations:
        lines += [
            "| Test | Category | Detail | Stderr signature |",
            "| --- | --- | --- | --- |",
        ]
        for r in investigations:
            sig = r.get("bucket_detail", "").replace("|", "\\|")
            lines.append(f"| `{r['test']}` | {r['category']} | {r['detail']} | {sig} |")
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


# --- main -------------------------------------------------------------------


def stage_coreroots() -> dict[str, Path]:
    """Stage both coreroots once; refresh the staged RokaJIT libclrjit.so
    when target/debug/librokajit.so is newer."""
    roots = {jit: _stage_coreroot(jit) for jit in ("ryujit", "rokajit")}
    staged = roots["rokajit"] / "libclrjit.so"
    source = _rokajit_lib("debug")
    if not source.is_file():
        raise ValueError(f"{source} missing — run build_rokajit first")
    if staged.stat().st_mtime < source.stat().st_mtime:
        staged.unlink()
        shutil.copy(source, staged)
        print(f"refreshed staged {staged} from {source}", file=sys.stderr)
    return roots


async def run(args: argparse.Namespace) -> None:
    print("staging coreroots…", file=sys.stderr)
    coreroots = stage_coreroots()
    prelink_runtime_pack()

    scanned, candidates = enumerate_candidates()
    if args.only:
        pattern = re.compile(args.only)
        candidates = [c for c in candidates if pattern.search(c["test"])]
    if args.limit:
        candidates = candidates[: args.limit]

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_results(results_path)
    todo = []
    for c in candidates:
        rec = records.get(c["test"])
        if (
            args.rerun
            or rec is None
            or rec.get("version") != RESULT_VERSION
            or rec.get("entry") != c["entry"]
            or rec.get("src_mtime") != (RUNTIME_TESTS / c["test"]).stat().st_mtime
        ):
            todo.append(c)
    skipped = len(candidates) - len(todo)
    print(
        f"{scanned} .cs scanned, {len(candidates)} candidates: "
        f"{len(todo)} to run, {skipped} cached",
        file=sys.stderr,
    )

    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(candidate: dict) -> dict:
        async with semaphore:
            return await triage_one(candidate, coreroots, args.timeout)

    started = time.monotonic()
    done = 0
    with results_path.open("a", encoding="utf-8") as out:
        tasks = [asyncio.ensure_future(guarded(c)) for c in todo]
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            records[rec["test"]] = rec
            out.write(json.dumps(rec, sort_keys=True) + "\n")
            out.flush()
            done += 1
            if done % 100 == 0 or done == len(todo):
                elapsed = int(time.monotonic() - started)
                print(f"  {done}/{len(todo)} triaged ({elapsed}s)", file=sys.stderr)

    # Aggregate over the full record set (cached + fresh), restricted to the
    # selected candidate set.
    selected = {c["test"]: records[c["test"]] for c in candidates if c["test"] in records}
    report = render_report(selected, scanned, candidates)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    cats = Counter(r["category"] for r in selected.values())
    print(f"categories: {dict(sorted(cats.items()))}", file=sys.stderr)
    print(f"report written to {report_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=10, help="per-test seconds")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="", help="regex over candidate relpaths")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--rerun", action="store_true", help="re-run the selected tests even if cached")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
