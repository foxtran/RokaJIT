#!/usr/bin/env python3
"""Measured-unlock analyzer for the RokaJIT feature ladder (step_10 tooling).

This is the measurement backing the feature ladder in
RokaJIT-internal/docs/step_10.md: it answers "which IL opcode, if the
importer learned it next, would make the most currently-failing tests
pass?"

Inputs (both produced by the triage harness, read-only here):
  - RokaJIT/target/triage/results.jsonl
      One JSON record per line with fields "test" (path like
      JIT/CodeGenBringUpTests/And1.cs) and "category"
      (MATCH/CRASH/MISMATCH/COMPILE_FAIL/TIMEOUT).
  - RokaJIT/target/triage/bin/*.dll
      The compiled test assemblies. Each dll's PE/CLI metadata is parsed
      by hand (struct-based, stdlib only) to enumerate method RVAs
      (excluding Xunit boilerplate and the synthesized TriageEntryPoint
      wrapper types), then each method body is linear-scanned to collect
      the set of IL opcode values it contains. 0xFE-prefixed opcodes are
      represented as 0xFE00 | second_byte. Methods with exception-handling
      sections (fat header MoreSects flag) are flagged as EH.

Opcode display names are parsed from the reference runtime checkout at
runtime/src/coreclr/inc/opcode.def (sibling of RokaJIT/ in the
workspace); if that file is unreadable, names fall back to hex.

Run from the workspace root:
  python3 RokaJIT/scripts/feature_unlock_analysis.py

Stdlib only, python3, no network. Output must be deterministic: all
rankings sort by count descending then opcode value ascending, all
scans iterate in sorted order, and no timestamps are printed.
"""
import argparse
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # RokaJIT/
WORKSPACE = ROOT.parent                                # workspace root
BIN = ROOT / "target" / "triage" / "bin"
RESULTS = ROOT / "target" / "triage" / "results.jsonl"
OPCODE_DEF = WORKSPACE / "runtime" / "src" / "coreclr" / "inc" / "opcode.def"

# Opcodes RokaJIT's importer accepts today.
# Keep in sync with the decode() gate in crates/rokajit/src/import.rs.
SUPPORTED = {
    0x00,                    # nop
    0x01,                    # break
    *range(0x02, 0x06),      # ldarg.0..3
    *range(0x06, 0x0A),      # ldloc.0..3
    *range(0x0A, 0x0E),      # stloc.0..3
    0x0E,                    # ldarg.s
    0x0F,                    # ldarga.s
    0x10,                    # starg.s
    0x11,                    # ldloc.s
    0x12,                    # ldloca.s
    0x13,                    # stloc.s
    0x14,                    # ldnull
    *range(0x15, 0x1F),      # ldc.i4.m1..ldc.i4.8
    0x1F,                    # ldc.i4.s
    0x20,                    # ldc.i4
    0x21,                    # ldc.i8 (step_10.2)
    0x22,                    # ldc.r4 (step_10.2)
    0x23,                    # ldc.r8 (step_10.2)
    0x25,                    # dup
    0x26,                    # pop
    0x28,                    # call
    0x29,                    # calli (step_10.12)
    0x2A,                    # ret
    *range(0x2B, 0x2E),      # br.s / brfalse.s / brtrue.s
    *range(0x2E, 0x38),      # short conditional branches
    0x38, 0x39, 0x3A,        # br / brfalse / brtrue
    *range(0x3B, 0x45),      # long conditional branches
    0x45,                    # switch
    *range(0x46, 0x51),      # ldind.i1/u1/i2/u2/i4/u4/i8/i/r4/r8/ref (10.13)
    *range(0x51, 0x58),      # stind.ref/i1/i2/i4/i8/r4/r8 (10.13)
    0x58, 0x59, 0x5A,        # add / sub / mul
    0x5B, 0x5C,              # div / div.un
    0x5D, 0x5E,              # rem / rem.un
    *range(0x5F, 0x67),      # and / or / xor / shl / shr / shr.un / neg / not
    *range(0x67, 0x6B),      # conv.i1 / conv.i2 / conv.i4 / conv.i8
    0x6B,                    # conv.r4 (step_10.2)
    0x6C,                    # conv.r8 (step_10.2)
    0x6D,                    # conv.u4
    0x6E,                    # conv.u8
    0x6F,                    # callvirt (step_10.4; full dispatch step_10.12)
    0x70,                    # cpobj (step_10.9)
    0x71,                    # ldobj (step_10.9)
    0x72,                    # ldstr (step_10.3)
    0x73,                    # newobj (step_10.4)
    0x74,                    # castclass (step_10.5)
    0x75,                    # isinst (step_10.5)
    0x76,                    # conv.r.un
    0x79,                    # unbox (step_10.5)
    0x7A,                    # throw (step_10.6)
    *range(0x82, 0x8C),      # conv.ovf.i1/i2/i4/i8/u1/u2/u4/u8/i/u .un (checked conv)
    0x7B,                    # ldfld (step_10.4)
    0x7C,                    # ldflda (step_10.4)
    0x7D,                    # stfld (step_10.4)
    0x7E,                    # ldsfld (step_10.7)
    0x7F,                    # ldsflda (step_10.7)
    0x80,                    # stsfld (step_10.7)
    0x81,                    # stobj (step_10.9)
    0x8C,                    # box (step_10.5)
    0x8D,                    # newarr (step_10.8)
    0x8E,                    # ldlen (step_10.8)
    0x8F,                    # ldelema (step_10.8)
    *range(0x90, 0x9B),      # ldelem.i1/u1/i2/u2/i4/u4/i8/i/r4/r8/ref (10.8)
    *range(0x9B, 0xA3),      # stelem.i/i1/i2/i4/i8/r4/r8/ref (10.8)
    0xA3,                    # ldelem (token form, step_10.8)
    0xA4,                    # stelem (token form, step_10.8)
    0xA5,                    # unbox.any (step_10.5)
    *range(0xB3, 0xBB),      # conv.ovf.i1/u1/i2/u2/i4/u4/i8/u8 (checked conv)
    0xC3,                    # ckfinite
    0xC2,                    # refanyval (TypedReference)
    0xC6,                    # mkrefany (TypedReference)
    0xD0,                    # ldtoken (step_10.10)
    0xD1,                    # conv.u2 (step_10.7)
    0xD2,                    # conv.u1 (step_10.7)
    0xD3,                    # conv.i (step_10.11, integer sources)
    0xD4, 0xD5,              # conv.ovf.i / conv.ovf.u (checked conv)
    *range(0xD6, 0xDC),      # add/sub/mul.ovf[.un] (checked arithmetic)
    0xDC,                    # endfinally (step_10.6)
    0xDD,                    # leave (step_10.6)
    0xDE,                    # leave.s (step_10.6)
    0xDF,                    # stind.i (step_10.13)
    0xE0,                    # conv.u (step_10.7)
}
# Supported 0xFE-prefixed opcodes, by second byte (opcode.def):
#   FE 01..05 = ceq, cgt, cgt.un, clt, clt.un
#   FE 06 = ldftn, FE 07 = ldvirtftn (step_10.12)
#   FE 09 = ldarg, FE 0A = ldarga, FE 0B = starg
#   FE 0C = ldloc, FE 0D = ldloca, FE 0E = stloc, FE 0F = localloc
#   FE 12 = unaligned., FE 13 = volatile. (prefixes, step_10.13)
#   FE 15 = initobj (step_10.9), FE 17 = cpblk, FE 18 = initblk
#   FE 1A = rethrow, FE 1E = readonly. (prefix)
#   FE 1C = sizeof (step_10.10)
#   FE 16 = constrained. (step_11.3C)
#   FE 1D = refanytype (TypedReference)
SUPPORTED_FE = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x12, 0x13, 0x15, 0x16, 0x17,
    0x18, 0x1A, 0x1C, 0x1D, 0x1E,
}

# --- IL linear-scan operand-size tables -------------------------------------
# Operand sizes for opcodes the scanner may encounter; anything not listed
# has a zero-byte operand. Only used to keep the linear scan in sync.
OP4 = {0x28, 0x29, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x79, 0x7B,
       0x7C, 0x7D, 0x7E, 0x7F, 0x80, 0x81, 0x8C, 0x8D, 0x8F, 0xA3, 0xA4,
       0xA5, 0xC2, 0xC6, 0xD0}
OP1 = {0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x1F}
IMM = {0x20: 4, 0x21: 8, 0x22: 4, 0x23: 8}
BRS = set(range(0x2B, 0x38))   # short branches: i8 operand
BRS.add(0xDE)                  # leave.s: i8 operand
BRL = set(range(0x38, 0x45))   # long branches: i32 operand
BRL.add(0xDD)                  # leave: i32 operand
FE_OP1 = {0x12}  # unaligned. takes a 1-byte alignment operand
FE_OP2 = {0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E}
FE_OP4 = {0x06, 0x07, 0x15, 0x16, 0x1C}

TABLE_LIMIT = 25

OPCODE_DEF_RE = re.compile(
    r'OPDEF\(\s*[A-Z0-9_]+\s*,\s*"([^"]+)"[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\s*'
    r'\d+\s*,\s*0x([0-9A-Fa-f]{2})\s*,\s*0x([0-9A-Fa-f]{2})\s*,'
)


def load_opcode_names(path):
    """Parse opcode.def into {opcode_value: display_name}.

    Byte1 0xFF means single-byte encoding (byte2 is the opcode);
    byte1 0xFE means two-byte encoding 0xFE xx (value 0xFE00 | byte2).
    Returns an empty dict if the file cannot be read.
    """
    names = {}
    try:
        text = path.read_text()
    except OSError:
        return names
    for m in OPCODE_DEF_RE.finditer(text):
        name, b1, b2 = m.group(1), int(m.group(2), 16), int(m.group(3), 16)
        if b1 == 0xFF:
            names[b2] = name
        elif b1 == 0xFE:
            names[0xFE00 | b2] = name
    return names


def hex_name(op):
    return f"0x{op:02X}" if op < 0x100 else f"0xFE {op & 0xFF:02X}"


# --- PE/CLI metadata parsing (stdlib struct only) ----------------------------

def rva_to_off(sections, rva):
    for va, vsz, raw, rsz in sections:
        if va <= rva < va + max(vsz, rsz):
            return raw + (rva - va)
    return None


def method_rvas(data):
    """Return (sections, [method RVAs]) for a managed dll.

    Methods belonging to Xunit.* types and the synthesized
    TriageEntryPoint wrapper types are excluded.
    """
    e = struct.unpack_from("<I", data, 0x3C)[0]
    nsec = struct.unpack_from("<H", data, e + 6)[0]
    opt = e + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    cli_rva = struct.unpack_from("<I", data, dd + 14 * 8)[0]
    secs = []
    soff = opt + struct.unpack_from("<H", data, e + 20)[0]
    for i in range(nsec):
        b = soff + 40 * i
        vsz, va, rsz, raw = struct.unpack_from("<IIII", data, b + 8)
        secs.append((va, vsz, raw, rsz))
    cli = rva_to_off(secs, cli_rva)
    md_rva = struct.unpack_from("<I", data, cli + 8)[0]
    md = rva_to_off(secs, md_rva)
    assert data[md:md + 4] == b"BSJB"
    vlen = struct.unpack_from("<I", data, md + 12)[0]
    p = md + 16 + ((vlen + 3) & ~3)
    p += 2  # flags
    nstreams = struct.unpack_from("<H", data, p)[0]
    p += 2
    streams = {}
    for _ in range(nstreams):
        off, size = struct.unpack_from("<II", data, p)
        p += 8
        end = data.index(b"\0", p)
        nm = data[p:end].decode()
        p = (end + 4) & ~3
        streams[nm] = (md + off, size)
    t, _ = streams["#~"] if "#~" in streams else streams["#-"]
    heap_sizes = data[t + 6]
    p = t + 8
    valid = struct.unpack_from("<Q", data, p)[0]
    p += 8
    p += 8  # sorted
    rows = {}
    for i in range(64):
        if valid >> i & 1:
            rows[i] = struct.unpack_from("<I", data, p)[0]
            p += 4
    strsz = 4 if heap_sizes & 1 else 2
    blobsz = 4 if heap_sizes & 4 else 2
    guidsz = 4 if heap_sizes & 2 else 2

    def idx(tid):
        return 4 if rows.get(tid, 0) >= 0x10000 else 2

    def coded2(*tids):  # 2-bit coded index (TypeDefOrRef, ResolutionScope, ...)
        return 4 if max(rows.get(t, 0) for t in tids) >= 0x4000 else 2

    def rs(tid):
        if tid == 0:
            return 2 + strsz + guidsz * 3
        if tid == 1:
            return coded2(0, 26, 35, 1) + strsz + strsz
        if tid == 2:
            return 4 + strsz + strsz + coded2(2, 1, 26) + idx(4) + idx(6)
        if tid == 3:
            return idx(4)
        if tid == 4:
            return 2 + strsz + blobsz
        if tid == 5:
            return idx(6)
        if tid == 6:
            return 4 + 2 + 2 + strsz + blobsz + idx(8)
        return None

    base = p
    typedefs = []  # (method_start_row_1based, namespace.name)
    strings_off = streams["#Strings"][0]

    def heap_str(i):
        end = data.index(b"\0", strings_off + i)
        return data[strings_off + i:end].decode(errors="replace")

    for tid in range(7):
        if tid not in rows:
            continue
        if tid == 2:
            for r in range(rows[2]):
                q = base + r * rs(2)
                name_i = struct.unpack_from("<I" if strsz == 4 else "<H", data, q + 4)[0]
                ns_i = struct.unpack_from("<I" if strsz == 4 else "<H", data, q + 4 + strsz)[0]
                mstart = struct.unpack_from(
                    "<I" if idx(6) == 4 else "<H",
                    data, q + 4 + strsz * 2 + coded2(2, 1, 26) + idx(4))[0]
                typedefs.append((mstart, heap_str(ns_i) + "." + heap_str(name_i)))
        if tid == 6:
            mrows = rows[6]
            excluded = set()
            for j, (mstart, tname) in enumerate(typedefs):
                mend = typedefs[j + 1][0] if j + 1 < len(typedefs) else mrows + 1
                if (tname.startswith("Xunit.") or "TriageEntryPoint" in tname
                        or tname.startswith(".Xunit")):
                    excluded |= set(range(mstart, mend))
            out = []
            for r in range(mrows):
                if r + 1 in excluded:
                    continue
                rva = struct.unpack_from("<I", data, base + r * rs(6))[0]
                if rva:
                    out.append(rva)
            return secs, out
        base += rows[tid] * rs(tid)
    return secs, []


def il_opcodes(data, secs, rva):
    """Linear-scan one method body. Returns ({opcode values}, has_eh)."""
    off = rva_to_off(secs, rva)
    if off is None:
        return set(), False
    b = data[off]
    if b & 3 == 2:  # tiny header
        size = b >> 2
        il = data[off + 1: off + 1 + size]
        eh = False
    else:           # fat header
        flags = struct.unpack_from("<H", data, off)[0]
        size = struct.unpack_from("<I", data, off + 4)[0]
        il = data[off + 12: off + 12 + size]
        eh = bool(flags & 0x8)  # MoreSects (bit 0x10 is InitLocals, not EH)
    ops, i = set(), 0
    while i < len(il):
        op = il[i]
        i += 1
        if op == 0xFE:
            op2 = il[i]
            i += 1
            ops.add(0xFE00 | op2)
            if op2 in FE_OP1:
                i += 1
            elif op2 in FE_OP2:
                i += 2
            elif op2 in FE_OP4:
                i += 4
        else:
            ops.add(op)
            if op == 0x45:  # switch
                n = struct.unpack_from("<I", il, i)[0]
                i += 4 + 4 * n
            elif op in OP4:
                i += 4
            elif op in OP1:
                i += 1
            elif op in BRS:
                i += 1
            elif op in BRL:
                i += 4
            elif op in IMM:
                i += IMM[op]
    return ops, eh


# --- Analysis ----------------------------------------------------------------

def is_supported(op):
    if op < 0x100:
        return op in SUPPORTED
    return (op & 0xFF) in SUPPORTED_FE


def load_results(path):
    results = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                results[r["test"]] = r["category"]
    return results


def scan_tests(results):
    """Scan every cached dll; return (per_test, unmatched, anomalies).

    per_test: {test_name: (frozenset of opcode values, has_eh)}
    unmatched: count of bin/ dlls with no test in results.jsonl
      (framework/runtime reference assemblies co-located in bin/)
    anomalies: sorted list of human-readable strings for dlls that
      failed to parse.
    """
    norm2test = {t.replace("/", "_").replace(".", "_"): t for t in results}
    per_test = {}
    unmatched = 0
    anomalies = []
    for dll in sorted(BIN.glob("*.dll")):
        test = norm2test.get(dll.stem)
        if test is None:
            unmatched += 1
            continue
        data = dll.read_bytes()
        try:
            secs, rvas = method_rvas(data)
        except Exception as exc:
            anomalies.append(f"{dll.name}: metadata parse failed ({exc!r})")
            continue
        allops, eh = set(), False
        for rva in rvas:
            ops, e = il_opcodes(data, secs, rva)
            allops |= ops
            eh |= e
        per_test[test] = (frozenset(allops), eh)
    return per_test, unmatched, sorted(anomalies)


def unsupported_by_test(per_test, results):
    """{test: frozenset of unsupported opcodes} over FAILING tests only.

    EH tests are no longer excluded: step_10.6 implemented
    try/catch/finally, so EH-bearing tests enter the unlock pool like
    any other. MATCH tests are excluded (step_11.1): an opcode already
    passing under it unlocks nothing — counting MATCH tests inflated
    every bucket with already-unlocked tests.
    """
    return {t: frozenset(o for o in ops if not is_supported(o))
            for t, (ops, _eh) in per_test.items()
            if results.get(t) not in (None, "MATCH")}


def greedy_ladder(unsup):
    """Greedy set-cover ladder over {test: unsupported-opcode set}.

    Each round picks the opcode that completes (empties the unsupported
    set of) the most tests; ties break by lower opcode value. Returns a
    list of (opcode, newly_unlocked, cumulative) rounds.
    """
    remaining = {t: s for t, s in unsup.items() if s}
    rounds = []
    cumulative = 0
    while True:
        # completion count for opcode X = tests whose unsupported set
        # is non-empty and a subset of {X}, i.e. exactly {X} since sets
        # shrink each round.
        completes = Counter()
        for s in remaining.values():
            if len(s) == 1:
                completes[next(iter(s))] += 1
        if not completes:
            break
        best = min(completes, key=lambda o: (-completes[o], o))
        n = completes[best]
        cumulative += n
        rounds.append((best, n, cumulative))
        remaining = {t: s - {best} for t, s in remaining.items()}
        remaining = {t: s for t, s in remaining.items() if s}
    return rounds, len(remaining)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measured-unlock analyzer for the RokaJIT feature "
                    "ladder (step_10 tooling). Scans the triage cache and "
                    "ranks IL opcodes by how many tests implementing them "
                    "would unlock.")
    parser.parse_args(argv)

    if not RESULTS.is_file() or not BIN.is_dir():
        print(f"error: triage cache not found (need {RESULTS} and {BIN}); "
              "run the triage harness first", file=sys.stderr)
        return 2

    names = load_opcode_names(OPCODE_DEF)

    def name_of(op):
        return names.get(op, hex_name(op))

    results = load_results(RESULTS)
    per_test, unmatched, anomalies = scan_tests(results)

    if unmatched:
        print(f"note: {unmatched} bin/ dlls have no matching test in "
              "results.jsonl (framework/runtime assemblies), skipped",
              file=sys.stderr)
    for a in anomalies:
        print(f"warning: {a}", file=sys.stderr)

    pool = unsupported_by_test(per_test, results)
    needs = {t: s for t, s in pool.items() if s}

    solo = Counter()       # tests whose ONLY unsupported opcode is X
    membership = Counter()  # tests where X appears in the unsupported set
    for s in needs.values():
        if len(s) == 1:
            solo[next(iter(s))] += 1
        for o in s:
            membership[o] += 1

    def ranked(counter):
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

    print(f"tests scanned: {len(per_test)}")

    print("\n== Solo-unlock: tests whose only unsupported opcode is X ==")
    for o, n in ranked(solo)[:TABLE_LIMIT]:
        print(f"{n:6d}  {name_of(o)}")

    print("\n== Membership upper bound: tests where X is in the "
          "unsupported set ==")
    for o, n in ranked(membership)[:TABLE_LIMIT]:
        print(f"{n:6d}  {name_of(o)}")

    rounds, residual = greedy_ladder(pool)
    print("\n== Greedy unlock ladder (opcode that completes the most "
          "tests, applied in order) ==")
    for rank, (o, n, cum) in enumerate(rounds, 1):
        print(f"{rank:3d}. {name_of(o):24s} +{n:5d}  cumulative {cum}")
    print(f"total unlocked by ladder: {rounds[-1][2] if rounds else 0}")
    print(f"residual tests needing multi-opcode progress: {residual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
