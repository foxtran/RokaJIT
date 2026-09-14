#!/usr/bin/env python3
"""Generic-shape analyzer for the step_11.3 phase-A scope decision.

Answers, over the triage generics bucket (RokaJIT/target/triage/
results.jsonl rows with bucket == "generics"):

  1. generic METHODS vs generic TYPES — which shape each failing test
     carries, measured two ways:
       - the first-failure marker split (bucket_detail: the GENERIC
         callconv gate vs the PARAMTYPE shared-code gate), and
       - local metadata: GenericParam table owners (TypeDef vs
         MethodDef) in the test assembly.
  2. reference-type vs value-type INSTANTIATIONS — MethodSpec and
     TypeSpec(GENERICINST) signature blobs decoded and classified:
     an instantiation whose argument list contains a value type is
     unsharable (per-instantiation body); all-reference or
     type-parameter-only instantiations share the canonical body.
     Instantiated type names are histogrammed to spot
     List<int>-shaped pools.

Inputs (read-only): the triage cache, like feature_unlock_analysis.py.

Run from the workspace root:
  python3 RokaJIT/scripts/generic_shape_analysis.py

Stdlib only, python3, no network. Output must be deterministic: all
rankings sort by count descending then name ascending, all scans
iterate in sorted order, and no timestamps are printed.
"""
import json
import struct
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # RokaJIT/
BIN = ROOT / "target" / "triage" / "bin"
RESULTS = ROOT / "target" / "triage" / "results.jsonl"

# CLI metadata table ids used here.
T_MODULEREF, T_TYPEDEF, T_PARAM, T_METHOD = 0x1A, 0x02, 0x08, 0x06
T_MEMBERREF, T_TYPESPEC = 0x0A, 0x1B
T_GENERICPARAM, T_METHODSPEC = 0x2A, 0x2B

# ELEMENT_TYPE_* codes (ECMA-335 II.23.1.16) relevant to sig decoding.
ET = {
    0x01: "void", 0x02: "bool", 0x03: "char", 0x04: "i1", 0x05: "u1",
    0x06: "i2", 0x07: "u2", 0x08: "i4", 0x09: "u4", 0x0A: "i8",
    0x0B: "u8", 0x0C: "r4", 0x0D: "r8", 0x0E: "string",
}
ET_PTR, ET_BYREF = 0x0F, 0x10
ET_VALUETYPE, ET_CLASS = 0x11, 0x12
ET_VAR, ET_ARRAY, ET_GENERICINST = 0x13, 0x14, 0x15
ET_I, ET_U = 0x18, 0x19
ET_OBJECT, ET_SZARRAY, ET_MVAR = 0x1C, 0x1D, 0x1E
ET_CMOD_REQD, ET_CMOD_OPT, ET_SENTINEL, ET_PINNED = 0x1F, 0x20, 0x41, 0x45

VALUE_ETS = {0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
             0x0C, 0x0D, ET_I, ET_U}                       # primitives
REF_ETS = {0x0E, ET_OBJECT}                                # string, object


def rva_to_off(sections, rva):
    for va, vsz, raw, rsz in sections:
        if va <= rva < va + max(vsz, rsz):
            return raw + (rva - va)
    return None


class Metadata:
    """Minimal ECMA-335 metadata reader: row counts, table offsets, the
    #Strings/#Blob heaps, and decoded TypeRef/TypeDef/GenericParam/
    MethodSpec/TypeSpec rows for tables 0x00..0x2C."""

    # Coded-index tags: name -> (tag bits, member table ids).
    CODED = {
        "TypeDefOrRef": (2, (T_TYPEDEF, 0x01, T_TYPESPEC)),
        "HasConstant": (2, (0x04, T_PARAM, 0x17)),
        "HasCustomAttribute": (5, (T_METHOD, 0x04, 0x01, T_TYPEDEF, T_PARAM,
                                   0x09, 0x0A, 0x00, 0x0E, 0x17, 0x14,
                                   0x11, 0x1A, 0x1B, 0x20, 0x23, 0x26,
                                   0x27, 0x28, 0x2A, 0x2B, 0x2C)),
        "HasFieldMarshal": (1, (0x04, T_PARAM)),
        "HasDeclSecurity": (2, (T_TYPEDEF, T_METHOD, 0x20)),
        "MemberRefParent": (3, (T_TYPEDEF, 0x01, T_MODULEREF, T_METHOD,
                                T_TYPESPEC)),
        "HasSemantics": (1, (0x14, 0x17)),
        "MethodDefOrRef": (1, (T_METHOD, T_MEMBERREF)),
        "MemberForwarded": (1, (0x04, T_METHOD)),
        "Implementation": (2, (0x26, 0x23, 0x27)),
        "CustomAttributeType": (3, (0, 0, T_METHOD, T_MEMBERREF, 0)),
        "ResolutionScope": (2, (0x00, T_MODULEREF, 0x23, 0x01)),
        "TypeOrMethodDef": (1, (T_TYPEDEF, T_METHOD)),
    }

    def __init__(self, data):
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
        self.sections = secs
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
        self.data = data
        self.strings_off = streams["#Strings"][0]
        self.blob_off = streams["#Blob"][0]
        t, _ = streams["#~"] if "#~" in streams else streams["#-"]
        heap_sizes = data[t + 6]
        self.strsz = 4 if heap_sizes & 1 else 2
        self.blobsz = 4 if heap_sizes & 4 else 2
        self.guidsz = 4 if heap_sizes & 2 else 2
        p = t + 8
        valid = struct.unpack_from("<Q", data, p)[0]
        p += 16  # valid + sorted
        self.rows = {}
        for i in range(64):
            if valid >> i & 1:
                self.rows[i] = struct.unpack_from("<I", data, p)[0]
                p += 4
        self.toff = {}
        base = p
        for tid in range(0x2D):
            if tid in self.rows:
                self.toff[tid] = base
                base += self.rows[tid] * self.row_size(tid)

    def idx(self, tid):
        return 4 if self.rows.get(tid, 0) >= 0x10000 else 2

    def coded(self, name):
        bits, tids = self.CODED[name]
        hi = max(self.rows.get(t, 0) for t in tids)
        return 4 if hi >= (1 << (16 - bits)) else 2

    def row_size(self, tid):
        s, b, g = self.strsz, self.blobsz, self.guidsz

        def c(name):
            return self.coded(name)

        def i(t):
            return self.idx(t)
        schema = {
            0x00: [2, s, g, g, g],
            0x01: [c("ResolutionScope"), s, s],
            0x02: [4, s, s, c("TypeDefOrRef"), i(0x04), i(T_METHOD)],
            0x03: [i(0x04)],
            0x04: [2, s, b],
            0x05: [i(T_METHOD)],
            0x06: [4, 2, 2, s, b, i(T_PARAM)],
            0x07: [i(T_PARAM)],
            0x08: [2, 2, s],
            0x09: [i(T_TYPEDEF), c("TypeDefOrRef")],
            0x0A: [c("MemberRefParent"), s, b],
            0x0B: [2, c("HasConstant"), b],
            0x0C: [c("HasCustomAttribute"), c("CustomAttributeType"), b],
            0x0D: [c("HasFieldMarshal"), b],
            0x0E: [2, c("HasDeclSecurity"), b],
            0x0F: [2, 4, i(T_TYPEDEF)],
            0x10: [4, i(0x04)],
            0x11: [b],
            0x12: [i(T_TYPEDEF), i(0x14)],
            0x13: [i(0x14)],
            0x14: [2, s, c("TypeDefOrRef")],
            0x15: [i(T_TYPEDEF), i(0x17)],
            0x16: [i(0x17)],
            0x17: [2, s, b],
            0x18: [2, i(T_METHOD), c("HasSemantics")],
            0x19: [i(T_TYPEDEF), c("MethodDefOrRef"), c("MethodDefOrRef")],
            0x1A: [s],
            0x1B: [b],
            0x1C: [2, c("MemberForwarded"), s, i(T_MODULEREF)],
            0x1D: [4, i(0x04)],
            0x1E: [4, 4],
            0x1F: [4],
            0x20: [4, 2, 2, 2, 2, 4, b, s, s],
            0x21: [4, 4, 4],
            0x22: [4, s, s, s],
            0x23: [2, 2, 2, 2, 4, b, s, s, b],
            0x24: [4, 4, i(0x26)],
            0x25: [4, 4, i(0x26)],
            0x26: [4, s, b],
            0x27: [4, 4, s, s, c("Implementation")],
            0x28: [4, 4, s, c("Implementation")],
            0x29: [i(T_TYPEDEF), i(T_TYPEDEF)],
            0x2A: [2, 2, c("TypeOrMethodDef"), s],
            0x2B: [c("MethodDefOrRef"), b],
            0x2C: [i(T_GENERICPARAM), c("TypeDefOrRef")],
        }
        return sum(schema[tid])

    def heap_str(self, i):
        if i == 0:
            return ""
        end = self.data.index(b"\0", self.strings_off + i)
        return self.data[self.strings_off + i:end].decode(errors="replace")

    def heap_blob(self, i):
        n, p = self.compressed(self.blob_off + i)
        return self.data[p:p + n]

    def compressed(self, p):
        b0 = self.data[p]
        if b0 & 0x80 == 0:
            return b0, p + 1
        if b0 & 0xC0 == 0x80:
            return ((b0 & 0x3F) << 8) | self.data[p + 1], p + 2
        v = ((b0 & 0x1F) << 24) | (self.data[p + 1] << 16) \
            | (self.data[p + 2] << 8) | self.data[p + 3]
        return v, p + 4

    def row(self, tid, ridx):  # ridx is 1-based; returns raw row slice
        off = self.toff[tid] + (ridx - 1) * self.row_size(tid)
        return off

    def read(self, off, size):
        if size == 2:
            return struct.unpack_from("<H", self.data, off)[0]
        return struct.unpack_from("<I", self.data, off)[0]

    def type_name(self, coded_index):
        """TypeDefOrRef coded index -> display name (best effort)."""
        tag, ridx = coded_index & 3, coded_index >> 2
        try:
            if tag == 0 and ridx:  # TypeDef
                off = self.row(T_TYPEDEF, ridx)
                name = self.heap_str(self.read(off + 4, self.strsz))
                ns = self.heap_str(self.read(off + 4 + self.strsz, self.strsz))
                return f"{ns}.{name}" if ns else name
            if tag == 1 and ridx:  # TypeRef
                off = self.row(0x01, ridx)
                sz = self.coded("ResolutionScope")
                name = self.heap_str(self.read(off + sz, self.strsz))
                ns = self.heap_str(self.read(off + sz + self.strsz, self.strsz))
                return f"{ns}.{name}" if ns else name
        except (struct.error, ValueError):
            pass
        return "?"

    def generic_param_owners(self):
        """GenericParam rows -> list of owner kinds: 'type' | 'method'."""
        out = []
        if T_GENERICPARAM not in self.rows:
            return out
        csz = self.coded("TypeOrMethodDef")
        for r in range(1, self.rows[T_GENERICPARAM] + 1):
            off = self.row(T_GENERICPARAM, r)
            owner = self.read(off + 4, csz)
            out.append("method" if owner & 1 else "type")
        return out

    def _decode_type(self, blob, p, names):
        """Decode one Type signature at blob[p:].
        Returns (classification, new_p) where classification is one of
        'ref' | 'val' | 'var' | 'void' | 'other'. GENERICINST
        occurrences are reported through names (list of display names).
        """
        et = blob[p]
        p += 1
        if et in VALUE_ETS:
            return "val", p
        if et in REF_ETS:
            return "ref", p
        if et in (ET_VAR, ET_MVAR):
            _, p = self._comp_in(blob, p)
            return "var", p
        if et in (ET_VALUETYPE, ET_CLASS):
            coded, p = self._comp_in(blob, p)
            names.append(self.type_name(coded))
            return ("val" if et == ET_VALUETYPE else "ref"), p
        if et == ET_GENERICINST:
            kind = blob[p]
            p += 1
            coded, p = self._comp_in(blob, p)
            names.append(self.type_name(coded))
            n, p = self._comp_in(blob, p)
            inner = "ref"
            for _ in range(n):
                c2, p = self._decode_type(blob, p, names)
                if c2 == "val":
                    inner = "val"
            own = "val" if kind == ET_VALUETYPE else "ref"
            return ("val" if inner == "val" or own == "val" else "ref"), p
        if et == ET_SZARRAY:
            _, p = self._decode_type(blob, p, names)
            return "ref", p
        if et == ET_ARRAY:
            _, p = self._decode_type(blob, p, names)
            rank, p = self._comp_in(blob, p)
            nsz, p = self._comp_in(blob, p)
            for _ in range(nsz):
                _, p = self._comp_in(blob, p)
            nlo, p = self._comp_in(blob, p)
            for _ in range(nlo):
                _, p = self._comp_in(blob, p)
            _ = rank
            return "ref", p
        if et in (ET_PTR, ET_BYREF):
            _, p = self._decode_type(blob, p, names)
            return "other", p
        if et in (ET_CMOD_REQD, ET_CMOD_OPT):
            _, p = self._comp_in(blob, p)
            return self._decode_type(blob, p, names)
        if et == ET_PINNED:
            return self._decode_type(blob, p, names)
        if et == ET_SENTINEL:
            return "other", p
        if et == 0x01:  # void
            return "void", p
        return "other", p  # FNPTR etc.

    def _comp_in(self, blob, p):
        b0 = blob[p]
        if b0 & 0x80 == 0:
            return b0, p + 1
        if b0 & 0xC0 == 0x80:
            return ((b0 & 0x3F) << 8) | blob[p + 1], p + 2
        v = ((b0 & 0x1F) << 24) | (blob[p + 1] << 16) | (blob[p + 2] << 8) \
            | blob[p + 3]
        return v, p + 4

    def instantiation_classes(self):
        """Decode MethodSpec + TypeSpec instantiation blobs.
        Returns (methodspec_arg_classes, typespec_records) where
        methodspec_arg_classes is a list of per-MethodSpec classification
        sets ({'ref','val','var',...}) and typespec_records is a list of
        (outer_name, classification) for each GENERICINST found.
        """
        ms_out, ts_out = [], []
        for r in range(1, self.rows.get(T_METHODSPEC, 0) + 1):
            off = self.row(T_METHODSPEC, r)
            bidx = self.read(off + self.coded("MethodDefOrRef"), self.blobsz)
            blob = self.heap_blob(bidx)
            if not blob or blob[0] != 0x0A:  # GENERICINST calling conv
                continue
            n, p = self._comp_in(blob, 1)
            names, classes = [], set()
            for _ in range(n):
                c, p = self._decode_type(blob, p, names)
                classes.add(c)
            ms_out.append(classes)
        for r in range(1, self.rows.get(T_TYPESPEC, 0) + 1):
            off = self.row(T_TYPESPEC, r)
            blob = self.heap_blob(self.read(off, self.blobsz))
            names = []
            try:
                c, _ = self._decode_type(blob, 0, names)
            except (IndexError, struct.error):
                continue
            if names:
                ts_out.append((names[0], c))
        return ms_out, ts_out


def load_generics_bucket(path):
    tests = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tests[r["test"]] = r          # resumable cache: last wins
    return {t: r for t, r in tests.items() if r.get("bucket") == "generics"}


def main():
    if not RESULTS.is_file() or not BIN.is_dir():
        print(f"error: triage cache not found (need {RESULTS} and {BIN}); "
              "run the triage harness first", file=sys.stderr)
        return 2

    bucket = load_generics_bucket(RESULTS)
    norm2test = {t.replace("/", "_").replace(".", "_"): t for t in bucket}

    print(f"generics-bucket tests: {len(bucket)}")

    # 1. First-failure marker split.
    markers = Counter(r.get("bucket_detail") for r in bucket.values())
    print("\n== First-failure marker (which gate rejected the test) ==")
    for k, v in sorted(markers.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"{v:6d}  {k}")

    # 2-3. Per-assembly metadata shapes.
    def_kind = Counter()       # local generic defs: method/type/both/neither
    ms_shape = Counter()       # MethodSpec arg classification
    ts_kind = Counter()        # TypeSpec GENERICINST outer: class/valuetype
    ts_arg = Counter()         # TypeSpec GENERICINST args: ref-only/has-val
    top_types = Counter()      # instantiated type names
    scanned = skipped = 0
    for dll in sorted(BIN.glob("*.dll")):
        test = norm2test.get(dll.stem)
        if test is None:
            continue
        try:
            md = Metadata(dll.read_bytes())
        except Exception:
            skipped += 1
            continue
        scanned += 1
        owners = set(md.generic_param_owners())
        def_kind["+".join(sorted(owners)) if owners else "none"] += 1
        mss, tss = md.instantiation_classes()
        if not mss:
            ms_shape["no methodspec"] += 1
        else:
            flat = set().union(*mss)
            if "val" in flat:
                ms_shape["has value-type arg"] += 1
            elif flat <= {"ref"}:
                ms_shape["ref-only"] += 1
            elif flat <= {"ref", "var"}:
                ms_shape["ref+type-param only"] += 1
            else:
                ms_shape["type-param only / other"] += 1
        saw = False
        for name, cls in tss:
            saw = True
            ts_kind[cls] += 1
            top_types[name] += 1
        if not saw:
            ts_kind["no genericinst typespec"] += 1
        if any(cls == "val" for _, cls in tss):
            ts_arg["has value-type shape"] += 1
        elif saw:
            ts_arg["ref-only"] += 1

    print(f"\nassemblies scanned: {scanned} (unparseable: {skipped})")

    print("\n== Local generic definitions (GenericParam table owners) ==")
    for k, v in sorted(def_kind.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{v:6d}  {k}")

    print("\n== Generic METHOD instantiations (MethodSpec arg shapes) ==")
    for k, v in sorted(ms_shape.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{v:6d}  {k}")

    print("\n== Generic TYPE instantiations (TypeSpec GENERICINST, per use) ==")
    for k, v in sorted(ts_kind.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{v:6d}  {k}")

    print("\n== Tests carrying a value-type-instantiation shape "
          "(TypeSpec view) ==")
    for k, v in sorted(ts_arg.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{v:6d}  {k}")

    print("\n== Top instantiated type names (TypeSpec GENERICINST) ==")
    for k, v in sorted(top_types.items(), key=lambda kv: (-kv[1], kv[0]))[:25]:
        print(f"{v:6d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
