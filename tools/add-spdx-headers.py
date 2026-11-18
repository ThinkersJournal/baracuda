#!/usr/bin/env python3
"""Add one `SPDX-License-Identifier` line to every source file that lacks one (CodeRipper SUP-016).

Rule (CireSnave 2026-11-17, via the PM): one header per file, no REUSE.toml shortcut.
The file set mirrors SUP-016 (coderipper src/module/neutral/licence.rs): `git ls-files`
(never a disk walk), the same extension list, the same skipped directory names, and a
file "has a header" when SPDX-License-Identifier appears in its first 15 lines.

Provenance (portfolio rule, EXPECTATIONS 6.4a: "I am not a plageurist"): a crate whose root
holds a provenance record (NOTICE*, ATTRIBUTION*, THIRD*PARTY*, LICENSE-thirdparty*) may contain
files derived from someone else's work. Stamping them with our identifier would claim a licence
that is not ours to claim, so those crates are SKIPPED and listed for a human decision.

    python tools/add-spdx-headers.py            # dry run: counts and the skipped list
    python tools/add-spdx-headers.py --apply    # write; prints every file touched and before/after counts
    python tools/add-spdx-headers.py --check    # exit 1 if any eligible file lacks a header
"""
import re
import subprocess
import sys

IDENT = "MIT OR Apache-2.0"  # `[workspace.package] license`
SLASH = {"rs", "h", "hpp", "cpp", "cc", "c", "js", "mjs", "cjs", "jsx", "ts", "tsx", "go", "java", "kt", "cs", "swift", "php"}
HASH = {"py", "sh", "ps1", "rb"}
EXTS = SLASH | HASH
SKIPPED = {"fixtures", "fixture", "testdata", "test_data", "conformance", "snapshots", "golden", "vendor", "vendors",
           "third_party", "third-party", "node_modules", "bower_components", "target"}
PROVENANCE = re.compile(r"^(NOTICE|ATTRIBUTION|THIRD[-_]?PARTY|LICENSE-thirdparty)", re.I)
HEADER_LINES = 15


def ls_files():
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def is_source(p):
    seg = p.split("/")
    if any(s in SKIPPED for s in seg[:-1]):
        return False
    name = seg[-1]
    return "." in name and name.rsplit(".", 1)[1].lower() in EXTS


def has_header(text):
    return any("SPDX-License-Identifier" in l for l in text.split("\n")[:HEADER_LINES])


def provenance_crates(files):
    roots = set()
    for p in files:
        seg = p.split("/")
        if len(seg) == 3 and seg[0] == "crates" and PROVENANCE.match(seg[2]):
            roots.add("crates/" + seg[1] + "/")
    return roots


def stamp(raw, ext):
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    text = raw[len(bom):].decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    line = ("// " if ext in SLASH else "# ") + "SPDX-License-Identifier: " + IDENT + nl
    if text.startswith("#!"):
        first, sep, rest = text.partition("\n")
        return bom + (first + sep + line + rest).encode("utf-8")
    return bom + (line + text).encode("utf-8")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dry"
    files = ls_files()
    sources = [p for p in files if is_source(p)]
    prov = provenance_crates(files)

    def missing_of(paths):
        return [p for p in paths if not has_header(open(p, "rb").read().decode("utf-8"))]

    missing = missing_of(sources)
    skipped = [p for p in missing if any(p.startswith(r) for r in prov)]
    eligible = [p for p in missing if p not in set(skipped)]
    print(f"enumerated: git ls-files ({len(files)} tracked), {len(sources)} source files (SUP-016 extensions, skipped dirs excluded)")
    print(f"BEFORE: {len(missing)} of {len(sources)} without SPDX-License-Identifier in the first {HEADER_LINES} lines")
    print(f"provenance crates skipped ({len(prov)}): {', '.join(sorted(r.rstrip('/') for r in prov)) or '-'}")
    print(f"skipped files needing a human decision: {len(skipped)}")
    for p in skipped:
        print("  SKIP", p)
    print(f"eligible: {len(eligible)}")
    if mode == "--check":
        sys.exit(1 if eligible else 0)
    if mode != "--apply":
        return
    for p in eligible:
        raw = open(p, "rb").read()
        open(p, "wb").write(stamp(raw, p.rsplit(".", 1)[1].lower()))
        print("  TOUCHED", p)
    after = missing_of(sources)
    print(f"touched: {len(eligible)} files")
    print(f"AFTER: {len(after)} of {len(sources)} without a header (the {len(skipped)} skipped; difference from skipped = {len(set(after) ^ set(skipped))})")


if __name__ == "__main__":
    main()
