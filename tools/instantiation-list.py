#!/usr/bin/env python3
"""PROTOTYPE: (symbol, launcher, template-args, source file, site line) from PREPROCESSED .cu.

    python tools/instantiation-list.py [kernels/elementwise] [--nvcc nvcc] [--arch sm_89] [--only SUBSTR]

Runs `nvcc -E` on every .cu in ONE directory under crates/baracuda-kernels-sys with the
flags build.rs gives that directory (-I kernels/include, -DBARACUDA_KERNELS_HAS_SM89,
-arch), so the INSTANTIATE macros are really expanded, then reads each `extern "C" ...
<symbol>_run(` wrapper the expansion produced and the first `launcher<args>(` call in its
body. TSV on stdout, one row per `_run` symbol:
    symbol  launcher  template_args  source_file  site_line
`site_line` is the line the preprocessor attributes to the expansion (the END of a
multi-line macro invocation). Needs nvcc and, on Windows, cl.exe on PATH.
Prototype limits: elementwise-style wrappers only (a body whose first template call is
not the launcher is reported as is); no feature gating; one directory.
ROADMAP row: "Machine-readable instantiation list".
"""
import os, re, subprocess, sys, tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "baracuda-kernels-sys")
WRAP = re.compile(r'extern\s+"C"\s+\w+\s+(baracuda_kernels_\w+?)_run\s*\(')
MARK = re.compile(r'^#line (\d+) "(.*)"')


def balanced(s, i):
    """s[i] == '<': index just past the matching '>' (nested <> only), or -1."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == '<':
            depth += 1
        elif s[j] == '>':
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def first_launch(body):
    for m in re.finditer(r'([A-Za-z_][\w:]*)\s*<', body):
        if m.group(1).endswith("_cast"):  # static_cast<...>(x) is not the launcher
            continue
        end = balanced(body, m.end() - 1)
        if end > 0 and body[end:].lstrip().startswith('('):
            return m.group(1), body[m.end():end - 1].strip()
    return "-", "-"


def rows(path, nvcc, arch):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "pp.i")
        cmd = [nvcc, "-E", "-I", "kernels/include", "-DBARACUDA_KERNELS_HAS_SM89",
               f"-arch={arch}", path, "-o", out]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"nvcc -E failed for {path}: {r.stderr}")
        want = os.path.normcase(os.path.abspath(os.path.join(ROOT, path)))
        cur_file, cur_line = None, 0
        with open(out, encoding="utf-8", errors="replace") as f:
            for text in f:
                m = MARK.match(text)
                if m:
                    cur_line = int(m.group(1))
                    cur_file = os.path.normcase(os.path.abspath(m.group(2).replace("\\\\", "\\")))
                    continue
                if cur_file == want and 'extern "C"' in text:
                    pieces = [(w.group(1), w.start()) for w in WRAP.finditer(text)]
                    for k, (sym, at) in enumerate(pieces):
                        stop = pieces[k + 1][1] if k + 1 < len(pieces) else len(text)
                        launcher, args = first_launch(text[at:stop])
                        yield sym, launcher, args, path.replace(os.sep, "/"), cur_line
                cur_line += 1


def main():
    args = [a for a in sys.argv[1:]]
    nvcc, arch = "nvcc", "sm_89"
    if "--nvcc" in args:
        i = args.index("--nvcc"); nvcc = args[i + 1]; del args[i:i + 2]
    if "--arch" in args:
        i = args.index("--arch"); arch = args[i + 1]; del args[i:i + 2]
    only = None
    if "--only" in args:  # substring filter on the file name (each nvcc -E takes ~5 s)
        i = args.index("--only"); only = args[i + 1]; del args[i:i + 2]
    d = args[0] if args else "kernels/elementwise"
    # Enumerate from the index, never by walking the disk (portfolio rule 5b).
    tracked = subprocess.run(["git", "ls-files", "--", d], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()
    files = sorted(p for p in tracked if p.endswith(".cu") and (only is None or only in p))
    print("# ran: git ls-files %s -> %d .cu; nvcc -E per file" % (d, len(files)), file=sys.stderr)
    print("symbol\tlauncher\ttemplate_args\tsource_file\tsite_line")
    for p in files:
        for r in rows(p, nvcc, arch):
            print("\t".join(str(x) for x in r))


if __name__ == "__main__":
    main()
