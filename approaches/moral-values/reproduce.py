#!/usr/bin/env python3
"""
Regenerate every number, table and figure of the measurement paper and its supplement from the cached predictions.

No GPU and no model training: the scripts below read the gold labels of ValueEval'24 (data/, not
redistributed) and the per-seed predictions tracked in this repository (approaches/*/output/).

Steps, in order (the first six are independent; `assets` and `supplement` read their CSVs):
    classification  threshold_schemes.py     macro-F1 by decision rule, gates, presence diagnostics
    measurement     measurement_analysis.py  prevalence, contrasts and slopes for every estimator
    shift           shift_tests.py           leave-one-language-out, genre transfer, news-only
    llm             llm_measurement.py       LLM labels used as counts
    reliability     gold_reliability.py      annotation agreement vs. per-language error
    worked          worked_example.py        manifesto vs. news, value by value
    assets          paper_assets.py          Figures 1-4 (PDF) and Tables I-V (LaTeX)
    supplement      supplement_assets.py     the supplement's tables and figures (supp-*.tex, figS*.pdf)

Every step runs in its own Python process, under an audit hook that records which files under
the repository root it reads and writes. The result is results/reproduce-manifest.tsv: the
SHA-256 of every input (gold labels, predictions) and every output. A run compares its outputs
with the manifest and says, file by file, whether they are identical, equal to the printed
precision (CSV numbers at most one unit apart in the fourth decimal), or different. It never overwrites the
reference manifest unless asked to (--update-manifest); if there is none, it writes one.

Usage (from this folder):
    python3 reproduce.py                    # run everything (about 40 min on an Apple M1), then compare
    python3 reproduce.py -j 4               # independent steps in parallel (the shift step alone takes ~22 min)
    python3 reproduce.py --steps shift,assets
    python3 reproduce.py --verify           # compare the files on disk with the manifest; run nothing
    python3 reproduce.py --update-manifest  # after an intended change: run, then record the new reference
    python3 reproduce.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = (HERE / ".." / "..").resolve()          # repository root; manifest paths are relative to it
MANIFEST = HERE / "results" / "reproduce-manifest.tsv"
PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "matplotlib"]
GOLD = [ROOT / "data" / f"{split}-english" / "labels-cat.tsv" for split in ["training", "validation", "test"]]
CSV_TOL = 1.5e-4                               # one unit in the 4th decimal (outputs have 4-6 decimals)
# One BLAS thread per step unless the caller sets these: the matrix products here are small, so extra
# threads cost more than they save (30-50% slower alone, much slower when steps run in parallel), and a
# fixed thread count keeps floating-point sums in the same order on every run.
SINGLE_THREAD = {k: "1" for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    what: str
    minutes: float
    final: bool = False                        # runs after all the others (reads their outputs)


STEPS = [
    # minutes: wall time on an Apple M1 with one BLAS thread per step, 19 Nov 2026 (rules refitted on
    # every validation resample, annotated-sample estimators, extra shift-test rules, subsampling)
    Step("classification", "threshold_schemes.py", "Table II: macro-F1 by decision rule, gates, presence", 0.3),
    Step("measurement", "measurement_analysis.py", "Table III, Fig. 2: prevalence, contrasts, slopes", 10.7),
    Step("shift", "shift_tests.py", "Fig. 3: a language or genre left out of the fit", 21.5),
    Step("llm", "llm_measurement.py", "Tables II-III: LLM labels used as counts", 0.8),
    Step("reliability", "gold_reliability.py", "Section on the gold standard: agreement vs. error", 0.1),
    Step("worked", "worked_example.py", "Fig. 4, Table IV: manifesto vs. news", 0.5),
    Step("subsample", "subsample_validation.py", "Supplement: how large the annotated sample must be", 4.6),
    Step("assets", "paper_assets.py", "Figures 1-4 (PDF) and Tables I-V (LaTeX)", 0.4, final=True),
    Step("supplement", "supplement_assets.py", "Supplement: Tables S1-S13 and Figures S1-S2", 0.3, final=True),
]
BY_NAME = {s.name: s for s in STEPS}

# Runs a script as __main__ and records the files under ROOT that it opens (argv: script, trace file, root).
TRACER = r'''
import json, os, runpy, sys
script, trace, root = sys.argv[1], sys.argv[2], os.path.realpath(sys.argv[3])
reads, writes = set(), set()
def hook(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.path.realpath(os.fsdecode(args[0]))
    if not path.startswith(root + os.sep) or path.endswith((".py", ".pyc")) or "__pycache__" in path:
        return
    mode, flags = args[1], args[2]
    writing = any(c in mode for c in "wax+") if isinstance(mode, str) else bool(flags & (os.O_WRONLY | os.O_RDWR))
    (writes if writing else reads).add(os.path.relpath(path, root))
sys.addaudithook(hook)
sys.argv = [script]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    with open(trace, "w") as f:
        json.dump({"reads": sorted(reads), "writes": sorted(writes)}, f)
'''


# ----------------------------------------------------------------------------- manifest
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions() -> str:
    parts = [f"python {platform.python_version()}"]
    for name in PACKAGES:
        try:
            parts.append(f"{name} {importlib.import_module(name).__version__}")
        except ImportError:
            parts.append(f"{name} missing")
    return ", ".join(parts)


def read_manifest() -> dict[tuple[str, str], dict]:
    rows = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or line.startswith("kind\t") or not line.strip():
                continue
            kind, steps, digest, size, path = line.split("\t")
            rows[(kind, path)] = {"steps": set(steps.split(",")), "sha256": digest, "bytes": int(size)}
    return rows


def write_manifest(rows: dict[tuple[str, str], dict]) -> None:
    order = {s.name: i for i, s in enumerate(STEPS)}
    lines = ["# Reference outputs of reproduce.py, and the inputs they were computed from.",
             f"# Recorded with {versions()}.",
             "# Paths are relative to the repository root. Gold labels (data/) are not redistributed.",
             "kind\tsteps\tsha256\tbytes\tpath"]
    for (kind, path), r in sorted(rows.items(), key=lambda kv: (kv[0][0] != "input", kv[0][1])):
        steps = ",".join(sorted(r["steps"], key=lambda s: order.get(s, 99)))
        lines.append(f"{kind}\t{steps}\t{r['sha256']}\t{r['bytes']}\t{path}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def current_rows(traces: dict[str, dict]) -> dict[tuple[str, str], dict]:
    """Manifest rows for the steps that just ran: inputs are files read but written by no step."""
    written = {p for t in traces.values() for p in t["writes"]}
    rows: dict[tuple[str, str], dict] = {}
    for step, t in traces.items():
        for kind, paths in [("output", t["writes"]), ("input", [p for p in t["reads"] if p not in written])]:
            for p in paths:
                if kind == "input" and p.startswith(os.path.relpath(MANIFEST.parent, ROOT)):
                    continue                    # outputs of a step that did not run this time
                row = rows.setdefault((kind, p), {"steps": set()})
                row["steps"].add(step)
    for (kind, p), row in rows.items():
        f = ROOT / p
        row["sha256"], row["bytes"] = sha256(f), f.stat().st_size
    return rows


# ----------------------------------------------------------------------------- comparison
def compare_csv(path: Path, ref_path: Path | None) -> str:
    if ref_path is None:
        return "differs"
    import numpy as np
    import pandas as pd
    a, b = pd.read_csv(ref_path), pd.read_csv(path)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return f"differs (shape {a.shape} -> {b.shape})"
    num = a.select_dtypes("number").columns
    other = [c for c in a.columns if c not in num]
    if not a[other].fillna("").astype(str).equals(b[other].fillna("").astype(str)):
        return "differs (text columns)"
    diff = np.abs(a[num].to_numpy(float) - b[num].to_numpy(float))
    both_nan = np.isnan(a[num].to_numpy(float)) & np.isnan(b[num].to_numpy(float))
    worst = float(np.nanmax(np.where(both_nan, 0.0, diff))) if diff.size else 0.0
    if not np.isfinite(worst):
        return "differs (missing values)"
    return f"equal to {CSV_TOL:g} (max |diff| {worst:.1e})" if worst <= CSV_TOL else f"differs (max |diff| {worst:.2g})"


def compare(ref: dict, now: dict, steps: set[str], snapshot: dict[str, Path]) -> bool:
    """Print a file-by-file comparison of the outputs of `steps` with the reference; True if all match."""
    ok = True
    changed_inputs = [p for (k, p), r in ref.items() if k == "input" and r["steps"] & steps
                      and (not (ROOT / p).exists() or sha256(ROOT / p) != r["sha256"])]
    if changed_inputs:
        print(f"\n{len(changed_inputs)} input(s) differ from the ones the reference was computed from:")
        for p in changed_inputs[:10]:
            print(f"  {'missing ' if not (ROOT / p).exists() else 'changed '} {p}")
    counts = {"identical": 0, "equal": 0, "differs": 0, "missing": 0, "new": 0}
    print("\nOutputs compared with results/reproduce-manifest.tsv:")
    for (kind, p), r in sorted(ref.items()):
        if kind != "output" or not r["steps"] & steps:
            continue
        f = ROOT / p
        if not f.exists():
            status = "missing"
        elif sha256(f) == r["sha256"]:
            status = "identical"
        elif p.endswith(".csv"):
            status = compare_csv(f, snapshot.get(p))
        elif p.endswith(".pdf"):
            status = "differs (figures also depend on the matplotlib version and the fonts installed)"
        else:
            status = "differs"
        key = status.split()[0]
        counts["equal" if key == "equal" else key] += 1
        ok &= key in {"identical", "equal"}
        if key != "identical":
            print(f"  {status:60s} {p}")
    for (kind, p) in sorted(now):
        if kind == "output" and (kind, p) not in ref:
            counts["new"] += 1
            print(f"  {'new (not in the reference)':60s} {p}")
    print("  " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    return ok


# ----------------------------------------------------------------------------- running
def preflight(steps: list[Step], ref: dict) -> None:
    problems = []
    for name in PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            problems.append(f"Python package '{name}' is not installed (see requirements-analysis.txt).")
    needed = {ROOT / p for (k, p), r in ref.items() if k == "input" and r["steps"] & {s.name for s in steps}}
    for f in sorted(needed | set(GOLD)):
        if not f.exists():
            problems.append(f"missing input: {f.relative_to(ROOT)}")
    if problems:
        print("Cannot run:\n  " + "\n  ".join(problems[:20]))
        if any("data" in p for p in problems):
            print("\nThe ValueEval'24 labels are not redistributed. Download the English release (data usage"
                  "\nagreement required), put each split's labels.tsv under data/<split>-english/, and run"
                  "\n    python3 data-prep/make_labels_cat.py\nfrom the repository root (see README.md).")
        sys.exit(2)


def run_step(step: Step, logdir: Path) -> dict:
    trace, log = logdir / f"{step.name}.json", logdir / f"{step.name}.log"
    start = time.time()
    with open(log, "w", encoding="utf-8") as out:
        proc = subprocess.run([sys.executable, "-c", TRACER, step.script, str(trace), str(ROOT)], cwd=HERE,
                              stdout=out, stderr=subprocess.STDOUT, env={**SINGLE_THREAD, **os.environ, "MPLBACKEND": "Agg"})
    if proc.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        raise RuntimeError(f"step '{step.name}' ({step.script}) failed; last lines of {log}:\n  " + "\n  ".join(tail))
    print(f"  {step.name:15s} done in {time.time() - start:6.0f} s   ({step.script})", flush=True)
    return json.loads(trace.read_text())


def run(steps: list[Step], jobs: int, logdir: Path) -> dict[str, dict]:
    traces: dict[str, dict] = {}
    first, last = [s for s in steps if not s.final], [s for s in steps if s.final]
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(run_step, s, logdir): s for s in first}
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for f in done:
            if f.exception():
                for g in futures:
                    g.cancel()
                raise f.exception()
        for f, s in futures.items():
            traces[s.name] = f.result()
    for s in last:
        traces[s.name] = run_step(s, logdir)
    return traces


def snapshot_reference(ref: dict, steps: set[str], into: Path) -> dict[str, Path]:
    """Copies of the current CSV outputs, so that a run can compare numbers, not only hashes."""
    snap = {}
    for (kind, p), r in ref.items():
        f = ROOT / p
        if kind == "output" and p.endswith(".csv") and r["steps"] & steps and f.exists() and sha256(f) == r["sha256"]:
            dest = into / p.replace("/", "__")
            dest.write_bytes(f.read_bytes())
            snap[p] = dest
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", help="comma-separated subset of: " + ", ".join(BY_NAME))
    ap.add_argument("-j", "--jobs", type=int, default=1, help="independent steps to run at once (default 1)")
    ap.add_argument("--verify", action="store_true", help="compare the files on disk with the manifest; run nothing")
    ap.add_argument("--update-manifest", action="store_true", help="record this run's outputs as the new reference")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    args = ap.parse_args()

    if args.list:
        for s in STEPS:
            print(f"{s.name:15s} {s.script:25s} ~{s.minutes:g} min  {s.what}")
        return
    names = [n.strip() for n in args.steps.split(",")] if args.steps else [s.name for s in STEPS]
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        ap.error(f"unknown step(s): {', '.join(unknown)}; choose from {', '.join(BY_NAME)}")
    steps = [s for s in STEPS if s.name in names]
    ref = read_manifest()

    if args.verify:
        if not ref:
            sys.exit("No manifest to verify against: run reproduce.py first.")
        sys.exit(0 if compare(ref, {}, set(names), {}) else 1)

    preflight(steps, ref)
    logdir = Path(tempfile.mkdtemp(prefix="reproduce-"))
    snap = snapshot_reference(ref, set(names), logdir)
    est = sum(s.minutes for s in steps)
    print(f"Running {len(steps)} step(s) (about {est:.0f} min run one at a time); logs in {logdir}")
    try:
        traces = run(steps, args.jobs, logdir)
    except RuntimeError as e:
        sys.exit(str(e))
    now = current_rows(traces)

    if not ref or args.update_manifest:
        merged = {}
        for key, r in ref.items():               # keep the reference for steps that did not run
            rest = r["steps"] - set(names)
            if rest:
                merged[key] = {**r, "steps": rest}
        for key, r in now.items():
            if key in merged and merged[key]["sha256"] == r["sha256"]:
                merged[key]["steps"] |= r["steps"]
            else:
                merged[key] = r
        write_manifest(merged)
        n_out = sum(k == "output" for k, _ in now)
        print(f"\nRecorded {n_out} outputs and {len(now) - n_out} inputs in {MANIFEST.relative_to(ROOT)}.")
        if ref:
            compare(ref, now, set(names), snap)  # what changed with respect to the old reference
        return
    sys.exit(0 if compare(ref, now, set(names), snap) else 1)


if __name__ == "__main__":
    main()
