#!/usr/bin/env python3
"""
Build labels-cat.tsv from the official ValueEval'24 labels.tsv, for each English split.

Same rule as Human_values_dataset_to_categories.ipynb (which needs Google Colab): a value is 1 when
it is attained or constrained (score > 0); Presence is 1 when any value is; each higher-order
category is 1 when any of its values is. Output columns: Text-ID, Sentence-ID, the 19 values,
Presence and the 8 higher-order categories, as floats (0.0 / 1.0).

Usage (from the repository root):
    python3 data-prep/make_labels_cat.py            # write data/<split>-english/labels-cat.tsv
    python3 data-prep/make_labels_cat.py --check    # compare with the existing files; write nothing
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

VALUES = [  # Schwartz's circular order
    "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism", "Achievement",
    "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition",
    "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring",
    "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance",
]


def span(first: str, last: str) -> list[str]:
    return VALUES[VALUES.index(first): VALUES.index(last) + 1]


HIGHER = {  # name -> values (contiguous on the circle; Face, Hedonism and Humility sit on borders)
    "Presence": VALUES,
    "Self-Transcendence": span("Humility", "Universalism: tolerance"),
    "Openness to Change": span("Self-direction: thought", "Hedonism"),
    "Self-Enhancement": span("Hedonism", "Face"),
    "Conservation": span("Face", "Humility"),
    "Social Focus": span("Security: societal", "Universalism: tolerance"),
    "Personal Focus": span("Self-direction: thought", "Security: personal"),
    "Growth Anxiety-Free": span("Humility", "Universalism: tolerance") + span("Self-direction: thought", "Achievement"),
    "Self-Protection Anxiety-Avoidance": span("Achievement", "Humility"),
}


def build(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels[["Text-ID", "Sentence-ID"]].copy()
    for v in VALUES:
        out[v] = (labels[[f"{v} attained", f"{v} constrained"]] > 0).any(axis=1).astype(float)
    for name, values in HIGHER.items():
        out[name] = (out[values] > 0).any(axis=1).astype(float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--check", action="store_true", help="compare with the existing labels-cat.tsv; write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite a labels-cat.tsv that differs")
    args = ap.parse_args()
    status = 0
    for split in ["training", "validation", "test"]:
        src, dest = args.data / f"{split}-english" / "labels.tsv", args.data / f"{split}-english" / "labels-cat.tsv"
        if not src.exists():
            print(f"{split:10s} skipped: {src} not found")
            status = 1
            continue
        buf = io.StringIO()
        build(pd.read_csv(src, sep="\t")).to_csv(buf, sep="\t", index=False)
        text = buf.getvalue()
        same = dest.exists() and dest.read_text(encoding="utf-8") == text
        if args.check:
            print(f"{split:10s} {'identical to' if same else 'DIFFERS from'} {dest}")
            status |= not same
        elif same:
            print(f"{split:10s} up to date: {dest}")
        elif dest.exists() and not args.force:
            print(f"{split:10s} NOT written: {dest} exists and differs (use --force to overwrite)")
            status = 1
        else:
            dest.write_text(text, encoding="utf-8")
            print(f"{split:10s} wrote {dest}")
    sys.exit(status)


if __name__ == "__main__":
    main()
