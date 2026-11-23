#!/usr/bin/env python3
"""
Which threshold rule and which configuration produced each number in the submitted
tables (step 1.1 of the revision plan).

For every main-text table that can be rebuilt from cached predictions, recompute the
reported numbers under the candidate rules and print them next to the paper's values:

  Table 1  HO detection by bipolar pair: fixed 0.5 versus the Eq. 8 rule
  Table 3  Presence-gated HO detection: fixed 0.5 versus the Eq. 8 rule
  Table 4  hard HO->values gating: reported threshold versus the validation sweep, and
           the reported configuration versus the one validation would select
  Table 6  "Single" column: the Direct model under the global sweep

Usage:
    python3 approaches/ho-revision/table_provenance.py
Writes results/table-provenance.txt.
"""
import contextlib
import io
import sys

from common import (APPROACHES, ID, PAIRS, RESULTS, SLICES, TABLE4_GATED, eq8_threshold,
                    global_sweep_threshold, GLOBAL_GRID, labels_of, load, macro_f1, label_f1s)

# Values printed in the submitted manuscript
T1 = {  # baseline fixed / tuned, best tuned (variant), poles of the best
    "Growth vs. Self-Protection": ("0.58/0.58", "0.58 (Baseline) 0.54/0.62"),
    "Social Focus vs. Personal Focus": ("0.41/0.43", "0.57 (NER / WorryWords / LIWC15) 0.59/0.54"),
    "Openness vs. Conservation": ("0.42/0.38", "0.42 (LIWC22 + Ling. Feat) 0.34/0.50"),
    "Self-Transcendence vs. Self-Enhancement": ("0.48/0.50", "0.51 (WorryWords / MFD-20) 0.53/0.48"),
}
T3 = {
    "Growth vs. Self-Protection": ("0.570/0.580", "0.570/0.560"),
    "Social Focus vs. Personal Focus": ("0.560/0.560", "0.550/0.550"),
    "Openness vs. Conservation": ("0.412/0.384", "0.416/0.414"),
    "Self-Transcendence vs. Self-Enhancement": ("0.490/0.490", "0.490/0.480"),
}
T4 = {"Growth": (0.271, 0.30), "Self-Protection": (0.306, 0.15), "Social Focus": (0.326, 0.25),
      "Personal Focus": (0.287, 0.70), "Openness": (0.235, 0.50), "Conservation": (0.297, 0.50),
      "Self-Transcendence": (0.307, 0.25), "Self-Enhancement": (0.320, 0.55)}
T6_SINGLE = {"Growth": 0.286, "Self-Protection": 0.321, "Social Focus": 0.330,
             "Personal Focus": 0.305, "Openness": 0.240, "Conservation": 0.301,
             "Self-Transcendence": 0.302, "Self-Enhancement": 0.338}


def table1() -> None:
    print("== Table 1: HO detection (test macro-F1 over the two poles)")
    for pair, (folder, labs) in PAIRS.items():
        out = APPROACHES / folder / "output"
        print(f"\n{pair}  [paper: baseline {T1[pair][0]}; best {T1[pair][1]}]")
        for vf in sorted(out.glob("*-val.tsv")):
            name = vf.name[:-8]
            yv, sv = load(vf, labs, "val")
            yt, st = load(out / f"{name}-test.tsv", labs, "test")
            th = [eq8_threshold(yv[:, i], sv[:, i])[0] for i in range(2)]
            fx = label_f1s(yt, st, 0.5); tu = label_f1s(yt, st, th)
            print(f"  {name:22s} fixed {fx.mean():.3f} ({fx[0]:.2f}/{fx[1]:.2f})"
                  f"   Eq. 8 {tu.mean():.3f} ({tu[0]:.2f}/{tu[1]:.2f}) at {[round(t, 2) for t in th]}")


def table3() -> None:
    print("\n== Table 3: Presence-gated HO detection, gate LIWC22+LingFeat, HO model Baseline")
    for pair, (folder, labs) in PAIRS.items():
        cells = []
        for tg in ["0.5", "0.1"]:
            base = APPROACHES / f"p_{folder}" / "output" / f"1_Lex-LIWC-22_LingFeat_{tg}_Baseline"
            yv, sv = load(base.with_name(base.name + "-val.tsv"), labs, "val")
            yt, st = load(base.with_name(base.name + "-test.tsv"), labs, "test")
            th = [eq8_threshold(yv[:, i], sv[:, i])[0] for i in range(2)]
            cells.append(f"tau_g={tg}: {macro_f1(yt, st, 0.5):.3f}/{macro_f1(yt, st, th):.3f}")
        print(f"  {pair:40s} paper {T3[pair][0]} | {T3[pair][1]}   ours {' | '.join(cells)}")


def slice_files(folder: str):
    out = APPROACHES / f"{folder}_moral-values" / "output"
    for vf in sorted(out.glob("1_*-val.tsv")):
        yield vf.name[:-8], vf, out / f"{vf.name[:-8]}-test.tsv"


def table4_and_6() -> None:
    print("\n== Table 4: hard HO->values gating (test macro-F1 over the slice's values)")
    for sl, (folder, _) in SLICES.items():
        rows = []
        for name, vf, tf in slice_files(folder):
            labs = labels_of(tf)
            yv, sv = load(vf, labs, "val"); yt, st = load(tf, labs, "test")
            t = global_sweep_threshold(yv, sv)
            best_test = max((macro_f1(yt, st, g), float(g)) for g in GLOBAL_GRID)
            rows.append((name, macro_f1(yv, sv, t), t, macro_f1(yt, st, t), best_test, yt, st))
        rep = next(r for r in rows if r[0] == TABLE4_GATED[sl])
        val_pick = max(rows, key=lambda r: r[1])
        f_paper, t_paper = T4[sl]
        print(f"  {sl:18s} paper {f_paper:.3f}@{t_paper:.2f} | reported config: at paper thr "
              f"{macro_f1(rep[5], rep[6], t_paper):.3f}, val-swept thr {rep[2]} -> {rep[3]:.3f}"
              f" | validation would pick {val_pick[0]} (thr {val_pick[2]}) -> test {val_pick[3]:.3f}"
              f" | n configs {len(rows)}")

    print("\n== Table 6, 'Single' column: Direct Baseline restricted to the slice's values")
    direct = APPROACHES / "moral-values" / "output"
    for sl, (folder, _) in SLICES.items():
        labs = labels_of(APPROACHES / f"{folder}_moral-values" / "output" / f"{TABLE4_GATED[sl]}-test.tsv")
        yv, sv = load(direct / "Baseline-val.tsv", labs, "val")
        yt, st = load(direct / "Baseline-test.tsv", labs, "test")
        t = global_sweep_threshold(yv, sv)
        print(f"  {sl:18s} paper {T6_SINGLE[sl]:.3f} | Baseline at 0.3 {macro_f1(yt, st, 0.3):.3f}"
              f" | val-swept thr {t} -> {macro_f1(yt, st, t):.3f} | at 0.5 {macro_f1(yt, st, 0.5):.3f}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        table1(); table3(); table4_and_6()
    text = buf.getvalue()
    (RESULTS / "table-provenance.txt").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
