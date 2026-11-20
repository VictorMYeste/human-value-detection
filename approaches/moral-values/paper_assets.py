#!/usr/bin/env python3
"""
Figures and tables for the TCSS manuscript (step 1.8 of paper/tcss-plan.md).

Reads the outputs of the analysis scripts, which must be run first:
    threshold_schemes.py   -> results/classification-table.csv
    measurement_analysis.py-> results/measurement-summary.csv
    llm_measurement.py     -> results/llm-measurement.csv
    worked_example.py      -> results/worked-example-values.csv, worked-example-summary.csv
Figure 1 and Table 1 are computed here from the gold labels and the cached predictions,
with the bootstrap shared by every script.

Writes to results/paper/:
    figures/fig1-prevalence.pdf   per-value prevalence bias of three decision rules (full width)
    figures/fig4-worked.pdf       worked example: manifesto-news differences (full width; Fig. 2 of the article)
    tables/tab1-data.tex ... tab5-recommendations.tex   LaTeX tables (IEEEtran + booktabs)
    tables/tabS1-classification-full.tex                Table II with every system, for the supplement

Figures are vector PDFs with embedded TrueType fonts (Times), 8 pt text, IEEE widths
(column 3.47 in, page 7.11 in, to fit IEEEtran). Colours: the first three slots of the validated categorical
palette, always paired with marker shape or fill, so the figures survive grayscale print.

Usage:
    python3 paper_assets.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analyze_provenance import VALUES, load_gold
from measurement_analysis import RES, RNG_SEED, SEEDS, RowFits, document_design, system_probs, wsum

OUT = RES / "paper"
FIG, TAB = OUT / "figures", OUT / "tables"
MAIN = "Feature ensemble"
OTHERS = ["DeBERTa", "RoBERTa", "Seed ensemble"]

# ---- style: validated palette (blue, orange, aqua; all-pairs CVD-safe), ink and recessive greys
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#9a9893", "#e6e6e3", "#d9d8d4"
COL_W, PAGE_W = 3.47, 7.11  # IEEEtran \columnwidth 251 pt, \textwidth 514 pt
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 0, "axes.edgecolor": INK2, "xtick.color": INK2,
    "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False, "legend.handlelength": 1.6,
})


def new_figure(width: float, height: float, nrows: int = 1, ncols: int = 1, **kw):
    """Figure laid out at its exact print size (constrained layout, no tight-bbox resizing)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.04)
    return fig, axes

LABEL = {  # math in $...$ renders in matplotlib mathtext and in LaTeX alike
    "CC@0.5": "CC, $t = 0.5$", "CC@t*": "CC, macro-F1 $t^*$", "CC@label-t*": "CC, per-label F1 $t_j$",
    "CC@prev": "CC, prevalence-matched", "PCC": "PCC, raw", "PCC-Platt": "PCC, Platt",
    "ACC@t*": "ACC, $t^*$", "ACC@prev": "ACC, prevalence-matched", "PACC": "PACC", "EM": "EM",
    "ACC-G@prev": "ACC-G, prevalence-matched", "SC@prev": "SC, prevalence-matched",
    "Gold-only": "Annotated share", "PPI@prev": "PPI, prevalence-matched", "PPI-Platt": "PPI, Platt",
    "Platt-G": "Platt per group",
}
RECOMMENDED = {"CC@prev", "PPI-Platt", "Platt-G"}   # the methods Table IV recommends (step 5.2 E)


def save(fig, name: str) -> None:
    """Vector PDF without a creation date, so that reruns are byte-identical."""
    fig.savefig(FIG / name, metadata={"CreationDate": None})


def style_axis(ax, grid_axis: str = "x") -> None:
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def ci_bar(ax, x, lo, hi, y, color, marker, filled=True, ms=3.6, lw=0.8, zorder=3, label=None):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)
    ax.plot([x], [y], marker=marker, ms=ms, color=color, mfc=color if filled else "white", mew=0.8,
            ls="none", zorder=zorder + 1, label=label)


# ============================================================================ figures
def figure1_prevalence() -> None:
    """Per-value estimated / reference prevalence for three decision rules (main system, mean of runs)."""
    gt, gv = load_gold("test"), load_gold("validation")
    yt = (gt[VALUES].to_numpy() >= 0.5).astype(int)
    yv = (gv[VALUES].to_numpy() >= 0.5).astype(int)
    d = document_design(gt)
    S, W, nsent = d["S"], d["W"], d["nsent"]
    denom = (W @ nsent)[:, None]
    gold = (W @ (S @ yt)) / denom
    rules = [("CC@0.5", AQUA, "v"), ("CC@t*", ORANGE, "^"), ("CC@prev", BLUE, "o")]
    ratio = {r: 0.0 for r, _, _ in rules}
    vdesign = document_design(gv, seed=RNG_SEED + 1)
    every = np.ones(S.shape[0], bool)
    for seed in SEEDS:   # rules refitted on every validation row, as in measurement_analysis.py
        pv, pt = system_probs(MAIN, seed, "val"), system_probs(MAIN, seed, "test")
        rf = RowFits(pv, yv, vdesign)
        for r, _, _ in rules:
            c = S @ (pt >= 0.5).astype(float) if r == "CC@0.5" else rf.counts(r, pt, S)
            ratio[r] = ratio[r] + (wsum(W, c, every) / denom) / gold / len(SEEDS)

    order = np.argsort(-gold[0])
    y = np.arange(len(VALUES))[::-1]
    fig, (a, b) = new_figure(PAGE_W, 3.2, 1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 3.3]})
    a.barh(y, gold[0, order] * 100, height=0.62, color=BAND, edgecolor="none")
    for yi, j in zip(y, order):
        a.text(gold[0, j] * 100 + 0.25, yi, f"{gold[0, j] * 100:.1f}", va="center", fontsize=6.5, color=INK2)
    a.set_xlim(0, gold[0].max() * 100 * 1.32)
    a.set_xlabel("% of sentences")
    a.set_yticks(y, [VALUES[j] for j in order])
    a.set_title("(a) Reference prevalence", loc="left")
    style_axis(a)

    # A ratio of 0 (the rule never predicts the value; Humility at t = 0.5) has no place on a log
    # axis. It is drawn at an explicit "0" left of an axis break, and intervals that reach 0 run to it.
    ZERO, BREAK = 0.032, 0.0395
    b.axvline(1, color=INK2, lw=0.7, zorder=1)
    b.axvline(BREAK, color=INK2, lw=0.5, ls=(0, (1, 2)), zorder=1)
    for k, (r, color, marker) in enumerate(rules):
        rr = ratio[r][:, order]
        lo, hi = np.percentile(rr[1:], [2.5, 97.5], axis=0)
        lo, hi = np.where(lo > 0, lo, ZERO), np.where(hi > 0, hi, ZERO)
        for i, yi in enumerate(y):
            yk = yi + (k - 1) * 0.24
            x = rr[0, i] if rr[0, i] > 0 else ZERO
            ci_bar(b, x, lo[i], hi[i], yk, color, marker, label=LABEL[r] if i == 0 else None)
    b.set_xscale("log")
    b.minorticks_off()
    ticks = [ZERO, 0.05, 0.1, 0.2, 0.5, 1, 2]
    b.set_xticks(ticks, ["0"] + [f"{t:g}" for t in ticks[1:]])
    b.set_xlim(0.027, 2.6)
    b.text(BREAK, 0, "//", transform=b.get_xaxis_transform(), ha="center", va="center", fontsize=7,
           color=INK2, clip_on=False, bbox={"fc": "white", "ec": "none", "pad": 0.1}, zorder=5)
    b.set_xlabel("Estimated / reference prevalence (log scale; 1 = count is right)")
    b.set_title("(b) Over- and undercounting by decision rule (95% intervals)", loc="left")
    b.text(0.06, y[0] + 0.55, "undercount", fontsize=6.5, color=INK2, ha="left", va="bottom")
    b.text(2.3, y[0] + 0.55, "overcount", fontsize=6.5, color=INK2, ha="right", va="bottom")
    handles, labels = b.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="outside upper center", ncol=3, handletextpad=0.3, columnspacing=1.6)
    style_axis(b)
    b.set_ylim(-0.7, len(VALUES) - 0.1)
    save(fig, "fig1-prevalence.pdf")
    plt.close(fig)


def figure4_worked(values: pd.DataFrame) -> None:
    """Manifesto-news differences per value: gold with CI next to four estimators (crude version).

    Estimates are averaged over the main system's runs, and the fill shows whether that averaged
    estimate is significant. The counts of found / missed / spurious are in Table 4."""
    v = values[values["version"] == "crude"]
    gold = v[v["estimator"] == "gold"].set_index("value")
    order = gold["diff_pp"].sort_values(ascending=False).index.tolist()
    y = np.arange(len(order))[::-1].astype(float)
    panels = [("CC@t*", "(a) CC, macro-F1 $t^*$"), ("CC@prev", "(b) CC, prevalence-matched"),
              ("PCC-Platt", "(c) PCC, Platt"), ("LLM", "(d) LLM labels")]
    fig, axes = new_figure(PAGE_W, 3.25, 1, 4, sharey=True, sharex=True)
    for ax, (est, title) in zip(axes, panels):
        e = v[v["estimator"] == est].set_index("value")
        ax.axvline(0, color=INK2, lw=0.6, zorder=1)
        for i, name in enumerate(order):
            g = gold.loc[name]
            ax.plot([g["lo_pp"], g["hi_pp"]], [y[i], y[i]], color=BAND, lw=3.2, solid_capstyle="butt", zorder=1)
            ax.plot([g["diff_pp"]], [y[i]], marker="|", ms=6, mew=1.1, color=INK, ls="none", zorder=2)
            x, sig = e.loc[name, "diff_pp"], bool(e.loc[name, "significant"])
            ax.plot([x], [y[i]], marker="o", ms=3.6, color=BLUE, mfc=BLUE if sig else "white", mew=0.8,
                    ls="none", zorder=3)
            if g["significant"] and sig and np.sign(x) != np.sign(g["diff_pp"]):
                ax.annotate("reversed", xy=(x, y[i]), xytext=(x + 0.3, y[i] - 1.8), fontsize=6.5, color=INK,
                            ha="left", arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.5})
        ax.set_title(title, loc="left")
        style_axis(ax)
    axes[0].set_yticks(y, order)
    axes[0].set_ylim(-0.7, len(order) - 0.3)
    axes[0].set_xlim(-4.5, 13.5)
    fig.supxlabel("Manifesto minus news, percentage points of sentences", fontsize=8)
    fig.legend(handles=[
        Line2D([], [], color=BAND, lw=3.2, label="reference 95% CI"),
        Line2D([], [], color=INK, marker="|", ms=6, mew=1.1, lw=0, label="reference"),
        Line2D([], [], color=BLUE, marker="o", ms=3.6, lw=0, label="estimate, significant"),
        Line2D([], [], color=BLUE, marker="o", ms=3.6, mfc="white", lw=0, label="estimate, not significant")],
        loc="outside upper center", ncol=4, handletextpad=0.4, columnspacing=1.6)
    save(fig, "fig4-worked.pdf")
    plt.close(fig)


# ============================================================================ tables
def tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write_table(name: str, body: str) -> None:
    (TAB / name).write_text(body.strip() + "\n", encoding="utf-8")


def table1_data() -> None:
    rows = {}
    for split, label in [("training", "Training"), ("validation", "Validation"), ("test", "Test")]:
        g = load_gold(split)
        y = (g[VALUES].to_numpy() >= 0.5).astype(int)
        docs = g["Text-ID"].drop_duplicates()
        man = docs.str.contains("_M_").sum()
        man_sent = g["Text-ID"].str.contains("_M_").sum()
        rows[label] = [f"{len(docs) - man:,} / {man:,}", f"{len(g):,}", f"{man_sent:,}",
                       f"{y.any(1).mean() * 100:.1f}", f"{y.sum(1).mean():.2f}"]
    names = ["Documents (news / manifestos)", "Sentences", r"\quad in manifestos", r"Sentences with $\geq$1 value (\%)",
             "Value mentions per sentence"]
    lines = [f"{n} & " + " & ".join(rows[s][k] for s in rows) + r" \\" for k, n in enumerate(names)]
    write_table("tab1-data.tex", r"""
\begin{table}[t]
\caption{Data: the English release of ValueEval'24 (ValuesML corpus)}
\label{tab:data}
\centering
\footnotesize
\begin{tabular}{@{}lrrr@{}}
\toprule
 & Training & Validation & Test \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\multicolumn{4}{@{}p{0.96\columnwidth}@{}}{\scriptsize Source languages: Bulgarian, German, Greek, English, French, Hebrew, Italian, Dutch and Turkish. Documents not written in English were machine-translated by the task organizers (DeepL, and Google Translate for Hebrew).}
\end{tabular}
\end{table}
""")


CLASSIFICATION_NOTE = (r"Thresholds are fitted on the validation split: F1-tuned ones on a grid of 0.01--0.99, "
                       r"prevalence-matched ones as quantiles of the validation probabilities. The feature ensemble "
                       r"(soft vote of DeBERTa, DeBERTa + LIWC-22 and DeBERTa + Prev-2) is the main system. The seed "
                       r"ensemble averages three training runs of DeBERTa and is reported as one system. LLM outputs "
                       r"are label sets, so no threshold can be tuned, and each LLM system was run once.")


def table2_classification(llm: pd.DataFrame) -> None:
    """Table II of the paper (main systems) and its full version for the supplement (all systems)."""
    classification_table(llm, ["DeBERTa", "RoBERTa", "Seed ensemble", "Feature ensemble"], "tab2-classification.tex",
                         "tab:classification",
                         r"Thresholds are fitted on the validation split: F1-tuned ones on a grid of 0.01--0.99, "
                         r"prevalence-matched ones as quantiles of the validation probabilities. Bold: main system. "
                         r"The seed ensemble averages three training runs of DeBERTa and is reported as one system. "
                         r"LLM labels are in Table~\ref{S-tab:llm}, and models with auxiliary inputs and presence "
                         r"gates in Table~\ref{S-tab:classification-full} of the supplement.", include_llm=False)
    classification_table(llm, ["DeBERTa", "DeBERTa + LIWC-22", "DeBERTa + MJD", "DeBERTa + Prev-2", "DeBERTa + BERTopic",
                               "DeBERTa + soft gate", "DeBERTa + hard gate 0.10", "DeBERTa + hard gate tuned", "RoBERTa",
                               "Seed ensemble", "Feature ensemble"], "tabS1-classification-full.tex",
                         "tab:classification-full", CLASSIFICATION_NOTE + " Gated systems use the global threshold only.",
                         caption_extra=", all systems")


def classification_table(llm: pd.DataFrame, systems: list[str], filename: str, label: str, note: str,
                         caption_extra: str = "", include_llm: bool = True) -> None:
    c = pd.read_csv(RES / "classification-table.csv")
    agg = c.groupby(["system", "rule"]).agg(f1=("macro_f1", "mean"), sd=("macro_f1", "std"), n=("macro_f1", "size"),
                                            ratio=("pred_gold_ratio", "mean"))
    rules = ["t=0.5", "global t*", "label t*", "label prev"]

    def cell(system, rule):
        if (system, rule) not in agg.index:
            return "--"
        a = agg.loc[(system, rule)]
        return f"{a.f1:.3f}" + (f" $\\pm$ {a.sd:.3f}" if a.n > 1 else "")

    def ratio(system, rule):
        return f"{agg.loc[(system, rule)].ratio:.2f}" if (system, rule) in agg.index else "--"

    lines = []
    for s in systems:
        name = r"\textbf{Feature ensemble}" if s == MAIN else tex_escape(s)
        lines.append(f"{name} & " + " & ".join(cell(s, r) for r in rules) + f" & {ratio(s, 'global t*')} & "
                     f"{ratio(s, 'label prev')}" + r" \\")
        if s in {"DeBERTa + BERTopic", "DeBERTa + hard gate tuned"} and s != systems[-1]:
            lines.append(r"\addlinespace")
    llm_names = {} if not include_llm else {"Gemma-2-9B few-shot + gate": "Gemma-2-9B, few-shot + gate", "Gemma-2-9B few-shot": "Gemma-2-9B, few-shot",
                 "Gemma-2-9B QLoRA": "Gemma-2-9B, QLoRA", "LLM ensemble (hard vote)": "LLM union (few-shot + QLoRA)"}
    if llm_names:
        lines.append(r"\midrule")
    for s, row_name in llm_names.items():
        r = llm.loc[(s, "CC")]
        lines.append(f"{row_name} & \\multicolumn{{4}}{{c}}{{{r.macro_f1:.3f} (labels as returned)}} & "
                     f"\\multicolumn{{2}}{{c}}{{{r.ratio:.2f}}}" + r" \\")
    write_table(filename, r"""
\begin{table*}[t]
\caption{Classification: test macro-F1 under each decision rule (mean $\pm$ sd over three training runs) and predicted/reference value mentions""" + caption_extra + r"""}
\label{""" + label + r"""}
\centering
\footnotesize
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{4}{c}{Macro-F1 by decision rule} & \multicolumn{2}{c}{Predicted / reference mentions} \\
\cmidrule(lr){2-5}\cmidrule(l){6-7}
System & $t = 0.5$ & Macro-F1 $t^*$ & Per-label F1 $t_j$ & Prevalence-matched $t^{\mathrm{prev}}$ & at $t^*$ & prev.-matched \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\multicolumn{7}{@{}p{0.98\textwidth}@{}}{\scriptsize """ + note + r"""}
\end{tabular}
\end{table*}
""")


def table3_measurement(summary: pd.DataFrame) -> None:
    """Main system only (step 5.2 E): accuracy (MAE) and coverage for totals and both families of contrasts,
    with the estimators that use each group's annotated documents. LLM rows are in Table S19."""
    def excl1(v, lo, hi):
        return f"{v:.2f}" + (r"$^\dagger$" if lo > 1 or hi < 1 else "")

    def slope_cell(v, lo, hi):
        return "--" if pd.isna(v) else f"{v:.2f} [{lo:.2f}, {hi:.2f}]"

    groups = [("1. Classify and count", ["CC@0.5", "CC@t*", "CC@label-t*", "CC@prev"]),
              ("2. Probability averages", ["PCC", "PCC-Platt"]),
              ("3. Adjusted counts", ["ACC@t*", "ACC@prev", "PACC", "EM", "ACC-G@prev"]),
              ("4. Slope correction", ["SC@prev"]),
              ("5. Per-group estimates, from each group's annotated documents", ["Gold-only", "PPI@prev", "PPI-Platt",
                                                                                  "Platt-G"])]
    lines = [r"\emph{Target} & 1 & 0 & $\approx$95 & 1 & 0 & $\approx$95 & 1 & 0 & $\approx$95 & 100 \\",
             r"\midrule"]
    for title, rules in groups:
        lines.append(f"\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{{title}}}}}" + r" \\")
        for r in rules:
            m = summary.loc[(MAIN, r)]
            name = f"\\textbf{{{LABEL[r]}}}" if r in RECOMMENDED else LABEL[r]
            lines.append(" & ".join([
                r"\quad " + name, excl1(m.ratio, m.ratio_lo, m.ratio_hi), f"{m.prev_mae:.2f}",
                f"{m.prev_cover:.0f}",
                slope_cell(m.genre_slope, m.genre_slope_lo, m.genre_slope_hi), f"{m.genre_mae:.2f}",
                f"{m.genre_cover:.0f}",
                slope_cell(m.lpair_slope, m.lpair_slope_lo, m.lpair_slope_hi), f"{m.lpair_mae:.2f}",
                f"{m.lpair_cover:.0f}", f"{m.lpair_detected:.0f}"]) + r" \\")
    write_table("tab3-measurement.tex", r"""
\begin{table*}[t]
\caption{Measurement: corpus prevalence of the 19 values, manifesto--news contrasts (19) and source-language pair contrasts (684), main system}
\label{tab:measurement}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{Corpus prevalence} & \multicolumn{3}{c}{Manifesto vs.\ news} & \multicolumn{4}{c}{Language pairs} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-11}
Method & Pred./ref. & MAE & Cover & Slope [95\% CI] & MAE & Cover & Slope [95\% CI] & MAE & Cover & Det. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\multicolumn{11}{@{}p{0.98\textwidth}@{}}{\scriptsize Main system: feature ensemble, metrics computed on the mean of three training runs. Bold: methods recommended in Table~\ref{tab:recommendations}. Target: the value of an accurate method. Pred./ref.: estimated over reference value mentions, $^\dagger$ 95\% CI excludes 1. MAE: mean absolute error in percentage points, over the 19 values or the contrasts. Cover: share of values or contrasts (\%) whose error is within sampling noise (paired document bootstrap, $p \geq 0.05$), about 95 for an unbiased method and higher for a noisy one, so it is read with the MAE. Slope: attenuation slope (Section~\ref{sec:attenuation}, 1 = differences keep their size), not shown for per-group estimates (Section~\ref{sec:metrics}). Det.: share (\%) of the 126 language contrasts that are significant in the reference that the method detects (single test; after correction for multiple tests in Table~\ref{S-tab:detection} of the supplement). Macro-F1: Table~\ref{tab:classification}. LLM labels: Table~\ref{S-tab:llm} of the supplement.}
\end{tabular*}
\end{table*}
""")


def table4_worked(worked: pd.DataFrame, values: pd.DataFrame) -> None:
    w = worked[worked["version"] == "crude"].set_index("estimator")
    ref = values[(values["version"] == "crude") & (values["estimator"] == "gold")]
    n_sig = int(ref["significant"].sum()); n_other = len(ref) - n_sig
    names = {"CC@0.5": "CC, $t = 0.5$", "CC@t*": "CC, macro-F1 $t^*$", "CC@prev": r"\textbf{CC, prev.-matched}",
             "PCC-Platt": "PCC, Platt", "ACC@prev": "ACC, prev.-matched", "SC@prev": r"\textbf{SC, prev.-matched}",
             "LLM": "Gemma few-shot + gate"}
    lines = [f"{names[e]} & {w.loc[e, 'found']:.0f} & {w.loc[e, 'missed']:.0f} & {w.loc[e, 'reversed']:.0f} & "
             f"{w.loc[e, 'spurious']:.0f} & {w.loc[e, 'slope']:.2f} & {w.loc[e, 'mae']:.2f}" + r" \\" for e in names]
    write_table("tab4-worked.tex", r"""
\begin{table}[t]
\caption{Worked example: how each method fares on the """ + str(n_sig) + r""" significant reference differences between manifestos and news}
\label{tab:worked}
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Method & Found & Missed & Rev. & Spur. & Slope & MAE \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\multicolumn{7}{@{}p{0.96\columnwidth}@{}}{\scriptsize Found: significant with the sign of the reference. Missed: not significant. Rev.: significant with the opposite sign. Spur.: significant where the reference is not (of """ + str(n_other) + r"""). Supervised methods: main system, on the mean of three training runs; LLM: Gemma-2-9B few-shot with the presence gate, single run. Slope: attenuation slope, and MAE: mean absolute error in percentage points, over the 19 differences.}
\end{tabular}
\end{table}
""")


def table_families() -> None:
    """The five families of estimators, in plain words (Section V-C; step 5.2 E5, readability check)."""
    rows = [
        (r"1. Classify and count (CC)", r"Label each sentence with one of the four rules of Section~\ref{sec:rules} and count the labels.", "No"),
        (r"2. Probability averages (PCC)", r"Average the probabilities, raw or after Platt scaling on validation data.", "No"),
        (r"3. Adjusted counts (ACC, PACC, EM, ACC-G)", r"Undo the classifier's error rates, measured on validation data, in each group for ACC-G.", "ACC-G"),
        (r"4. Slope correction (SC)", r"Divide each contrast by the shrinkage measured on validation contrasts of the same kind.", "No"),
        (r"5. Per-group estimates (annotated share, PPI, Platt per group)", r"Use each group's annotated documents, alone or to correct the classifier's estimate.", "Yes"),
    ]
    lines = [r"\raggedright " + f"{a} & {b} & {c}" + r" \\" for a, b, c in rows]
    write_table("tab-families.tex", r"""
\begin{table}[t]
\caption{Five families of methods for estimating prevalence from a classifier}
\label{tab:families}
\centering
\footnotesize
\begin{tabular}{@{}p{0.31\columnwidth}p{0.43\columnwidth}c@{}}
\toprule
Family (labels in the tables) & How it estimates & Each group \\
\midrule
""" + "\n\\addlinespace\n".join(lines) + r"""
\bottomrule
\multicolumn{3}{@{}p{0.96\columnwidth}@{}}{\scriptsize Each group: whether the method needs annotated documents from every group compared.}
\end{tabular}
\end{table}
""")


def table5_recommendations() -> None:
    # step 5.2 E: from the results of steps B-D (annotated-sample estimators, subsampling, LLM checks)
    rows = [
        ("Any analysis", r"Report how the thresholds were chosen, and on how many annotated documents from which groups."),
        ("Corpus totals", r"Set each value's threshold so that it is labeled as often as in the annotated sample, or average calibrated probabilities. With 50 annotated documents, totals were within about 5\% here. Both assume that the corpus resembles the sample, and neither makes each group's total right. Do not count with thresholds tuned for macro-F1."),
        ("Group differences, about 20 or more annotated documents per group", r"Add the classifier's mean error on each group's annotated documents to its count (prediction-powered estimate), or calibrate it on each group's documents."),
        ("Group differences, fewer annotated documents", r"Count with prevalence-matched thresholds, and report that differences may be too small (here, those between languages by about 40\%; genre differences kept their size only because two errors canceled). Dividing them by the shrinkage measured on validation data (slope correction) restores their average size but makes single differences less accurate. Take the slope from comparisons of the same kind."),
        ("Group differences, to avoid", r"Calibrated averages, which shrink differences. Adjusted counts, which inflate them when the classifier's error rates differ between groups, and are too noisy with error rates measured in each group."),
        ("LLM labels", r"Check them against annotated documents from each group before counting. Here no correction brought their differences as close to the expert labels as the classifier's counts."),
    ]
    lines = [r"\raggedright " + f"{a} & {b}" + r" \\" for a, b in rows]   # ragged goal labels, local to the cell
    write_table("tab5-recommendations.tex", r"""
\begin{table}[t]
\caption{Recommendations for measuring values with a classifier, from the comparisons tested here: two genres and nine source languages of one translated corpus}
\label{tab:recommendations}
\centering
\footnotesize
\begin{tabular}{@{}p{0.27\columnwidth}p{0.67\columnwidth}@{}}
\toprule
Goal & Recommendation \\
\midrule
""" + "\n\\addlinespace\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RES / "measurement-summary.csv").set_index(["system", "rule"])
    llm = pd.read_csv(RES / "llm-measurement.csv").set_index(["system", "estimator"])
    worked_values = pd.read_csv(RES / "worked-example-values.csv")
    worked_summary = pd.read_csv(RES / "worked-example-summary.csv")

    figure1_prevalence()
    figure4_worked(worked_values)
    table1_data()
    table2_classification(llm)
    table3_measurement(summary)
    table4_worked(worked_summary, worked_values)
    table_families()
    table5_recommendations()
    print(f"Wrote {len(list(FIG.glob('*.pdf')))} figures to {FIG} and {len(list(TAB.glob('*.tex')))} tables to {TAB}")


if __name__ == "__main__":
    main()
