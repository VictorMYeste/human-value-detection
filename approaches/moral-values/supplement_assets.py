#!/usr/bin/env python3
"""
Tables and figures for the supplement of the TCSS manuscript (step 4.7 of paper/tcss-plan.md).

Reads the outputs of the analysis scripts, which must be run first (reproduce.py runs this
script last), together with the gold labels and the cached predictions:
    threshold_schemes.py     -> results/classification-comparisons.csv
    measurement_analysis.py  -> results/measurement-summary.csv, measurement-per-value.csv,
                                measurement-language-contrasts.csv
    shift_tests.py           -> results/shift-tests.csv
    llm_measurement.py       -> results/llm-measurement.csv
    worked_example.py        -> results/worked-example-values.csv
    subsample_validation.py  -> results/subsample-validation.csv
    shift_tests.py           -> results/shift-lolo-languages.csv
    measurement_analysis.py  -> results/measurement-error-rates.csv
    ../llm_moral-values/output/<prompt>/<model>-val-results.txt
                                validation macro-F1 logged by each LLM run. The log is read
                                rather than the prediction file because one prediction file
                                (Llama-3.1-8B, direct prompt) was saved incomplete; its log
                                holds the score of the full run.

Writes to results/paper/ (the paper's own tables and figures come from paper_assets.py):
    tables/supp-values.tex            the 19 values: paraphrased definitions, reference prevalence
    tables/supp-languages.tex         source languages: sentences, value share, agreement, error
    tables/supp-prompts.tex           LLM validation macro-F1 by prompt and number of examples
    tables/supp-comparisons.tex       paired macro-F1 comparisons behind Section VI-A
    tables/supp-prevalence.tex        per-value prevalence under each method (main system)
    tables/supp-calibration.tex       calibration of the probabilities, raw and after Platt scaling
    tables/supp-systems.tex           totals and slopes for the other three supervised systems
    tables/supp-worked-values.tex     the 19 manifesto-news differences under each method
    tables/supp-shift.tex             shift tests for every supervised system
    tables/supp-llm.tex               LLM labels with each correction
    tables/supp-subsample.tex         accuracy by size of the annotated sample (subsample_validation.py)
    tables/supp-lolo.tex              each language left out of the fit (shift_tests.py)
    tables/supp-composition.tex       language contrasts on value composition
    tables/supp-error-rates.tex       tpr and fpr per value by genre, validation and test
    tables/supp-group-totals.tex      totals of single groups, also with the group left out of the fit
    tables/supp-detection.tex         significant reference differences found by each method
    figures/figS1-reliability.pdf     reliability diagram of the main system
    figures/figS2-language-contrasts.pdf   the 684 language contrasts, estimated against reference

Usage:
    python3 supplement_assets.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from analyze_provenance import VALUES, load_gold
from gold_reliability import ALPHA_ALL, ALPHA_REFINED
from measurement_analysis import SEEDS, apply_rules, ece, fit_rules, macro_f1, system_probs, system_seeds
from paper_assets import (BLUE, COL_W, FIG, GRID, INK2, LABEL, MAIN, MUTED, ORANGE, RES, TAB, new_figure, save,
                          style_axis, write_table)

LLM_OUT = RES.parent / ".." / "llm_moral-values" / "output"
SUPERVISED = ["Feature ensemble", "DeBERTa", "RoBERTa", "Seed ensemble"]
OTHERS = ["DeBERTa", "RoBERTa", "Seed ensemble"]

# Paraphrased from Table 1 of Schwartz et al. (2012); short working definitions, not quotations.
DEFINITIONS = {
    "Self-direction: thought": "Freedom to develop one's own ideas and abilities",
    "Self-direction: action": "Freedom to decide one's own actions",
    "Stimulation": "Excitement, novelty and change",
    "Hedonism": "Pleasure and sensory enjoyment",
    "Achievement": "Success by the standards of society",
    "Power: dominance": "Power through control over people",
    "Power: resources": "Power through control of material and social resources",
    "Face": "Security and power from protecting one's public image and avoiding humiliation",
    "Security: personal": "Safety in one's immediate surroundings",
    "Security: societal": "Safety and stability of the wider society",
    "Tradition": "Keeping cultural, family or religious traditions",
    "Conformity: rules": "Compliance with rules, laws and formal obligations",
    "Conformity: interpersonal": "Avoiding upsetting or harming other people",
    "Humility": "Recognizing one's small place in the larger scheme of things",
    "Benevolence: caring": "Devotion to the welfare of the members of one's group",
    "Benevolence: dependability": "Being a reliable and trustworthy member of one's group",
    "Universalism: concern": "Commitment to equality, justice and protection for all people",
    "Universalism: nature": "Preservation of the natural environment",
    "Universalism: tolerance": "Acceptance and understanding of people who differ from oneself",
}
LANGUAGES = {"BG": "Bulgarian", "DE": "German", "EL": "Greek", "EN": "English", "FR": "French",
             "HE": "Hebrew", "IT": "Italian", "NL": "Dutch", "TR": "Turkish"}
LLMS = {"google-gemma-2-9b-it": "Gemma-2-9B", "meta-llama-Llama-3-1-8B-Instruct": "Llama-3.1-8B",
        "mistralai-Ministral-8B-Instruct-2410": "Ministral-8B", "Qwen-Qwen2-5-7B-Instruct": "Qwen-2.5-7B"}
ZERO_SHOT = ["direct", "cot_hidden", "definition", "qa"]
SHOTS = [1, 2, 4, 8, 12, 16, 20]
LLM_SYSTEMS = {"Gemma-2-9B few-shot + gate": "Gemma few-shot + gate", "Gemma-2-9B few-shot": "Gemma few-shot",
               "Gemma-2-9B QLoRA": "Gemma QLoRA", "LLM ensemble (hard vote)": "LLM union"}
ALL_RULES = ["CC@0.5", "CC@t*", "CC@label-t*", "CC@prev", "PCC", "PCC-Platt", "ACC@t*", "ACC@prev", "PACC", "EM",
             "ACC-G@prev", "SC@prev", "Gold-only", "PPI@prev", "PPI-Platt", "Platt-G"]
COVER_NOTE = (r"Cover: share of totals or contrasts whose error is within sampling noise (\%, paired document "
              r"bootstrap, $p \geq 0.05$), about 95 for an unbiased method.")


def slope_or_dash(x: float) -> str:
    """Slopes are not reported for the estimators from each group's annotated documents (article, Section V-D)."""
    return "--" if pd.isna(x) else f"{x:.2f}"


# ============================================================================ formatting
def signed(x: float, nd: int = 1) -> str:
    """A signed number with a typographic minus, e.g. $+$10.2 or $-$1.8."""
    s = f"{x:+.{nd}f}"
    if float(s) == 0:
        return f"{0:.{nd}f}"
    return ("$-$" if s[0] == "-" else "$+$") + s[1:]


def plain(x: float, nd: int = 2) -> str:
    s = f"{x:.{nd}f}"
    return "$-$" + s[1:] if s.startswith("-") else s


def interval(v: float, lo: float, hi: float, nd: int = 2) -> str:
    return f"{plain(v, nd)} [{plain(lo, nd)}, {plain(hi, nd)}]"


def excl1(v: float, lo: float, hi: float) -> str:
    return f"{v:.2f}" + (r"$^\dagger$" if lo > 1 or hi < 1 else "")


def table(env: str, caption: str, label: str, spec: str, header: str, rows: list[str], note: str,
          width: str = "", size: str = r"\footnotesize", colsep: str = "") -> str:
    """A booktabs table as wide as the text (table*) or the column (table), with a note row below.

    The table is a tabular* of exactly that width with \\extracolsep{\\fill}, so the space left over
    is spread evenly between the columns. The note is a paragraph of the same width below it, not a
    row: a full-width row is wider than the columns under it, and TeX gives the excess to the last
    column before tabular* can spread it."""
    width = width or (r"\textwidth" if env == "table*" else r"\columnwidth")
    assert spec.startswith("@{}"), spec
    return "\n".join([
        rf"\begin{{{env}}}[t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering", size,
        *([rf"\setlength{{\tabcolsep}}{{{colsep}}}"] if colsep else []),
        rf"\begin{{tabular*}}{{{width}}}{{@{{\extracolsep{{\fill}}}}{spec[3:]}}}",
        r"\toprule", header, r"\midrule", *rows, r"\bottomrule", r"\end{tabular*}",
        r"\par\vspace{2pt}", rf"\begin{{minipage}}{{{width}}}\scriptsize {note}\end{{minipage}}",
        rf"\end{{{env}}}"])


# ============================================================================ data and values
def table_values() -> None:
    g = load_gold("test")
    y = (g[VALUES].to_numpy() >= 0.5)
    man = g["Text-ID"].str.contains("_M_").to_numpy()
    rows = []
    for j, v in enumerate(VALUES):
        rows.append(f"{v} & {DEFINITIONS[v]} & {100 * y[:, j].mean():.1f} & {100 * y[man, j].mean():.1f} & "
                    f"{100 * y[~man, j].mean():.1f}" + r" \\")
    write_table("supp-values.tex", table(
        "table*", "The 19 values of the refined theory and their prevalence in the test split",
        "tab:values", r"@{}lp{0.52\textwidth}rrr@{}",
        r"Value & Definition (paraphrased) & All (\%) & Manifestos (\%) & News (\%) \\", rows,
        r"Definitions paraphrased from Schwartz et al.~\cite{Schwartz2012}. Prevalence: share of test sentences labeled with "
        r"the value by the annotators (reference labels), attained and constrained merged."))


def table_languages() -> None:
    splits = {s: load_gold(s) for s in ["training", "validation", "test"]}
    gv, gt = splits["validation"], splits["test"]
    yv = (gv[VALUES].to_numpy() >= 0.5).astype(int)
    yt = (gt[VALUES].to_numpy() >= 0.5).astype(int)
    lang_t = gt["Text-ID"].str.split("_").str[0].to_numpy()
    # the same computation as gold_reliability.py: main system, prevalence-matched thresholds, mean of runs
    f1, err = {l: [] for l in LANGUAGES}, {l: [] for l in LANGUAGES}
    for seed in SEEDS:
        pv, pt = system_probs(MAIN, seed, "val"), system_probs(MAIN, seed, "test")
        dec = (pt >= fit_rules(pv, yv)["tprev"]).astype(int)
        for l in LANGUAGES:
            m = lang_t == l
            f1[l].append(macro_f1(yt[m], dec[m]))
            err[l].append((dec[m].sum() / yt[m].sum() - 1) * 100)
    rows = []
    for l, name in LANGUAGES.items():
        n = [int((s["Text-ID"].str.split("_").str[0] == l).sum()) for s in splits.values()]
        docs = gt.loc[lang_t == l, "Text-ID"].drop_duplicates()
        allsent = pd.concat([s[s["Text-ID"].str.split("_").str[0] == l] for s in splits.values()])
        valued = (allsent[VALUES].to_numpy() >= 0.5).any(1).mean() * 100
        engine = "none (original)" if l == "EN" else ("Google" if l == "HE" else "DeepL")
        rows.append(f"{l} & {name} & {engine} & {n[0]:,} / {n[1]:,} / {n[2]:,} & "
                    f"{(~docs.str.contains('_M_')).sum()} / {docs.str.contains('_M_').sum()} & {valued:.1f} & "
                    f"{ALPHA_REFINED[l]:.3f} & {np.mean(f1[l]):.3f} & {signed(np.mean(err[l]))}" + r" \\")
    write_table("supp-languages.tex", table(
        "table*", "Source languages: size, annotation and the main system's errors on each language's test sentences",
        "tab:languages", r"@{}llllrrrrr@{}",
        r"Code & Language & Translation & Sentences (train / val. / test) & Test docs (news / man.) & "
        r"With a value (\%) & $\alpha$ & Macro-F1 & Count error (\%) \\", rows,
        rf"Translation into English by the task organizers. With a value: share of sentences with at least one "
        rf"value, all splits. $\alpha$: Krippendorff's alpha before curation for the 19 values, by language "
        rf"(ValuesML, Table 5~\cite{{Scharfbillig2026}}; all languages {ALPHA_ALL}). Macro-F1 and count error: main system with "
        rf"prevalence-matched thresholds on the test sentences of each language, mean of three training runs; "
        rf"count error = predicted over reference value mentions minus 1 (positive = overcount).", colsep="4pt"))


# ============================================================================ classifiers and LLMs
def logged_f1(setting: str, model: str) -> float | None:
    path = LLM_OUT / setting / f"{model}-val-results.txt"
    if not path.exists():
        return None
    m = re.search(r"Macro-average F1 \(fixed threshold\) across 19 labels: ([0-9.]+)", path.read_text())
    return float(m.group(1)) if m else None


def table_prompts() -> None:
    rows = []
    for model, name in LLMS.items():
        zero = [logged_f1(s, model) for s in ZERO_SHOT]
        prompt = "direct" if model.startswith("Qwen") else "definition"  # few-shot runs used each model's best style
        few = [logged_f1(f"{prompt}-{k}", model) for k in SHOTS]
        best = max(x for x in zero if x is not None)
        cells = [(r"\textbf{%.3f}" % x if x == best else f"{x:.3f}") if x is not None else "--" for x in zero]
        cells += [f"{x:.3f}" if x is not None else "--" for x in few]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    gated = logged_f1("definition-20", "google-gemma-2-9b-it-gated_Presence")
    write_table("supp-prompts.tex", table(
        "table*", "LLM labels on the validation split: macro-F1 by prompt style (zero-shot) and by number of "
                  "labeled examples in the prompt (few-shot)",
        "tab:prompts", r"@{}lcccc ccccccc@{}",
        r" & \multicolumn{4}{c}{Zero-shot prompt style} & \multicolumn{7}{c}{Few-shot, number of examples $k$} \\"
        "\n" r"\cmidrule(lr){2-5}\cmidrule(l){6-12}"
        "\n" r"Model & direct & hidden reasoning & definitions & question & 1 & 2 & 4 & 8 & 12 & 16 & 20 \\", rows,
        r"Validation macro-F1 of the returned label sets, as logged by each run (single runs). Bold: best "
        r"zero-shot style. Few-shot prompts use the definitions style (Qwen: the direct style, its best). "
        rf"With the presence gate, Gemma-2-9B with $k = 20$ reaches {gated:.3f}. The four styles are described "
        r"in Section~\ref{sec:llms}.", colsep="4pt"))


def table_comparisons() -> None:
    c = pd.read_csv(RES / "classification-comparisons.csv")
    rename = [(" [global t*]", r" (macro-F1 $t^*$)"), (" [label prev]", " (prevalence-matched)"),
              ("global t* vs t=0.5", r"macro-F1 $t^*$ vs.\ $t = 0.5$"), ("label t* vs global t*",
              r"per-label F1 vs.\ macro-F1 $t^*$"), ("label prev vs global t*", r"prevalence-matched vs.\ macro-F1 $t^*$"),
              (" vs no gate", r" vs.\ no gate"), (" vs ", r" vs.\ ")]
    families = {"Decision rules": "Decision rules (same system)",
                "Auxiliary inputs (vs text-only DeBERTa, same rule)": "Auxiliary inputs (vs.\\ text-only DeBERTa, same rule)",
                "Encoders and ensembles (same rule)": "Encoders and ensembles (same rule)",
                "Presence gate (same value classifier, vs no gate, global t*)":
                    r"Presence gate (same value classifier, macro-F1 $t^*$)"}
    rows = []
    for fam, title in families.items():
        rows.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for _, r in c[c.family == fam].iterrows():
            name = r.comparison
            for a, b in rename:
                name = name.replace(a, b)
            if fam.startswith("Auxiliary"):
                name = name.replace("DeBERTa + ", "+ ")
            same = "--" if pd.isna(r.same_sign) else r.same_sign
            q = "$<$0.01" if r.q < 0.01 else f"{r.q:.2f}"
            rows.append(rf"\quad {name} & {plain(r.delta, 3)} [{plain(r.lo, 3)}, {plain(r.hi, 3)}] & {q} & {same} \\")
    write_table("supp-comparisons.tex", table(
        "table*", "Paired comparisons of test macro-F1", "tab:comparisons", r"@{}lccc@{}",
        r"Comparison (second minus first) & $\Delta$ macro-F1 [95\% CI] & $q$ & Same sign in every run \\", rows,
        r"Differences in mean macro-F1 over three training runs, with the second system or rule minus the first. "
        r"Intervals: document bootstrap (2,000 resamples, stratified by language and genre), the same resamples "
        r"for every system. $q$: Benjamini--Hochberg within each block. Same sign: the difference has the same "
        r"sign in each of the three runs (--: the seed ensemble is a single system)."))


# ============================================================================ prevalence and calibration
def table_prevalence() -> None:
    v = pd.read_csv(RES / "measurement-per-value.csv")
    v = v[v.system == MAIN].sort_values("gold_prev", ascending=False)
    cols = ["CC@0.5", "CC@t*", "CC@prev", "PCC", "PCC-Platt", "ACC@prev", "EM", "Gold-only"]
    rows = []
    for _, r in v.iterrows():
        rows.append(f"{r.value} & {100 * r.gold_prev:.2f} [{100 * r.gold_lo:.2f}, {100 * r.gold_hi:.2f}] & "
                    + " & ".join(f"{100 * r[c]:.2f}" for c in cols) + r" \\")
    # drop the method name where the group header already gives it ("PCC, raw" -> "raw"); keep "ACC, ..."
    head = " & ".join(re.sub(r"^P?CC, ", "", LABEL[c]).replace("prevalence-matched", "prev.-matched")
                      for c in cols)
    write_table("supp-prevalence.tex", table(
        "table*", "Corpus prevalence of each value on the test split: reference and estimates of the main system (\\%)",
        "tab:prevalence", r"@{}lccccccccc@{}",
        r" & & \multicolumn{3}{c}{Classify and count} & \multicolumn{2}{c}{Probability averages} & "
        r"\multicolumn{2}{c}{Adjusted counts} & No classifier \\" "\n"
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-10}"
        "\n" rf"Value & Reference [95\% CI] & {head} \\", rows,
        r"Share of test sentences with the value, in percent; estimates are means over three training runs. "
        r"Values sorted by reference prevalence. Annotated share: the value's share of the validation sentences, "
        r"with no classifier. SC leaves the corpus total unchanged, so it would repeat prevalence-matched classify "
        r"and count. Fig.~\ref{M-fig:prevalence} of the article plots the ratios of the first three columns.",
        size=r"\scriptsize", colsep="3pt"))


def calibration() -> pd.DataFrame:
    gv, gt = load_gold("validation"), load_gold("test")
    yv = (gv[VALUES].to_numpy() >= 0.5).astype(int)
    yt = (gt[VALUES].to_numpy() >= 0.5).astype(int)
    rows, pooled = [], {"raw": [], "platt": []}
    for system in SUPERVISED:
        for seed in system_seeds(system):
            pv, pt = system_probs(system, seed, "val"), system_probs(system, seed, "test")
            fit = fit_rules(pv, yv)
            platt = apply_rules(pt, fit)["PCC-Platt"]
            rows.append({"system": system, "tstar": fit["tstar"], "ece_raw": ece(pt, yt), "ece_platt": ece(platt, yt),
                         "citl_raw": pt.mean() / yt.mean(), "citl_platt": platt.mean() / yt.mean()})
            if system == MAIN:
                pooled["raw"].append(pt.ravel()); pooled["platt"].append(platt.ravel())
    pooled["y"] = np.tile(yt.ravel(), len(SEEDS))
    figure_reliability({k: np.concatenate(v) if k != "y" else v for k, v in pooled.items()})
    return pd.DataFrame(rows)


def table_calibration(cal: pd.DataFrame) -> None:
    agg = cal.groupby("system", sort=False).agg(["mean", "min", "max", "size"])
    rows = []
    for s in SUPERVISED:
        a = agg.loc[s]
        t = (f"{a[('tstar', 'min')]:.2f}--{a[('tstar', 'max')]:.2f}" if a[("tstar", "size")] > 1
             else f"{a[('tstar', 'mean')]:.2f}")
        name = r"\textbf{Feature ensemble}" if s == MAIN else s
        rows.append(f"{name} & {a[('ece_raw', 'mean')]:.4f} & {a[('ece_platt', 'mean')]:.4f} & "
                    f"{a[('citl_raw', 'mean')]:.3f} & {a[('citl_platt', 'mean')]:.3f} & {t}" + r" \\")
    write_table("supp-calibration.tex", table(
        "table", "Calibration of the probabilities on the test split, raw and after Platt scaling",
        "tab:calibration", r"@{}lccccc@{}",
        r" & \multicolumn{2}{c}{ECE} & \multicolumn{2}{c}{Mean prob. / rate} & \\"
        "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
        "\n" r"System & raw & Platt & raw & Platt & $t^*$ \\", rows,
        r"ECE: expected calibration error over all sentence--value pairs, 15 equal-width bins. Mean prob.\ / "
        r"rate: mean predicted probability over the reference rate (1 = the average is right). Means of three "
        r"training runs (seed ensemble: one system). $t^*$: macro-F1 threshold, range over runs.",
        colsep="4pt"))


def figure_reliability(p: dict[str, np.ndarray]) -> None:
    bins = 15
    fig, (ax, axh) = new_figure(COL_W, 2.9, 2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for key, color, marker, name in [("raw", ORANGE, "o", "Raw probabilities"),
                                     ("platt", BLUE, "s", "After Platt scaling")]:
        x = p[key]
        idx = np.minimum((x * bins).astype(int), bins - 1)
        n = np.bincount(idx, minlength=bins)
        ok = n > 0
        mp = np.bincount(idx, weights=x, minlength=bins)[ok] / n[ok]
        obs = np.bincount(idx, weights=p["y"], minlength=bins)[ok] / n[ok]
        ax.plot(mp, obs, color=color, lw=1.0, marker=marker, ms=3.6, mfc=color if key == "platt" else "white",
                mew=0.8, label=name, zorder=3)
        centres = (np.arange(bins) + 0.5) / bins
        axh.plot(centres[ok], n[ok] / n.sum(), color=color, lw=1.0, marker=marker, ms=3.0,
                 mfc=color if key == "platt" else "white", mew=0.8, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_ylabel("Share with the value")
    ax.legend(loc="upper left")
    axh.set_yscale("log")
    axh.set_ylabel("Share of pairs")
    axh.set_xlabel("Predicted probability (bin mean)")
    for a in (ax, axh):
        style_axis(a, "y")
        a.spines["left"].set_visible(True); a.spines["left"].set_color(INK2)
        a.tick_params(axis="y", length=2.5)
    save(fig, "figS1-reliability.pdf")


# ============================================================================ contrasts
def table_systems() -> None:
    m = pd.read_csv(RES / "measurement-summary.csv").set_index(["system", "rule"])
    rows = []
    for r in ALL_RULES:
        cells = []
        for s in OTHERS:
            x = m.loc[(s, r)]
            cells += [excl1(x.ratio, x.ratio_lo, x.ratio_hi), slope_or_dash(x.genre_slope), f"{x.genre_cover:.0f}",
                      slope_or_dash(x.lpair_slope), f"{x.lpair_cover:.0f}"]
        rows.append(f"{LABEL[r]} & " + " & ".join(cells) + r" \\")
    head = " & ".join(rf"\multicolumn{{5}}{{c}}{{{s}}}" for s in OTHERS)
    write_table("supp-systems.tex", table(
        "table*", "Measurement with the other supervised systems: totals, manifesto--news contrasts and "
                  "language-pair contrasts", "tab:systems", r"@{}l ccccc ccccc ccccc@{}",
        rf" & {head} \\" "\n" r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(l){12-16}"
        "\n" r"Method & " + " & ".join(["Ratio", "G.\\ slope", "G.\\ cover", "L.\\ slope", "L.\\ cover"] * 3) + r" \\",
        rows,
        r"Ratio: estimated over reference value mentions, $^\dagger$ 95\% CI excludes 1. G.: the 19 manifesto--news "
        r"contrasts; L.: the 684 language-pair contrasts. Slope: attenuation slope, not shown for per-group "
        r"estimates (Section~\ref{M-sec:metrics} of the article). " + COVER_NOTE + r" DeBERTa "
        r"and RoBERTa: means of three training runs; seed ensemble: one system. The main system is in "
        r"Table~\ref{M-tab:measurement} of the article.",
        size=r"\scriptsize", colsep="2.5pt"))


def figure_language_contrasts() -> None:
    lc = pd.read_csv(RES / "measurement-language-contrasts.csv")
    lc = lc[lc.system == MAIN]
    sig = lc.gold_sig.to_numpy(bool)
    g = 100 * lc.gold_diff.to_numpy()
    fig, axes = new_figure(COL_W, 2.15, 1, 2, sharey=True)
    lim = np.ceil(np.abs(np.r_[g, 100 * lc["CC@prev"], 100 * lc["SC@prev"]]).max())
    for ax, rule, tag in [(axes[0], "CC@prev", "(a)"), (axes[1], "SC@prev", "(b)")]:
        e = 100 * lc[rule].to_numpy()
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.scatter(g[~sig], e[~sig], s=4, color=GRID, edgecolors=MUTED, linewidths=0.3, zorder=2)
        ax.scatter(g[sig], e[sig], s=5, color=BLUE, edgecolors="none", zorder=3)
        b = np.polyfit(g, e, 1)
        ax.plot([-lim, lim], np.polyval(b, [-lim, lim]), color=ORANGE, lw=1.0, zorder=4)
        ax.set_title(f"{tag} {LABEL[rule].replace('prevalence-matched', 'prev.-matched')}", loc="left", fontsize=7.5)
        ax.text(0.97, 0.04, f"slope {b[0]:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Reference contrast (pp)")
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=2.5)
    axes[0].set_ylabel("Estimated contrast (pp)")
    save(fig, "figS2-language-contrasts.pdf")


def table_worked_values() -> None:
    w = pd.read_csv(RES / "worked-example-values.csv")
    w = w[w.version == "crude"]
    gold = w[w.estimator == "gold"].set_index("value")
    order = gold.sort_values("diff_pp", ascending=False).index
    est = ["CC@0.5", "CC@t*", "CC@prev", "PCC-Platt", "ACC@prev", "SC@prev", "LLM"]
    piv = w.set_index(["value", "estimator"])
    rows = []
    for v in order:
        gr = gold.loc[v]
        cells = [signed(gr.diff_pp) + (r"$^*$" if gr.significant else "")
                 + f" [{signed(gr.lo_pp)}, {signed(gr.hi_pp)}]"]
        for e in est:
            x = piv.loc[(v, e)]
            mark = ""
            if x.significant:
                mark = r"$^\ddagger$" if gr.significant and np.sign(x.diff_pp) != np.sign(gr.diff_pp) else r"$^*$"
            cells.append(signed(x.diff_pp) + mark)
        rows.append(f"{v} & " + " & ".join(cells) + r" \\")
    head = ["CC, $t = 0.5$", "CC, $t^*$", "CC, prev.", "PCC, Platt", "ACC, prev.", "SC, prev.", "LLM"]
    write_table("supp-worked-values.tex", table(
        "table*", "Worked example: manifesto minus news difference for each value (percentage points)",
        "tab:worked-values", r"@{}lcccccccc@{}",
        "Value & Reference [95\\% CI] & " + " & ".join(head) + r" \\", rows,
        r"Supervised methods: main system, mean of three training runs; LLM: Gemma-2-9B few-shot with the "
        r"presence gate, single run. $^*$ significant (document bootstrap, Benjamini--Hochberg $q < 0.05$ over "
        r"the 19 values); $^\ddagger$ significant with the sign opposite to a significant reference difference. "
        r"Significance is tested on the mean of the three runs. "
        r"Values sorted by the reference difference. Fig.~\ref{M-fig:worked} of the article plots four of these columns.",
        size=r"\scriptsize", colsep="3pt"))


def table_shift() -> None:
    s = pd.read_csv(RES / "shift-tests.csv").set_index(["system", "scenario", "rule"])
    rows = []
    for sysname in SUPERVISED:
        def get(sc, r):
            return s.loc[(sysname, sc, r)]
        lo_cc, lo_sc = get("A leave-one-language-out", "CC@prev"), get("A leave-one-language-out", "SC@prev")
        nw_cc, nw_sc = get("B news-only fit", "CC@prev"), get("B news-only fit", "SC@prev")
        wn_cc, wn_sc = get("C news only", "CC@prev"), get("C news only", "SC@prev")
        name = r"\textbf{Feature ensemble}" if sysname == MAIN else sysname
        rows.append(" & ".join([
            name, f"{lo_cc.slope:.2f}", interval(lo_sc.slope, lo_sc.slope_lo, lo_sc.slope_hi),
            f"{lo_cc.mae:.2f} / {lo_sc.mae:.2f}", f"{lo_cc.total_err:.1f}", f"{nw_cc.slope:.2f}",
            interval(nw_sc.slope, nw_sc.slope_lo, nw_sc.slope_hi), f"{nw_cc.mae:.2f} / {nw_sc.mae:.2f}",
            f"{wn_cc.slope:.2f}", f"{wn_sc.slope:.2f}"]) + r" \\")
    write_table("supp-shift.tex", table(
        "table*", "Shift tests for every supervised system", "tab:shift", r"@{}l cccc ccc cc@{}",
        r" & \multicolumn{4}{c}{(a) One language left out of the fit} & \multicolumn{3}{c}{(b) Fitted on news only} "
        r"& \multicolumn{2}{c}{(c) Within news} \\"
        "\n" r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(l){9-10}"
        "\n" r"System & CC slope & SC slope [95\% CI] & MAE CC / SC & CC total err.\ (\%) & CC slope & "
        r"SC slope [95\% CI] & MAE CC / SC & CC slope & SC slope \\", rows,
        r"CC: prevalence-matched thresholds; SC: slope correction on the same counts. (a) Thresholds and "
        r"slope fitted without the language whose contrasts are estimated (contrasts of that language with the "
        r"other eight, pooled over languages); total err.: mean absolute error of the left-out language's value "
        r"mentions. MAE: mean absolute error of the contrasts, percentage points. "
        r"(b) Rules fitted on news validation documents; manifesto--news contrasts, with SC borrowing the "
        r"language slope. (c) Language-pair contrasts within news documents. Means of three training runs (seed "
        r"ensemble: one system).", size=r"\scriptsize", colsep="3pt"))


def table_llm() -> None:
    m = pd.read_csv(RES / "llm-measurement.csv").set_index(["system", "estimator"])
    rows = []
    for s, name in LLM_SYSTEMS.items():
        for e in ["CC", "ACC", "SC"]:
            x = m.loc[(s, e)]
            f1 = "--" if pd.isna(x.macro_f1) else f"{x.macro_f1:.3f}"
            rows.append(" & ".join([
                f"{name}, {e}", f1, excl1(x.ratio, x.ratio_lo, x.ratio_hi), f"{x.mae:.2f}", f"{x.cover:.0f}",
                interval(x.genre_slope, x.genre_slope_lo, x.genre_slope_hi), f"{x.genre_mae:.2f}",
                f"{x.genre_cover:.0f}", f"{x.genre_reversed:.0f}",
                interval(x.lpair_slope, x.lpair_slope_lo, x.lpair_slope_hi), f"{x.lpair_mae:.2f}",
                f"{x.lpair_cover:.0f}", f"{x.lpair_reversed:.1f}"]) + r" \\")
        if s != list(LLM_SYSTEMS)[-1]:
            rows.append(r"\addlinespace")
    write_table("supp-llm.tex", table(
        "table*", "LLM labels used as counts, with each correction", "tab:llm", r"@{}lcccccccccccc@{}",
        r" & & \multicolumn{3}{c}{Corpus prevalence} & \multicolumn{4}{c}{Manifesto vs.\ news} & "
        r"\multicolumn{4}{c}{Language pairs} \\"
        "\n" r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}"
        "\n" r"System, estimator & Macro-F1 & Ratio & MAE & Cover & Slope [95\% CI] & MAE & Cover & Rev.\ (\%) "
        r"& Slope [95\% CI] & MAE & Cover & Rev.\ (\%) \\", rows,
        r"Single runs. CC: the labels as returned; ACC: adjusted with error rates measured on validation; SC: "
        r"contrasts divided by the validation slope. Ratio: estimated over reference value mentions, $^\dagger$ "
        r"95\% CI excludes 1. MAE: mean absolute error in percentage points, over the 19 values or the contrasts. "
        + COVER_NOTE + r" Rev.: share of the significant reference contrasts that the method finds significant "
        r"with the opposite sign (Benjamini--Hochberg $q < 0.05$).", size=r"\scriptsize", colsep="2.5pt"))


# ============================================================================ step 5.2: new analyses
def table_subsample() -> None:
    """Accuracy of each estimator when it is fitted on fewer annotated documents (medians over 100 subsamples)."""
    d = pd.read_csv(RES / "subsample-validation.csv", dtype={"size": str}).set_index(["size", "rule"])
    sizes = ["25", "50", "100", "200", "all"]
    cols = [("Corpus total (\\%)", "total_err", ["CC@t*", "CC@prev"], 1),
            ("Languages (\\%)", "lang_total_err", ["CC@prev", "PPI-Platt"], 1),
            ("Language pairs, MAE", "lpair_mae", ["CC@prev", "SC@prev", "PPI-Platt", "Platt-G"], 2),
            ("Large differences, MAE", "lpair_mae_refsig", ["CC@prev", "SC@prev", "PPI-Platt"], 2),
            ("Genre, MAE", "genre_mae", ["CC@prev", "PPI-Platt"], 2),
            ("Detected (\\%)", "lpair_detected", ["CC@prev", "SC@prev", "PPI-Platt"], 0)]
    rows = []
    for z in sizes:
        x = d.loc[(z, "CC@prev")]
        cells = [f"{x.docs:.0f} ({x.sentences:,.0f})"]
        for _, k, rules, nd in cols:
            cells += [f"{d.loc[(z, r), k]:.{nd}f}" for r in rules]
        rows.append(" & ".join(cells) + r" \\")
    short = {"CC@t*": "$t^*$", "CC@prev": "CC", "SC@prev": "SC", "PPI-Platt": "PPI", "Platt-G": "P-G"}
    head1 = " & ".join(rf"\multicolumn{{{len(rs)}}}{{c}}{{{t}}}" for t, _, rs, _ in cols)
    spans, start = [], 2
    for _, _, rs, _ in cols:
        spans.append(rf"\cmidrule(lr){{{start}-{start + len(rs) - 1}}}")
        start += len(rs)
    rule_heads = " & ".join(short[r] for _, _, rs, _ in cols for r in rs)
    write_table("supp-subsample.tex", table(
        "table*", "Accuracy by size of the annotated sample: every method refitted on random subsamples of the "
                  "validation documents (main system)", "tab:subsample",
        "@{}l" + "c" * sum(len(rs) for _, _, rs, _ in cols) + "@{}",
        rf" & {head1} \\" + "\n" + "".join(spans) + "\n" + rf"Annotated documents (sentences) & {rule_heads} \\",
        rows,
        r"Medians over 100 random subsamples of each size; detection is the mean over 20 of them, each with 50 "
        r"bootstrap resamples. Manifestos (about 11\% of the documents, at least two) and news (at least two per "
        r"language, the rest in proportion to each language's size) are drawn separately. The last row is the full "
        r"validation split, about 58 documents per language. Corpus total and languages: error of the total value "
        r"mentions, \%, for the corpus and on average over the nine languages. MAE in percentage points over the "
        r"684 language-pair contrasts, the 126 that are significant in the reference (large differences) and the "
        r"19 manifesto--news contrasts. Detected: share of the 126 found significant with the reference's sign "
        r"(single test); with 50 resamples, the last row differs by a few points from the 2,000-resample shares of "
        r"Table~\ref{M-tab:measurement} (49\% and 86\%). CC: prevalence-matched thresholds; $t^*$: macro-F1 threshold; SC: slope correction; PPI: "
        r"prediction-powered estimate with Platt-scaled probabilities; P-G: Platt per group.",
        size=r"\scriptsize", colsep="3pt"))


def table_lolo() -> None:
    """Each language left out of the fit in turn: prevalence-matched counts and the slope correction."""
    d = pd.read_csv(RES / "shift-lolo-languages.csv")
    d = d[(d.system == MAIN) & (d.fit == "leave-one-language-out")].set_index(["rule", "language"])
    rows = []
    for code, name in LANGUAGES.items():
        cc, sc = d.loc[("CC@prev", code)], d.loc[("SC@prev", code)]
        rows.append(" & ".join([
            name, f"{cc.fitted_slope:.2f}", interval(cc.contrast_slope, cc.contrast_slope_lo, cc.contrast_slope_hi),
            interval(sc.contrast_slope, sc.contrast_slope_lo, sc.contrast_slope_hi), f"{cc.mae:.2f}", f"{sc.mae:.2f}",
            f"{cc.total_err:.1f}", f"{sc.total_err:.1f}"]) + r" \\")
    write_table("supp-lolo.tex", table(
        "table*", "Each source language left out of the fit in turn (main system)", "tab:lolo", "@{}lccccccc@{}",
        r" & & \multicolumn{2}{c}{Slope of its contrasts} & \multicolumn{2}{c}{MAE (pp)} & "
        r"\multicolumn{2}{c}{Total error (\%)} \\" + "\n" + r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}"
        + "\n" + r"Left-out language & Fitted slope & CC [95\% CI] & SC [95\% CI] & CC & SC & CC & SC \\", rows,
        r"Thresholds and slope are fitted on the validation documents of the other eight languages. Fitted slope: "
        r"the validation slope of language-pair contrasts without the language, which SC divides by. Slope and MAE "
        r"over the 152 contrasts between the left-out language and the other eight; total error: of the left-out "
        r"language's value mentions. CC: prevalence-matched thresholds; SC: slope correction of the same counts. "
        r"Means of three training runs.", size=r"\scriptsize", colsep="3pt"))


def table_composition() -> None:
    """Language contrasts on value composition, which removes the teams' differences in labeling rate."""
    m = pd.read_csv(RES / "measurement-summary.csv").set_index(["system", "rule"])
    rules = ["CC@t*", "CC@prev", "PCC-Platt", "ACC@prev", "SC@prev", "Gold-only", "PPI@prev", "PPI-Platt", "Platt-G"]
    rows = []
    for r in rules:
        x = m.loc[(MAIN, r)]
        slope = "--" if pd.isna(x.comp_slope) else interval(x.comp_slope, x.comp_slope_lo, x.comp_slope_hi)
        rows.append(" & ".join([LABEL[r], slope, f"{x.comp_mae:.2f}", f"{x.comp_cover:.0f}", f"{x.comp_detected:.0f}",
                                slope_or_dash(x.lpair_slope), f"{x.lpair_mae:.2f}"]) + r" \\")
    write_table("supp-composition.tex", table(
        "table*", "Language-pair contrasts on value composition, each value's share of all value mentions in the "
                  "language (main system)", "tab:composition", "@{}lcccccc@{}",
        r" & \multicolumn{4}{c}{Composition} & \multicolumn{2}{c}{Prevalence} \\"
        + "\n" + r"\cmidrule(lr){2-5}\cmidrule(l){6-7}"
        + "\n" + r"Method & Slope [95\% CI] & MAE (pp) & Cover & Detected (\%) & Slope & MAE (pp) \\", rows,
        r"684 contrasts, 64 of them significant in the reference. Composition removes the differences between the "
        r"teams in how many sentences they labeled, so a contrast compares which values a language's texts stress. "
        r"Estimates are turned into shares the same way, with negative estimates set to zero first. Detected: share "
        r"of the significant reference contrasts found significant with the same sign (single test). " + COVER_NOTE
        + r" Prevalence columns repeat Table~\ref{M-tab:measurement} of the article.",
        size=r"\scriptsize", colsep="3pt"))


def table_error_rates() -> None:
    """tpr and fpr of the prevalence-matched decisions per value, by genre, on validation and test."""
    e = pd.read_csv(RES / "measurement-error-rates.csv")
    e = e[e.system == MAIN].set_index(["split", "group", "value"])
    langs = list(LANGUAGES)

    def pct(x: float) -> str:
        return "--" if pd.isna(x) else f"{100 * x:.0f}"

    rows = []
    for v in VALUES:
        cells = [f"{pct(e.loc[(sp, g, v), 'tpr'])} / {100 * e.loc[(sp, g, v), 'fpr']:.1f}"
                 for sp in ["validation", "test"] for g in ["manifestos", "news"]]
        lt = np.array([e.loc[("test", l, v), "tpr"] for l in langs], dtype=float)
        lf = np.array([e.loc[("test", l, v), "fpr"] for l in langs], dtype=float)
        cells += [f"{100 * np.nanmin(lt):.0f}--{100 * np.nanmax(lt):.0f}",
                  f"{100 * np.nanmin(lf):.1f}--{100 * np.nanmax(lf):.1f}"]
        rows.append(f"{v} & " + " & ".join(cells) + r" \\")
    write_table("supp-error-rates.tex", table(
        "table*", r"Error rates of the prevalence-matched decisions by genre and language (main system, \%)",
        "tab:error-rates", "@{}lcccccc@{}",
        r" & \multicolumn{2}{c}{Validation, tpr / fpr} & \multicolumn{2}{c}{Test, tpr / fpr} & "
        r"\multicolumn{2}{c}{Test, range over languages} \\"
        + "\n" + r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}"
        + "\n" + r"Value & Manifestos & News & Manifestos & News & tpr & fpr \\", rows,
        r"tpr: share of the sentences with the value that are labeled with it; fpr: share of those without it that "
        r"are. --: no manifesto sentence with the value. Means of three training runs. Section~\ref{M-sec:res-acc} "
        r"of the article correlates the manifesto--news gaps with the reference contrasts.",
        size=r"\scriptsize", colsep="3pt"))


def table_group_totals() -> None:
    """Totals of single groups, where prevalence differs from the annotated sample (AE item 1)."""
    m = pd.read_csv(RES / "measurement-summary.csv").set_index(["system", "rule"])
    s = pd.read_csv(RES / "shift-tests.csv").set_index(["system", "scenario", "rule"])
    rows = []
    for r in ALL_RULES:
        x = m.loc[(MAIN, r)]
        lolo = s.loc[(MAIN, "A leave-one-language-out", r)] if (MAIN, "A leave-one-language-out", r) in s.index else None
        news = s.loc[(MAIN, "B news-only fit", r)] if (MAIN, "B news-only fit", r) in s.index else None
        rows.append(" & ".join([
            LABEL[r], f"{x.man_total_err:.1f}", f"{x.news_total_err:.1f}",
            f"{x.lang_total_err:.1f} ({x.lang_total_err_max:.1f})",
            "--" if lolo is None else f"{lolo.total_err:.1f} ({lolo.total_err_max:.1f})",
            "--" if news is None else f"{news.total_err:.1f}"]) + r" \\")
    write_table("supp-group-totals.tex", table(
        "table*", "Error of the total value mentions in single groups (main system, \\%)", "tab:group-totals",
        "@{}lccccc@{}",
        r" & \multicolumn{3}{c}{Every rule fitted on all validation documents} & "
        r"\multicolumn{2}{c}{Group left out of the fit} \\" "\n" r"\cmidrule(lr){2-4}\cmidrule(l){5-6}"
        "\n" r"Method & Manifestos & News & Languages, mean (max) & Left-out language, mean (max) & "
        r"Manifestos, fitted on news \\", rows,
        r"Error: estimated over reference value mentions in the group, minus 1, in absolute value. Languages: "
        r"each of the nine languages, mean and largest error. Left-out language: thresholds, maps, error rates "
        r"and slopes fitted without the language's validation documents. Manifestos, fitted on news: every rule "
        r"fitted on news validation documents only. --: not run in that test. Means of three training runs.",
        size=r"\scriptsize", colsep="3pt"))


def table_detection() -> None:
    """How many of the significant reference differences each method finds (AE item 11)."""
    m = pd.read_csv(RES / "measurement-summary.csv").set_index(["system", "rule"])
    rows = []
    for r in ALL_RULES:
        x = m.loc[(MAIN, r)]
        rows.append(" & ".join([LABEL[r], f"{x.genre_detected:.0f}", f"{x.genre_reproduced:.0f}",
                                f"{x.lpair_detected:.0f}", f"{x.lpair_reproduced:.0f}", f"{x.lpair_sign_ok:.0f}"])
                    + r" \\")
    write_table("supp-detection.tex", table(
        "table", "Significant reference differences found by each method (main system, \\%)", "tab:detection",
        "@{}lccccc@{}",
        r" & \multicolumn{2}{c}{Genre (9)} & \multicolumn{3}{c}{Language pairs (126)} \\"
        "\n" r"\cmidrule(lr){2-3}\cmidrule(l){4-6}"
        "\n" r"Method & Single & BH & Single & BH & Sign \\", rows,
        r"Share of the significant reference differences that the method also finds significant with the same "
        r"sign. Single: each difference tested on its own ($p < 0.05$). BH: after Benjamini--Hochberg correction "
        r"over the 19 genre or 684 language contrasts ($q < 0.05$). Sign: share estimated with the right sign, "
        r"significant or not. After the BH correction, no method finds any of them significant with the wrong sign.",
        size=r"\scriptsize", colsep="2.5pt"))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    table_values()
    table_languages()
    table_prompts()
    table_comparisons()
    table_prevalence()
    table_calibration(calibration())
    table_systems()
    figure_language_contrasts()
    table_worked_values()
    table_shift()
    table_llm()
    table_subsample()
    table_lolo()
    table_composition()
    table_error_rates()
    table_group_totals()
    table_detection()
    print(f"Wrote the supplement's tables and figures to {TAB.parent}")


if __name__ == "__main__":
    main()
