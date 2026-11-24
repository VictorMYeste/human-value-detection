#!/usr/bin/env python3
"""
LaTeX for Supplementary Section S8 ("Results under per-label F1 thresholds"), step 2.4d of
the revision plan. It only formats results already computed by threshold_sensitivity.py
rebuild_tables.py, rebuild_ensembles.py, soft_conditioning.py and multitask_eval.py; nothing is refitted here.

Usage (after threshold_sensitivity.py, rebuild_tables.py, rebuild_ensembles.py and
threshold_gain_direct.py, soft_conditioning.py and multitask_eval.py):
    python3 approaches/ho-revision/make_supplement_s8.py
Writes results/supplement-s8.tex, to be pasted (or \\input) at the end of the supplement.
"""
import ast
import json
import re

import pandas as pd

from common import PAIRS, RESULTS, SLICES

TOK = {
    "Baseline": "Baseline", "Previous-Sentences-2": "Prev.\\ sentences", "NER": "NER",
    "LingFeat": "Ling.\\ Feat.", "MultiLayer": "MultiLayer",
    "Lex-LIWC": "LIWC-15", "Lex-LIWC-15": "LIWC-15", "Lex-LIWC-22": "LIWC-22", "Lex-MFD": "MFD",
    "Lex-MFD-20": "MFD-20", "Lex-eMFD": "eMFD", "Lex-MJD": "MJD", "Lex-Schwartz": "Schwartz lex.",
    "Lex-VAD": "NRC VAD", "Lex-EmoLex": "EmoLex", "Lex-EmotionIntensity": "Emotion Int.",
    "Lex-WorryWords": "WorryWords", "TD-LDA": "LDA topics", "TD-NMF": "NMF topics",
    "TD-BERTopic": "BERTopic topics",
}
ABBR = {
    "Self-direction: thought": "SD-t", "Self-direction: action": "SD-a", "Stimulation": "Stim",
    "Hedonism": "Hed", "Achievement": "Ach", "Power: dominance": "Pow-d", "Power: resources": "Pow-r",
    "Face": "Face", "Security: personal": "Sec-per", "Security: societal": "Sec-soc", "Tradition": "Trad",
    "Conformity: rules": "Conf-r", "Conformity: interpersonal": "Conf-i", "Humility": "Hum",
    "Benevolence: caring": "Ben-c", "Benevolence: dependability": "Ben-d",
    "Universalism: concern": "Uni-c", "Universalism: nature": "Uni-n", "Universalism: tolerance": "Uni-t",
}
CASCADE = re.compile(r"^1_(?P<gate>.+)_(?P<tg>0\.\d+)_(?P<model>.+)$")
RULES = ["fixed 0.5", "Eq. 8, floor 0.30", "Eq. 8, floor 0.40", "Eq. 8, floor 0.50", "label F1", "global F1"]
RULE_TEX = ["0.5", "$\\pi{=}0.30$", "$\\pi{=}0.40$", "$\\pi{=}0.50$", "Label $F_1$", "Global $F_1$"]


def pretty(name: str) -> str:
    out = []
    for p in name.split("_"):
        if p.startswith("Previous-Sentences-2-"):
            out += [TOK["Previous-Sentences-2"], TOK[p[len("Previous-Sentences-2-"):]]]
        else:
            out.append(TOK[p])
    return " + ".join(out)


def short_pair(pair: str) -> str:
    return (pair.replace("Self-Transcendence", "Self-Trans.").replace("Self-Enhancement", "Self-Enh.")
            .replace("Self-Protection", "Self-Prot.").replace("Openness", "Openness")
            .replace(" Focus", "").replace("vs.", "vs.\\"))


def f3(x: float) -> str:
    return f"{x:.3f}".replace("-", "$-$")


def table(caption, label, colspec, header, rows, size="\\footnotesize"):
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    return (f"\\begin{{table}}[!htbp]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}\n\\hline\n{header} \\\\\n\\hline\n{body}\n"
            f"\\hline\n\\end{{tabularx}}\n\\end{{table}}\n")


def longtable(caption, label, colspec, header, rows):
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    ncol = header.count("&") + 1
    return (f"{{\\footnotesize\n\\begin{{longtable}}{{{colspec}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n"
            f"\\hline\n{header} \\\\\n\\hline\n\\endfirsthead\n"
            f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{Table~\\ref{{{label}}}, continued}}}}\\\\\n\\hline\n{header} \\\\\n\\hline\n\\endhead\n"
            f"\\hline\n\\endfoot\n{body}\n\\end{{longtable}}\n}}\n")


def sig_mark(cmp, task, unit, system, rule):
    if rule == "fixed 0.5":
        return ""
    r = cmp[(cmp.task == task) & (cmp.unit == unit) & (cmp.system == system) & (cmp.baseline == "fixed 0.5")
            & (cmp["rule"] == rule)]
    if r.empty:
        return ""
    r = r.iloc[0]
    return "$^{+}$" if r.p_bh < 0.05 else "$^{-}$" if r.p_worse_bh < 0.05 else ""


def main() -> None:
    sens = pd.read_csv(RESULTS / "threshold-sensitivity.csv")
    cmp = pd.read_csv(RESULTS / "threshold-comparisons.csv")
    cand = pd.read_csv(RESULTS / "rebuild-candidates.csv")
    sel = pd.read_csv(RESULTS / "rebuild-selected.csv")
    chosen = json.loads((RESULTS / "rebuild-selections.json").read_text())
    out = []

    out.append(r"""\section{Results under per-label $F_1$ thresholds}
\label{sec:per-label-f1}

This section reports the systems of the main text under its threshold protocol (main text, Section~3.6). Each label receives the threshold on the grid $0.01,0.02,\dots,0.99$ that maximizes its validation $F_1$, and ties go to the threshold closest to 0.5. Within each family (HO detectors, \emph{Presence} cascades, \emph{Direct} value models, HO-gated pipelines), we select the system with the highest validation Macro-$F_1$ and use the test set once. The candidates are all the systems whose validation and test predictions were saved, which are the systems taken to test in Sections~S2--S4. Paired comparisons use a one-sided sentence bootstrap with $B=2000$ resamples, run in both directions, with Benjamini--Hochberg correction across the systems or slices of a table. The scripts that produce these tables are in \texttt{approaches/ho-revision} of the released code.

\subsection{Threshold rules compared}
\label{sec:s8-rules}

Tables~\ref{tab:s8-rules-ho} and~\ref{tab:s8-rules-values} compare six rules, each fitted on validation and applied once to test. Besides per-label $F_1$, they are the fixed threshold 0.5 and one global threshold chosen to maximize validation Macro-$F_1$ (grid 0.05--1.00). The other three are a recall rule with floors $\pi\in\{0.30,0.40,0.50\}$: it maximizes each label's recall subject to precision $\ge\pi$ on the grid 0.10--0.89 and falls back to 0.5 when no threshold meets the floor. Table~\ref{tab:s8-rules-summary} counts significant differences.
""")

    # ---- S8.1 HO detection by rule
    rows = []
    for pair in PAIRS:
        s = sens[(sens.task == "HO detection") & (sens.unit == pair)]
        for k, sysname in enumerate(s.system.unique()):
            g = s[s.system == sysname].set_index("rule")
            v = cand[(cand.family == "HO detector") & (cand.unit == pair) & (cand.name == sysname)].val_f1.iloc[0]
            name = pretty(sysname)
            if chosen["ho_detector"][pair] == sysname:
                name = f"\\textbf{{{name}}}"
            rows.append([short_pair(pair) if k == 0 else "", name, f3(v)] + [f3(g.loc[r, "test_f1"]) + sig_mark(cmp, "HO detection", pair, sysname, r) for r in RULES])
    out.append(table(
        "HO detection: test Macro-$F_1$ of every detector under six threshold rules. Val: validation Macro-$F_1$ under per-label $F_1$ thresholds, used for selection; the selected detector is in bold. $^{+}$/$^{-}$: significantly better/worse than the fixed threshold 0.5 (one-sided paired bootstrap, Benjamini--Hochberg correction over the 16 detectors and five rules).",
        "tab:s8-rules-ho", "llcccccccc",
        "\\textbf{Pair} & \\textbf{Detector} & \\textbf{Val} & " + " & ".join(f"\\textbf{{{r}}}" for r in RULE_TEX),
        rows, size="\\scriptsize"))

    # ---- S8.1 value detection by rule
    rows = []
    for sl in SLICES:
        s = sens[(sens.task == "Value detection") & (sens.unit == sl)]
        for sysname in s.system.unique():
            g = s[s.system == sysname].set_index("rule")
            m = re.match(r"Gated \((.+)\)", sysname)
            if m:
                c = CASCADE.match(m[1])
                label = f"Gated: {pretty(c['gate'])}@{float(c['tg']):.2f} / {pretty(c['model'])}"
            else:
                label = "Direct: Baseline"
            fb = "/".join(str(int(g.loc[r, "fallbacks"])) for r in RULES[1:4])
            rows.append([sl if sysname == s.system.unique()[0] else "", label]
                        + [f3(g.loc[r, "test_f1"]) + sig_mark(cmp, "Value detection", sl, sysname, r) for r in RULES]
                        + [f"{fb} of {int(g.loc['label F1', 'n_labels'])}"])
    out.append(table(
        "Value detection: test Macro-$F_1$ over the values of each HO slice under six threshold rules. In each slice, the \\emph{Direct} Baseline and the HO-gated configuration with the highest test score in Section~S4; these two systems are fixed for the comparison of rules and were not selected on validation. $^{+}$/$^{-}$: significantly better/worse than 0.5. Fallbacks: labels for which the recall rule found no threshold meeting the floor and used 0.5, for $\\pi=0.30/0.40/0.50$.",
        "tab:s8-rules-values", "lXcccccccc",
        "\\textbf{Slice} & \\textbf{System} & " + " & ".join(f"\\textbf{{{r}}}" for r in RULE_TEX) + " & \\textbf{Fallbacks}",
        rows, size="\\scriptsize"))

    # ---- S8.1 summary of comparisons
    rows = []
    for task in ["HO detection", "Value detection"]:
        for base in ["fixed 0.5", "Eq. 8, floor 0.40"]:
            for r, rt in zip(RULES, RULE_TEX):
                c = cmp[(cmp.task == task) & (cmp.baseline == base) & (cmp["rule"] == r)]
                if c.empty:
                    continue
                rows.append([task, RULE_TEX[RULES.index(base)], rt, str(len(c)), f3(c.delta.mean()),
                             str(int((c.delta > 0).sum())), str(int((c.p_bh < 0.05).sum())), str(int((c.p_worse_bh < 0.05).sum()))])
    seen = set()
    for r in rows:
        key = r[0]
        if key:
            if key in seen:
                r[0] = ""
            seen.add(key)
    out.append(table(
        "Each threshold rule against a reference rule (the fixed threshold 0.5, or the recall rule with $\\pi=0.40$): number of systems, mean test difference, and number of systems for which the rule scores higher, significantly better, and significantly worse (one-sided paired bootstrap, Benjamini--Hochberg correction within each block).",
        "tab:s8-rules-summary", "llXccccc",
        "\\textbf{Task} & \\textbf{Reference} & \\textbf{Rule} & \\textbf{Systems} & \\textbf{Mean $\\Delta$} & \\textbf{Higher} & \\textbf{Sig.\\ better} & \\textbf{Sig.\\ worse}",
        rows))

    # ---- S8.2 Presence cascades
    out.append(r"""
\subsection{HO detection and Presence cascades}
\label{sec:s8-presence}

The HO detectors and their selection are in Table~\ref{tab:s8-rules-ho} (columns Val and Label $F_1$). Table~\ref{tab:s8-cascades} lists every \emph{Presence}$\rightarrow$HO cascade. In every pair, validation selects a cascade whose gate is the plain \emph{Presence} detector at $\tau_g=0.5$, which lets through about 60\% of the validation sentences.
""")
    rows = []
    for pair in PAIRS:
        c = cand[(cand.family == "Presence cascade") & (cand.unit == pair)].sort_values("val_f1", ascending=False)
        first = True
        for _, r in c.iterrows():
            m = CASCADE.match(r["name"])
            gate, model = pretty(m["gate"]), pretty(m["model"])
            if chosen["presence_cascade"][pair] == r["name"]:
                gate, model = f"\\textbf{{{gate}}}", f"\\textbf{{{model}}}"
            rows.append([short_pair(pair) if first else "", gate, f"{float(m['tg']):.2f}", model, f3(r.val_f1), f3(r.test_f1)])
            first = False
    out.append(longtable(
        "Every \\emph{Presence}$\\rightarrow$HO cascade, sorted by validation Macro-$F_1$ within each pair (per-label $F_1$ thresholds on the HO labels). The selected cascade is in bold.",
        "tab:s8-cascades", "llclcc",
        "\\textbf{Pair} & \\textbf{Presence gate} & \\textbf{$\\tau_g$} & \\textbf{HO model} & \\textbf{Val} & \\textbf{Test}",
        rows))

    # ---- S8.3 value detection
    out.append(r"""
\subsection{HO-gated value detection}
\label{sec:s8-values}

Table~\ref{tab:s8-values-rules} repeats the comparison of main-text Table~4 with the selection redone under each of three rules. Table~\ref{tab:s8-mcnemar} gives the values for which the per-label McNemar test finds a significant difference under the main rule. Table~\ref{tab:s8-thr-direct} gives the ``Thr.\ tuning'' column of main-text Table~8. Tables~\ref{tab:s8-direct} and~\ref{tab:s8-gated} list every candidate.
""")
    rules3 = [("label F1", "Label $F_1$"), ("global F1", "Global $F_1$"), ("fixed 0.5", "Fixed 0.5")]
    rows = []
    cr = sel[sel.part == "C-rules"]
    for rule, rt in rules3:
        for i, sl in enumerate(SLICES):
            r = cr[(cr.unit == sl) & (cr["rule"] == rule)].iloc[0]
            c = CASCADE.match(r.gated)
            sig = "Direct" if r.p_direct_better_bh < 0.05 else "Gated" if r.p_gated_better_bh < 0.05 else "n.s."
            rows.append([rt if i == 0 else "", sl, pretty(r.direct), f3(r.direct_test),
                         f"{pretty(c['gate'])}@{float(c['tg']):.2f} / {pretty(c['model'])}", f3(r.gated_test),
                         f3(r["diff"]), f3(r.lower95), sig])
    out.append(table(
        "\\emph{Direct} vs.\\ HO-gated value detection with the selection on validation redone under each threshold rule (test Macro-$F_1$ over the values of the slice). $\\Delta$: \\emph{Direct} minus gated; LB: one-sided 95\\% lower bound; Sig.: the system that is significantly better after Benjamini--Hochberg correction across the eight slices, or n.s.",
        "tab:s8-values-rules", "llccXcccc",
        "\\textbf{Rule} & \\textbf{Slice} & \\textbf{Direct} & \\textbf{$F_1$} & \\textbf{Gated: gate@$\\tau_c$ / value model} & \\textbf{$F_1$} & \\textbf{$\\Delta$} & \\textbf{LB} & \\textbf{Sig.}",
        rows, size="\\scriptsize"))

    rows = []
    for sl in SLICES:
        r = sel[(sel.part == "C") & (sel.unit == sl)].iloc[0]
        labs = ast.literal_eval(r.mcnemar_sig)
        n = len(ast.literal_eval(r.direct_thresholds))
        rows.append([sl, f"{len(labs)} of {n}", ", ".join(ABBR[x] for x in labs) or "none"])
    out.append(table(
        "Per-label exact McNemar tests on the gold positives, \\emph{Direct} vs.\\ HO-gated (main rule, systems of main-text Table~4), with Benjamini--Hochberg correction within each slice. Abbreviations as in Table~\\ref{tab:abbr-moral-values}.",
        "tab:s8-mcnemar", "lcX", "\\textbf{Slice} & \\textbf{Significant} & \\textbf{Values}", rows))

    thr = pd.read_csv(RESULTS / "threshold-gain-direct.csv")
    rows = []
    for _, r in thr.iterrows():
        labs = ast.literal_eval(r.mcnemar_sig)
        sig = "+" if r.p_bh < 0.05 else "$-$" if r.p_worse_bh < 0.05 else "n.s."
        rows.append([r.slice, pretty(r.model), f3(r.fixed_test), f3(r.tuned_test), f3(r["diff"]), f3(r.lower95), sig,
                     f"{r.median_threshold:.3f}", f"{len(labs)} of {r.n_values}"])
    out.append(table(
        "Threshold tuning for the selected \\emph{Direct} model of each slice: test Macro-$F_1$ at the fixed threshold 0.5 and at the per-label $F_1$ thresholds fitted on validation. $\\Delta$: tuned minus 0.5; LB: one-sided 95\\% lower bound; Sig.: ``+'' significant after Benjamini--Hochberg correction across the eight slices; Median $\\tau$: median tuned threshold over the values of the slice; McNemar: values whose decisions on the sentences that carry them change significantly (exact test, Benjamini--Hochberg within the slice).",
        "tab:s8-thr-direct", "lXccccccc",
        "\\textbf{Slice} & \\textbf{Direct} & \\textbf{0.5} & \\textbf{Tuned} & \\textbf{$\\Delta$} & \\textbf{LB} & \\textbf{Sig.} & \\textbf{Median $\\tau$} & \\textbf{McNemar}",
        rows))

    dmodels = list(dict.fromkeys(cand[cand.family == "Direct"].name))
    rows = []
    for sl in SLICES:
        cells = []
        for m in dmodels:
            r = cand[(cand.family == "Direct") & (cand.unit == sl) & (cand.name == m)].iloc[0]
            cell = f"{f3(r.val_f1)} / {f3(r.test_f1)}"
            cells.append(f"\\textbf{{{cell}}}" if chosen["direct"][sl] == m else cell)
        rows.append([sl] + cells)
    out.append(table(
        "Every \\emph{Direct} candidate: validation / test Macro-$F_1$ over the values of each slice, per-label $F_1$ thresholds. The 19-value models are the same in every slice; only the scored values change. The selected model is in bold.",
        "tab:s8-direct", "l" + "c" * len(dmodels),
        "\\textbf{Slice} & " + " & ".join(f"\\textbf{{{pretty(m)}}}" for m in dmodels), rows, size="\\scriptsize"))

    rows = []
    for sl in SLICES:
        c = cand[(cand.family == "HO-gated") & (cand.unit == sl)].sort_values("val_f1", ascending=False)
        first = True
        for _, r in c.iterrows():
            m = CASCADE.match(r["name"])
            gate, model = pretty(m["gate"]), pretty(m["model"])
            if chosen["ho_gated"][sl] == r["name"]:
                gate, model = f"\\textbf{{{gate}}}", f"\\textbf{{{model}}}"
            rows.append([sl if first else "", gate, f"{float(m['tg']):.2f}", model, f3(r.val_f1), f3(r.test_f1)])
            first = False
    out.append(longtable(
        "Every HO-gated candidate, sorted by validation Macro-$F_1$ within each slice (per-label $F_1$ thresholds on the slice's values). The selected pipeline is in bold.",
        "tab:s8-gated", "llclcc",
        "\\textbf{Slice} & \\textbf{HO gate detector} & \\textbf{$\\tau_c$} & \\textbf{Value model} & \\textbf{Val} & \\textbf{Test}",
        rows))

    # ---- S8.4 ensembles (rebuild_ensembles.py)
    ens = pd.read_csv(RESULTS / "ensembles.csv")
    out.append(r"""
\subsection{Ensembles}
\label{sec:s8-ensembles}

Tables~\ref{tab:s8-ens} and~\ref{tab:s8-hybrid} rebuild the ensembles of main-text Tables~6--8 under the main rule. In each slice, the transformer ensemble starts from the selected \emph{Direct} model and adds other \emph{Direct} models by forward selection on validation (main text, Section~3.7). The pool holds the five \emph{Direct} models of Table~\ref{tab:s8-direct}. Members are combined by soft voting, and every candidate ensemble receives its own per-label $F_1$ thresholds on validation. A candidate is accepted when the one-sided 95\% bootstrap lower bound of its validation gain is at least 0.001 and at least 1\% of the current Macro-$F_1$. The Transformer+LLM ensemble applies the same procedure with majority voting to the binarized transformer ensemble and the Gemma-2-9B-it systems of the slice: zero-shot and few-shot prompting, each with and without the SBERT gate, QLoRA hier, and the LLM-only ensemble where one was formed (Section~S6). A tie counts as a positive vote, so a two-member ensemble predicts the union of its members' labels. Each ensemble is compared once on test with the system it extends, and the Benjamini--Hochberg correction runs across the slices where an ensemble was formed.
""")

    def lb(x):
        return "$<$0.001" if 0 < x < 0.0005 else f3(x)

    def ens_sig(formed, p, p_worse):
        if not formed:
            return "--"
        return "+" if p < 0.05 else "$-$" if p_worse < 0.05 else "n.s."

    def llm_name(n):
        m = re.match(r"^definition(-\d+)?( \+gate)?$", n)
        if m:
            return ("Few-shot" if m[1] else "Zero-shot") + (" + gate" if m[2] else "")
        return {"qlora +gate": "QLoRA hier", "LLM ensemble": "LLM-only ensemble"}[n]

    rows = []
    for _, r in ens.iterrows():
        added = [pretty(m) for m in r.tr_members.split(" + ")[1:]]
        rows.append([r.slice, pretty(r.single), f3(r.single_val), f3(r.single_test), ", ".join(added) or "none",
                     f3(r.tr_val), f3(r.tr_test), f3(r.tr_test - r.single_test) if r.ens_formed else "--",
                     lb(r.ens_lower95) if r.ens_formed else "--",
                     ens_sig(r.ens_formed, r.ens_p_bh, r.ens_p_worse_bh)])
    out.append(table(
        "Transformer soft-voting ensembles (main rule). Single: the selected \\emph{Direct} model; Added: members accepted by forward selection on validation. Validation and test Macro-$F_1$ over the values of the slice. $\\Delta$: ensemble minus single on test; LB: one-sided 95\\% lower bound; Sig.: ``+'' ensemble significantly better, ``$-$'' significantly worse, n.s.\\ not significant, -- no member added.",
        "tab:s8-ens", "llccXccccc",
        "\\textbf{Slice} & \\textbf{Single} & \\textbf{Val} & \\textbf{Test} & \\textbf{Added} & \\textbf{Val} & \\textbf{Test} & \\textbf{$\\Delta$} & \\textbf{LB} & \\textbf{Sig.}",
        rows, size="\\scriptsize"))

    rows = []
    for _, r in ens.iterrows():
        added = [llm_name(m) for m in r.hy_members.split(" + ")[1:]]
        rows.append([r.slice, ", ".join(added) or "none", f3(r.hy_val), f3(r.hy_test),
                     f3(r.hy_test - r.tr_test) if r.hyb_formed else "--",
                     lb(r.hyb_lower95) if r.hyb_formed else "--",
                     ens_sig(r.hyb_formed, r.hyb_p_bh, r.hyb_p_worse_bh), llm_name(r.best_llm), f3(r.best_llm_test)])
    out.append(table(
        "Transformer+LLM ensembles (main rule). Added: LLM systems accepted by forward selection on validation. $\\Delta$: Transformer+LLM minus the transformer ensemble on test; LB and Sig.\\ as in Table~\\ref{tab:s8-ens}. Best LLM: the LLM system with the highest validation Macro-$F_1$, with its test Macro-$F_1$.",
        "tab:s8-hybrid", "lXccccclc",
        "\\textbf{Slice} & \\textbf{Added} & \\textbf{Val} & \\textbf{Test} & \\textbf{$\\Delta$} & \\textbf{LB} & \\textbf{Sig.} & \\textbf{Best LLM} & \\textbf{Test}",
        rows, size="\\scriptsize"))

    rows = []
    for _, r in ens[ens.ens_formed].iterrows():
        labs = ast.literal_eval(r.ens_mcnemar_sig)
        rows.append([r.slice, f"{len(labs)} of {r.n_values}", ", ".join(ABBR[x] for x in labs) or "none"])
    out.append(table(
        "Per-label exact McNemar tests on the gold positives, transformer ensemble vs.\\ single model, in the slices where an ensemble was formed, with Benjamini--Hochberg correction within each slice. Abbreviations as in Table~\\ref{tab:abbr-moral-values}.",
        "tab:s8-ens-mcnemar", "lcX", "\\textbf{Slice} & \\textbf{Significant} & \\textbf{Values}", rows))

    soft = pd.read_csv(RESULTS / "soft-conditioning.csv")
    out.append(r"""
\subsection{Soft uses of the HO probability}
\label{sec:s8-soft}

Table~\ref{tab:s8-soft} details main-text Table~5. All systems use per-label $F_1$ thresholds fitted on validation. The HO prior multiplies the \emph{Direct} probability of each value by $\hat{p}_c^{\alpha}$, where $\hat{p}_c$ is the probability of the slice's category from the HO detector selected in Table~\ref{tab:s8-rules-ho}, and $\alpha$ is chosen on validation from $\{0.25,0.5,1,2\}$. The parent stack and the HO stack are logistic regressions fitted for each value on the log-odds of the \emph{Direct} probability and of the parent HO probability, or of all eight HO probabilities. Their validation scores are out-of-fold (five folds, all the sentences of a text in the same fold), and they are refitted on the whole validation split before being applied once to test. Masked Direct places the \emph{Direct} model behind the gate of the selected hard-gated pipeline, with the same detector and gate threshold, so it differs from that pipeline only in the value model. Benjamini--Hochberg correction runs across the 24 soft entries and, separately, across the eight masked Direct entries of each comparison.
""")

    def soft_sig(r, ref):
        return ("+" if r[f"p_better_than_{ref}_bh"] < 0.05 else
                "$-$" if r[f"p_worse_than_{ref}_bh"] < 0.05 else "n.s.")

    names = {"Direct": "Direct", "hard gate": "Hard-gated", "masked Direct": "Masked Direct",
             "HO prior": "HO prior", "parent stack": "Parent stack", "HO stack": "HO stack"}
    rows = []
    for sl in SLICES:
        for sys_, label in names.items():
            r = soft[(soft.slice == sl) & (soft.system == sys_)].iloc[0]
            if sys_ == "HO prior":
                label += f" ($\\alpha={r.alpha:g}$)"
            row = [sl if sys_ == "Direct" else "", label, f3(r.val_f1), f3(r.test_f1),
                   f3(r.test_precision), f3(r.test_recall)]
            for ref in ("Direct", "hard gate"):
                if sys_ in ("Direct", "hard gate"):
                    row += ["", "", ""]
                else:
                    row += [f3(r[f"diff_vs_{ref}"]), f3(r[f"lower95_vs_{ref}"]), soft_sig(r, ref)]
            rows.append(row)
    out.append("{\\setlength{\\tabcolsep}{4pt}\n" + longtable(
        "Hard and soft uses of the HO probability (main rule). Val and Test: Macro-$F_1$ over the values of the slice (validation scores of the stackers are out-of-fold). P and R: mean precision and recall over the values on test. $\\Delta$: difference on test from the \\emph{Direct} model or from the hard-gated pipeline; LB: one-sided 95\\% lower bound; Sig.: ``+'' significantly higher, ``$-$'' significantly lower, n.s.\\ not significant.",
        "tab:s8-soft", "llcccccccccc",
        "\\textbf{Slice} & \\textbf{System} & \\textbf{Val} & \\textbf{Test} & \\textbf{P} & \\textbf{R} & "
        "\\textbf{$\\Delta$ Direct} & \\textbf{LB} & \\textbf{Sig.} & \\textbf{$\\Delta$ gated} & \\textbf{LB} & \\textbf{Sig.}",
        rows) + "}\n")

    mt = pd.read_csv(RESULTS / "multitask.csv")
    mt_ho = pd.read_csv(RESULTS / "multitask-ho.csv")
    seeds = list(dict.fromkeys(mt.seed))
    units = list(dict.fromkeys(mt.unit))
    mean = mt.groupby("unit")["diff"].mean()
    slice_means = mean.drop("All 19 values")
    n_neg = int((mt["diff"] < 0).sum())
    out.append(rf"""
\subsection{{Multi-task training with HO labels}}
\label{{sec:s8-multitask}}

The multi-task model of main-text Section~3.3.3 fine-tunes one DeBERTa-base encoder with a linear head over 27 labels, the 19 values and the eight HO categories, with the hyperparameters of the \emph{{Direct}} Baseline. Checkpoint selection and early stopping use the validation Macro-$F_1$ over the 27 labels at the threshold 0.5. The model was trained with the three seeds of the \emph{{Direct}} runs ({", ".join(map(str, seeds[:-1]))} and {seeds[-1]}), and each run is compared with the \emph{{Direct}} Baseline of the same seed. Table~\ref{{tab:s8-multitask}} uses only the 19 value outputs. Both systems use per-label $F_1$ thresholds fitted on validation, and Benjamini--Hochberg correction runs across the nine units of each seed. Averaged over the seeds, the difference on test is {f3(mean["All 19 values"])} on all 19 values and between {f3(slice_means.max())} and {f3(slice_means.min())} in the slices, and {n_neg} of the {len(mt)} differences are negative. Table~\ref{{tab:s8-multitask-ho}} scores the eight HO outputs of the same model as HO detectors, with per-label thresholds, against the detectors selected in main-text Table~1.
""")
    rows = []
    for u in units:
        for i, seed in enumerate(seeds):
            r = mt[(mt.unit == u) & (mt.seed == seed)].iloc[0]
            sig = "+" if r.p_mt_better_bh < 0.05 else "$-$" if r.p_mt_worse_bh < 0.05 else "n.s."
            rows.append([u if i == 0 else "", str(seed), f3(r.direct_val), f3(r.mt_val), f3(r.direct_test),
                         f3(r.mt_test), f3(r["diff"]), f3(r.lower95), sig])
    out.append(table(
        "Multi-task model (19 values + 8 HO labels) vs.\\ the \\emph{Direct} Baseline of the same seed, on the 19 value outputs (per-label $F_1$ thresholds). Val and Test: Macro-$F_1$ over all 19 values or over the values of the slice. $\\Delta$: multi-task minus \\emph{Direct} on test; LB: one-sided 95\\% lower bound of $\\Delta$; Sig.: ``+'' multi-task significantly higher, ``$-$'' significantly lower, n.s.\\ not significant.",
        "tab:s8-multitask", "Xcccccccc",
        "\\textbf{Unit} & \\textbf{Seed} & \\textbf{Val Direct} & \\textbf{Val MT} & \\textbf{Test Direct} & "
        "\\textbf{Test MT} & \\textbf{$\\Delta$} & \\textbf{LB} & \\textbf{Sig.}", rows))
    rows = []
    for pair in dict.fromkeys(mt_ho.pair):
        for i, seed in enumerate(seeds):
            r = mt_ho[(mt_ho.pair == pair) & (mt_ho.seed == seed)].iloc[0]
            rows.append([short_pair(pair) if i == 0 else "", str(seed), f3(r.mt_test),
                         pretty(r.detector) if i == 0 else "", f3(r.detector_test) if i == 0 else "",
                         f3(r.mt_test - r.detector_test)])
    out.append(table(
        "The HO outputs of the multi-task model scored as HO detectors (test Macro-$F_1$ over the two poles of each pair, per-label $F_1$ thresholds), against the detector selected for the pair in main-text Table~1, which was trained once with seed 42. $\\Delta$: multi-task minus the selected detector.",
        "tab:s8-multitask-ho", "Xccccc",
        "\\textbf{Pair} & \\textbf{Seed} & \\textbf{Multi-task} & \\textbf{Selected detector} & \\textbf{Test} & "
        "\\textbf{$\\Delta$}", rows))

    (RESULTS / "supplement-s8.tex").write_text("\n".join(out))
    print(f"wrote {RESULTS / 'supplement-s8.tex'}")


if __name__ == "__main__":
    main()
