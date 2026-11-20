"""Figures (Phases 8-15) -> results/figures/.

    python -m synth.plots
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import FIGURES, MANIFESTS, METRICS, load_json  # noqa: E402

C = {"real": "#7F7F7F", "fresh": "#000000", "filt": "#0072B2", "raw": "#D55E00", "syn": "#CC79A7",
     "abl": "#009E73", "acc": "#0072B2"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, name), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


REASON_ORDER = ["accepted", "schema:truncated", "ending", "length", "repetition", "language", "schema:malformed",
                "near_duplicate", "exact_duplicate", "copies_real", "contamination"]
REASON_LABEL = {"accepted": "accepted", "schema:truncated": "hit token limit", "ending": "no proper ending",
                "length": "too short/long", "repetition": "repetitive", "language": "unusual words",
                "schema:malformed": "odd characters", "near_duplicate": "near duplicate",
                "exact_duplicate": "exact duplicate", "copies_real": "copies a real story",
                "contamination": "matches test story"}
REASON_COL = ["#0072B2", "#E69F00", "#F0E442", "#56B4E9", "#D55E00", "#CC79A7", "#999999", "#009E73", "#004D40",
              "#882255", "#000000"]


def fig1_funnel():
    rows = []
    for gen in ("main", "lowT", "plain"):
        p = os.path.join(MANIFESTS, f"filter_{gen}.json")
        if os.path.exists(p):
            v = load_json(p)["variants"]["filt"]
            rows.append((gen, v))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 2.9))
    labels = {"main": "main (T 1.0)", "lowT": "low temperature (T 0.6)", "plain": "no prompt diversity"}
    labelled = set()
    for y, (gen, v) in enumerate(rows):
        left = 0
        n = v["candidates"]
        counts = {"accepted": v["accepted"], **v["rejected_by"]}
        for k, col in zip(REASON_ORDER, REASON_COL):
            w = counts.get(k, 0) / n
            if w:
                ax.barh(y, w, left=left, color=col, edgecolor="white", linewidth=0.8,
                        label=None if k in labelled else REASON_LABEL[k])
                labelled.add(k)
                left += w
        ax.text(1.01, y, f"{100 * v['accepted'] / n:.1f}% kept", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [labels[g] for g, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the 15,000 teacher candidates")
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        if ll and ll not in seen:
            seen[ll] = hh
    ax.legend(seen.values(), seen.keys(), ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              frameon=False)
    ax.set_title("What the quality filter removed")
    save(fig, "fig1_filter_funnel.png")


def fig2_diversity():
    p = os.path.join(METRICS, "dataset_report.json")
    if not os.path.exists(p):
        return
    d = load_json(p)
    names = [("real_seeds", "real seeds"), ("real_fresh", "real (other)"), ("main_raw", "main, unfiltered"),
             ("main_filt", "main, filtered"), ("lowT_filt", "low temperature"), ("plain_filt", "no prompt diversity")]
    names = [(k, l) for k, l in names if k in d]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    cols = [C["real"], C["real"], C["raw"], C["filt"], C["abl"], C["syn"]]
    for ax, key, title in ((axs[0], "distinct_2", "distinct word pairs (2,000 stories)"),
                           (axs[1], "top5_name_share", "share of characters named by the top-5 names")):
        vals = [d[k][key] for k, _ in names]
        ax.barh(range(len(names)), vals, color=cols[:len(names)], height=0.65)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        ax.set_yticks(range(len(names)), [l for _, l in names])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
    axs[1].set_yticks([])
    save(fig, "fig2_diversity.png")


def table():
    p = os.path.join(METRICS, "summary_table.json")
    return {r["system"]: r for r in load_json(p)} if os.path.exists(p) else {}


LABEL = {"real_only": "real only (5k stories)", "real_fresh": "new real data (reference)", "syn_only": "synthetic only",
         "mix50_raw": "50% unfiltered synthetic", "mix25_filt": "25% filtered synthetic",
         "mix50_filt": "50% filtered synthetic", "mix75_filt": "75% filtered synthetic",
         "mix50_lowT": "50%, teacher T 0.6", "mix50_plain": "50%, no prompt diversity",
         "mix50_cand1": "50%, 1 candidate/prompt", "mix50_nodedup": "50%, no dedup", "mix50_judge": "50%, + judge filter"}


def fig3_test_loss():
    t = table()
    order = [s for s in ("real_fresh", "mix25_filt", "mix50_filt", "mix75_filt", "mix50_raw", "real_only", "syn_only")
             if s in t]
    if not order:
        return
    order.sort(key=lambda s: t[s]["test_loss"])
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, s in enumerate(order):
        col = C["fresh"] if s == "real_fresh" else C["real"] if s == "real_only" else C["raw"] if "raw" in s \
            else C["syn"] if s == "syn_only" else C["filt"]
        ax.scatter(t[s]["per_seed"], [i] * len(t[s]["per_seed"]), color=col, s=18, alpha=0.6, zorder=3)
        ax.scatter([t[s]["test_loss"]], [i], color=col, s=70, marker="|", linewidths=3, zorder=4)
        ax.text(max(t[s]["per_seed"]) + 0.004, i, f"{t[s]['test_loss']:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(order)), [LABEL[s] for s in order])
    ax.invert_yaxis()
    ax.set_xlabel("test loss on 2,000 held-out real stories (nats/token, lower is better; dots = seeds)")
    ax.grid(axis="x", color="#e6e6e6")
    ax.set_title("Same 6M-token budget, different training data")
    save(fig, "fig3_test_loss.png")


def fig4_ratio():
    t = table()
    pts = [(0, "real_only"), (25, "mix25_filt"), (50, "mix50_filt"), (75, "mix75_filt"), (100, "syn_only")]
    pts = [(x, s) for x, s in pts if s in t]
    if len(pts) < 3:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = [x for x, _ in pts]
    ax.plot(xs, [t[s]["test_loss"] for _, s in pts], color=C["filt"], lw=2, marker="o", label="real + filtered synthetic")
    for x, s in pts:
        ax.scatter([x] * len(t[s]["per_seed"]), t[s]["per_seed"], color=C["filt"], s=12, alpha=0.5)
    if "mix50_raw" in t:
        ax.scatter([50], [t["mix50_raw"]["test_loss"]], color=C["raw"], marker="s", s=50, zorder=4,
                   label="50% unfiltered synthetic")
    if "real_fresh" in t:
        ax.axhline(t["real_fresh"]["test_loss"], color=C["fresh"], ls="--", lw=1)
        ax.text(1, t["real_fresh"]["test_loss"], " 6M tokens of new real stories", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xlabel("synthetic share of the 6M training tokens (%)")
    ax.set_ylabel("test loss (nats/token)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#e6e6e6")
    save(fig, "fig4_mixture_ratio.png")


def fig5_val_curves():
    runs = [("real_only_s0", C["real"], "real only"), ("mix50_filt_s0", C["filt"], "50% filtered synthetic"),
            ("syn_only_s0", C["syn"], "synthetic only"), ("real_fresh_s0", C["fresh"], "new real data")]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    any_ = False
    for r, col, lab in runs:
        p = os.path.join(METRICS, "train", r, "log.csv")
        if not os.path.exists(p):
            continue
        rows = [x for x in csv.DictReader(open(p)) if x["val_loss"]]
        ax.plot([int(x["tokens"]) / 1e6 for x in rows], [float(x["val_loss"]) for x in rows], color=col, marker="o",
                ms=3, lw=1.8, label=lab)
        any_ = True
    if not any_:
        plt.close(fig)
        return
    ax.set_xlabel("training tokens seen (millions)")
    ax.set_ylabel("validation loss (real stories)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#eeeeee")
    ax.set_title("No run overfits within 6M tokens; the gaps open early and persist")
    save(fig, "fig5_validation_curves.png")


def fig6_ablations():
    p = os.path.join(METRICS, "comparisons.json")
    if not os.path.exists(p):
        return
    comps = [c for c in load_json(p)["pairs"] if c["kind"] == "ablation"]
    if not comps:
        return
    fig, ax = plt.subplots(figsize=(7, 2.9))
    ref = comps[0]["ref_seed_range"]
    t = table()
    base = t["mix50_filt"]["per_seed"]
    s0 = [x for x in load_json(METRICS, "eval", "mix50_filt_s0.json")["per_story"]]
    lo, hi = min(base) - base[0], max(base) - base[0]
    ax.axvspan(lo, hi, color="#dddddd", alpha=0.7, label="range of the 3 seeds of the main run")
    for i, c in enumerate(comps):
        d, ci = c["vs_ref_seed0_diff"], c["vs_ref_seed0_ci95"]
        ax.errorbar(d, i, xerr=[[d - ci[0]], [ci[1] - d]], fmt="o", color=C["abl"] if c["counts"] else "#555555",
                    capsize=3)
        ax.text(ci[1] + 0.001, i, f"{d:+.4f}", va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(comps)), [LABEL[c["a"]] for c in comps])
    ax.invert_yaxis()
    ax.set_xlabel("test loss: ablation − main run seed 0 (nats/token, 95% CI; < 0 = better)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Ablations (seed 0) against the main 50% filtered run; green = counts under the rule", fontsize=9)
    save(fig, "fig6_ablations.png")


def fig7_generation():
    t = table()
    order = [s for s in ("real continuations", "real_fresh", "real_only", "mix25_filt", "mix50_filt", "mix75_filt",
                         "syn_only", "mix50_raw") if s in t and "scorer_nll" in t[s]]
    if len(order) < 3:
        return
    lab = dict(LABEL, **{"real continuations": "real continuations"})
    order = [s for s in order if s != "real continuations"]     # real text is far off-scale; quoted in the title
    real = t.get("real continuations", {})
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, key, title in ((axs[0], "scorer_nll", "scorer loss (SmolLM2; lower = more fluent)"),
                           (axs[1], "distinct_2", "distinct word pairs (variety)"),
                           (axs[2], "ended", "share that finish the story")):
        vals = [t[s].get(key, 0) for s in order]
        cols = [C["real"] if "real" in s else C["raw"] if "raw" in s else C["syn"] if s == "syn_only" else C["filt"]
                for s in order]
        ax.hlines(range(len(order)), min(vals) - 0.02 * (max(vals) - min(vals) + 1e-9), vals, color="#e0e0e0", lw=1)
        ax.scatter(vals, range(len(order)), color=cols, s=40, zorder=3)
        pad = (max(vals) - min(vals)) * 0.35
        ax.set_xlim(min(vals) - pad * 0.3, max(vals) + pad)
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v:.3f}", va="center", fontsize=7.5)
        rv = f"\nreal continuations: {real[key]:.3f}" if key in real else ""
        ax.set_title(title + rv, fontsize=9)
    axs[0].set_yticks(range(len(order)), [lab[s] for s in order])
    axs[0].invert_yaxis()
    save(fig, "fig7_generations.png")


def main():
    for f in (fig1_funnel, fig2_diversity, fig3_test_loss, fig4_ratio, fig5_val_curves, fig6_ablations,
              fig7_generation):
        f()
    print("figures ->", FIGURES, sorted(os.listdir(FIGURES)) if os.path.exists(FIGURES) else [])


if __name__ == "__main__":
    main()
