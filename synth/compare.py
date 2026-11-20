"""Tables and paired comparisons between students (Phases 13 and 14).

    python -m synth.compare

A *system* is a mixture (+ size); its runs are the seeds evaluated (e.g. mix50_filt_s0/s1/s2).
Test loss of a system = total nats / total tokens over the 2,000 test stories, per seed; the table shows the mean
and the spread over seeds. Paired comparison A vs B: per story, the loss averaged over each system's seeds, then a
bootstrap over stories (10,000 resamples, fixed seed) of the token-weighted difference
    sum_stories(nats_A - nats_B) / sum_stories(tokens)
(negative = A better). The same on the test stories without non-ASCII characters (robustness), and for the
generation metrics per prompt (scorer loss, copied). Pairs: configs/systems.yaml. Ablation rule (hypotheses.md):
an ablation counts only if its interval against the reference seed excludes 0 and its loss lies outside the range
of the reference's seeds. Also reported (added after training, stricter): seed_pairs_a_better = share of all
seed pairings (3 x 3 for main systems) in which A has the lower test loss. Writes results/metrics/comparisons.json and results/metrics/summary_table.json.
"""
from __future__ import annotations

import os
import re

import numpy as np

from . import DATA, METRICS, load_config, load_json, read_jsonl, save_json

NONASCII = re.compile(r"[^\x20-\x7e]")


def runs_of(system):
    d = os.path.join(METRICS, "eval")
    pat = re.compile(rf"^{re.escape(system)}_s(\d+)\.json$")
    return sorted(f[:-5] for f in os.listdir(d) if pat.match(f))


def load(system):
    return [load_json(METRICS, "eval", f"{r}.json") for r in runs_of(system)]


def story_arrays(evals):
    nll = np.mean([[a for a, _ in e["per_story"]] for e in evals], axis=0)
    n = np.array([b for _, b in evals[0]["per_story"]], dtype=float)
    return nll, n


def boot_ratio(da, n, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(da), size=(n_boot, len(da)))
    b = da[idx].sum(1) / n[idx].sum(1)
    return float(da.sum() / n.sum()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def boot_mean(d, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(d.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def summary(system):
    ev = load(system)
    if not ev:
        return None
    tl = [e["test_loss"] for e in ev]
    out = {"system": system, "seeds": len(ev), "test_loss": float(np.mean(tl)), "test_loss_sd": float(np.std(tl)),
           "test_loss_min": float(min(tl)), "test_loss_max": float(max(tl)), "test_ppl": float(np.exp(np.mean(tl))),
           "per_seed": tl}
    if all("generation" in e for e in ev):
        for k in ev[0]["generation"]:
            out[k] = float(np.mean([e["generation"][k] for e in ev]))
    return out


def pair(a, b, kind, ascii_mask):
    ea, eb = load(a), load(b)
    if not ea or not eb:
        return None
    na, n = story_arrays(ea)
    nb, _ = story_arrays(eb)
    d = na - nb
    res = {"a": a, "b": b, "kind": kind, "seeds_a": len(ea), "seeds_b": len(eb)}
    res["test_loss_diff"], res["ci95"] = boot_ratio(d, n)
    res["ascii_only_diff"], res["ascii_only_ci95"] = boot_ratio(d[ascii_mask], n[ascii_mask])
    # seed-level check (added after training, stricter than the pre-registered story bootstrap, which averages
    # seeds first and so ignores seed-to-seed variation): share of all seed pairings in which A has lower loss
    la_s, lb_s = [e["test_loss"] for e in ea], [e["test_loss"] for e in eb]
    res["seed_pairs_a_better"] = float(np.mean([x < y for x in la_s for y in lb_s]))
    if kind == "ablation":
        e0 = [e for e in eb if e["run"].endswith("_s0")][0]
        ns0 = np.array([x for x, _ in e0["per_story"]])
        res["vs_ref_seed0_diff"], res["vs_ref_seed0_ci95"] = boot_ratio(na - ns0, n)
        rng_b = [min(e["test_loss"] for e in eb), max(e["test_loss"] for e in eb)]
        la = float(np.mean([e["test_loss"] for e in ea]))
        res["ref_seed_range"] = rng_b
        res["outside_seed_range"] = not (rng_b[0] <= la <= rng_b[1])
        res["counts"] = (res["outside_seed_range"] and
                         (res["vs_ref_seed0_ci95"][1] < 0 or res["vs_ref_seed0_ci95"][0] > 0))
    if all("gen_per_prompt" in e for e in ea + eb):
        for m in ("scorer_nll", "copy_jaccard"):
            pa = np.mean([[p[m] for p in e["gen_per_prompt"]] for e in ea], axis=0)
            pb = np.mean([[p[m] for p in e["gen_per_prompt"]] for e in eb], axis=0)
            res[f"{m}_diff"], res[f"{m}_ci95"] = boot_mean(pa - pb)
        ca = np.mean([[p["copy_jaccard"] >= 0.5 for p in e["gen_per_prompt"]] for e in ea], axis=0)
        cb = np.mean([[p["copy_jaccard"] >= 0.5 for p in e["gen_per_prompt"]] for e in eb], axis=0)
        res["copied_diff"], res["copied_ci95"] = boot_mean(ca - cb)
    return res


def main():
    cfg = load_config("systems.yaml")
    test = read_jsonl(DATA, "seeds", "test_loss.jsonl")
    ascii_mask = np.array([not NONASCII.search(r["text"]) for r in test])
    table = [s for s in (summary(x) for x in cfg["systems"]) if s]
    real = os.path.join(METRICS, "eval", "real.json")
    if os.path.exists(real):
        table.append({"system": "real continuations", **load_json(real)["generation"]})
    save_json(table, METRICS, "summary_table.json")
    comps = [c for c in (pair(a, b, k, ascii_mask) for a, b, k in cfg["pairs"]) if c]
    save_json({"n_test_stories": len(test), "ascii_only_stories": int(ascii_mask.sum()), "pairs": comps},
              METRICS, "comparisons.json")
    for s in table:
        if "test_loss" in s:
            print(f"{s['system']:>18} loss {s['test_loss']:.4f} ±{s['test_loss_sd']:.4f} (n={s['seeds']}) "
                  f"ppl {s['test_ppl']:.2f}  scorer {s.get('scorer_nll', float('nan')):.3f} "
                  f"d2 {s.get('distinct_2', float('nan')):.3f} rep {s.get('repetitive', float('nan')):.3f} "
                  f"copied {s.get('copied', float('nan')):.3f} ended {s.get('ended', float('nan')):.3f}")
    for c in comps:
        extra = f"  counts={c['counts']}" if c["kind"] == "ablation" else ""
        print(f"{c['a']:>18} - {c['b']:<14} {c['test_loss_diff']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]"
              f"  ascii {c['ascii_only_diff']:+.4f}  seedpairs A<B {c['seed_pairs_a_better']:.2f}{extra}")


if __name__ == "__main__":
    main()
