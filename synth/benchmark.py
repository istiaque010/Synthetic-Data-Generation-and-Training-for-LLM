"""Efficiency and cost of the whole pipeline (Phase 16).

    python -m synth.benchmark

Teacher and student run on the same CPU here (no GPU); both are reported separately:
  teacher   per pool: candidates, teacher tokens, generation seconds, teacher tokens/s, accepted stories per hour of
            generation (after the main filter), student-tokens accepted per teacher-second
  filter    seconds per stage (rules, contamination, dedup, judge) from the filter manifests
  student   per run: training seconds, tokens/s, peak memory, parameters
  value     quality gain per unit of cost: (test loss real_only - test loss mix50_filt) per 1M teacher tokens and per
            teacher CPU-hour, next to what the same gain costs with real data (real_fresh)
Writes results/metrics/benchmark.json.
"""
from __future__ import annotations

import os

import numpy as np

from . import MANIFESTS, METRICS, load_json, save_json


def main():
    out = {"hardware": load_json(METRICS, "env.json") if os.path.exists(os.path.join(METRICS, "env.json")) else {},
           "teacher": {}, "filter": {}, "student": {}}
    for gen in ("main", "lowT", "plain"):
        p = os.path.join(MANIFESTS, f"raw_{gen}.json")
        if not os.path.exists(p):
            continue
        r = load_json(p)
        f = load_json(MANIFESTS, f"filter_{gen}.json")
        acc = f["variants"]["filt"]
        hrs = r["generation_seconds"] / 3600
        out["teacher"][gen] = {"candidates": r["candidates"], "teacher_tokens": r["teacher_tokens"],
                               "generation_seconds": r["generation_seconds"],
                               "teacher_tokens_per_s": r["teacher_tokens"] / r["generation_seconds"],
                               "hit_limit": r["hit_limit"], "accepted": acc["accepted"],
                               "accepted_per_hour": acc["accepted"] / hrs,
                               "accepted_student_tokens": acc["accepted_tokens"],
                               "accepted_student_tokens_per_teacher_s": acc["accepted_tokens"] / r["generation_seconds"]}
        out["filter"][gen] = {v: d["seconds"] for v, d in f["variants"].items()}
        out["filter"][gen]["contamination_index"] = f["contamination_check_seconds"]
    tdir = os.path.join(METRICS, "train")
    pilot_sec = 0.0
    for run in sorted(os.listdir(tdir)):
        p = os.path.join(tdir, run, "summary.json")
        if run.startswith("lr") and os.path.exists(p):      # learning-rate pilot: counted separately
            pilot_sec += load_json(p)["train_seconds"]
            continue
        if os.path.exists(p):
            s = load_json(p)
            out["student"][run] = {"train_seconds": s["train_seconds"], "tokens": s["tokens"],
                                   "tokens_per_s": s["tokens"] / s["train_seconds"], "peak_rss_mb": s["peak_rss_mb"],
                                   "params": s["params"]["total"], "best_step": s["best_step"]}
    out["lr_pilot_cpu_hours"] = pilot_sec / 3600
    st = os.path.join(METRICS, "summary_table.json")
    if os.path.exists(st) and "main" in out["teacher"]:
        t = {r["system"]: r for r in load_json(st)}
        if all(k in t for k in ("real_only", "mix50_filt", "real_fresh")):
            gain = t["real_only"]["test_loss"] - t["mix50_filt"]["test_loss"]
            gain_fresh = t["real_only"]["test_loss"] - t["real_fresh"]["test_loss"]
            tm = out["teacher"]["main"]
            out["value"] = {"gain_mix50_filt_nats": gain, "gain_real_fresh_nats": gain_fresh,
                            "gain_per_1M_teacher_tokens": gain / (tm["teacher_tokens"] / 1e6),
                            "gain_per_teacher_cpu_hour": gain / (tm["generation_seconds"] / 3600),
                            "share_of_fresh_real_gain": gain / gain_fresh if gain_fresh else None,
                            "note": "mix50_filt = 3M tokens of the 5k real seeds (3.1 epochs) + 3M filtered synthetic tokens; real_fresh = 6M tokens of other real stories (no repetition, ~31k stories)"}
    save_json(out, METRICS, "benchmark.json")
    for k in ("teacher", "filter"):
        for g, v in out[k].items():
            print(k, g, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()}
                  if isinstance(v, dict) else v)
    sec = [v["train_seconds"] for v in out["student"].values()]
    if sec:
        print("student runs", len(sec), "median train min", round(float(np.median(sec)) / 60, 1),
              "total CPU-hours", round(sum(sec) / 3600, 2))
    if "value" in out:
        print(out["value"])


if __name__ == "__main__":
    main()
