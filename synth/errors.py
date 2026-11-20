"""Error analysis and failure modes of student generations (Phase 15).

    python -m synth.errors --stage auto        # automatic measures for every evaluated system
    python -m synth.errors --stage sample      # blind review sample (systems hidden, order shuffled)
    (label results/error_review/blind.jsonl -> results/error_review/labels_ai.json; AI labels, disclosed as such)
    python -m synth.errors --stage count       # un-blind and count the review labels per system

Automatic measures over the 300 continuations of each run (seed-averaged per system):
  repetitive, unfinished, copied (from the evaluation), plus
  moral_tag      share containing "The moral of the story" (a phrase the teacher uses far more often than real
                 TinyStories: over-fitting to teacher phrasing)
  name_drift     for prompts that introduce a named character ("... named Ben."), share of continuations that never
                 mention that name again (prompt mismatch)
  top5_names     share of all "named X" introductions taken by the five most common names (stylistic collapse)
Real continuations of the same prompts are the reference row.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re

import numpy as np

from . import DATA, METRICS, RESULTS, SAMPLES, load_config, load_json, read_jsonl, save_json, write_jsonl
from .compare import runs_of

NAMED = re.compile(r"\bnamed ([A-Z][a-z]+)")
MORAL = re.compile(r"moral of (the|this) story", re.I)
REVIEW = os.path.join(RESULTS, "error_review")
REVIEW_SYSTEMS = ["real_only", "mix50_raw", "mix50_filt", "syn_only", "real_fresh"]
N_REVIEW = 20


def auto_measures(prompts, conts):
    named_prompts = [(i, NAMED.findall(p)) for i, p in enumerate(prompts)]
    drift = [not any(n in conts[i] for n in names) for i, names in named_prompts if names]
    names = collections.Counter(n for c in conts for n in NAMED.findall(c))
    tot = sum(names.values())
    return {"moral_tag": float(np.mean([bool(MORAL.search(c)) for c in conts])),
            "name_drift": float(np.mean(drift)) if drift else None, "named_prompts": len(drift),
            "top5_names": sum(c for _, c in names.most_common(5)) / max(1, tot),
            "top_names": names.most_common(5)}


def stage_auto():
    tp = read_jsonl(DATA, "seeds", "test_prompt.jsonl")
    prompts = [r["opening"] for r in tp]
    out = {"real continuations": auto_measures(prompts, [r["text"][len(r["opening"]):] for r in tp])}
    for s in load_config("systems.yaml")["systems"]:
        per = []
        for r in runs_of(s):
            rows = read_jsonl(SAMPLES, f"{r}.jsonl")
            m = auto_measures(prompts, [x["continuation"] for x in rows])
            e = load_json(METRICS, "eval", f"{r}.json")["generation"]
            m.update({"repetitive": e["repetitive"], "unfinished": 1 - e["ended"], "copied": e["copied"]})
            per.append(m)
        if per:
            out[s] = {k: float(np.mean([p[k] for p in per])) for k in per[0] if isinstance(per[0][k], float)}
            out[s]["top_names_seed0"] = per[0]["top_names"]
    # teacher phrasing in the training data itself
    for name, path in (("data:real_seeds", ("seeds", "seeds.jsonl")), ("data:main_filt", ("filtered", "main_filt.jsonl"))):
        texts = [r["text"] for r in read_jsonl(DATA, *path)]
        out[name] = {"moral_tag": float(np.mean([bool(MORAL.search(t)) for t in texts]))}
    save_json(out, METRICS, "errors_auto.json")
    for k, v in out.items():
        print(f"{k:>20}", {a: round(b, 3) for a, b in v.items() if isinstance(b, float)})


def stage_sample():
    tp = read_jsonl(DATA, "seeds", "test_prompt.jsonl")
    rng = random.Random(77)
    pick = sorted(rng.sample(range(len(tp)), N_REVIEW))
    items = []
    for s in REVIEW_SYSTEMS:
        rows = read_jsonl(SAMPLES, f"{s}_s0.jsonl")
        for i in pick:
            items.append({"system": s, "prompt": rows[i]["prompt"], "continuation": rows[i]["continuation"],
                          "ended": rows[i]["ended"], "prompt_index": i})
    for i in pick:
        r = tp[i]
        items.append({"system": "real", "prompt": r["opening"], "continuation": r["text"][len(r["opening"]):].strip(),
                      "ended": True, "prompt_index": i})
    rng.shuffle(items)
    blind, key = [], []
    for k, it in enumerate(items):
        blind.append({"n": k, "prompt": it["prompt"], "continuation": it["continuation"]})
        key.append({"n": k, "system": it["system"], "prompt_index": it["prompt_index"], "ended": it["ended"]})
    write_jsonl(blind, REVIEW, "blind.jsonl")
    write_jsonl(key, REVIEW, "key.jsonl")
    print(f"{len(blind)} blind items -> {REVIEW}")


def stage_count():
    key = {r["n"]: r for r in read_jsonl(REVIEW, "key.jsonl")}
    lab = load_json(REVIEW, "labels_ai.json")["labels"]
    by = collections.defaultdict(list)
    for l in lab:
        by[key[l["n"]]["system"]].append(l)
    out = {}
    for s, ls in by.items():
        issues = collections.Counter(i for l in ls for i in l["issues"] if i != "none")
        out[s] = {"n": len(ls), "coherence_mean": float(np.mean([l["coherence"] for l in ls])),
                  "relevance_mean": float(np.mean([l["relevance"] for l in ls])),
                  "issues": dict(issues.most_common())}
    save_json({"annotator": "AI (Claude), blind to system; not human", "systems": out}, METRICS, "errors_review.json")
    for s, v in out.items():
        print(f"{s:>12} coherence {v['coherence_mean']:.2f} relevance {v['relevance_mean']:.2f} {v['issues']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["auto", "sample", "count"], required=True)
    a = ap.parse_args()
    {"auto": stage_auto, "sample": stage_sample, "count": stage_count}[a.stage]()


if __name__ == "__main__":
    main()
