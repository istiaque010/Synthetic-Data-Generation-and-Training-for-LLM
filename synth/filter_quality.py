"""Multi-stage quality filter (Phase 8) + deduplication/contamination (Phase 9) -> dataset versions.

    python -m synth.filter_quality --calibrate         # once, before any student training: thresholds from real data
    python -m synth.filter_quality --gen main          # writes data/filtered/main_<variant>.jsonl + a manifest

Stages, in order; a record is rejected at the first stage it fails (all flags are also counted):

  1 schema        empty, truncated (teacher hit its limit), malformed characters        (from parse_outputs)
  2 length        80-450 student tokens (the same range as the real seed stories)
  3 ending        the story ends with . ! or ? (optionally followed by a closing quote)
  4 repetition    repeated word 4-grams above the real stories' 99th percentile, or one sentence used 3+ times
  5 language      share of words never seen in 40,000 real "fresh" stories above the real 99th percentile
                  (made-up words, garbled text)
  6 contamination near duplicate (Jaccard >= 0.5) of any validation or test story
  7 dedup         exact or near duplicate of an earlier synthetic story in the pool
  8 copies_real   near duplicate of one of the 5,000 real seed stories (adds nothing new)
  9 judge         (only the `judge` variant) Project 1's small model finds the story more surprising than 90% of
                  real seed stories: a model-based quality score, calibrated on real data, used as an ablation only

Thresholds come from real seed stories (`--calibrate`, data_manifests/filter_calibration.json) and were fixed before
any student was trained. Variants written per pool:

  raw      only empty and contaminated records removed: "unfiltered synthetic" (contamination must go in every
           version or the test would be invalid)
  filt     stages 1-8: the main filtered pool
  nodedup  stages 1-6 (deduplication off)
  judge    stages 1-9
  cand1    stages 1-8 run on candidate 0 of each prompt only (1 candidate per prompt instead of 3)
"""
from __future__ import annotations

import argparse
import collections
import re
import time

import numpy as np

from . import DATA, MANIFESTS, load_json, read_jsonl, save_json, write_jsonl
from . import deduplicate as dd

MIN_TOK, MAX_TOK = 80, 450
ENDING = re.compile(r'[.!?]["\']?$')
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def rep4(text: str) -> float:
    w = dd.words(text)
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g else 1.0 - len(set(g)) / len(g)


def max_sentence_repeat(text: str) -> int:
    s = [x.strip().lower() for x in SENT_SPLIT.split(text) if len(x.split()) >= 3]
    return max(collections.Counter(s).values()) if s else 0


def oov_rate(text: str, vocab) -> float:
    w = dd.words(text)
    return 0.0 if not w else sum(x not in vocab for x in w) / len(w)


def real_vocab():
    c = collections.Counter(w for r in read_jsonl(DATA, "seeds", "fresh.jsonl") for w in dd.words(r["text"]))
    return {w for w, n in c.items() if n >= 2}


def calibrate():
    seeds = read_jsonl(DATA, "seeds", "seeds.jsonl")
    vocab = real_vocab()
    r4 = np.array([rep4(s["text"]) for s in seeds])
    oov = np.array([oov_rate(s["text"], vocab) for s in seeds])
    msr = np.array([max_sentence_repeat(s["text"]) for s in seeds])
    from .judge import judge_losses
    jl = np.array(judge_losses([s["text"] for s in seeds]))
    cal = {"computed_on": "5,000 real seed stories (vocabulary from the 40,000 fresh stories)",
           "rep4_p99": float(np.percentile(r4, 99)), "rep4_p50": float(np.median(r4)),
           "oov_p99": float(np.percentile(oov, 99)), "oov_p50": float(np.median(oov)),
           "max_sentence_repeat_share_ge3": float(np.mean(msr >= 3)),
           "judge_loss_p90": float(np.percentile(jl, 90)), "judge_loss_p50": float(np.median(jl)),
           "real_rejected_by_rules": None}
    # how many REAL stories the rule stages would reject (a sanity check of the thresholds)
    rej = collections.Counter()
    for s in seeds:
        rej[rule_reason({"text": s["text"], "n_tokens": s["n_tokens"], "quality_flags": []}, cal, vocab) or "ok"] += 1
    cal["real_rejected_by_rules"] = dict(rej)
    save_json(cal, MANIFESTS, "filter_calibration.json")
    print(cal)


def rule_reason(rec, cal, vocab):
    if rec["quality_flags"]:
        return "schema:" + rec["quality_flags"][0]
    if not (MIN_TOK <= rec["n_tokens"] <= MAX_TOK):
        return "length"
    if not ENDING.search(rec["text"]):
        return "ending"
    if rep4(rec["text"]) > cal["rep4_p99"] or max_sentence_repeat(rec["text"]) >= 3:
        return "repetition"
    if oov_rate(rec["text"], vocab) > cal["oov_p99"]:
        return "language"
    return None


def run_pipeline(recs, cal, vocab, contam, seeds_index, stages):
    """stages: subset of {'rules','contam','dedup','copies','judge'}; returns (accepted, reasons per record)."""
    reasons = [None] * len(recs)
    timing = {}
    t = time.perf_counter()
    for i, r in enumerate(recs):
        if "rules" in stages:
            reasons[i] = rule_reason(r, cal, vocab)
        elif not r["response"]:
            reasons[i] = "schema:empty"
    timing["rules"] = time.perf_counter() - t
    t = time.perf_counter()
    for i, r in enumerate(recs):
        if reasons[i] is None and contam[r["id"]] >= dd.THRESHOLD:
            reasons[i] = "contamination"
    timing["contamination"] = time.perf_counter() - t
    if "dedup" in stages:
        t = time.perf_counter()
        live = [i for i in range(len(recs)) if reasons[i] is None]
        keep, why = dd.dedup_within([recs[i]["text"] for i in live])
        for i, k, w in zip(live, keep, why):
            if not k:
                reasons[i] = w
        live = [i for i in range(len(recs)) if reasons[i] is None]
        sims = dd.max_sim_against([recs[i]["text"] for i in live], seeds_index)
        for i, s in zip(live, sims):
            if s >= dd.THRESHOLD:
                reasons[i] = "copies_real"
        timing["dedup"] = time.perf_counter() - t
    if "judge" in stages:
        t = time.perf_counter()
        from .judge import judge_losses
        live = [i for i in range(len(recs)) if reasons[i] is None]
        jl = judge_losses([recs[i]["text"] for i in live])
        for i, l in zip(live, jl):
            if l > cal["judge_loss_p90"]:
                reasons[i] = "judge"
        timing["judge"] = time.perf_counter() - t
    acc = [r for r, why in zip(recs, reasons) if why is None]
    return acc, reasons, timing


VARIANTS = {"raw": set(), "filt": {"rules", "dedup"}, "nodedup": {"rules"}, "judge": {"rules", "dedup", "judge"},
            "cand1": {"rules", "dedup"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--gen")
    ap.add_argument("--variants", nargs="+", default=None)
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
        return
    cal = load_json(MANIFESTS, "filter_calibration.json")
    vocab = real_vocab()
    recs = sorted(read_jsonl(DATA, "parsed", f"{args.gen}.jsonl"), key=lambda r: r["id"])
    t = time.perf_counter()
    heldout = [r["text"] for s in ("val", "test_loss", "test_prompt") for r in read_jsonl(DATA, "seeds", f"{s}.jsonl")]
    hidx = dd.build_index(heldout)
    contam = dict(zip([r["id"] for r in recs], dd.max_sim_against([r["text"] for r in recs], hidx)))
    gset = dd.ngram_set(heldout)
    seeds = read_jsonl(DATA, "seeds", "seeds.jsonl")
    seeds_index = dd.build_index([s["text"] for s in seeds])
    t_contam = time.perf_counter() - t
    variants = args.variants or (list(VARIANTS) if args.gen == "main" else ["filt"])
    man = {"gen": args.gen, "n_raw": len(recs), "contamination_check_seconds": t_contam,
           "share_with_13gram_overlap_to_heldout": float(np.mean([dd.shares_ngram(r["text"], gset) for r in recs])),
           "real_seed_share_with_13gram_overlap_to_heldout":
               float(np.mean([dd.shares_ngram(s["text"], gset) for s in seeds])),
           "near_duplicate_of_heldout": int(sum(v >= dd.THRESHOLD for v in contam.values())),
           "max_jaccard_to_heldout_p99": float(np.percentile(list(contam.values()), 99)),
           "variants": {}}
    for v in variants:
        sub = [r for r in recs if r["candidate"] == 0] if v == "cand1" else recs
        acc, reasons, timing = run_pipeline(sub, cal, vocab, contam, seeds_index, VARIANTS[v])
        write_jsonl(acc, DATA, "filtered", f"{args.gen}_{v}.jsonl")
        c = collections.Counter(x or "accepted" for x in reasons)
        man["variants"][v] = {"stages": sorted(VARIANTS[v]), "candidates": len(sub), "accepted": len(acc),
                              "acceptance_rate": len(acc) / max(1, len(sub)),
                              "accepted_tokens": int(sum(r["n_tokens"] for r in acc)),
                              "rejected_by": {k: n for k, n in c.most_common() if k != "accepted"},
                              "seconds": timing}
        print(f"{args.gen}_{v}: {len(acc)}/{len(sub)} accepted ({100 * len(acc) / max(1, len(sub)):.1f}%), "
              f"{man['variants'][v]['accepted_tokens']:,} tokens; rejected {dict(c.most_common())}")
    save_json(man, MANIFESTS, f"filter_{args.gen}.json")


if __name__ == "__main__":
    main()
