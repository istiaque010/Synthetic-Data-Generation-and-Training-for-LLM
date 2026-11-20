"""Dataset quality and diversity report (Phase 10), written before any student is trained.

    python -m synth.dataset_report

For the real seed stories, the fresh real stories (reference) and every synthetic dataset version:
  stories, tokens, acceptance rate (from the filter manifest), length distribution (student tokens),
  distinct-1/2 and vocabulary size on a fixed-size random sample (2,000 stories; distinct-n falls as more text is
  pooled, so every set is measured on the same amount), coverage of the real vocabulary, repeated-4-gram rate,
  protagonist-name concentration (the word after "named": the teacher's favourite names show up here),
  template balance, exact/near duplicate counts, and copies of real seed stories.
Writes results/metrics/dataset_report.json.
"""
from __future__ import annotations

import collections
import os
import random
import re

import numpy as np

from . import DATA, MANIFESTS, METRICS, load_json, read_jsonl, save_json
from . import deduplicate as dd
from .filter_quality import real_vocab, rep4

NAMED = re.compile(r"\bnamed ([A-Z][a-z]+)")


def describe(texts, ntok, vocab, rng_seed=0, n_sample=2000):
    rng = random.Random(rng_seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    samp = [texts[i] for i in idx[:n_sample]]
    words = [w for t in samp for w in dd.words(t)]
    bi = [tuple(dd.words(t)[i:i + 2]) for t in samp for i in range(len(dd.words(t)) - 1)]
    names = collections.Counter(m for t in texts for m in NAMED.findall(t))
    n_named = sum(names.values())
    types = set(words)
    return {"stories": len(texts), "tokens": int(sum(ntok)),
            "len_mean": float(np.mean(ntok)), "len_p5": float(np.percentile(ntok, 5)),
            "len_p50": float(np.median(ntok)), "len_p95": float(np.percentile(ntok, 95)),
            "distinct_1": len(types) / len(words), "distinct_2": len(set(bi)) / len(bi),
            "vocab_types_2000": len(types), "real_vocab_coverage_2000": len(types & vocab) / len(vocab),
            "rep4_mean": float(np.mean([rep4(t) for t in samp])),
            "distinct_names": len(names), "top_name": names.most_common(1)[0] if names else None,
            "top5_name_share": sum(c for _, c in names.most_common(5)) / max(1, n_named)}


def main():
    vocab = real_vocab()
    out = {}
    for name in ("seeds", "fresh"):
        rows = read_jsonl(DATA, "seeds", f"{name}.jsonl")
        if name == "fresh":
            rows = rows[:5000]
        out["real_" + name] = describe([r["text"] for r in rows], [r["n_tokens"] for r in rows], vocab)
    for f in sorted(os.listdir(os.path.join(DATA, "filtered"))):
        if f.startswith("pilot"):
            continue
        name = f[:-6]
        rows = read_jsonl(DATA, "filtered", f)
        d = describe([r["text"] for r in rows], [r["n_tokens"] for r in rows], vocab)
        gen, var = name.split("_", 1)
        man = load_json(MANIFESTS, f"filter_{gen}.json")
        v = man["variants"][var]
        d.update({"acceptance_rate": v["acceptance_rate"], "candidates": v["candidates"],
                  "rejected_by": v["rejected_by"],
                  "templates": dict(collections.Counter(r["template_id"] for r in rows))})
        out[name] = d
    for gen in ("main", "lowT", "plain"):
        p = os.path.join(MANIFESTS, f"filter_{gen}.json")
        if os.path.exists(p):
            m = load_json(p)
            out[f"contamination_{gen}"] = {k: m[k] for k in m if k not in ("variants",)}
    save_json(out, METRICS, "dataset_report.json")
    keys = ["stories", "tokens", "acceptance_rate", "len_p50", "distinct_2", "real_vocab_coverage_2000", "rep4_mean",
            "distinct_names", "top5_name_share"]
    for n, d in out.items():
        if "stories" in d:
            print(f"{n:>14} " + " ".join(f"{k}={d[k]:.3f}" if isinstance(d.get(k), float) else f"{k}={d.get(k)}"
                                         for k in keys))


if __name__ == "__main__":
    main()
