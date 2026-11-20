"""Training mixtures at a fixed token budget (Phase 11).

    python -m synth.build_mixtures --mixture mix50_filt --seed 0

Every student sees exactly the same number of training tokens (the budget in configs/student.yaml). A mixture says
what share of those tokens is real and which synthetic pool supplies the rest:

    tokens_real = round(real_share x budget),  tokens_syn = budget - tokens_real

Each story is one unit <BOS> story <EOS> (Project 1 tokenizer). A pool smaller than its quota is repeated in epochs,
reshuffled each epoch; the last story of a quota may be cut. Real and synthetic units are then shuffled together
(story level) and concatenated into one token stream, data/mixtures/<mixture>_s<seed>.bin (uint16). The seed sets
both shuffles, so the three training seeds also see different orders. The manifest records exact token counts,
stories, unique stories and epochs per source, so "same number of tokens" is checkable (same number of *stories*
would not be: synthetic and real stories differ in length).
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np

from . import DATA, MANIFESTS, load_config, load_tokenizer, read_jsonl, save_json


def units(texts, tok):
    return [np.array(tok.encode(t, add_bos=True, add_eos=True), dtype=np.uint16) for t in texts]


def fill(pool, quota, rng):
    """Stories from `pool` (list of arrays) until `quota` tokens, epoch by epoch, reshuffled each epoch."""
    out, n, epochs, used = [], 0, 0, set()
    while n < quota:
        order = list(range(len(pool)))
        rng.shuffle(order)
        epochs += 1
        for i in order:
            u = pool[i]
            if n + len(u) > quota:
                u = u[: quota - n]
            out.append(u)
            used.add(i)
            n += len(u)
            if n >= quota:
                break
    full_epochs = (sum(len(u) for u in out)) / max(1, sum(len(u) for u in pool))
    return out, {"tokens": int(n), "stories": len(out), "unique_stories": len(used), "pool_stories": len(pool),
                 "pool_tokens": int(sum(len(u) for u in pool)), "epochs": round(full_epochs, 3)}


def source_texts(name):
    if name == "real":
        return [r["text"] for r in read_jsonl(DATA, "seeds", "seeds.jsonl")]
    if name == "fresh":
        return [r["text"] for r in read_jsonl(DATA, "seeds", "fresh.jsonl")]
    return [r["text"] for r in read_jsonl(DATA, "filtered", f"{name}.jsonl")]


def build(mixture: str, seed: int, tok=None):
    cfg = load_config("student.yaml")
    mix = load_config("mixtures.yaml")["mixtures"][mixture]
    budget = int(cfg["budget_tokens"]) + 1  # +1 so the last target exists
    tok = tok or load_tokenizer()
    rng = random.Random(1000 * seed + 7)
    q_real = round(mix["real_share"] * budget)
    parts, man = [], {"mixture": mixture, "seed": seed, "budget_tokens": budget - 1, **mix, "sources": {}}
    if q_real:
        u, m = fill(units(source_texts(mix["real"]), tok), q_real, rng)
        parts += u
        man["sources"]["real:" + mix["real"]] = m
    if budget - q_real:
        u, m = fill(units(source_texts(mix["syn"]), tok), budget - q_real, rng)
        parts += u
        man["sources"]["syn:" + mix["syn"]] = m
    rng.shuffle(parts)
    stream = np.concatenate(parts)
    assert len(stream) == budget, (len(stream), budget)
    path = os.path.join(DATA, "mixtures", f"{mixture}_s{seed}.bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stream.tofile(path)
    man["file"] = os.path.relpath(path, DATA)
    man["real_token_share_actual"] = round(q_real / budget, 4)
    save_json(man, MANIFESTS, "mixtures", f"{mixture}_s{seed}.json")
    return path, man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _, man = build(args.mixture, args.seed)
    print({k: v for k, v in man.items() if k != "sources"}, man["sources"])


if __name__ == "__main__":
    main()
