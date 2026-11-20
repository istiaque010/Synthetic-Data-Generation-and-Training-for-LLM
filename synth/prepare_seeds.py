"""Seed data and held-out evaluation sets (Phase 3). Frozen before any teacher generation.

    python -m synth.prepare_seeds

All stories come from Project 1's cleaned TinyStories splits (data/clean/*.parquet, read only):

  seeds       5,000 training stories. They are BOTH the student's whole real-data pool (the scarce real data) and
              the source of the teacher's seed prompts.
  fresh       40,000 *other* training stories, used only for the "real_fresh" reference (what the same budget of
              new real text would give: the ceiling a synthetic-data method is trying to approach).
  val         500 validation stories: checkpoint selection (real validation loss).
  test_loss   2,000 test stories: the primary metric, loss of the student on held-out real stories.
  test_prompt 300 other test stories: the student writes a continuation of the first sentence (generation quality).

Project 1's validation and test splits are the two halves of TinyStories' own validation split, which the teacher
(TinyStories-33M, trained on the training split) never saw. Checks written to data_manifests/seeds.json:
no story text shared between any two sets (normalised exact match), and no test prompt sentence equal to any seed
opening sentence (those test stories are skipped, so a teacher prompt never copies an evaluation prompt).
"""
from __future__ import annotations

import random
import re

import pyarrow.parquet as pq

from . import CLEAN, DATA, MANIFESTS, load_tokenizer, save_json, text_hash, write_jsonl

N_SEEDS, N_FRESH, N_VAL, N_TEST_LOSS, N_TEST_PROMPT = 5_000, 40_000, 500, 2_000, 300
TRAIN_WINDOW = 200_000   # sample seeds/fresh from the first 200k cleaned training stories (fixed order)
MIN_TOK, MAX_TOK = 80, 450
SENT = re.compile(r'^(.+?[.!?]["\']?)(?:\s|$)')


def first_sentence(text: str):
    m = SENT.match(text)
    return m.group(1) if m else None


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def well_formed(text: str, n_tok: int) -> bool:
    """Starts like a story (capital letter), has a 4-30 word first sentence and at least 3 sentences, fits the
    student's context comfortably."""
    if not text[:1].isupper() or not (MIN_TOK <= n_tok <= MAX_TOK):
        return False
    s = first_sentence(text)
    if s is None or not (4 <= len(s.split()) <= 30):
        return False
    return len(re.findall(r"[.!?]", text)) >= 3


def load(split, n=None):
    col = pq.read_table(f"{CLEAN}/{split}.parquet", columns=["text"]).column("text")
    return (col if n is None else col.slice(0, n)).to_pylist()


def pick(texts, tok, n, rng, exclude=frozenset()):
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        t = texts[i]
        if norm(t) in exclude:
            continue
        if well_formed(t, len(tok.encode(t))):
            out.append(i)
            if len(out) == n:
                break
    assert len(out) == n, (len(out), n)
    return out


def main():
    tok = load_tokenizer()
    rng = random.Random(9)
    train = load("train", TRAIN_WINDOW)
    sel = pick(train, tok, N_SEEDS + N_FRESH, rng)
    seeds_i, fresh_i = sel[:N_SEEDS], sel[N_SEEDS:]

    val_t, test_t = load("validation"), load("test")
    val_i = pick(val_t, tok, N_VAL, random.Random(10))
    test_order = list(range(len(test_t)))
    random.Random(11).shuffle(test_order)

    seed_openings = {norm(first_sentence(train[i])) for i in seeds_i}
    test_loss_i, test_prompt_i, skipped = [], [], 0
    for i in test_order:
        t = test_t[i]
        if not well_formed(t, len(tok.encode(t))):
            continue
        if len(test_loss_i) < N_TEST_LOSS:
            test_loss_i.append(i)
        elif len(test_prompt_i) < N_TEST_PROMPT:
            if norm(first_sentence(t)) in seed_openings:
                skipped += 1
                continue
            test_prompt_i.append(i)
        else:
            break

    def rows(texts, ids, split):
        out = []
        for k, i in enumerate(ids):
            t = texts[i]
            out.append({"id": f"{split}_{k:05d}", "src_index": i, "text": t, "opening": first_sentence(t),
                        "n_tokens": len(tok.encode(t)), "hash": text_hash(norm(t))})
        return out

    sets = {"seeds": rows(train, seeds_i, "seed"), "fresh": rows(train, fresh_i, "fresh"),
            "val": rows(val_t, val_i, "val"), "test_loss": rows(test_t, test_loss_i, "testl"),
            "test_prompt": rows(test_t, test_prompt_i, "testp")}
    for name, r in sets.items():
        write_jsonl(r, DATA, "seeds", f"{name}.jsonl")

    # leakage checks: no shared story between any two sets
    hashes = {k: {r["hash"] for r in v} for k, v in sets.items()}
    overlaps = {f"{a}&{b}": len(hashes[a] & hashes[b]) for a in hashes for b in hashes if a < b}
    assert all(v == 0 for v in overlaps.values()), overlaps
    man = {"source": "Project 1 data/clean/{train,validation,test}.parquet (TinyStories, CDLA-Sharing-1.0)",
           "train_window": TRAIN_WINDOW, "token_range": [MIN_TOK, MAX_TOK],
           "counts": {k: len(v) for k, v in sets.items()},
           "tokens": {k: sum(r["n_tokens"] for r in v) for k, v in sets.items()},
           "pairwise_story_overlap": overlaps,
           "test_prompts_skipped_because_opening_equals_a_seed_opening": skipped,
           "ids_sha1": {k: text_hash(",".join(r["id"] + ":" + r["hash"] for r in v)) for k, v in sets.items()}}
    save_json(man, MANIFESTS, "seeds.json")
    print(man["counts"], man["tokens"], "skipped", skipped)


if __name__ == "__main__":
    main()
