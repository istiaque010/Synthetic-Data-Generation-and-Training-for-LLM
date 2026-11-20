"""Deduplication and contamination checks (Phase 9).

Exact duplicates: identical text after lower-casing and whitespace normalisation (sha1).

Near duplicates: MinHash over word 5-gram "shingles". Two stories' Jaccard similarity = shared shingles / all
shingles. Each story gets 64 MinHash values; locality-sensitive hashing (16 bands x 4 rows) proposes candidate pairs
(pairs with Jaccard about 0.5 are found with high probability), and every proposed pair is then checked with the
*exact* Jaccard, so the MinHash only makes the search fast, never decides alone. Threshold: Jaccard >= 0.5.

All hashing is deterministic (crc32 and a fixed-seed permutation), not Python's per-process `hash`.

Used for: (1) duplicates inside a synthetic pool (keep the first by id), (2) synthetic stories that copy a real seed
story (the teacher was trained on TinyStories and may reproduce one), (3) contamination: a synthetic story that is
a near duplicate of any validation or test story is removed, and the share of stories sharing any 13-word sequence
with validation/test text is reported (TinyStories is formulaic, so this is also measured on real seed stories as
the natural base rate). No evaluation label or score is used to choose training data.
"""
from __future__ import annotations

import hashlib
import re
import zlib
from collections import defaultdict

import numpy as np

WORD = re.compile(r"[a-z0-9']+")
K, BANDS, ROWS = 64, 16, 4
PRIME = (1 << 61) - 1
_rng = np.random.default_rng(12345)
A = _rng.integers(1, 1 << 29, size=K, dtype=np.uint64)
B = _rng.integers(0, 1 << 29, size=K, dtype=np.uint64)
THRESHOLD = 0.5


def words(text: str):
    return WORD.findall(text.lower())


def exact_key(text: str) -> str:
    return hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()


def shingles(text: str, n: int = 5) -> np.ndarray:
    w = words(text)
    if len(w) < n:
        grams = [" ".join(w)] if w else [""]
    else:
        grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return np.unique(np.array([zlib.crc32(g.encode()) for g in grams], dtype=np.uint64))


def signature(sh: np.ndarray) -> np.ndarray:
    return ((A[:, None] * sh[None, :] + B[:, None]) % PRIME).min(axis=1)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


class Index:
    """LSH index over reference stories."""

    def __init__(self):
        self.buckets = defaultdict(list)
        self.sh = []

    def add(self, sh, sig):
        k = len(self.sh)
        self.sh.append(sh)
        for band in range(BANDS):
            self.buckets[(band, sig[band * ROWS:(band + 1) * ROWS].tobytes())].append(k)
        return k

    def candidates(self, sig):
        out = set()
        for band in range(BANDS):
            out.update(self.buckets.get((band, sig[band * ROWS:(band + 1) * ROWS].tobytes()), ()))
        return out

    def best(self, sh, sig):
        best, who = 0.0, None
        for k in self.candidates(sig):
            j = jaccard(sh, self.sh[k])
            if j > best:
                best, who = j, k
        return best, who


def build_index(texts):
    idx = Index()
    for t in texts:
        sh = shingles(t)
        idx.add(sh, signature(sh))
    return idx


def dedup_within(texts):
    """Returns (keep mask, reason list). Keeps the first of every exact or near-duplicate group (input order)."""
    seen_exact, idx = set(), Index()
    keep, reason = [], []
    for t in texts:
        e = exact_key(t)
        if e in seen_exact:
            keep.append(False)
            reason.append("exact_duplicate")
            continue
        sh = shingles(t)
        sig = signature(sh)
        j, _ = idx.best(sh, sig)
        if j >= THRESHOLD:
            keep.append(False)
            reason.append("near_duplicate")
            continue
        seen_exact.add(e)
        idx.add(sh, sig)
        keep.append(True)
        reason.append(None)
    return keep, reason


def max_sim_against(texts, ref_index):
    """Highest exact Jaccard of each text to any reference story proposed by the LSH index."""
    out = []
    for t in texts:
        sh = shingles(t)
        out.append(ref_index.best(sh, signature(sh))[0])
    return out


def ngram_set(texts, n=13):
    s = set()
    for t in texts:
        w = words(t)
        for i in range(len(w) - n + 1):
            s.add(zlib.crc32(" ".join(w[i:i + n]).encode()))
    return s


def shares_ngram(text, gset, n=13):
    w = words(text)
    return any(zlib.crc32(" ".join(w[i:i + n]).encode()) in gset for i in range(len(w) - n + 1))
