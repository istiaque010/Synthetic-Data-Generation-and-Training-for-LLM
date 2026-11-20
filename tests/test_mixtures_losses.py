import random

import numpy as np
import torch

import synth  # noqa: F401
from synth.build_mixtures import fill
from synth.losses import _windows
from synth.sampling import cached_logits_check

from src.model import GPT, GPTConfig


def test_fill_exact_quota_and_epochs():
    pool = [np.arange(10, dtype=np.uint16), np.arange(30, dtype=np.uint16)]  # 40 tokens
    out, m = fill(pool, 100, random.Random(0))
    assert sum(len(u) for u in out) == 100 == m["tokens"]
    assert m["epochs"] == 2.5 and m["unique_stories"] == 2


def test_fill_smaller_than_pool():
    pool = [np.arange(10, dtype=np.uint16)] * 5
    out, m = fill(pool, 25, random.Random(0))
    assert m["tokens"] == 25 and m["epochs"] == 0.5


def test_windows_cover_every_target_once():
    for L in (5, 257, 258, 300, 513):
        ids = list(range(L))
        covered = []
        for w, s in _windows(ids, 256):
            off = ids.index(w[0])
            covered += [off + j + 1 for j in range(s, len(w) - 1)]
        assert sorted(covered) == list(range(1, L)), L


def test_cached_sampler_matches_gpt_forward():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, n_embd=16, dropout=0.0)).eval()
    ids = [3, 7, 9, 11, 4, 8]
    ref, _ = m(torch.tensor([ids]))
    got = cached_logits_check(m, ids)
    assert torch.allclose(ref[0], got, atol=1e-5)
