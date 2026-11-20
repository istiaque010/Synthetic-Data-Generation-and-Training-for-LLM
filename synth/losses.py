"""Loss of a student on whole real stories (validation and test use this same function).

Each story is <BOS> story <EOS> in the Project 1 tokenizer; every token after <BOS> (including <EOS>) is a target.
A story longer than the 256-token context is scored in two windows: the first 257 tokens, then the last 257 tokens,
counting in the second window only the targets the first did not cover, so every target is predicted exactly once
with up to 256 tokens of context. Returns the corpus loss (total nats / total targets) and per-story (nats, targets)
for paired bootstrap tests.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import load_tokenizer

_tok = None


def _windows(ids, block=256):
    L = len(ids)
    if L <= block + 1:
        return [(ids, 0)]
    return [(ids[: block + 1], 0), (ids[L - block - 1:], block - (L - block - 1))]  # (window, first scored target)


@torch.no_grad()
def story_losses(model, texts, batch=64, block=256):
    global _tok
    _tok = _tok or load_tokenizer()
    items = []  # (story index, window ids, first scored target position within window)
    for k, t in enumerate(texts):
        for win, start in _windows(_tok.encode(t, add_bos=True, add_eos=True), block):
            items.append((k, win, start))
    nll = [0.0] * len(texts)
    cnt = [0] * len(texts)
    for b in range(0, len(items), batch):
        chunk = items[b:b + batch]
        L = max(len(w) for _, w, _ in chunk)
        x = torch.full((len(chunk), L), _tok.pad_id, dtype=torch.long)
        m = torch.zeros((len(chunk), L - 1))
        for i, (_, w, s) in enumerate(chunk):
            x[i, : len(w)] = torch.tensor(w)
            m[i, s: len(w) - 1] = 1
        logits, _ = model(x[:, :-1])
        l = F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="none") * m
        for i, (k, _, _) in enumerate(chunk):
            nll[k] += float(l[i].sum())
            cnt[k] += int(m[i].sum())
    return sum(nll) / sum(cnt), list(zip(nll, cnt))
