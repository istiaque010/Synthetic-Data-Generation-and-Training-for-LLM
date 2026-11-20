"""Model-based quality judge (Phase 8, ablation only): Project 1's small GPT (13.9M, trained on all 1.77M real
TinyStories training stories) scores how surprising a story is, as mean loss per token (nats) over its first 256
tokens. Lower = more like real TinyStories. It is calibrated on real seed stories (threshold = their 90th
percentile) and is *not* treated as ground truth: it only defines the `judge` dataset variant.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from . import P1_SMALL, load_tokenizer

from src.model import load_checkpoint  # Project 1 (read only)


@torch.no_grad()
def judge_losses(texts, batch=64, block=256):
    torch.set_num_threads(os.cpu_count())
    model, _ = load_checkpoint(P1_SMALL)
    model.eval()
    tok = load_tokenizer()
    out = []
    for b in range(0, len(texts), batch):
        ids = [tok.encode(t, add_bos=True, add_eos=True)[: block + 1] for t in texts[b:b + batch]]
        L = max(len(x) for x in ids)
        x = torch.full((len(ids), L), tok.pad_id, dtype=torch.long)
        m = torch.zeros((len(ids), L - 1))
        for i, s in enumerate(ids):
            x[i, : len(s)] = torch.tensor(s)
            m[i, : len(s) - 1] = 1
        logits, _ = model(x[:, :-1])
        nll = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none")
        out += ((nll * m).sum(1) / m.sum(1)).tolist()
    return out
