"""Batched, key/value-cached sampling from a Project 1 GPT student (evaluation, Phase 13).

Copied from Project 8 (grpo/policy.py, tested there against GPT.forward) with top-k added. Project 1's own
generate() recomputes the whole sequence for every new token; this keeps the keys/values of earlier tokens.
Prompts are left-padded; positions count real tokens only, so a padded prompt sees the positions it would see alone.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def _attend(attn_mod, x, mask, cache=None, start=None):
    B, T, C = x.shape
    h, hd = attn_mod.n_head, attn_mod.head_dim
    q, k, v = attn_mod.qkv(x).split(C, dim=2)
    q = q.view(B, T, h, hd).transpose(1, 2)
    k = k.view(B, T, h, hd).transpose(1, 2)
    v = v.view(B, T, h, hd).transpose(1, 2)
    if cache is not None:
        K, V = cache
        K[:, :, start:start + T] = k
        V[:, :, start:start + T] = v
        k, v = K[:, :, :start + T], V[:, :, :start + T]
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return attn_mod.proj(y.transpose(1, 2).contiguous().view(B, T, C))


def _blocks(model, x, mask, caches=None, start=None):
    for i, blk in enumerate(model.blocks):
        x = x + _attend(blk.attn, blk.ln1(x), mask, None if caches is None else caches[i], start)
        x = x + blk.mlp(blk.ln2(x))
    return model.head(model.ln_f(x))


@torch.no_grad()
def sample(model, prompts: List[List[int]], max_new: int, eos_id: int, pad_id: int, temperature: float = 1.0,
           top_k: int = 0, generator: Optional[torch.Generator] = None):
    """Returns per row the generated token ids (ending with <EOS> if one was produced)."""
    model.eval()
    B = len(prompts)
    P = max(len(p) for p in prompts)
    max_new = min(max_new, model.cfg.block_size - P)
    L = P + max_new
    h, hd = model.blocks[0].attn.n_head, model.blocks[0].attn.head_dim
    caches = [(torch.zeros(B, h, L, hd), torch.zeros(B, h, L, hd)) for _ in model.blocks]
    ids = torch.full((B, P), pad_id, dtype=torch.long)
    keymask = torch.zeros(B, L, dtype=torch.bool)
    lens = torch.tensor([len(p) for p in prompts])
    for i, p in enumerate(prompts):
        ids[i, P - len(p):] = torch.tensor(p)
        keymask[i, P - len(p):P] = True
    pos = (keymask[:, :P].long().cumsum(1) - 1).clamp(min=0)
    causal = torch.ones(P, P, dtype=torch.bool).tril()
    mask = (causal[None] & keymask[:, None, :P]) | torch.eye(P, dtype=torch.bool)[None]
    logits = _blocks(model, model.tok_emb(ids) + model.pos_emb(pos), mask[:, None], caches, 0)[:, -1]
    out = torch.full((B, max_new), pad_id, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    n = torch.zeros(B, dtype=torch.long)
    for t in range(max_new):
        lg = logits.float() / temperature
        if top_k:
            kth = lg.topk(top_k, dim=-1).values[:, -1:]
            lg = lg.masked_fill(lg < kth, float("-inf"))
        nxt = torch.multinomial(F.softmax(lg, dim=-1), 1, generator=generator).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        out[:, t] = nxt
        n = n + (~done).long()
        done = done | (nxt == eos_id)
        if done.all() or t == max_new - 1:
            break
        keymask[:, P + t] = True
        pos_t = (lens + t)[:, None]
        mask_t = keymask[:, None, None, :P + t + 1]
        x = model.tok_emb(nxt[:, None]) + model.pos_emb(pos_t)
        logits = _blocks(model, x, mask_t, caches, P + t)[:, -1]
    return [out[i, : int(n[i])].tolist() for i in range(B)]


@torch.no_grad()
def cached_logits_check(model, ids: List[int]):
    """Logits of the last position via the cached path (used by tests to compare with GPT.forward)."""
    P = len(ids)
    h, hd = model.blocks[0].attn.n_head, model.blocks[0].attn.head_dim
    caches = [(torch.zeros(1, h, P, hd), torch.zeros(1, h, P, hd)) for _ in model.blocks]
    x = torch.tensor([ids])
    mask = torch.ones(P, P, dtype=torch.bool).tril()[None, None]
    return _blocks(model, model.tok_emb(x) + model.pos_emb(torch.arange(P)[None]), mask, caches, 0)[0]
