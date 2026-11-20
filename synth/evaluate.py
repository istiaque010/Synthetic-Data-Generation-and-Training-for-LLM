"""Evaluate a student on truly held-out data (Phase 13).

    python -m synth.evaluate --run mix50_filt_s0            # best.pt (lowest real-validation loss)
    python -m synth.evaluate --real                         # the real continuations, as a reference row

1. Test loss (primary): mean loss per token (nats) on 2,000 held-out real test stories (TinyStories validation half
   that neither the teacher nor any student saw), same tokenizer and same function for every student, with
   per-story values kept for paired tests. Perplexity = exp(loss).
2. Generations: the student continues the opening sentence of 300 other held-out test stories (temperature 0.8,
   top-k 50, up to 220 new tokens, fixed sampling seed), saved to results/samples/<run>.jsonl. Measured:
     scorer_nll    mean loss per token of the continuation under an independent model, SmolLM2-135M (general web
                   text, not the teacher and not trained on TinyStories only); lower = more fluent / predictable
     distinct_1/2  distinct words / word pairs over the 300 continuations (variety)
     repetitive    share of continuations whose repeated word 4-gram rate is above the real stories' 99th
                   percentile (the same threshold as the data filter)
     ended         share that end with <EOS> within the limit (a complete story)
     copied        share that are near duplicates (Jaccard >= 0.5) of one of the 5,000 real seed stories
                   (memorisation of the scarce real data)
   Writes results/metrics/eval/<run>.json.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import CHECKPOINTS, DATA, MANIFESTS, METRICS, SAMPLES, SCORER_DIR, load_json, load_tokenizer, read_jsonl
from . import save_json, write_jsonl
from . import deduplicate as dd
from .filter_quality import rep4
from .losses import story_losses
from .sampling import sample

from src.model import load_checkpoint  # Project 1 (read only)

GEN = {"temperature": 0.8, "top_k": 50, "max_new": 220, "seed": 4321}


def distinct(texts, n):
    grams, total = set(), 0
    for t in texts:
        w = dd.words(t)
        g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        grams.update(g)
        total += len(g)
    return len(grams) / max(1, total)


_scorer = None


@torch.no_grad()
def scorer_nll(prompts, conts, batch=32):
    """Per-continuation mean NLL (nats/token) under SmolLM2-135M, conditioned on the prompt."""
    global _scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _scorer is None:
        tok = AutoTokenizer.from_pretrained(SCORER_DIR)
        model = AutoModelForCausalLM.from_pretrained(SCORER_DIR, dtype=torch.float32).eval()
        _scorer = (tok, model)
    tok, model = _scorer
    out = []
    for b in range(0, len(prompts), batch):
        ids, starts = [], []
        for p, c in zip(prompts[b:b + batch], conts[b:b + batch]):
            pi = tok(p)["input_ids"]
            full = tok(p + " " + c)["input_ids"][:512]
            ids.append(full)
            starts.append(len(pi))
        L = max(len(x) for x in ids)
        x = torch.full((len(ids), L), tok.eos_token_id or 0, dtype=torch.long)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        m = torch.zeros((len(ids), L - 1))
        for i, (s, st) in enumerate(zip(ids, starts)):
            x[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
            m[i, st - 1:len(s) - 1] = 1
        logits = model(input_ids=x, attention_mask=att).logits[:, :-1]
        nll = F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="none") * m
        out += (nll.sum(1) / m.sum(1).clamp(min=1)).tolist()
    return out


_seed_index = None


def gen_metrics(prompts, conts, ended, ntoks):
    global _seed_index
    cal = load_json(MANIFESTS, "filter_calibration.json")
    if _seed_index is None:
        _seed_index = dd.build_index([r["text"] for r in read_jsonl(DATA, "seeds", "seeds.jsonl")])
    full = [p + " " + c for p, c in zip(prompts, conts)]
    sn = scorer_nll(prompts, conts)
    copied = dd.max_sim_against(full, _seed_index)
    per = [{"scorer_nll": a, "rep4": rep4(f), "copy_jaccard": j} for a, f, j in zip(sn, full, copied)]
    return {"scorer_nll": float(np.mean(sn)), "distinct_1": distinct(conts, 1), "distinct_2": distinct(conts, 2),
            "repetitive": float(np.mean([p["rep4"] > cal["rep4_p99"] for p in per])),
            "ended": float(np.mean(ended)), "mean_tokens": float(np.mean(ntoks)),
            "copied": float(np.mean([j >= dd.THRESHOLD for j in copied]))}, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--no-gen", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    tok = load_tokenizer()
    tp = read_jsonl(DATA, "seeds", "test_prompt.jsonl")
    prompts = [r["opening"] for r in tp]
    if args.real:
        conts = [r["text"][len(r["opening"]):].strip() for r in tp]
        ntok = [len(tok.encode(c)) for c in conts]
        summ, per = gen_metrics(prompts, conts, [1.0] * len(conts), ntok)
        save_json({"run": "real", "generation": summ, "gen_per_prompt": per}, METRICS, "eval", "real.json")
        print("real", summ)
        return
    name = args.run if args.ckpt == "best" else f"{args.run}__{args.ckpt}"
    model, ck = load_checkpoint(os.path.join(CHECKPOINTS, args.run, f"{args.ckpt}.pt"))
    t0 = time.perf_counter()
    test = [r["text"] for r in read_jsonl(DATA, "seeds", "test_loss.jsonl")]
    loss, per_story = story_losses(model, test)
    res = {"run": name, "ckpt_step": ck.get("step"), "test_loss": loss, "test_ppl": float(np.exp(loss)),
           "per_story": per_story, "loss_seconds": time.perf_counter() - t0}
    if not args.no_gen:
        g = torch.Generator().manual_seed(GEN["seed"])
        enc = [tok.encode(p, add_bos=True) for p in prompts]
        conts, ended, ntoks = [], [], []
        t0 = time.perf_counter()
        for b in range(0, len(enc), 100):
            for r in sample(model, enc[b:b + 100], GEN["max_new"], tok.eos_id, tok.pad_id, GEN["temperature"],
                            GEN["top_k"], g):
                e = bool(r) and r[-1] == tok.eos_id
                conts.append(tok.decode(r[:-1] if e else r).strip())
                ended.append(float(e))
                ntoks.append(len(r))
        res["gen_seconds"] = time.perf_counter() - t0
        summ, per = gen_metrics(prompts, conts, ended, ntoks)
        res["generation"] = summ
        res["gen_per_prompt"] = per
        write_jsonl([{"id": r["id"], "prompt": p, "continuation": c, "ended": bool(e)}
                     for r, p, c, e in zip(tp, prompts, conts, ended)], SAMPLES, f"{name}.jsonl")
    save_json(res, METRICS, "eval", f"{name}.json")
    print(name, f"test loss {loss:.4f} (ppl {np.exp(loss):.2f})", res.get("generation", ""))


if __name__ == "__main__":
    main()
