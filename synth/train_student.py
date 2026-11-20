"""Train one student from scratch on one mixture (Phase 12).

    python -m synth.train_student --mixture mix50_filt --seed 0 [--arch base] [--lr 1e-3] [--run name]

Student: Project 1's GPT (decoder-only, causal next-token cross-entropy on every token: these are plain stories, not
prompt/answer pairs, so nothing is masked), randomly initialised with the run's seed. Data: the mixture's token
stream (built here if missing), cut into consecutive 257-token windows (256 inputs + next-token targets); one
update = 16 windows. AdamW, linear warm-up then cosine decay, gradient clipping. Ten times per run the loss on the
500 real validation stories is measured (same function as the test loss); the checkpoint with the lowest validation
loss is kept as best.pt (selection uses validation only), the last as final.pt.

Logs: results/metrics/train/<run>/log.csv (step, lr, loss, tokens, secs, val_loss) and summary.json (best step and
val loss, wall-clock, tokens, peak memory, parameter count, mixture manifest).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
import psutil
import torch

from . import CHECKPOINTS, DATA, METRICS, load_config, load_json, MANIFESTS, read_jsonl, save_json
from .build_mixtures import build
from .losses import story_losses

from src.model import GPT, GPTConfig, count_parameters  # Project 1 (read only)


def lr_at(step, total, cfg, peak):
    if step < cfg["warmup"]:
        return peak * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, total - cfg["warmup"])
    lo = peak * cfg["min_lr_ratio"]
    return lo + 0.5 * (peak - lo) * (1 + math.cos(math.pi * min(1.0, p)))


def make_model(arch, seed, vocab=8192, block=256):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=vocab, block_size=block, bias=True, **arch))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixture", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="base")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--budget", type=int, default=None, help="pilot only: fewer tokens")
    args = ap.parse_args()
    cfg = load_config("student.yaml")
    torch.set_num_threads(os.cpu_count())
    run = args.run or f"{args.mixture}{'' if args.arch == 'base' else '_' + args.arch}_s{args.seed}"
    peak = args.lr or cfg["lr"]
    path = os.path.join(DATA, "mixtures", f"{args.mixture}_s{args.seed}.bin")
    if not os.path.exists(path):
        build(args.mixture, args.seed)
    stream = np.fromfile(path, dtype=np.uint16)
    man = load_json(MANIFESTS, "mixtures", f"{args.mixture}_s{args.seed}.json")
    T, Bsz = cfg["block_size"], cfg["batch_size"]
    budget = args.budget or cfg["budget_tokens"]
    steps = budget // (T * Bsz)
    n_win = (len(stream) - 1) // T
    model = make_model(cfg["archs"][args.arch], args.seed, block=T)
    params = count_parameters(model)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=peak, betas=tuple(cfg["betas"]))
    val = [r["text"] for r in read_jsonl(DATA, "seeds", "val.jsonl")]
    out_dir = os.path.join(METRICS, "train", run)
    ck_dir = os.path.join(CHECKPOINTS, run)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ck_dir, exist_ok=True)
    eval_at = set(int(round(steps * k / cfg["evals"])) for k in range(1, cfg["evals"] + 1))
    log = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "lr", "loss", "tokens", "secs", "val_loss"])
    best, best_step, evals = float("inf"), None, []
    proc = psutil.Process()
    peak_rss = 0
    t_train = 0.0
    model.train()
    for step in range(steps):
        t0 = time.perf_counter()
        idx = [(step * Bsz + b) % n_win for b in range(Bsz)]
        x = torch.from_numpy(np.stack([stream[i * T: i * T + T + 1].astype(np.int64) for i in idx]))
        lr = lr_at(step, steps, cfg, peak)
        for g in opt.param_groups:
            g["lr"] = lr
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        t_train += time.perf_counter() - t0
        vl = ""
        if step + 1 in eval_at:
            model.eval()
            vl, _ = story_losses(model, val)
            model.train()
            evals.append({"step": step + 1, "val_loss": vl, "secs": t_train})
            if vl < best:
                best, best_step = vl, step + 1
                torch.save({"model": model.state_dict(), "model_config": model.cfg.to_dict(), "step": step + 1,
                            "run": run, "val_loss": vl}, os.path.join(ck_dir, "best.pt"))
            print(f"[{run}] step {step + 1}/{steps} loss {loss.item():.3f} val {vl:.4f} "
                  f"{(step + 1) * T * Bsz / t_train:.0f} tok/s", flush=True)
            peak_rss = max(peak_rss, proc.memory_info().rss)
        w.writerow([step + 1, f"{lr:.3e}", f"{loss.item():.4f}", (step + 1) * T * Bsz, f"{t_train:.1f}",
                    f"{vl:.4f}" if vl != "" else ""])
    log.close()
    torch.save({"model": model.state_dict(), "model_config": model.cfg.to_dict(), "step": steps, "run": run},
               os.path.join(ck_dir, "final.pt"))
    save_json({"run": run, "mixture": args.mixture, "seed": args.seed, "arch": args.arch, "lr": peak,
               "steps": steps, "tokens": steps * T * Bsz, "train_seconds": t_train, "best_step": best_step,
               "best_val_loss": best, "final_val_loss": evals[-1]["val_loss"], "evals": evals,
               "params": params, "peak_rss_mb": peak_rss / 2 ** 20, "mixture_manifest": man},
              out_dir, "summary.json")


if __name__ == "__main__":
    main()
