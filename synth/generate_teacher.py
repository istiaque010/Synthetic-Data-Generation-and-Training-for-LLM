"""Teacher generation (Phases 4 and 6): several candidates per seed prompt, raw outputs kept before any filtering.

    python -m synth.generate_teacher --gen main            # configs/teacher.yaml -> generations.main
    python -m synth.generate_teacher --gen main --limit 34 # pilot

Teacher: TinyStories-33M (GPT-Neo, 4 layers, 768 wide, 50,257-token vocabulary), MIT licence, pinned revision,
loaded from Project 4's download (read only). CPU, fp32. Prompts are left-padded and generated in batches with
sampling (temperature, top-k) and a token limit; every candidate is written to data/raw/<gen>.jsonl with its prompt
id, template, parameters, teacher id and revision, decoding settings, random seed, whether it stopped by itself (EOS)
or hit the limit, its token count and the batch's generation time. Appends after every batch and skips prompt ids
already written, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from . import DATA, MANIFESTS, TEACHER_DIR, TEACHER_REPO, TEACHER_REVISION, load_config, read_jsonl, save_json
from .templates import build_prompts


def load_teacher():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER_DIR)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(TEACHER_DIR, dtype=torch.float32).eval()
    return model, tok


CHUNKS = (112, 112, 96)  # generate in rounds; finished stories leave the batch between rounds


@torch.no_grad()
def generate_batch(model, tok, prompts, n, temperature, top_k, max_new, seed):
    """n candidates per prompt. Generation runs in rounds of CHUNKS tokens: after each round, candidates that ended
    (EOS) leave the batch and the rest continue from their exact text so far (prompt + tokens generated), so the
    batch does not keep computing padding for finished stories. Sampling is the same as one long generate call:
    each next token depends only on the tokens before it (left padding is masked)."""
    eos = tok.eos_token_id
    rows = [tok(p)["input_ids"] for p in prompts for _ in range(n)]
    gen = [[] for _ in rows]
    done = [False] * len(rows)
    torch.manual_seed(seed)
    for chunk in CHUNKS:
        act = [i for i in range(len(rows)) if not done[i]]
        if not act:
            break
        step = min(chunk, max_new - len(gen[act[0]]))
        enc = tok.pad({"input_ids": [rows[i] + gen[i] for i in act]}, return_tensors="pt", padding=True)
        out = model.generate(**enc, do_sample=True, temperature=temperature, top_k=top_k, top_p=1.0,
                             max_new_tokens=step, pad_token_id=eos, eos_token_id=eos)
        P = enc["input_ids"].shape[1]
        for j, i in enumerate(act):
            new = out[j, P:].tolist()
            if eos in new:
                gen[i] += new[: new.index(eos)]
                done[i] = "eos"
            else:
                gen[i] += new
                if len(gen[i]) >= max_new:
                    done[i] = "length"
    res = []
    for i, ids in enumerate(gen):
        res.append((tok.decode(ids, skip_special_tokens=True), done[i] or "length", len(ids)))
    return res  # grouped: prompt 0 candidates 0..n-1, prompt 1 ...


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config("teacher.yaml")
    g = cfg["generations"][args.gen]
    torch.set_num_threads(os.cpu_count())
    seeds = read_jsonl(DATA, "seeds", "seeds.jsonl")
    prompts = build_prompts(seeds, g["pool"], seed=cfg["prompt_seed"])
    if args.limit:
        prompts = prompts[: args.limit]
    out_name = args.out or args.gen
    path = os.path.join(DATA, "raw", f"{out_name}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {r["prompt_id"] for r in read_jsonl(path)} if os.path.exists(path) else set()
    todo = [p for p in prompts if p["prompt_id"] not in done]
    model, tok = load_teacher()
    dec = {"temperature": g["temperature"], "top_k": g["top_k"], "top_p": 1.0, "max_new_tokens": cfg["max_new_tokens"],
           "candidates": g["candidates"]}
    bs = cfg["batch_prompts"]
    t_start = time.perf_counter()
    n_tok = 0
    import json
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for b in range(0, len(todo), bs):
            batch = todo[b:b + bs]
            seed = cfg["base_seed"] + int(batch[0]["prompt_id"].split("_")[-1])
            t0 = time.perf_counter()
            res = generate_batch(model, tok, [p["prompt"] for p in batch], g["candidates"], g["temperature"],
                                 g["top_k"], cfg["max_new_tokens"], seed)
            dt = time.perf_counter() - t0
            for i, p in enumerate(batch):
                for c in range(g["candidates"]):
                    text, finish, k = res[i * g["candidates"] + c]
                    n_tok += k
                    f.write(json.dumps({
                        "id": f"syn_{out_name}_{p['prompt_id'].split('_')[-1]}_{c}", **p, "candidate": c,
                        "output": text, "teacher": TEACHER_REPO, "revision": TEACHER_REVISION, "decoding": dec,
                        "seed": seed, "finish": finish, "teacher_tokens": k,
                        "batch_seconds": round(dt, 3), "batch_size": len(batch) * g["candidates"]},
                        ensure_ascii=False) + "\n")
            f.flush()
            el = time.perf_counter() - t_start
            print(f"[{args.gen}] {b + len(batch)}/{len(todo)} prompts  {n_tok / el:.0f} teacher tok/s  "
                  f"{el / 60:.1f} min", flush=True)
    rows = read_jsonl(path)
    save_json({"gen": out_name, "config": g, "decoding": dec, "teacher": TEACHER_REPO, "revision": TEACHER_REVISION,
               "prompts": len({r["prompt_id"] for r in rows}), "candidates": len(rows),
               "teacher_tokens": sum(r["teacher_tokens"] for r in rows),
               "hit_limit": sum(r["finish"] == "length" for r in rows),
               "generation_seconds": sum(r["batch_seconds"] / r["batch_size"] for r in rows),
               "hardware": {"device": "cpu", "threads": torch.get_num_threads(), "precision": "fp32"}},
              MANIFESTS, f"raw_{out_name}.json")


if __name__ == "__main__":
    main()
