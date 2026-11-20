"""Parse and normalise raw teacher outputs into one record schema (Phase 7).

    python -m synth.parse_outputs --gen main

Each raw candidate becomes

    {"id", "prompt_id", "seed_id", "template_id", "params", "candidate", "prompt", "response", "text",
     "teacher", "revision", "decoding", "seed", "finish", "teacher_tokens", "n_tokens", "quality_flags"}

`text` = prompt + response, whitespace and curly quotes normalised exactly as Project 1 normalised the real stories
(src.dataset.clean_story), so real and synthetic stories share one format. Nothing is re-worded. `n_tokens` counts the
student's tokenizer (Project 1 BPE, 8,192 tokens), the unit of every training budget. Schema-level problems are
recorded as flags (the record is kept; the filter decides):

  empty       no text after the prompt
  truncated   the teacher hit its token limit instead of ending the story (finish == "length")
  malformed   characters outside printable ASCII after normalisation, or leftover special-token text
"""
from __future__ import annotations

import argparse
import re

from . import DATA, load_tokenizer, read_jsonl, write_jsonl

from src.dataset import clean_story  # Project 1 (read only)

BAD = re.compile(r"[^\x20-\x7e]")
SCHEMA = ["id", "prompt_id", "seed_id", "template_id", "params", "candidate", "prompt", "response", "text",
          "teacher", "revision", "decoding", "seed", "finish", "teacher_tokens", "n_tokens", "quality_flags"]


def normalise(prompt: str, output: str):
    resp = clean_story(output)
    text = clean_story(prompt + " " + output)
    return resp, text


def schema_flags(resp: str, text: str, finish: str):
    flags = []
    if not resp:
        flags.append("empty")
    if finish != "eos":
        flags.append("truncated")
    if BAD.search(text) or "<|endoftext|>" in text or "<BOS>" in text or "<EOS>" in text:
        flags.append("malformed")
    return flags


def parse_record(raw: dict, tok) -> dict:
    resp, text = normalise(raw["prompt"], raw["output"])
    rec = {k: raw.get(k) for k in SCHEMA if k in raw}
    rec.update({"response": resp, "text": text, "n_tokens": len(tok.encode(text)),
                "quality_flags": schema_flags(resp, text, raw["finish"])})
    return {k: rec.get(k) for k in SCHEMA}


def validate(rec: dict):
    assert list(rec) == SCHEMA, list(rec)
    assert isinstance(rec["text"], str) and isinstance(rec["quality_flags"], list)
    assert rec["finish"] in ("eos", "length") and rec["n_tokens"] >= 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    args = ap.parse_args()
    tok = load_tokenizer()
    recs = [parse_record(r, tok) for r in read_jsonl(DATA, "raw", f"{args.gen}.jsonl")]
    for r in recs:
        validate(r)
    write_jsonl(recs, DATA, "parsed", f"{args.gen}.jsonl")
    from collections import Counter
    c = Counter(f for r in recs for f in r["quality_flags"])
    print(f"{args.gen}: {len(recs)} records, flags {dict(c)}")


if __name__ == "__main__":
    main()
