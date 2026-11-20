"""Project 9: synthetic data generation and training (teacher -> filter -> dedup -> mixtures -> student).

Code package `synth`. Importing it makes Project 1's `src` package importable (model, tokenizer, data utilities);
Projects 1 and 4 are only read, never changed, and bytecode writing is off so nothing is written into them.
"""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True

PROJECT9 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PROJECT9)
PROJECT1 = os.path.join(ROOT, "project1")
PROJECT4 = os.path.join(ROOT, "project4")
CONFIGS = os.path.join(PROJECT9, "configs")
PROMPTS = os.path.join(PROJECT9, "prompts")
DATA = os.path.join(PROJECT9, "data")             # seeds, raw/filtered synthetic data, mixtures (token files)
MANIFESTS = os.path.join(PROJECT9, "data_manifests")
CHECKPOINTS = os.path.join(PROJECT9, "checkpoints")
RESULTS = os.path.join(PROJECT9, "results")
METRICS = os.path.join(RESULTS, "metrics")
SAMPLES = os.path.join(RESULTS, "samples")
FIGURES = os.path.join(RESULTS, "figures")

# Project 1: tokenizer, cleaned TinyStories splits, the small model (used as an optional quality judge)
TOKENIZER = os.path.join(PROJECT1, "data", "tokenizer.json")
CLEAN = os.path.join(PROJECT1, "data", "clean")
P1_SMALL = os.path.join(PROJECT1, "checkpoints", "small", "model_best.pt")

# Teacher: TinyStories-33M (Eldan & Li 2023, MIT licence), pinned revision, downloaded in Project 4 (read only)
TEACHER_REPO = "roneneldan/TinyStories-33M"
TEACHER_REVISION = "2ad0a164221b7c4d21cac7c46aec74f6f98dbfc8"
TEACHER_DIR = os.path.join(PROJECT4, "checkpoints", "scorer_tinystories33m")

# Independent scorer for student generations: SmolLM2-135M (Hugging Face, Apache-2.0), general web text,
# not trained by the teacher's authors and not on TinyStories only; pinned revision, stored inside project9
SCORER_REPO = "HuggingFaceTB/SmolLM2-135M"
SCORER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SCORER_DIR = os.path.join(CHECKPOINTS, "scorer_smollm2_135m")

if PROJECT1 not in sys.path:
    sys.path.insert(0, PROJECT1)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def save_json(obj, *path):
    p = os.path.join(*path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return p


def load_json(*path):
    with open(os.path.join(*path), encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows, *path):
    p = os.path.join(*path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def read_jsonl(*path):
    with open(os.path.join(*path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config(name: str) -> dict:
    import yaml

    with open(os.path.join(CONFIGS, name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tokenizer():
    from src.tokenizer import StoryTokenizer

    return StoryTokenizer(TOKENIZER)
