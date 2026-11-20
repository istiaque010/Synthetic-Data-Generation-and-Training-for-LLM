"""Phase 1: environment report + identity of every model/data file read from other projects -> results/metrics/env.json"""
import os, platform, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import synth
import psutil, torch, tokenizers, numpy, transformers

files = {"tokenizer (Project 1)": synth.TOKENIZER, "judge model (Project 1 small)": synth.P1_SMALL,
         "teacher weights (Project 4 download)": os.path.join(synth.TEACHER_DIR, "pytorch_model.bin"),
         "scorer weights": os.path.join(synth.SCORER_DIR, "model.safetensors")}
for s in ("train", "validation", "test"):
    files[f"TinyStories cleaned {s} (Project 1)"] = os.path.join(synth.CLEAN, f"{s}.parquet")
info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "torch": torch.__version__,
        "transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(), "device": "cpu", "precision": "fp32",
        "teacher": {"repo": synth.TEACHER_REPO, "revision": synth.TEACHER_REVISION, "license": "MIT"},
        "scorer": {"repo": synth.SCORER_REPO, "revision": synth.SCORER_REVISION, "license": "Apache-2.0"},
        "data_license": "TinyStories (roneneldan/TinyStories): CDLA-Sharing-1.0",
        "sha256": {k: synth.sha256(v) for k, v in files.items()}}
synth.save_json(info, synth.METRICS, "env.json")
print({k: v for k, v in info.items() if k != "sha256"})
