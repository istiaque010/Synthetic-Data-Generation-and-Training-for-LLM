"""Download the independent generation scorer, SmolLM2-135M (Hugging Face, Apache-2.0), at a pinned revision, into
checkpoints/scorer_smollm2_135m (not committed). The teacher is read from Project 4's pinned download."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import snapshot_download
import synth

p = snapshot_download(synth.SCORER_REPO, revision=synth.SCORER_REVISION, local_dir=synth.SCORER_DIR,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE*", "README*"])
print(p, os.listdir(p))
