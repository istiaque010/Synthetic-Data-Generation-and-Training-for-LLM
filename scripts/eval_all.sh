#!/usr/bin/env bash
# Phase 13: every trained student on the held-out test sets (best.pt = lowest real-validation loss), + real reference.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
[ -f results/metrics/eval/real.json ] || .venv/Scripts/python -m synth.evaluate --real 2>&1 | tail -1
for d in checkpoints/*/; do
  r=$(basename $d)
  case $r in scorer_*|lr*) continue;; esac
  [ -f results/metrics/eval/$r.json ] || .venv/Scripts/python -m synth.evaluate --run $r 2>&1 | grep -v -i warn | tail -1
done
echo "eval done"
