#!/usr/bin/env bash
# Phase 12 pilot: peak learning rate for the 5.3M student, chosen on REAL VALIDATION loss only (no test data),
# averaged over a real-only and a 50% synthetic run so the choice does not favour one data type. Seed 100.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
for lr in 5e-4 1e-3 2e-3; do
  for m in real_only mix50_filt; do
    .venv/Scripts/python -m synth.train_student --mixture $m --seed 100 --lr $lr --run lr${lr}_$m > results/metrics/logs/lr${lr}_$m.log 2>&1
    tail -1 results/metrics/logs/lr${lr}_$m.log
  done
done
