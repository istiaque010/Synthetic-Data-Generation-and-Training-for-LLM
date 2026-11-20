#!/usr/bin/env bash
# Phase 6: the three teacher pools (configs/teacher.yaml). Resumable: rerun to continue an interrupted pool.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
for g in main lowT plain; do
  .venv/Scripts/python -m synth.generate_teacher --gen $g 2>&1 | grep --line-buffered -v -i "warn\|unexpected\|ignored\|Loading\|LOAD REPORT\|^Key \|^---\|^Notes\|^$" > results/metrics/logs/gen_$g.log
  echo "$g done $(date +%H:%M)"
done
echo "all generation done"
