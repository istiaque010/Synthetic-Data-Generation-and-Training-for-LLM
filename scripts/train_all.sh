#!/usr/bin/env bash
# Phase 12/14: every student. Main systems x 3 seeds, then the ablations (seed 0), then the size ablation.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run(){ # mixture seed arch
  local name=$1$([ "$3" = small ] && echo _small)_s$2
  if [ -f checkpoints/$name/final.pt ]; then echo "$name exists"; return; fi
  .venv/Scripts/python -m synth.train_student --mixture $1 --seed $2 --arch $3 > results/metrics/logs/$name.log 2>&1
  echo "$name done $(date +%H:%M) $(tail -1 results/metrics/logs/$name.log)"
}
for s in 0 1 2; do
  for m in real_only mix50_filt mix50_raw syn_only mix25_filt mix75_filt real_fresh; do run $m $s base; done
done
# mix50_nodedup is not trained: dedup removed 0 records, so its mixture is byte-identical to mix50_filt_s0
# (results/metrics/dedup_check.json); training it would repeat mix50_filt_s0 exactly.
for m in mix50_plain mix50_lowT mix50_cand1 mix50_judge; do run $m 0 base; done
for m in real_only mix50_filt; do run $m 0 small; done
echo "all training done $(date +%H:%M)"
