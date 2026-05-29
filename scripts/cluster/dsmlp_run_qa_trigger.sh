#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/private/agentpoison_reproduction"
REPO="$ROOT/AgentPoison"
OUT="$ROOT/outputs/agentpoison_runs/triggers/qa/ap/dpr-ctx_encoder-single-nq-base/len10_seed0"
LOGDIR="$ROOT/outputs/logs"
mkdir -p "$OUT" "$LOGDIR"

cd "$REPO"

LOG="$LOGDIR/qa_dpr_trigger_len10_seed0_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="$LOGDIR/qa_dpr_trigger_len10_seed0.pid"

nohup python algo/trigger_optimization.py \
  --agent qa \
  --algo ap \
  --model dpr-ctx_encoder-single-nq-base \
  --save_dir "$OUT" \
  --num_iter 1000 \
  --num_grad_iter 30 \
  --per_gpu_eval_batch_size 64 \
  --num_cand 100 \
  --num_adv_passage_tokens 10 \
  --asr_threshold 0.5 \
  --ppl_filter \
  --seed 0 \
  --plot \
  > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "started pid $(cat "$PIDFILE")"
echo "log $LOG"
echo "pidfile $PIDFILE"
