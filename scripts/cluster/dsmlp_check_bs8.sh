#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/private/agentpoison_reproduction"
PIDFILE="$ROOT/outputs/logs/qa_dpr_trigger_len10_seed0_bs8.pid"
OUTROOT="$ROOT/outputs/agentpoison_runs/triggers/qa/ap/dpr-ctx_encoder-single-nq-base/len10_seed0_bs8"

echo "time=$(date -Is)"
echo "pidfile=$PIDFILE"
if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  echo "pid=$PID"
  ps -p "$PID" -o pid,etime,pcpu,pmem,cmd || true
else
  echo "pidfile missing"
fi

echo "--- gpu ---"
nvidia-smi || true

echo "--- latest stdout ---"
LATEST="$(find "$OUTROOT" -name stdout.txt 2>/dev/null | sort | tail -1 || true)"
echo "$LATEST"
if [ -n "$LATEST" ]; then
  tail -120 "$LATEST"
  echo "--- iteration summary ---"
  grep -o 'Iteration: [0-9]*' "$LATEST" | tail -20 || true
fi
