#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/private/agentpoison_reproduction"
LOGDIR="$ROOT/outputs/logs"
PIDFILE="$LOGDIR/qa_dpr_trigger_len10_seed0.pid"

echo "ROOT=$ROOT"
echo "LOGDIR=$LOGDIR"
echo

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  echo "PID=$PID"
  ps -p "$PID" -o pid,etime,pcpu,pmem,cmd || true
else
  echo "PIDFILE missing: $PIDFILE"
fi

echo
echo "GPU:"
nvidia-smi || true

echo
echo "Logs:"
ls -l "$LOGDIR" || true

echo
echo "Latest qa trigger log:"
python - <<'PY'
from pathlib import Path
logdir = Path.home() / "private" / "agentpoison_reproduction" / "outputs" / "logs"
logs = sorted(logdir.glob("qa_dpr_trigger_len10_seed0_*.log"))
print("num logs", len(logs))
if logs:
    p = logs[-1]
    print("path", p)
    text = p.read_text(errors="replace")
    print(text[-6000:])
PY

echo
echo "Latest AgentPoison stdout:"
python - <<'PY'
from pathlib import Path
root = Path.home() / "private" / "agentpoison_reproduction" / "outputs" / "agentpoison_runs"
files = sorted(root.rglob("stdout.txt"))
print("num stdout", len(files))
if files:
    p = files[-1]
    print("path", p)
    print(p.read_text(errors="replace")[-6000:])
PY
