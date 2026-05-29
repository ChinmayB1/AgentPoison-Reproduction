#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/agentpoison_reproduction"

echo "Smoke files:"
python3 - <<'PY'
from pathlib import Path
root = Path("outputs/smoke")
print("exists", root.exists(), root.resolve())
for p in sorted(root.rglob("*"))[:120]:
    print(p)
PY

echo
echo "Latest stdout tail:"
python3 - <<'PY'
from pathlib import Path
files = sorted(Path("outputs/smoke").rglob("stdout.txt"))
print("stdout files", len(files))
if files:
    p = files[-1]
    print(p)
    print(p.read_text(errors="replace")[-4000:])
PY
