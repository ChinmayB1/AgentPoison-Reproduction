#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/agentpoison_reproduction"
cd "$ROOT"

# Ensure log directories exist
mkdir -p "$ROOT/outputs/logs"

echo "=================================================="
echo "Starting Parallelized Offline Runs on cnadgir GPU"
echo "=================================================="

# We launch the runs in the background. Since the DPR retriever is lightweight,
# we can fit multiple runs on the same 1080 Ti GPU.
# We run them using the local LLaMA-3 backbone to bypass OpenAI API key requirements.

echo "Launching Seed 0..."
python3 run_experiments.py --mode all --agent qa --model dpr --seed 0 --backbone llama3 --save_root "$ROOT/outputs/seed0" > "$ROOT/outputs/logs/run_seed0.log" 2>&1 &

echo "Launching Seed 1..."
python3 run_experiments.py --mode all --agent qa --model dpr --seed 1 --backbone llama3 --save_root "$ROOT/outputs/seed1" > "$ROOT/outputs/logs/run_seed1.log" 2>&1 &

echo "Launching Seed 2..."
python3 run_experiments.py --mode all --agent qa --model dpr --seed 2 --backbone llama3 --save_root "$ROOT/outputs/seed2" > "$ROOT/outputs/logs/run_seed2.log" 2>&1 &

echo "Launching Adaptive Attacker (lambda=0.5)..."
python3 run_experiments.py --mode all --agent qa --model dpr --seed 0 --adaptive_lambda 0.5 --backbone llama3 --save_root "$ROOT/outputs/adaptive" > "$ROOT/outputs/logs/run_adaptive.log" 2>&1 &

echo "All runs started. Waiting for completion..."
wait

echo "=================================================="
echo "All parallel runs finished successfully!"
echo "=================================================="
