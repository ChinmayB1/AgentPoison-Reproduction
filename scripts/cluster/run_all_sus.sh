#!/usr/bin/env bash
set -euo pipefail
cd /home/sus028/agentpoison_reproduction

# Ensure logs directory exists
mkdir -p outputs/logs

echo "=================================================="
echo "Installing Python dependencies inside container..."
echo "=================================================="
pip3 install --user -q "pyautogen<0.3.0" jsonlines shortuuid termcolor wandb openai peft timm webdataset beautifulsoup4 gym python-Levenshtein wolframalpha replicate omegaconf iopath sentence-transformers datasets seaborn scikit-learn

export PYTHONPATH="/home/sus028/agentpoison_reproduction:/home/sus028/.local/lib/python3.11/site-packages:/home/sus028/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export AUTOGEN_USE_DOCKER="False"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=================================================="
echo "Starting EHRAgent & StrategyQA Runs on sus028 GPU"
echo "=================================================="


echo "1. Launching EHRAgent Seed 0 Baseline..."
python3 run_experiments.py --mode evaluate --agent ehr --model dpr --seed 0 --num_iter 50 --backbone llama3 --save_root outputs/ehr_seed0 > outputs/logs/ehr_seed0.log 2>&1

echo "2. Launching EHRAgent Seed 1 Baseline..."
python3 run_experiments.py --mode all --agent ehr --model dpr --seed 1 --num_iter 50 --backbone llama3 --save_root outputs/ehr_seed1 > outputs/logs/ehr_seed1.log 2>&1

echo "3. Launching EHRAgent Seed 2 Baseline..."
python3 run_experiments.py --mode all --agent ehr --model dpr --seed 2 --num_iter 50 --backbone llama3 --save_root outputs/ehr_seed2 > outputs/logs/ehr_seed2.log 2>&1

echo "4. Launching EHRAgent Seed 0 Adaptive (lambda=0.5)..."
python3 run_experiments.py --mode all --agent ehr --model dpr --seed 0 --adaptive_lambda 0.5 --num_iter 50 --backbone llama3 --save_root outputs/ehr_adaptive_0.5 > outputs/logs/ehr_adaptive.log 2>&1

echo "5. Launching StrategyQA Seed 0 Adaptive (lambda=0.5)..."
python3 run_experiments.py --mode all --agent qa --model dpr --seed 0 --adaptive_lambda 0.5 --num_iter 20 --backbone llama3 --save_root outputs/qa_adaptive_0.5 > outputs/logs/qa_adaptive.log 2>&1

echo "=================================================="
echo "All sus028 Runs completed!"
echo "=================================================="
