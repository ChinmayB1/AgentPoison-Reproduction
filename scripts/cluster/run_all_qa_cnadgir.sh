#!/usr/bin/env bash
set -euo pipefail
cd /home/cnadgir/agentpoison_reproduction

# Ensure logs directory exists
mkdir -p outputs/logs

echo "=================================================="
echo "Installing Python dependencies inside container..."
echo "=================================================="
pip3 install --user -q "pyautogen<0.3.0" jsonlines shortuuid termcolor wandb openai peft timm webdataset beautifulsoup4 gym python-Levenshtein wolframalpha replicate omegaconf iopath sentence-transformers datasets seaborn scikit-learn

export PYTHONPATH="/home/cnadgir/agentpoison_reproduction:/home/cnadgir/.local/lib/python3.11/site-packages:/home/cnadgir/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export AUTOGEN_USE_DOCKER="False"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=================================================="
echo "Starting StrategyQA Runs on cnadgir GPU"
echo "=================================================="


echo "1. Launching StrategyQA Seed 0 Baseline..."
python3 run_experiments.py --mode all --agent qa --model dpr --seed 0 --num_iter 20 --backbone llama3 --save_root outputs/qa_seed0 > outputs/logs/qa_seed0.log 2>&1

echo "2. Launching StrategyQA Seed 1 Baseline..."
python3 run_experiments.py --mode all --agent qa --model dpr --seed 1 --num_iter 20 --backbone llama3 --save_root outputs/qa_seed1 > outputs/logs/qa_seed1.log 2>&1

echo "3. Launching StrategyQA Seed 2 Baseline..."
python3 run_experiments.py --mode all --agent qa --model dpr --seed 2 --num_iter 20 --backbone llama3 --save_root outputs/qa_seed2 > outputs/logs/qa_seed2.log 2>&1

echo "=================================================="
echo "All cnadgir StrategyQA Runs completed!"
echo "=================================================="
