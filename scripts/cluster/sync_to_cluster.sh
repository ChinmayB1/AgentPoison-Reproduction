#!/usr/bin/env bash
set -euo pipefail

# Exclude list for rsync
EXCLUDES=(
  --exclude='*.pkl'
  --exclude='outputs'
  --exclude='AgentPoison/agentdriver/data'
  --exclude='AgentPoison/ReAct/database/embeddings'
  --exclude='.DS_Store'
  --exclude='*.pdf'
  --exclude='.git'
)

echo "--------------------------------------------------"
echo "Syncing local code to sus028 account (dsmlp-sus)..."
echo "--------------------------------------------------"
rsync -avz "${EXCLUDES[@]}" \
  "/Users/chinmay/Documents/UCSD Study/Spring/291/Project Proposal Overleaf/" \
  dsmlp-sus:"~/agentpoison_reproduction/"

echo
echo "--------------------------------------------------"
echo "Syncing local code to cnadgir account (dsmlp-cnadgir)..."
echo "--------------------------------------------------"
rsync -avz "${EXCLUDES[@]}" \
  "/Users/chinmay/Documents/UCSD Study/Spring/291/Project Proposal Overleaf/" \
  dsmlp-cnadgir:"/dsmlp/workspaces-fs05/DSC190_SP26_B00/home/cnadgir/agentpoison_reproduction/"

echo
echo "Sync completed successfully!"
