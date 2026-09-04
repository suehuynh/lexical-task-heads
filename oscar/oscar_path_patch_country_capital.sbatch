#!/bin/bash
#SBATCH -J path_patch_country_capital
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

REPO_ROOT="$SLURM_SUBMIT_DIR"

module load python/3.11.11-5e66
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/src"

MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Pilot run: small --exp_size to gauge runtime before scaling up or running
# the other two tasks. ep_index/ip_index/k match the shared_lexical_heads.py
# output already computed for country-capital (EP-10 vs IP-4, k=20 -> 16 shared heads).
python path_patch_upstream_heads.py \
    --model_name "$MODEL" --d_name country-capital \
    --ep_index 10 --ip_index 4 --k 20 \
    --exp_size 20 --batch_size 8 --variant both
