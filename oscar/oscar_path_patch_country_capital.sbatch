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

cd "$REPO_ROOT/fewshot_pp"

MODEL="meta-llama/Llama-3.2-1B-Instruct"

python run_fewshot_pp.py \
    --model_name "$MODEL" --d_name country-capital --original_d_name country-capital \
    --k 20 --threshold 0.2 \
    --pp_prompt_index 10 --exp_size 30 --batch_size 8 \
    --receiver_input v \
    --save_root output