#!/bin/bash
#SBATCH -J lth_fewshot_pp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

REPO_ROOT="$SLURM_SUBMIT_DIR"

module load python/3.11.11-5e66
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/fewshot_pp"

# Keep this small for now -- one small model, one task, few examples.
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TASK="${TASK:-country-capital}"
N_SHOT="${N_SHOT:-10}"
EXP_SIZE="${EXP_SIZE:-20}"

echo "=== identify_lth (few-shot PP): model=$MODEL task=$TASK n_shot=$N_SHOT exp_size=$EXP_SIZE ==="
python identify_lth.py \
    --model_name "$MODEL" \
    --d_name "$TASK" \
    --n_shot "$N_SHOT" \
    --exp_size "$EXP_SIZE"
