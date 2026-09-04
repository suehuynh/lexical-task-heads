#!/bin/bash
#SBATCH -J lth_multi_task
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs

REPO_ROOT="$SLURM_SUBMIT_DIR"

module load python/3.11.11-5e66
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/src"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASKS=("country-capital" "antonym" "country-currency")
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Task: $TASK (array index $SLURM_ARRAY_TASK_ID) ==="

echo "=== $TASK: identify lexical-task heads (TransformerLens) ==="
python identify_heads.py --model_name "$MODEL" --d_name "$TASK" \
    --prompt_type EP --component_type lexical_task --prompt_template_index 10
python identify_heads.py --model_name "$MODEL" --d_name "$TASK" \
    --prompt_type IP --component_type lexical_task
