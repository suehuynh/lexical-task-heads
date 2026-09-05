#!/bin/bash
#SBATCH -J lth_multi_task
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-0
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs

REPO_ROOT="$SLURM_SUBMIT_DIR"

module load python/3.11.11-5e66
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/fewshot_pp"

MODEL="meta-llama/Llama-3.2-1B-Instruct"
TASKS=("country-capital")

# Safety check: catch a mismatched --array range before it silently no-ops
# or errors on an out-of-bounds index.
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASKS[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID but TASKS has only ${#TASKS[@]} entries." >&2
    echo "Fix #SBATCH --array=0-$((${#TASKS[@]} - 1)) to match TASKS length." >&2
    exit 1
fi

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Task: $TASK (array index $SLURM_ARRAY_TASK_ID of ${#TASKS[@]}) ==="

echo "=== $TASK: identify lexical-task heads (TransformerLens) ==="
python identify_lth.py --model_name "$MODEL" --d_name "$TASK" \
    --n_shot 10 --exp_size 100 --k_list 20 25 --batch_size 10