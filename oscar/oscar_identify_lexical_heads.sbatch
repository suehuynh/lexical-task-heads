#!/bin/bash
#SBATCH -J lth_multi_task
#SBATCH --partition=gpu
#SBATCH --account=nhuynh15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Run oscar_behavior_variance.sbatch FIRST and wait for it to complete --
# identify_heads.py reads the behavior JSONs that job produces. Submit this
# one with a dependency, e.g.:
#   BEHAVIOR_JOB=$(sbatch --parsable oscar/oscar_behavior_variance.sbatch)
#   sbatch --dependency=afterok:$BEHAVIOR_JOB oscar/oscar_identify_lexical_heads.sbatch

set -euo pipefail
mkdir -p logs

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

module load python/3.11.11-5e66
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/src"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
# identify_heads.py's outputs are per-task files (no shared state across tasks),
# so it's safe to run these three as a parallel array job, unlike behavior_variance.py.
TASKS=("country-capital" "antonym" "country-currency")
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Task: $TASK (array index $SLURM_ARRAY_TASK_ID) ==="

echo "=== $TASK: identify lexical-task heads (TransformerLens) ==="
python identify_heads.py --model_name "$MODEL" --d_name "$TASK" \
    --prompt_type EP --component_type lexical_task --prompt_template_index 10
python identify_heads.py --model_name "$MODEL" --d_name "$TASK" \
    --prompt_type IP --component_type lexical_task
