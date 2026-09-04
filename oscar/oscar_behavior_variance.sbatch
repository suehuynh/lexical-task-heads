#!/bin/bash
#SBATCH -J behavior_multi_task
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

module load python/3.11.11-5e66
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/src"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASKS=("country-capital" "antonym" "country-currency")


for TASK in "${TASKS[@]}"; do
    echo "=== $TASK: behavior/correctness (nnsight) ==="
    python behavior_variance.py --model_name "$MODEL" --d_name "$TASK" --prompt_type EP
    python behavior_variance.py --model_name "$MODEL" --d_name "$TASK" --prompt_type IP
done
