#!/bin/bash
#SBATCH -J path_patch_country_capital
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
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

MODEL="meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT="${MODEL##*/}"

# args:  $1 = receiver input (q|k|v, default v)   $2 = exp_size (default 189 = full)
RECV="${1:-v}"
EXP="${2:-189}"

MAPS_PKL="output/${MODEL_SHORT}/country-capital/Heads/MAPS/Relation_across_tasks/country-capital_MAPS_Relation_heads_across_tasks_EP_10_correct.pkl"
if [ ! -f "$MAPS_PKL" ]; then
    echo "ERROR: missing MAPS pkl: $(pwd)/$MAPS_PKL"
    echo "Run identify_lth.py (or scp the pkl) before submitting this job."
    exit 1
fi

BEHAVIOR_JSON="output/${MODEL_SHORT}/across_tasks/Behavior/EP_vary_n_shot_behavior.json"
if [ -f "$BEHAVIOR_JSON" ]; then
    echo "=== behavior file present, skipping regeneration: $BEHAVIOR_JSON ==="
else
    echo "=== generating behavior file for $MODEL ==="
    python behavior_variance_tl.py \
        --model_name "$MODEL" --d_name country-capital \
        --n_shot_list 10 --save_root output
fi

echo "=== running path-patching sweep (receiver_input=$RECV, exp_size=$EXP) ==="
python run_fewshot_pp.py \
    --model_name "$MODEL" --d_name country-capital --original_d_name country-capital \
    --k 20 --threshold 0.2 \
    --pp_prompt_index 10 --exp_size "$EXP" --batch_size 8 \
    --receiver_input "$RECV" --metric l2_norm lprr \
    --save_root output