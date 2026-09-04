#!/bin/bash
#SBATCH -J lth_country_capital
#SBATCH --partition=gpu  
#SBATCH --account=nhuynh15            
#SBATCH --gres=gpu:1                    
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00   
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

module load <python-module>              # e.g. `module load python/3.11.0s-ixrhc3q` -- check `module avail python`
module load <cuda-module-if-required>    # e.g. `module load cuda/12.1` -- check `module avail cuda`
source "$REPO_ROOT/.venv/bin/activate"

cd "$REPO_ROOT/src"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TASK="country-capital"

# Step 1: behavior/correctness (nnsight) -- both prompt styles
python behavior_variance.py --model_name "$MODEL" --d_name "$TASK" --prompt_type EP
python behavior_variance.py --model_name "$MODEL" --d_name "$TASK" --prompt_type IP

# Step 2: identify lexical-task heads (TransformerLens)
python identify_heads.py --model_name "$MODEL" --d_name "$TASK" \
    --prompt_type EP --component_type lexical_task --prompt_template_index 10
python identify_heads.py --model_name "$MODEL" --d_name "$TASK" \
    --prompt_type IP --component_type lexical_task
