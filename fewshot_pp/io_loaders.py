import os
import json
import pickle
import numpy as np


def load_receiver_list(save_root, model_name, d_name, prompt_type, prompt_index, k,
                        component_type="Relation", threshold=0.1):
    folder = os.path.join(
        save_root, model_name, d_name, "Heads", "MAPS", f"{component_type}_across_tasks"  # dropped _vary_k
    )
    fname = f"{d_name}_MAPS_{component_type}_heads_across_tasks_{prompt_type}_{prompt_index}_correct.pkl"
    path = os.path.join(folder, fname)

    if not os.path.exists(path):
        available = os.listdir(folder) if os.path.exists(folder) else []
        raise FileNotFoundError(f"{path} not found. Files present in {folder}: {available}")

    with open(path, "rb") as f:
        maps_scores = pickle.load(f)

    if k not in maps_scores:
        raise KeyError(f"k={k} not found; available keys: {list(maps_scores.keys())}")
    if d_name not in maps_scores[k]:
        raise KeyError(f"d_name={d_name!r} not found under k={k}; available: {list(maps_scores[k].keys())}")

    scores = maps_scores[k][d_name]  # (n_prompts, n_layers, n_heads), values in {0,1}
    per_head_fraction = scores.mean(axis=0)  # (n_layers, n_heads)

    layers, heads = np.where(per_head_fraction >= threshold)
    receiver_list = list(zip(layers.tolist(), heads.tolist()))

    print(f"loaded {len(receiver_list)} receivers at k={k}, threshold={threshold}: {receiver_list}")
    return receiver_list


def load_correct_indices(behavior_json_path, d_name, n_shot):
    """
    Load correct-example indices from behavior_variance.py's
    EP_vary_n_shot_behavior.json for a given task and n_shot.
    """
    with open(behavior_json_path) as f:
        result_dict = json.load(f)

    if d_name not in result_dict:
        raise KeyError(f"d_name={d_name!r} not found; available: {list(result_dict.keys())}")

    key = str(n_shot)
    if key not in result_dict[d_name]:
        raise KeyError(
            f"n_shot={n_shot} not found for {d_name}; available: {list(result_dict[d_name].keys())}"
        )

    return result_dict[d_name][key]["correct_index"]

if __name__ == "__main__":
    receivers = load_receiver_list(
        save_root="fewshot_pp/output", model_name="Llama-3.2-1B-Instruct",
        d_name="country-capital", prompt_type="EP", prompt_index=10, k=20, threshold=0.2,
    )
    print(receivers)
    