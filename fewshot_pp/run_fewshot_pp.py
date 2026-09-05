import os
import sys
import json
import argparse
import torch
from transformer_lens import ActivationCache, HookedTransformer, utilities
from transformer_lens.components import MLP, Embed, LayerNorm, Unembed
from transformer_lens.hook_points import HookPoint

from prompt_builders import create_few_shot_prompts, create_corrupt_prompts, _load_dataset
from metrics import *
from path_patching_tl import get_model_specs_tl, find_earliest_receiver, _resolve_pos, patch_head_input,patch_or_freeze_head_vectors, get_path_patch_head_to_heads
from io_loaders import load_receiver_list, load_correct_indices
from plot import plot_sender_head_effect

torch.set_grad_enabled(False)

def rank_heads(scores_tensor, threshold=None):
    """
    Ranks heads by their score from a 2D tensor of shape [layer, head].

    Args:
        scores_tensor (np.ndarray): A 2D numpy array where the value at
                                    scores_tensor[i, j] is the score for
                                    layer i and head j.

    Returns:
        list: A list of tuples (layer, head, score), sorted in descending
              order of the score.
    """
    # Get the dimensions of the tensor
    num_layers, num_heads = scores_tensor.shape

    # Create an empty list to store the (layer, head, score) tuples
    ranked_head_list = []

    # Iterate through each layer and head to build the list
    for layer in range(num_layers):
        for head in range(num_heads):
            score = scores_tensor[layer, head]
            # If a threshold is provided, only add the head if its score is greater than the threshold
            if threshold is None or score >= threshold:
                ranked_head_list.append((layer, head, score))
            

    # reverse=True ensures the sorting is from high to low.
    ranked_head_list.sort(key=lambda item: item[2], reverse=True)

    return ranked_head_list

class TokDataset:
    """Minimal wrapper so a batch of prompts exposes the `.toks` attribute
    that get_path_patch_head_to_heads expects."""
    def __init__(self, toks):
        self.toks = toks
## HEAD-TO-HEAD PATH-PATCHING ##

## EXECUTION
if __name__ == "__main__":
    """
    Path patching: find which upstream heads/MLPs feed the shared few-shots
    lexical-task heads for a given task (the receiver set).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
        help="model name, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--d_name", type=str, default="country-capital")
    parser.add_argument("--original_d_name", type=str, default="country-capital",
                        help="task the final query is drawn from")
    parser.add_argument("--corrupt_d_name", type=str, default="present-past",
                        help="task the few-shot demos are drawn from")
    parser.add_argument("--k", type=int, required=True,
        help="which k (20 or 25) to take the lexical-task head MAPS scores from")
    parser.add_argument("--threshold", type=float, default=0.1,
        help="min fraction of prompts a head must fire on (in load_receiver_list) to count as a receiver")
    parser.add_argument("--receiver_input", type=str, default="v", choices=["q", "k", "v"],
        help="which input stream of the receiver heads to path-patch into")
    parser.add_argument("--exp_size", type=int, default=50,
        help="number of model-correct prompts to path-patch over")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--include_mlp_receivers", action="store_true",
        help="also add (layer, -1) MLP receivers for each layer that has a shared head")
    parser.add_argument("--dataset", type=str, default="datasets/abstractive")
    parser.add_argument("--save_root", type=str, default="output/")
    parser.add_argument("--component_type", type=str, default="Relation")
    parser.add_argument("--behavior_json_path", type=str, default=None,
        help="path to EP_vary_n_shot_behavior.json from behavior_variance_tl.py "
             "(default: <save_root>/<model_name>/across_tasks/Behavior/EP_vary_n_shot_behavior.json)")

    args = parser.parse_args()
    model_name_short = args.model_name.split("/")[-1]
    behavior_json_path = args.behavior_json_path or os.path.join(
        args.save_root, model_name_short, "across_tasks", "Behavior", "EP_vary_n_shot_behavior.json")

    print("model_name", args.model_name)
    print("d_name", args.d_name)
    print("pp_prompt_index", args.pp_prompt_index, "k", args.k)

    receiver_list = load_receiver_list(
        save_root=args.save_root, model_name=model_name_short, d_name=args.d_name,
        prompt_type=args.pp_prompt_type, prompt_index=args.pp_prompt_index, k=args.k,
        component_type=args.component_type, threshold=args.threshold,
    )
    if not receiver_list:
        raise ValueError(
            f"No lexical-task heads found for {args.d_name} at k={args.k}, threshold={args.threshold}. "
            "Re-run identify_heads_tl.py or lower --threshold."
        )
    print(f"receiver_list ({len(receiver_list)} heads): {receiver_list}")

    if args.include_mlp_receivers:
        raise NotImplementedError(
            "--include_mlp_receivers is not supported by get_path_patch_head_to_heads yet -- "
            "it only patches q/k/v inputs of attention heads."
        )

    # Load model
    model = HookedTransformer.from_pretrained(args.model_name)

    # Load which examples the model answers correctly at this n_shot (from behavior_variance_tl.py),
    # and build EP clean/corrupt/answer prompts restricted to those examples.
    correct_index = load_correct_indices(behavior_json_path, args.d_name, args.pp_prompt_index)[:args.exp_size]
    print(f"using {len(correct_index)} model-correct examples")
    data = _load_dataset(args.original_d_name, args.dataset_folder)
    clean_prompts, clean_answers = create_few_shot_prompts(data, n_shot=args.n_shot)
    corrupt_prompts, corrupt_answers = create_corrupt_prompts(
            n_shot=args.n_shot,
            original_d_name=args.original_d_name,
            corrupt_d_name=args.corrupt_d_name,
            dataset_folder=args.dataset_folder,
        )
    # Tokenize clean/corrupt together so both batches share one padded length --
    # get_path_patch_head_to_heads indexes new_cache/orig_cache positionally and
    # needs matching tensor shapes.
    n_clean = len(clean_prompts)
    all_toks = model.to_tokens(clean_prompts + corrupt_prompts, padding_side="left")
    clean_dataset = TokDataset(all_toks[:n_clean])
    corrupt_dataset = TokDataset(all_toks[n_clean:])

    results = get_path_patch_head_to_heads(
        receiver_heads=receiver_list,
        receiver_input=args.receiver_input,
        model=model,
        patching_metric=l2_norm_effect,
        new_dataset=corrupt_dataset,
        orig_dataset=clean_dataset,
    )  # [layer, head]

    n_heads = get_model_specs_tl(model)["n_heads"]
    save_dir = os.path.join(args.save_root, model_name_short, args.d_name, "Heads", "causal_mediation", "path_patching")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tag = f"{args.pp_prompt_type}{args.pp_prompt_index}_k{args.k}"

    torch.save(results, os.path.join(save_dir, f"sender_to_shared_lexical_heads_{tag}.pt"))

    plot_path = os.path.join(save_dir, f"sender_to_shared_lexical_heads_{tag}_heatmap.html")
    plot_sender_head_effect(results, receiver_list, args.receiver_input, save_path=plot_path)

    heads_ranked = rank_heads(results.cpu().numpy(), threshold=None)
    heads_ranked = [(int(layer), int(head), float(score)) for layer, head, score in heads_ranked]

    ranked = {
        "meta": {
            "model_name": model_name_short, "d_name": args.d_name, "k": args.k,
            "threshold": args.threshold,
            "pp_prompt_type": args.pp_prompt_type, "pp_prompt_index": args.pp_prompt_index,
            "receiver_input": args.receiver_input, "n_examples": len(clean_prompts),
            "receiver_list": receiver_list,
        },
        "heads_ranked": heads_ranked,
    }
    print(f"\nTop 15 upstream heads (sender -> receiver_input={args.receiver_input}):")
    for layer, head, score in heads_ranked[:15]:
        print(f"  L{layer}H{head}: {score:.4f}")

    ranked_path = os.path.join(save_dir, f"sender_to_shared_lexical_heads_{tag}_ranked.json")
    with open(ranked_path, "w") as f:
        json.dump(ranked, f, indent=2)
    print("\nsaved to", save_dir)
