import os
import sys
import json
import pickle
import argparse

import torch
import numpy as np
from transformer_lens import HookedTransformer

from prompt_builders import create_few_shot_prompts, check_correctness
from metrics import early_decode, topk_match_count

torch.set_grad_enabled(False)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def convert_task_words_to_token_ids(model, task_words):
    """
    Convert a task's descriptive word list into a set of token IDs, trying
    several casing/spacing variants (leading space, capitalized, all-caps)
    since a word like "capital" and " capital" can be different tokens.
    Only strings that tokenize to exactly one token are kept -- this mirrors
    how the head-output projection can only ever "vote" for single tokens.

    Doing this ONCE up front, rather than decoding每 top-k token back to a
    string at score time, is the main speedup: membership becomes an
    integer tensor comparison instead of a per-token string operation.
    """
    token_ids = set()
    variants = lambda w: [w, " " + w, w.upper(), w.capitalize(), w.lower()]

    for word in task_words:
        for variant in variants(word):
            ids = model.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(ids[0])

    return torch.tensor(sorted(token_ids))


def compute_maps_scores(model, prompts, task_relation_words, k_list, n_match, batch_size=10):
    """
    Vectorized MAPS scoring: batches across prompts AND across heads within
    a layer, and checks top-k matches via integer token-ID comparison
    instead of decoding each candidate token back to a string.

    Returns: {k: np.ndarray of shape (n_prompts, n_layers, n_heads)}
    """
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    n_prompts = len(prompts)
    candidate_ids = convert_task_words_to_token_ids(model, task_relation_words).to(model.cfg.device)

    scores = {k: np.zeros((n_prompts, n_layers, n_heads), dtype=np.int8) for k in k_list}

    for batch_start in range(0, n_prompts, batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        tokens = model.to_tokens(batch_prompts, padding_side="left")
        current_bs = tokens.shape[0]

        _, cache = model.run_with_cache(
            tokens, names_filter=lambda name: name.endswith("hook_z")
        )

        for layer in range(n_layers):
            z = cache["z", layer][:, -1, :, :]              # (batch, n_heads, d_head)
            W_O = model.W_O[layer]                            # (n_heads, d_head, d_model)

            # one batched matmul across ALL heads, not a per-head loop
            head_outputs = torch.einsum("bnd,ndm->bnm", z, W_O)   # (batch, n_heads, d_model)
            normalized = model.ln_final(head_outputs)               # (batch, n_heads, d_model)

            # ONE matmul projects every head to vocab space at once
            projections = normalized @ model.W_U                     # (batch, n_heads, d_vocab)

            for k in k_list:
                topk_ids = projections.topk(k, dim=-1).indices        # (batch, n_heads, k)
                # integer comparison against candidate_ids, no decoding
                matches = (topk_ids.unsqueeze(-1) == candidate_ids.view(1, 1, 1, -1))  # (batch, n_heads, k, n_candidates)
                n_hits = matches.any(dim=-1).sum(dim=-1)                # (batch, n_heads)
                scores[k][batch_start:batch_start + current_bs, layer, :] = (n_hits >= n_match).cpu().numpy()

        print(f"  scored batch {batch_start}-{batch_start + current_bs}/{n_prompts}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, required=True,
        help="e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True,
        help="task name, e.g. country-capital -- must match a file in dataset_folder")
    parser.add_argument("--n_shot", type=int, default=10,
        help="number of few-shot examples per prompt")
    parser.add_argument("--dataset_folder", type=str,
        default=os.path.join(SCRIPT_DIR, "datasets", "abstractive"))
    parser.add_argument("--task_relation_dict_path", type=str,
        default=os.path.join(SCRIPT_DIR, "datasets", "dataset_info", "task_relation_dict.json"),
        help="JSON file mapping each task name to its list of descriptive words")
    parser.add_argument("--save_root", type=str,
        default=os.path.join(SCRIPT_DIR, "output"))
    parser.add_argument("--k_list", type=int, nargs="+", default=[20, 25])
    parser.add_argument("--n_match", type=int, default=1)
    parser.add_argument("--exp_size", type=int, default=100,
        help="max number of correct examples to score")

    args = parser.parse_args()
    model_name_short = args.model_name.split("/")[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("model_name:", args.model_name)
    print("d_name:", args.d_name)
    print("n_shot:", args.n_shot)

    model = HookedTransformer.from_pretrained(args.model_name, device=device, dtype=torch.bfloat16)

    with open(os.path.join(args.dataset_folder, f"{args.d_name}.json")) as f:
        dataset = json.load(f)

    with open(args.task_relation_dict_path) as f:
        task_relation_dict = json.load(f)
    task_relation_words = task_relation_dict[args.d_name]

    prompts, answers = create_few_shot_prompts(
        dataset, n_shot=args.n_shot, delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":"
    )

    correctness = check_correctness(model=model, prompts=prompts, answers=answers, batch_size=10)
    correct_index = correctness["correct_index"][: args.exp_size]
    correct_prompts = [prompts[i] for i in correct_index]
    print(f"scoring {len(correct_prompts)} model-correct examples (of {len(prompts)})")

    scores = compute_maps_scores(
        model, correct_prompts, task_relation_words,
        k_list=args.k_list, n_match=args.n_match,
    )

    save_dir = os.path.join(
        args.save_root, model_name_short, args.d_name, "Heads", "MAPS", "Relation_across_tasks"
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"{args.d_name}_MAPS_Relation_heads_across_tasks_EP_{args.n_shot}_correct.pkl"
    )

    pickled_scores = {k: {args.d_name: arr} for k, arr in scores.items()}
    with open(save_path, "wb") as f:
        pickle.dump(pickled_scores, f)

    print(f"saved MAPS scores to {os.path.abspath(save_path)}")
    for k in args.k_list:
        fraction = scores[k].mean(axis=0)  # (n_layers, n_heads)
        top_heads = np.argwhere(fraction >= 0.1)
        print(f"k={k}: {len(top_heads)} heads at fraction >= 0.1")