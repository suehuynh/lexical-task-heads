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


def compute_maps_scores(model, prompts, task_relation_words, k_list, n_match):
    """
    Compute the MAPS 0/1 hit array for every (prompt, layer, head), for each
    k in k_list.

    For each prompt, at the final token position:
      1. Run the model once, caching every layer's attention head outputs
         (hook_z).
      2. For each (layer, head), compute that head's individual contribution
         to the residual stream: z_h @ W_O_h (shape d_model).
      3. Project through early_decode (final layernorm + unembedding) to get
         a vocabulary distribution -- "what this head alone would say if
         read out directly."
      4. Take the top-k tokens and count how many appear in
         task_relation_words. Score 1 if the count is >= n_match, else 0.

    Args:
        model: a HookedTransformer.
        prompts: list[str], already filtered to prompts the model answers
            correctly (see check_correctness) -- MAPS scoring is only
            meaningful on examples the model actually gets right.
        task_relation_words: list[str], the task's descriptive term set
            (e.g. from task_relation_dict.json).
        k_list: list[int], the different top-k values to score under.
        n_match: int, minimum number of top-k matches required to count a
            head as a "hit" on a given prompt.

    Returns:
        dict[int, np.ndarray] mapping each k to an array of shape
        (n_prompts, n_layers, n_heads) of 0/1 hit scores.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    max_k = max(k_list)
    relation_words = {w.lower() for w in task_relation_words}

    scores = {k: np.zeros((len(prompts), n_layers, n_heads), dtype=np.int8) for k in k_list}

    W_O = model.W_O  # (n_layers, n_heads, d_head, d_model)

    for prompt_idx, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        # only hook_z is needed -- don't cache every activation
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n.endswith("hook_z")
        )
        last_pos = tokens.shape[-1] - 1

        # (n_layers, n_heads, d_head) at the final position
        z = torch.stack(
            [cache["z", layer][0, last_pos] for layer in range(n_layers)], dim=0
        )
        # each head's residual-stream contribution: (n_layers, n_heads, d_model)
        head_out = torch.einsum("lhd,lhdm->lhm", z, W_O)

        # decode all heads in one unembed matmul instead of n_layers*n_heads
        flat = head_out.reshape(n_layers * n_heads, -1)
        logits = model.unembed(model.ln_final(flat))     # (n_layers*n_heads, d_vocab)
        topk_ids = logits.topk(max_k, dim=-1).indices.cpu()  # (n_lh, max_k)

        # decode each distinct token id once, not once per (head, k)
        id_is_relation = {
            tid: model.tokenizer.decode([tid]).strip().lower() in relation_words
            for tid in torch.unique(topk_ids).tolist()
        }
        match_mask = torch.tensor(
            [[id_is_relation[t] for t in row] for row in topk_ids.tolist()]
        )  # (n_lh, max_k) bool

        for k in k_list:
            hit = (match_mask[:, :k].sum(dim=-1) >= n_match)
            scores[k][prompt_idx] = hit.to(torch.int8).reshape(n_layers, n_heads).numpy()

        if (prompt_idx + 1) % 10 == 0:
            print(f"  scored {prompt_idx + 1}/{len(prompts)} prompts")

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
    parser.add_argument("--k_list", type=list, nargs="+", default=[10])
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

    model = HookedTransformer.from_pretrained(args.model_name, device=device)

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