import os
import json
import argparse
import torch
from nnsight import LanguageModel

from Shared_utils.prompt_utils import create_few_shot_prompts, create_random_query_prompts, create_zs_prompts, check_correctness
from Shared_utils.shared_utils import rank_heads
from Shared_utils.wrapper import get_model_specs
from Shared_utils.path_patching import path_patch_sender_to_receivers_batch, path_patch_sender_to_logits_via_receivers_batch

torch.set_grad_enabled(False)


def load_receiver_list(save_root, model_name, d_name, ep_index, ip_index, k, component_type="Relation"):
    """
    Load the EP<ep_index>-IP<ip_index> shared lexical-task heads for d_name at
    the given k, as produced by shared_lexical_heads.py, and return them as a
    flat list[(layer, head)] -- the receiver_list format path_patching expects.
    """
    path = os.path.join(save_root, model_name, d_name, "Heads", "MAPS",
        f"{component_type}_across_tasks_vary_k", f"shared_lexical_heads_EP{ep_index}_IP{ip_index}.json")
    with open(path, "r") as f:
        data = json.load(f)
    shared = data[str(k)]["shared"]
    return [tuple(x) for x in shared]


def build_ep_clean_corrupt_answers(dataset_folder, d_name, n_shot, corrupt_type="random_query"):
    with open(os.path.join(dataset_folder, f"{d_name}.json")) as f:
        dataset = json.load(f)

    clean_prompts, answers, _ = create_few_shot_prompts(
        dataset, n_shot=n_shot, delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":")

    if corrupt_type == "random_query":
        corrupt_prompts = create_random_query_prompts(
            dataset, n_shot=n_shot, delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":")
    elif corrupt_type == "zs":
        corrupt_prompts, _ = create_zs_prompts(dataset, return_answers=True)
    else:
        raise ValueError(f"corrupt_type {corrupt_type} not supported")

    return clean_prompts, corrupt_prompts, answers


if __name__ == "__main__":
    """
    Path patching: find which upstream heads/MLPs feed the shared EP/IP
    lexical-task heads for a given task (the receiver set).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
        help="model name, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--d_name", type=str, default="country-capital")
    parser.add_argument("--ep_index", type=int, default=10,
        help="EP n_shot count used when running identify_heads.py / shared_lexical_heads.py")
    parser.add_argument("--ip_index", type=int, required=True,
        help="IP instruction-template index used when running identify_heads.py / shared_lexical_heads.py "
             "(the value shared_lexical_heads.py printed/auto-resolved)")
    parser.add_argument("--k", type=int, required=True,
        help="which k (20 or 25) to take the shared lexical-task head set from")
    parser.add_argument("--pp_prompt_type", type=str, default="EP", choices=["EP"],
        help="prompt style used for the path-patching runs")
    parser.add_argument("--pp_prompt_index", type=int, default=10,
        help="EP n_shot count for the path-patching prompts (default matches --ep_index)")
    parser.add_argument("--corrupt_type", type=str, default="random_query", choices=["random_query", "zs"])
    parser.add_argument("--exp_size", type=int, default=50,
        help="number of model-correct prompts to path-patch over")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--variant", type=str, default="both", choices=["receiver_patch", "strict", "both"])
    parser.add_argument("--include_mlp_receivers", action="store_true",
        help="also add (layer, -1) MLP receivers for each layer that has a shared head")
    parser.add_argument("--dataset_folder", type=str, default="../datasets/abstractive")
    parser.add_argument("--save_root", type=str, default="../output")
    parser.add_argument("--component_type", type=str, default="Relation")
    parser.add_argument("--remote", action="store_true")

    args = parser.parse_args()
    model_name_short = args.model_name.split("/")[-1]

    print("model_name", args.model_name)
    print("d_name", args.d_name)
    print("ep_index", args.ep_index, "ip_index", args.ip_index, "k", args.k)

    receiver_list = load_receiver_list(
        save_root=args.save_root, model_name=model_name_short, d_name=args.d_name,
        ep_index=args.ep_index, ip_index=args.ip_index, k=args.k,
        component_type=args.component_type,
    )
    if not receiver_list:
        raise ValueError(
            f"No shared lexical-task heads found for {args.d_name} at k={args.k}. "
            "Re-run shared_lexical_heads.py with a lower --maps_threshold, or pick a different k."
        )
    print(f"receiver_list ({len(receiver_list)} heads): {receiver_list}")

    if args.include_mlp_receivers:
        layers_with_receivers = sorted({layer for layer, _ in receiver_list})
        receiver_list = receiver_list + [(layer, -1) for layer in layers_with_receivers]
        print(f"receiver_list with MLPs ({len(receiver_list)}): {receiver_list}")

    # Load model
    model = LanguageModel(args.model_name, device_map="auto", dispatch=not args.remote)

    # Build EP clean/corrupt/answer prompts, filter to model-correct examples
    clean_prompts, corrupt_prompts, answers = build_ep_clean_corrupt_answers(
        args.dataset_folder, args.d_name, n_shot=args.pp_prompt_index, corrupt_type=args.corrupt_type)

    correctness = check_correctness(model=model, prompts=clean_prompts, answers=answers,
        batch_size=args.batch_size, remote=args.remote)
    correct_index = correctness["correct_index"][:args.exp_size]
    print(f"using {len(correct_index)} model-correct examples (of {len(clean_prompts)})")

    clean_prompts = [clean_prompts[i] for i in correct_index]
    corrupt_prompts = [corrupt_prompts[i] for i in correct_index]
    answers = [answers[i] for i in correct_index]

    results = {}
    if args.variant in ("receiver_patch", "both"):
        print("Running receiver-patch path patching (path_patch_sender_to_receivers_batch)...")
        results["receiver_patch"] = path_patch_sender_to_receivers_batch(
            model=model, clean_prompts=clean_prompts, corrupt_prompts=corrupt_prompts,
            answers=answers, receiver_list=receiver_list,
            batch_size=args.batch_size, remote=args.remote,
        )
    if args.variant in ("strict", "both"):
        print("Running strict path patching (path_patch_sender_to_logits_via_receivers_batch)...")
        results["strict"] = path_patch_sender_to_logits_via_receivers_batch(
            model=model, clean_prompts=clean_prompts, corrupt_prompts=corrupt_prompts,
            answers=answers, receiver_list=receiver_list,
            batch_size=args.batch_size, remote=args.remote,
        )

    n_heads = get_model_specs(model)["n_heads"]
    save_dir = os.path.join(args.save_root, model_name_short, args.d_name, "Heads", "causal_mediation", "path_patching")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tag = f"{args.pp_prompt_type}{args.pp_prompt_index}_{args.corrupt_type}_k{args.k}"

    ranked = {"meta": {
        "model_name": model_name_short, "d_name": args.d_name,
        "ep_index": args.ep_index, "ip_index": args.ip_index, "k": args.k,
        "pp_prompt_type": args.pp_prompt_type, "pp_prompt_index": args.pp_prompt_index,
        "corrupt_type": args.corrupt_type, "n_examples": len(clean_prompts),
        "receiver_list": receiver_list,
    }}
    for variant, scores in results.items():
        torch.save(scores, os.path.join(save_dir, f"sender_to_shared_lexical_heads_{tag}_{variant}.pt"))
        heads_ranked = rank_heads(scores[:, :n_heads].numpy(), threshold=None)
        mlp_ranked = sorted(
            [(layer, scores[layer, n_heads].item()) for layer in range(scores.shape[0])],
            key=lambda x: x[1], reverse=True,
        )
        ranked[variant] = heads_ranked
        ranked[f"{variant}_mlp"] = mlp_ranked
        print(f"\nTop 15 upstream heads ({variant}):")
        for layer, head, score in heads_ranked[:15]:
            print(f"  L{layer}H{head}: {score:.4f}")

    ranked_path = os.path.join(save_dir, f"sender_to_shared_lexical_heads_{tag}_ranked.json")
    with open(ranked_path, "w") as f:
        json.dump(ranked, f, indent=2)
    print("\nsaved to", save_dir)
