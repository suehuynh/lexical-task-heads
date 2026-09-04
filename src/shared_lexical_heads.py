import os
import json
import argparse
import numpy as np

from Shared_utils.shared_utils import (
    get_task_heads_list_k,
    group_ranked_locations_by_layer,
    check_intersection,
)


def resolve_ip_index(save_root: str, model_name: str, d_name: str) -> int:
    """
    Resolve the best-accuracy IP instruction-template index, same logic
    identify_heads.py uses internally (argmax accuracy over templates 0..4).
    """
    path = os.path.join(save_root, model_name, "across_tasks", "Behavior",
        "IP_vary_n_inst_behavior.json")
    with open(path, "r") as f:
        data = json.load(f)
    acc_list = [data[d_name][str(i)]["accuracy"] for i in range(5)]
    return int(np.argmax(acc_list))


def get_shared_lexical_heads(
    model_name: str, d_name: str,
    ep_index: int, ip_index: int, k: int,
    maps_threshold: float = 0.1,
    save_root: str = "../output",
    component_type: str = "Relation",
) -> dict:
    """
    Load the EP-<ep_index> and IP-<ip_index> lexical-task head sets for d_name
    (from identify_heads.py's k-keyed MAPS pickles) and intersect them.

    Returns:
        dict with keys: "ep", "ip" (flat lists of (layer, head)),
        "shared_by_layer" (dict[layer -> [head, ...]]), "shared" (flat list),
        "total" (int, len of "shared").
    """
    ep = get_task_heads_list_k(
        target_task=d_name, other_task=d_name, prompt_type="EP",
        prompt_template_index=ep_index, k=k, MAPS_score_threshold=maps_threshold,
        correct_incorrect="correct", model_name=model_name,
        save_root=save_root, component_type=component_type,
    )
    ip = get_task_heads_list_k(
        target_task=d_name, other_task=d_name, prompt_type="IP",
        prompt_template_index=ip_index, k=k, MAPS_score_threshold=maps_threshold,
        correct_incorrect="correct", model_name=model_name,
        save_root=save_root, component_type=component_type,
    )
    shared_by_layer, total = check_intersection(
        group_ranked_locations_by_layer(ep),
        group_ranked_locations_by_layer(ip),
    )
    flat_shared = [(layer, head) for layer, heads in shared_by_layer.items() for head in heads]

    return {
        "ep": ep, "ip": ip,
        "shared_by_layer": shared_by_layer,
        "shared": flat_shared,
        "total": total,
    }


def report_shared_heads_across_k(
    model_name: str, d_name: str,
    ep_index: int, ip_index: int,
    k_list=(20, 25),
    maps_threshold: float = 0.1,
    save_root: str = "../output",
    component_type: str = "Relation",
) -> dict:
    """
    Compute get_shared_lexical_heads for each k in k_list and print a
    per-k summary table of |EP|, |IP|, |shared| head counts.
    """
    out = {}
    for k in k_list:
        info = get_shared_lexical_heads(
            model_name=model_name, d_name=d_name,
            ep_index=ep_index, ip_index=ip_index, k=k,
            maps_threshold=maps_threshold, save_root=save_root,
            component_type=component_type,
        )
        out[k] = info
        print(f"k={k}: |EP-{ep_index}|={len(info['ep'])}  "
              f"|IP-{ip_index}|={len(info['ip'])}  |shared|={info['total']}")
    return out


if __name__ == "__main__":
    """
    Compute the shared lexical-task heads across EP and IP prompt styles
    for a given task, from identify_heads.py's (TransformerLens) k-keyed
    MAPS score pickles.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
        help="model name, e.g. meta-llama/Llama-3.1-8B-Instruct (or its short form, "
             "e.g. Llama-3.1-8B-Instruct, matching identify_heads.py's output dirs)")
    parser.add_argument("--d_name", type=str, default="country-capital")
    parser.add_argument("--ep_index", type=int, default=10,
        help="EP n_shot count used when running identify_heads.py")
    parser.add_argument("--ip_index", type=int, default=None,
        help="IP instruction-template index used when running identify_heads.py; "
             "if omitted, auto-resolved from IP_vary_n_inst_behavior.json (argmax accuracy)")
    parser.add_argument("--k_list", type=int, nargs="+", default=[20, 25],
        help="top-k decoded-token windows to report on (identify_heads.py computes both by default)")
    parser.add_argument("--maps_threshold", type=float, default=0.1,
        help="mean-over-examples MAPS score threshold defining a lexical-task head (p)")
    parser.add_argument("--save_root", type=str, default="../output")
    parser.add_argument("--component_type", type=str, default="Relation")

    args = parser.parse_args()
    model_name = args.model_name.split("/")[-1]

    ip_index = args.ip_index
    if ip_index is None:
        ip_index = resolve_ip_index(args.save_root, model_name, args.d_name)
    print("ep_index", args.ep_index)
    print("ip_index", ip_index)
    print("maps_threshold", args.maps_threshold)

    results = report_shared_heads_across_k(
        model_name=model_name, d_name=args.d_name,
        ep_index=args.ep_index, ip_index=ip_index,
        k_list=args.k_list, maps_threshold=args.maps_threshold,
        save_root=args.save_root, component_type=args.component_type,
    )

    save_dir = os.path.join(args.save_root, model_name, args.d_name,
        "Heads", "MAPS", f"{args.component_type}_across_tasks_vary_k")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,
        f"shared_lexical_heads_EP{args.ep_index}_IP{ip_index}.json")

    serializable = {
        str(k): {
            "ep": info["ep"], "ip": info["ip"],
            "shared_by_layer": info["shared_by_layer"],
            "shared": info["shared"], "total": info["total"],
        }
        for k, info in results.items()
    }
    serializable["meta"] = {
        "model_name": model_name, "d_name": args.d_name,
        "ep_index": args.ep_index, "ip_index": ip_index,
        "maps_threshold": args.maps_threshold,
    }
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print("saved to", save_path)
