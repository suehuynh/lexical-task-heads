import os
import json
import argparse
import torch
from transformer_lens import HookedTransformer

from prompt_builders import create_few_shot_prompts, check_correctness

torch.set_grad_enabled(False)

if __name__ == "__main__":
    """
    EP-only behavior variance: for each n_shot in the sweep, check which
    examples the model answers correctly under few-shot prompting.
    
    Return:
    (result_dict[d_name][n_shot] = {accuracy, correct_index, n_correct_index,
    n_dataset}).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
        help="model name e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="output/")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--dataset_folder", type=str, default="datasets/abstractive")
    parser.add_argument("--n_shot_list", type=int, nargs="+", default=[10])

    args = parser.parse_args()
    model_name_short = args.model_name.split("/")[-1]

    print("model_name", args.model_name)
    print("d_name", args.d_name)
    print("n_shot_list", args.n_shot_list)

    print("loading model...")
    model = HookedTransformer.from_pretrained(args.model_name)
    print("model loaded")

    file_name = "EP_vary_n_shot_behavior.json"
    save_path = os.path.join(args.save_root, model_name_short, "across_tasks", "Behavior")
    print("save_path:", os.path.abspath(save_path))

    if os.path.exists(os.path.join(save_path, file_name)):
        with open(os.path.join(save_path, file_name), "r") as f:
            result_dict = json.load(f)
        print(f"Behavior file {file_name} loaded")
    else:
        print("Behavior file does not exist, creating a new dictionary")
        result_dict = {}


    with open(os.path.join(args.dataset_folder, f"{args.d_name}.json")) as f:
        dataset = json.load(f)

    result_dict[args.d_name] = {}
    for n_shot in args.n_shot_list:
        prompts, answers = create_few_shot_prompts(
            dataset, n_shot=n_shot, delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":"
        )

        prompt_dict = check_correctness(
            model=model, prompts=prompts, answers=answers,
            batch_size=args.batch_size,
        )
        correct_index = prompt_dict["correct_index"]
        acc = len(correct_index) / len(prompts)

        result_dict[args.d_name][n_shot] = {
            "accuracy": acc,
            "correct_index": correct_index,
            "n_correct_index": len(correct_index),
            "n_dataset": len(prompts),
        }
        print(f"  n_shot={n_shot}: accuracy={acc:.3f} ({len(correct_index)}/{len(prompts)})")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, file_name), "w") as f:
        json.dump(result_dict, f)
    print(f"Behavior file {file_name} saved to {save_path}")