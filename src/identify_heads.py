import os
import json
import argparse
from transformer_lens import HookedTransformer
import torch

from Submodules.identify_heads_early_decode import *

torch.set_grad_enabled(False)

if __name__ == "__main__":
    """
    Identify lexical task heads or retrieval heads
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
        help="model name e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True,)
    parser.add_argument("--prompt_type", type=str, required=True, help="prompt type: EP or IP")
    parser.add_argument("--component_type", type=str, required=True, 
        help="component type: lexical_task or Retrieval heads")
    parser.add_argument("--monitor_other_task", type=bool, default=False, 
        help="whether to monitor other task's lexical task heads while running the prompts for a giventarget task")
    parser.add_argument("--save_root", type=str, 
        default="../output",)
    parser.add_argument("--project_root", type=str, 
        default="../",
        help="directory of the codebase ")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--k", type=int, default=10, help="top k decoded tokens to look for match")
    parser.add_argument("--exp_size", type=int, default=100, help="number of examples to sample from the dataset")
    parser.add_argument("--prompt_template_index", type=int, default=None,
        help="EP n_shot count to use (default 5 if not set). Ignored for IP, "
             "which auto-selects its best-accuracy instruction template.")

    args = parser.parse_args()
    model_name = args.model_name
    save_root = args.save_root
    project_root = args.project_root
    d_name = args.d_name
    prompt_type = args.prompt_type
    monitor_other_task = args.monitor_other_task
    batch_size = args.batch_size
    k = args.k
    exp_size = args.exp_size
    component_type = args.component_type
    if component_type == "lexical_task":
        component_type = "Relation"

    print("d_name", d_name)
    print("prompt_type", prompt_type)

    # Load
    with open(os.path.join(project_root, "datasets", "dataset_info", 
        f"instruction_dict.json"), "r"
    ) as f:
        instruction_dict = json.load(f)
    with open(os.path.join(project_root, "datasets", "dataset_info", 
        "task_relation_dict.json"), "r"
    ) as f:
        task_relation_dict = json.load(f)

    # Parameters 
    # n_match_list=[1,2,3]
    # k_list=None
    # n_match = None
    
    n_match_list = None
    k_list=[20,25]
    n_match = 1

    if monitor_other_task:
        task_list = task_relation_dict.keys()
    else:
        task_list = [d_name]
    print("monitor_other_task", monitor_other_task)
    print("n_match_list", n_match_list)
    print("component_type", component_type)
    print("d_name", d_name)
    print("prompt_type", prompt_type)

    # Load model 
    print("model_name", model_name)
    model = HookedTransformer.from_pretrained(model_name, device='cuda')
    model_name = model_name.split("/")[-1]

    if prompt_type == "IP":
        ## load behavior results
        with open(os.path.join(save_root, model_name,
            "across_tasks", "Behavior", "IP_vary_n_inst_behavior.json"), "r") as f:
            result_dict_IP_tasks = json.load(f)
            
        ## get best acc index
            acc_list = [result_dict_IP_tasks[d_name][str(i)]['accuracy'] for i in range(5)]
            best_index = np.argmax(acc_list).item()
            prompt_template_index = best_index
    elif prompt_type == "EP":
        prompt_template_index = args.prompt_template_index if args.prompt_template_index is not None else 5
    else:
        raise ValueError(f"prompt_type {prompt_type} not supported")

    # Run and save the results
    result = get_save_MAPS_scores_across_tasks(
        model=model, model_name=model_name,
        prompt_type=prompt_type, prompt_template_index=prompt_template_index,
        d_name=d_name, correct_or_incorrect="correct",
        task_relation_dict=task_relation_dict,
        batch_size=batch_size, k=k, exp_size=exp_size,
        n_match_list=n_match_list, task_list=task_list, 
        result_type="each_sample_in_batch",
        component_type=component_type,
        instruction_dict=instruction_dict,
        save=True, save_root=save_root,     
        k_list=k_list, n_match=n_match,
    )