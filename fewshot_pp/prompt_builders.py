import argparse
import json
import os
import torch

ORIGINAL_D_NAME = "country-capital"
CORRUPT_D_NAME = "present-past"
DATASET_FOLDER = "datasets/abstractive"


def _load_dataset(d_name, dataset_folder=DATASET_FOLDER):
    with open(os.path.join(dataset_folder, f"{d_name}.json"), encoding="utf-8") as f:
        return json.load(f)


def create_few_shot_prompts(data, n_shot, delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":"):
    """Clean few-shot prompts: n_shot demos + a query, all from `data`."""
    prompts, answers = [], []
    for i in range(len(data)):
        context = ""
        for j in range(i - n_shot, i):
            ex = data[j]
            context += f"{q_bos}{ex['input']}{qa_delimiter}{a_bos}{ex['output']}{delimiter}"
        query = data[i]
        context += f"{q_bos}{query['input']}{qa_delimiter}"
        prompts.append(context)
        answers.append(a_bos + query["output"])
    return prompts, answers


def create_corrupt_prompts(n_shot, original_d_name=ORIGINAL_D_NAME, corrupt_d_name=CORRUPT_D_NAME,
                           dataset_folder=DATASET_FOLDER,
                           delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":"):
    """Corrupted few-shot prompts: demos from `corrupt_d_name`, query from `original_d_name`.

    Both datasets are loaded here from their names.
    """
    query_data = _load_dataset(original_d_name, dataset_folder)
    corrupt_data = _load_dataset(corrupt_d_name, dataset_folder)

    prompts, answers = [], []
    for i in range(len(query_data)):
        context = ""
        for j in range(i - n_shot, i):
            ex = corrupt_data[j % len(corrupt_data)]
            context += f"{q_bos}{ex['input']}{qa_delimiter}{a_bos}{ex['output']}{delimiter}"
        context += f"{q_bos}{query_data[i]['input']}{qa_delimiter}"
        prompts.append(context)
        answers.append(a_bos + query_data[i]["output"])
    return prompts, answers

def check_correctness(
    model, prompts: list, answers: list,
    batch_size=10, return_pred_tokens=False, return_answer_tokens=False,
):
    """
    Check correctness and filter only correct generation.
    """
    correct_index = []
    pred_tokens = []

    tokenizer_out = model.tokenizer(
        prompts, padding=True, padding_side="left", return_tensors="pt",
    )
    prompt_tokens = tokenizer_out["input_ids"]
    attention_mask = tokenizer_out["attention_mask"]

    answer_tokens = model.tokenizer(
        answers, add_special_tokens=False, padding=True,
        padding_side="right", return_tensors="pt",
    )["input_ids"][:, 0]

    for i in range(0, len(prompts), batch_size):
        batch_prompt_tokens = prompt_tokens[i : i + batch_size]
        batch_attention_mask = attention_mask[i : i + batch_size]
        batch_answer_tokens = answer_tokens[i : i + batch_size]

        logits = model(
            batch_prompt_tokens,
            attention_mask=batch_attention_mask,
            return_type="logits",
        )
        batch_pred_tokens = logits[:, -1, :].argmax(dim=-1).cpu()

        if return_pred_tokens:
            pred_tokens.extend(batch_pred_tokens.tolist())

        batch_correct = torch.where(batch_answer_tokens == batch_pred_tokens)[0].tolist()
        correct_index += [idx + i for idx in batch_correct]

    return_dict = {"correct_index": correct_index, "prompt_tokens": prompt_tokens}
    if return_pred_tokens:
        return_dict["pred_tokens"] = pred_tokens
    if return_answer_tokens:
        return_dict["answer_tokens"] = answer_tokens

    print(f"Accuracy: {len(correct_index)/len(prompts):.3f} ({len(correct_index)}/{len(prompts)})")
    return return_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_d_name", type=str, default=ORIGINAL_D_NAME,
                        help="task the final query is drawn from")
    parser.add_argument("--corrupt_d_name", type=str, default=CORRUPT_D_NAME,
                        help="task the few-shot demos are drawn from")
    parser.add_argument("--dataset_folder", type=str, default=DATASET_FOLDER)
    parser.add_argument("--n_shot", type=int, default=10)
    parser.add_argument("--n_preview", type=int, default=3)
    args = parser.parse_args()

    data = _load_dataset(args.original_d_name, args.dataset_folder)
    clean_prompts, clean_answers = create_few_shot_prompts(data, n_shot=args.n_shot)
    corrupt_prompts, corrupt_answers = create_corrupt_prompts(
        n_shot=args.n_shot,
        original_d_name=args.original_d_name,
        corrupt_d_name=args.corrupt_d_name,
        dataset_folder=args.dataset_folder,
    )

    for i in range(args.n_preview):
        print(f"[{i}] clean:   {clean_prompts[i]!r} -> {clean_answers[i]!r}")
        print(f"[{i}] corrupt: {corrupt_prompts[i]!r} -> {corrupt_answers[i]!r}")
