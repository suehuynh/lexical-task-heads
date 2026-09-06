"""
Stage 2 -- Upstream Task-Vector Patching.

Cache a mean-pooled residual-stream "task vector" from a SOURCE task (Antonym),
inject it into the residual stream feeding a target Lexical Task Head (L9H4 by
default) on a TARGET task (Country-Capital), and measure:

  (a) Vocabulary shift: what the target LTH's output now projects to
      (antonym-relation words vs city-relation words).
  (b) Behavioural degradation/steering: target-task accuracy collapse and the
      probability mass the final model puts on antonym- vs city-relation tokens.

Primary result  : hard REPLACE of resid_pre[inject_layer] at the final position
                  with the pooled source vector.
Secondary sweep : additive  h[:, -1] += alpha * (v_source - v_target)  for a
                  steering curve over alpha.
Controls        : self-patch (inject v_target), random norm-matched vector.
"""

import os
import json
import argparse
from functools import partial

import torch
import numpy as np
import plotly.graph_objects as go
from transformer_lens import HookedTransformer, utilities

from prompt_builders import create_few_shot_prompts, check_correctness
from metrics import early_decode, topk_match_count

torch.set_grad_enabled(False)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_KW = dict(delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":")


# --------------------------------------------------------------------------- #
# prompt / data helpers
# --------------------------------------------------------------------------- #
def load_correct_prompts(model, dataset_folder, d_name, n_shot, exp_size, batch_size):
    """Few-shot prompts for `d_name`, filtered to the ones the model answers correctly."""
    with open(os.path.join(dataset_folder, f"{d_name}.json"), encoding="utf-8") as f:
        dataset = json.load(f)

    prompts, answers = create_few_shot_prompts(dataset, n_shot=n_shot, **PROMPT_KW)
    correctness = check_correctness(
        model=model, prompts=prompts, answers=answers, batch_size=batch_size,
        return_answer_tokens=True,
    )
    idx = correctness["correct_index"][:exp_size]
    correct_prompts = [prompts[i] for i in idx]
    # first answer-token id per kept prompt (gold "capital" token for the target task)
    gold_tokens = correctness["answer_tokens"][idx].to(model.cfg.device)
    print(f"  {d_name}: kept {len(correct_prompts)} model-correct prompts")
    return correct_prompts, gold_tokens


def real_token_mask(tokens, pad_id):
    if pad_id is None:
        return torch.ones_like(tokens, dtype=torch.bool)
    return tokens != pad_id


# --------------------------------------------------------------------------- #
# task vector
# --------------------------------------------------------------------------- #
def pooled_task_vector(model, prompts, layer, batch_size=16):
    """Mean-pool resid_pre[layer] over non-pad positions, then over all prompts."""
    hook_name = utils.get_act_name("resid_pre", layer)
    pad_id = model.tokenizer.pad_token_id

    sums, counts = [], 0
    acc = torch.zeros(model.cfg.d_model, device=model.cfg.device, dtype=torch.float32)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        toks = model.to_tokens(batch, padding_side="left")
        _, cache = model.run_with_cache(
            toks, names_filter=lambda n: n == hook_name, return_type=None
        )
        resid = cache[hook_name].float()                     # (b, pos, d_model)
        mask = real_token_mask(toks, pad_id).unsqueeze(-1)   # (b, pos, 1)
        per_prompt = (resid * mask).sum(1) / mask.sum(1).clamp(min=1)  # (b, d_model)
        acc += per_prompt.sum(0)
        counts += per_prompt.shape[0]
    return acc / counts                                      # (d_model,)


# --------------------------------------------------------------------------- #
# injection hook
# --------------------------------------------------------------------------- #
def inject_hook(resid, hook, mode, vec=None, diff=None, alpha=1.0):
    """Patch the residual stream at the FINAL position only."""
    if mode == "replace":
        resid[:, -1, :] = vec.to(resid.dtype)
    elif mode == "add":
        resid[:, -1, :] = resid[:, -1, :] + alpha * diff.to(resid.dtype)
    return resid


# --------------------------------------------------------------------------- #
# readout
# --------------------------------------------------------------------------- #
def run_condition(model, prompts, gold_tokens, inject_layer, head,
                  relation_words, batch_size, hook_fn=None):
    """
    One forward pass (optionally with an injection hook) over `prompts`.
    Returns dict of aggregate metrics + summed vocab logits of the target head
    at the final position (for a top-k token table).
    """
    z_name = utils.get_act_name("z", inject_layer)
    W_O_h = model.W_O[inject_layer, head]                    # (d_head, d_model)

    n = 0
    correct = 0
    gold_logit_sum = 0.0
    gold_prob_sum = 0.0
    head_antonym_hits = 0
    head_city_hits = 0
    head_logit_acc = torch.zeros(model.cfg.d_vocab, device=model.cfg.device, dtype=torch.float32)
    final_logit_acc = torch.zeros(model.cfg.d_vocab, device=model.cfg.device, dtype=torch.float32)

    ant_words = relation_words["antonym"]
    city_words = relation_words["country-capital"]

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        gold = gold_tokens[start:start + batch_size]
        toks = model.to_tokens(batch, padding_side="left")

        model.reset_hooks()
        if hook_fn is not None:
            model.add_hook(utils.get_act_name("resid_pre", inject_layer), hook_fn, level=1)

        logits, cache = model.run_with_cache(
            toks, names_filter=lambda nm: nm == z_name, return_type="logits",
        )
        model.reset_hooks()

        final_logits = logits[:, -1, :].float()             # (b, d_vocab)
        pred = final_logits.argmax(-1)
        correct += (pred == gold).sum().item()
        gold_logit_sum += final_logits.gather(1, gold[:, None]).sum().item()
        gold_prob_sum += final_logits.softmax(-1).gather(1, gold[:, None]).sum().item()
        final_logit_acc += final_logits.sum(0)

        z = cache[z_name][:, -1, head, :].float()           # (b, d_head)
        head_out = z @ W_O_h                                # (b, d_model)
        head_logits = early_decode(head_out, model).float() # (b, d_vocab)
        head_logit_acc += head_logits.sum(0)
        for row in head_logits:
            head_antonym_hits += topk_match_count(row, ant_words, model.tokenizer, k=15)
            head_city_hits += topk_match_count(row, city_words, model.tokenizer, k=15)

        n += len(batch)

    return dict(
        n=n,
        accuracy=correct / n,
        gold_logit=gold_logit_sum / n,
        gold_prob=gold_prob_sum / n,
        head_antonym_match=head_antonym_hits / n,
        head_city_match=head_city_hits / n,
        head_logit_sum=head_logit_acc,
        final_logit_sum=final_logit_acc,
    )


def top_tokens(logit_sum, model, k=15):
    ids = logit_sum.topk(k).indices.tolist()
    return [model.tokenizer.decode([i]) for i in ids]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--source_d_name", type=str, default="antonym")
    p.add_argument("--target_d_name", type=str, default="country-capital")
    p.add_argument("--n_shot", type=int, default=10)
    p.add_argument("--inject_layer", type=int, default=9)
    p.add_argument("--head", type=int, default=4)
    p.add_argument("--exp_size", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--dataset_folder", type=str,
                   default=os.path.join(SCRIPT_DIR, "datasets", "abstractive"))
    p.add_argument("--task_relation_dict_path", type=str,
                   default=os.path.join(SCRIPT_DIR, "datasets", "dataset_info",
                                        "task_relation_dict.json"))
    p.add_argument("--save_root", type=str, default=os.path.join(SCRIPT_DIR, "output"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_short = args.model_name.split("/")[-1]
    print(f"device={device}  model={args.model_name}")
    print(f"inject resid_pre L{args.inject_layer} -> read L{args.inject_layer}H{args.head}")

    model = HookedTransformer.from_pretrained(
        args.model_name, device=device, dtype=torch.bfloat16
    )

    with open(args.task_relation_dict_path) as f:
        rel = json.load(f)
    relation_words = {"antonym": rel["antonym"], "country-capital": rel["country-capital"]}

    # ---- prompts -----------------------------------------------------------
    src_prompts, _ = load_correct_prompts(
        model, args.dataset_folder, args.source_d_name, args.n_shot,
        args.exp_size, args.batch_size,
    )
    tgt_prompts, tgt_gold = load_correct_prompts(
        model, args.dataset_folder, args.target_d_name, args.n_shot,
        args.exp_size, args.batch_size,
    )

    # ---- task vectors ----------------------------------------------------- #
    v_source = pooled_task_vector(model, src_prompts, args.inject_layer)
    v_target = pooled_task_vector(model, tgt_prompts, args.inject_layer)
    diff = v_source - v_target
    print(f"|v_source|={v_source.norm():.2f}  |v_target|={v_target.norm():.2f}  "
          f"|diff|={diff.norm():.2f}")

    g = torch.Generator(device=device).manual_seed(args.seed)
    v_random = torch.randn(model.cfg.d_model, generator=g, device=device, dtype=torch.float32)
    v_random = v_random / v_random.norm() * v_source.norm()

    common = dict(model=model, prompts=tgt_prompts, gold_tokens=tgt_gold,
                  inject_layer=args.inject_layer, head=args.head,
                  relation_words=relation_words, batch_size=args.batch_size)

    conditions = {}

    print("\n[clean]")
    conditions["clean"] = run_condition(**common, hook_fn=None)

    print("[replace: source task vector]")
    conditions["replace_source"] = run_condition(
        **common, hook_fn=partial(inject_hook, mode="replace", vec=v_source))

    print("[control: self-patch (replace with target task vector)]")
    conditions["control_self_patch"] = run_condition(
        **common, hook_fn=partial(inject_hook, mode="replace", vec=v_target))

    print("[control: random norm-matched vector]")
    conditions["control_random"] = run_condition(
        **common, hook_fn=partial(inject_hook, mode="replace", vec=v_random))

    for a in args.alphas:
        print(f"[add: alpha={a}]")
        conditions[f"add_alpha_{a}"] = run_condition(
            **common, hook_fn=partial(inject_hook, mode="add", diff=diff, alpha=a))

    # ---- serialise ------------------------------------------------------- #
    save_dir = os.path.join(args.save_root, model_short, args.target_d_name,
                            "Stage2_task_vector_patch")
    os.makedirs(save_dir, exist_ok=True)

    results = {
        "meta": {
            "model": model_short, "source_d_name": args.source_d_name,
            "target_d_name": args.target_d_name, "n_shot": args.n_shot,
            "inject_layer": args.inject_layer, "head": args.head,
            "n_source": len(src_prompts), "n_target": len(tgt_prompts),
            "alphas": args.alphas,
            "v_source_norm": v_source.norm().item(),
            "v_target_norm": v_target.norm().item(),
            "diff_norm": diff.norm().item(),
        },
        "conditions": {},
    }
    for name, c in conditions.items():
        results["conditions"][name] = {
            "n": c["n"],
            "cc_accuracy": c["accuracy"],
            "gold_logit": c["gold_logit"],
            "gold_prob": c["gold_prob"],
            "head_antonym_match@15": c["head_antonym_match"],
            "head_city_match@15": c["head_city_match"],
            "head_top15_tokens": top_tokens(c["head_logit_sum"], model),
            "final_top15_tokens": top_tokens(c["final_logit_sum"], model),
        }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved", os.path.join(save_dir, "results.json"))

    # ---- vocab-shift markdown table ------------------------------------- #
    lines = ["# Stage 2 vocab shift -- L%dH%d output (top-15 tokens, final position)\n"
             % (args.inject_layer, args.head)]
    for name in ["clean", "replace_source", "control_self_patch", "control_random"]:
        c = results["conditions"][name]
        lines.append(f"## {name}  (cc_acc={c['cc_accuracy']:.3f}, "
                     f"antonym@15={c['head_antonym_match@15']:.2f}, "
                     f"city@15={c['head_city_match@15']:.2f})")
        lines.append("`" + "` `".join(t.strip() for t in c["head_top15_tokens"]) + "`\n")
    with open(os.path.join(save_dir, "vocab_shift_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---- steering curve ------------------------------------------------- #
    xs = [0.0] + list(args.alphas)
    acc = [conditions["clean"]["accuracy"]] + [conditions[f"add_alpha_{a}"]["accuracy"]
                                               for a in args.alphas]
    ant = [conditions["clean"]["head_antonym_match"]] + [
        conditions[f"add_alpha_{a}"]["head_antonym_match"] for a in args.alphas]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=acc, name="Country-Capital accuracy", yaxis="y1")
    fig.add_scatter(x=xs, y=ant, name="L%dH%d antonym match@15" % (args.inject_layer, args.head),
                    yaxis="y2")
    fig.update_layout(
        title="Additive steering: h[:, -1] += alpha * (v_antonym - v_cc)",
        xaxis_title="alpha",
        yaxis=dict(title="accuracy", range=[0, 1]),
        yaxis2=dict(title="antonym match@15", overlaying="y", side="right"),
        width=800, height=500,
    )
    fig.write_html(os.path.join(save_dir, "steering_curve.html"))
    print("saved", os.path.join(save_dir, "steering_curve.html"))

    # ---- console summary ---------------------------------------------------
    print("\n=== summary ===")
    print(f"{'condition':<22} {'cc_acc':>7} {'gold_p':>8} {'ant@15':>7} {'city@15':>8}")
    for name, c in results["conditions"].items():
        print(f"{name:<22} {c['cc_accuracy']:>7.3f} {c['gold_prob']:>8.4f} "
              f"{c['head_antonym_match@15']:>7.2f} {c['head_city_match@15']:>8.2f}")
