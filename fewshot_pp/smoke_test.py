import torch
from transformer_lens import HookedTransformer

from path_patching_tl import (
    get_path_patch_head_to_heads,
    get_path_patch_head_to_LTH_vocab,
)
from metrics import l2_norm_effect
from identify_lth import convert_task_words_to_token_ids

torch.set_grad_enabled(False)


class TokDataset:
    def __init__(self, toks):
        self.toks = toks


def run_smoke_test(model_name="meta-llama/Llama-3.2-1B-Instruct", receiver_input="v"):
    """
    Minimal end-to-end check of both path-patching entry points: tiny prompt
    set, a single shallow receiver head (so the sender search space is small),
    no dependency on io_loaders / behavior-json / dataset files. Meant to fail
    fast (seconds, not minutes) on shape / indexing / metric-signature bugs
    before committing to a full run_fewshot_pp.py run.
    """
    print(f"loading {model_name} ...")
    model = HookedTransformer.from_pretrained(model_name)

    clean_prompts = [
        " France: Paris; Germany: Berlin; Italy:",
        " Spain: Madrid; Japan: Tokyo; China:",
    ]
    corrupt_prompts = [
        " run: ran; eat: ate; Italy:",
        " go: went; see: saw; China:",
    ]

    n_clean = len(clean_prompts)
    all_toks = model.to_tokens(clean_prompts + corrupt_prompts, padding_side="left")
    clean_dataset = TokDataset(all_toks[:n_clean])
    corrupt_dataset = TokDataset(all_toks[n_clean:])

    # One shallow receiver head keeps the sender loop tiny: it only searches
    # layers < max(receiver_layers).
    receiver_list = [(2, 3)]
    expected_shape = (max(layer for layer, _ in receiver_list), model.cfg.n_heads)

    z_filter = lambda name: name.endswith("z")
    _, clean_z_cache = model.run_with_cache(clean_dataset.toks, names_filter=z_filter, return_type=None)
    _, corrupt_z_cache = model.run_with_cache(corrupt_dataset.toks, names_filter=z_filter, return_type=None)

    l2 = get_path_patch_head_to_heads(
        receiver_heads=receiver_list,
        receiver_input=receiver_input,
        model=model,
        patching_metric=l2_norm_effect,
        new_dataset=corrupt_dataset,
        orig_dataset=clean_dataset,
        new_cache=corrupt_z_cache,
        orig_cache=clean_z_cache,
    )
    print("l2_norm results.shape:", tuple(l2.shape))
    assert tuple(l2.shape) == expected_shape, f"l2: expected {expected_shape}, got {tuple(l2.shape)}"
    assert torch.isfinite(l2).all(), "l2_norm results contain NaN/Inf"

    task_token_ids = convert_task_words_to_token_ids(
        model, ["city", "cities", "capital", "capitals"]
    ).to(model.cfg.device)

    lprr = get_path_patch_head_to_LTH_vocab(
        receiver_heads=receiver_list,
        receiver_input=receiver_input,
        model=model,
        task_token_ids=task_token_ids,
        clean_dataset=clean_dataset,
        corrupt_dataset=corrupt_dataset,
        clean_z_cache=clean_z_cache,
        corrupt_z_cache=corrupt_z_cache,
    )
    print("lprr results.shape:", tuple(lprr.shape))
    assert tuple(lprr.shape) == expected_shape, f"lprr: expected {expected_shape}, got {tuple(lprr.shape)}"
    # lprr may legitimately be all-NaN on this toy set (tiny V_task separation);
    # only require that it isn't Inf and that at least the shape/plumbing works.
    assert not torch.isinf(lprr).any(), "lprr results contain Inf"

    print(f"[PASS] smoke test ok (receiver_input={receiver_input!r})")


if __name__ == "__main__":
    run_smoke_test()
