import os
import json
import argparse
import torch as t
from torch import Tensor
from collections import defaultdict
from jaxtyping import Bool, Float, Int
from typing import Callable, Literal
from tqdm.auto import tqdm
import einops
import numpy
import functools
from functools import partial
from itertools import product

from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.components import MLP, Embed, LayerNorm, Unembed
from transformer_lens.hook_points import HookPoint

# PATH PATCHING
# Step 1: Run and cache the head activation through 
# clean and corrupted prompts
# Step 2: Run clean prompt with sender node's output
# is patched from corrupted prompts and other heads/MLP
# activations are forced to freeze and cache receiver's
# inputs
# Step 3: Run through clean prompts with receiver's
# inputs are patched from previous run while upstream
# activations are frozen. Record the outputs of run 2.
# Step 4: Measure changes in metrics from orginal clean
# run with one from receiver-corrupt run.

device = "cuda" if t.cuda.is_available() else "cpu"

def get_model_specs_tl(model: HookedTransformer) -> dict:
    """
    TransformerLens equivalent of Shared_utils.wrapper.get_model_specs
    """
    cfg = model.cfg
    return dict(
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        d_mlp=cfg.d_mlp,
    )


def find_earliest_receiver(receiver_list: list[tuple[int, int]]) -> tuple[int, int]:
    """
    Finds the computationally earliest component in a list of receivers.
    Order: Layer index first, then Attention Heads < MLP within a layer.
    """

    def compare_receivers(item1: tuple[int, int], item2: tuple[int, int]) -> int:
        layer1, comp1 = item1
        layer2, comp2 = item2
        if layer1 < layer2:
            return -1
        elif layer1 > layer2:
            return 1
        else:
            if comp1 >= 0 and comp2 == -1:
                return -1  # Attn < MLP
            elif comp1 == -1 and comp2 >= 0:
                return 1  # MLP > Attn
            else:
                return 0  # Order between heads doesn't matter

    if not receiver_list:
        raise ValueError("receiver_list cannot be empty for find_earliest_receiver")
    return sorted(receiver_list, key=functools.cmp_to_key(compare_receivers))[0]


def _resolve_pos(pos: int, seq_len: int) -> int:
    return pos if pos >= 0 else seq_len + pos

def patch_head_input(
    orig_activation: Float[Tensor, "batch pos head_idx d_head"],
    hook: HookPoint,
    patched_cache: ActivationCache,
    head_list: list[tuple[int, int]],
) -> Float[Tensor, "batch pos head_idx d_head"]:
    """
    Function which can patch any combination of heads in layers,
    according to the heads in head_list.
    """
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
    return orig_activation

def patch_or_freeze_head_vectors(
    orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: tuple[int, int],
) -> Float[Tensor, "batch pos head_index d_head"]:
    """
    This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them to their
    values in orig_cache), except for head_to_patch (if it's in this layer) which we patch with the
    value from new_cache.

    head_to_patch: tuple of (layer, head)
    """
    # Setting using ..., otherwise changing orig_head_vector will edit cache value too
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector

def get_path_patch_head_to_heads(
    receiver_heads: list[tuple[int, int]],
    receiver_input: str,
    model: HookedTransformer,
    patching_metric: Callable,
    new_dataset,
    orig_dataset,
    new_cache: ActivationCache | None = None,
    orig_cache: ActivationCache | None = None,
) -> Float[Tensor, "layer head"]:
    """
    Performs path patching (see algorithm at the top), with:

        sender head = each head above the LTHs, loop through one at a time
        receiver node = input to a set of LTHs

    The receiver node is specified by receiver_heads and receiver_input, for example if
    receiver_input = "v" and receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)], we're doing path
    patching from each head to the value inputs of the LTHs.

    Returns:
        tensor of metric values for every possible sender head
    """
    model.reset_hooks()

    assert receiver_input in ("k", "q", "v")
    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    # Under grouped-query attention, hook_k/hook_v are indexed by KV-head (fewer
    # heads than hook_q/hook_z). receiver_heads is in query/z-head space (0..n_heads-1),
    # so remap into KV-head space before indexing k/v tensors with it.
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", None) or model.cfg.n_heads
    if receiver_input in ("k", "v") and n_kv_heads != model.cfg.n_heads:
        group_size = model.cfg.n_heads // n_kv_heads
        receiver_heads_for_patch = [(layer, head // group_size) for layer, head in receiver_heads]
    else:
        receiver_heads_for_patch = receiver_heads

    results = t.zeros(max(receiver_layers), model.cfg.n_heads, device=device, dtype=t.float32)

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we
    # only cache the things we need (in this case, just attn head outputs).
    z_name_filter = lambda name: name.endswith("z")
    if new_cache is None:
        _, new_cache = model.run_with_cache(new_dataset.toks, names_filter=z_name_filter, return_type=None)
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(orig_dataset.toks, names_filter=z_name_filter, return_type=None)

    # Clean baseline for the receiver's input (no patching at all) -- this is the
    # "clean_head_output" that patching_metric compares each sender's effect against.
    # It doesn't depend on (sender_layer, sender_head), so compute it once.
    _, clean_receiver_cache = model.run_with_cache(
        orig_dataset.toks, names_filter=receiver_hook_names_filter, return_type=None
    )

    def _gather_receiver_vec(cache):
        return t.stack(
            [cache[utils.get_act_name(receiver_input, layer)][:, :, head]
             for layer, head in receiver_heads_for_patch],
            dim=0,
        )  # [n_receiver_heads, batch, pos, d_head]

    clean_receiver_vec = _gather_receiver_vec(clean_receiver_cache)

    # Note, the sender layer will always be before the final receiver layer, otherwise there will
    # be no causal effect from sender -> receiver. So we only need to loop this far.
    for sender_layer, sender_head in tqdm(list(product(range(max(receiver_layers)), range(model.cfg.n_heads)))):
        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen.
        # This directly gives us the receiver's input under this single-sender
        # intervention -- exactly the "patched_head_output" patching_metric wants,
        # so there's no need for a further step 3 run through the rest of the model.

        model.reset_hooks()
        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=new_cache,
            orig_cache=orig_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn, level=1)

        _, patched_cache = model.run_with_cache(
            orig_dataset.toks, names_filter=receiver_hook_names_filter, return_type=None
        )
        assert set(patched_cache.keys()) == set(receiver_hook_names)

        patched_receiver_vec = _gather_receiver_vec(patched_cache)

        # Save the results
        results[sender_layer, sender_head] = patching_metric(clean_receiver_vec, patched_receiver_vec)

    model.reset_hooks()
    return results