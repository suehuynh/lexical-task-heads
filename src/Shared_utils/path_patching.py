import functools
from collections import defaultdict

import torch
from nnsight import LanguageModel
from torch import Tensor
from tqdm.auto import tqdm

from Shared_utils.wrapper import ModelAccessor, get_accessor_config, get_model_specs


def path_patch_component_to_logits_batch(
    model: LanguageModel,
    clean_prompts: list[str],
    corrupt_prompts: list[str],
    answers: list[str],
    batch_size: int = 8,
    remote: bool = True,
) -> torch.Tensor:
    """
    Batched version that runs path patching from each component to logits.

    Args:
        model: Language model to analyze
        clean_prompts: List of original unmodified prompts
        corrupt_prompts: List of modified/corrupted prompts
        answers: List of expected answer strings
        batch_size: Size of each batch
        remote: Whether to run model remotely

    Returns:
        Tensor of path patching results (component effect on logits)
    """
    # Verify inputs
    n_samples = len(clean_prompts)
    if n_samples != len(corrupt_prompts) or n_samples != len(answers):
        raise ValueError("All input lists must have the same length")

    # Get model specs
    spec = get_model_specs(model)
    n_layers, n_heads = spec["n_layers"], spec["n_heads"]

    # Pre-tokenize all inputs
    clean_tokens = model.tokenizer(
        clean_prompts,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )["input_ids"]

    corrupt_tokens = model.tokenizer(
        corrupt_prompts,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )["input_ids"]

    answer_tokens = model.tokenizer(
        answers,
        padding=True,
        padding_side="right",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][:, 0]

    if not remote and torch.cuda.is_available():
        clean_tokens = clean_tokens.to("cuda")
        corrupt_tokens = corrupt_tokens.to("cuda")
        answer_tokens = answer_tokens.to("cuda")

    # Initialize accumulator for results
    all_results = torch.zeros((n_layers, n_heads + 1), device="cpu")

    # Process in batches
    for i in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, n_samples)

        # print(f"Working on batch: {i} to {batch_end}")

        # Extract current batch
        clean_batch = clean_tokens[i:batch_end]
        corrupt_batch = corrupt_tokens[i:batch_end]
        answer_batch = answer_tokens[i:batch_end]

        # Process this batch
        batch_results = path_patch_component_to_logits(
            model=model,
            clean_tokens=clean_batch,
            corrupt_tokens=corrupt_batch,
            answer_tokens=answer_batch,
            remote=remote,
        )

        # Accumulate batch results (weighted by batch size)
        current_batch_size = batch_end - i
        all_results += batch_results * (current_batch_size / n_samples)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def path_patch_component_to_logits(
    model: LanguageModel,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    answer_tokens: torch.Tensor,
    remote: bool = True,
) -> torch.Tensor:
    """
    Core implementation of path patching from components to logits.
    Works with pre-tokenized inputs for batch processing.

    Args:
        model: Language model to analyze
        clean_tokens: Tokenized clean prompts
        corrupt_tokens: Tokenized corrupt prompts
        answer_tokens: Tokenized answer tokens
        remote: Whether to run model remotely

    Returns:
        Tensor of path patching results
    """
    # Get model specs
    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]

    # Get accessor for the model
    accessor_config = get_accessor_config(model)
    accessor = ModelAccessor(model, accessor_config)

    batch_size = len(clean_tokens)

    # Get clean and corrupt logits at answer idx
    clean_logits = model.trace(clean_tokens, trace=False, remote=remote)["logits"]
    corrupt_logits = model.trace(corrupt_tokens, trace=False, remote=remote)["logits"]

    clean_answer_logits = clean_logits[torch.arange(batch_size), -1, answer_tokens]
    corrupt_answer_logits = corrupt_logits[torch.arange(batch_size), -1, answer_tokens]

    # Store clean and corrupt activations
    clean_attn_out = []
    clean_mlp = []
    corrupt_attn_out = []
    corrupt_mlp = []

    # Initialize results tensor
    results = torch.zeros((n_layers, n_heads + 1), device="cpu")

    # Collect clean activations
    with accessor.trace(remote=remote) as tracer:
        with tracer.invoke(clean_tokens):
            for layer in range(n_layers):
                attn_out = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                attn_out_reshaped = attn_out.reshape(batch_size, n_heads, d_head).save()
                clean_attn_out.append(attn_out_reshaped)

                mlp_out = accessor.layers[layer].mlp.unwrap().output[:, -1].save()
                clean_mlp.append(mlp_out)

    # Collect corrupt activations
    with accessor.trace(remote=remote) as tracer:
        with tracer.invoke(corrupt_tokens):
            for layer in range(n_layers):
                attn_out = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                attn_out_reshaped = attn_out.reshape(batch_size, n_heads, d_head).save()
                corrupt_attn_out.append(attn_out_reshaped)

                mlp_out = accessor.layers[layer].mlp.unwrap().output[:, -1].save()
                corrupt_mlp.append(mlp_out)

    # Patch each attention head
    for layer in tqdm(range(n_layers), desc="Patching layers"):
        for head in range(n_heads):
            with accessor.trace(remote=remote) as tracer:
                with tracer.invoke(corrupt_tokens):
                    # Freeze all components to corrupt values
                    for freeze_layer in range(n_layers):
                        freeze_attn_out = accessor.layers[freeze_layer].attention.output.unwrap().input[:, -1]
                        freeze_attn_out_reshaped = freeze_attn_out.reshape(batch_size, n_heads, d_head)
                        freeze_attn_out_reshaped[...] = corrupt_attn_out[freeze_layer]

                        # Patch the target head with clean activation
                        if freeze_layer == layer:
                            freeze_attn_out_reshaped[:, head, :] = clean_attn_out[layer][:, head, :]

                        # Freeze MLPs to corrupt values
                        freeze_mlp = accessor.layers[freeze_layer].mlp.unwrap().output[:, -1]
                        freeze_mlp[...] = corrupt_mlp[freeze_layer]

                # Get patched logits
                patched_logits = accessor.lm_head.unwrap().output[:, -1][torch.arange(batch_size), answer_tokens].save()

            # Calculate normalized intervention effect
            intervention_diff = patched_logits.to("cpu") - corrupt_answer_logits.to("cpu")
            baseline = clean_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
            intervention_effect = intervention_diff.mean() / baseline.mean()
            results[layer, head] = intervention_effect

        # Patch MLP
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(corrupt_tokens):
                # Freeze all components to corrupt values
                for freeze_layer in range(n_layers):
                    freeze_attn_out = accessor.layers[freeze_layer].attention.output.unwrap().input[:, -1]
                    freeze_attn_out_reshaped = freeze_attn_out.reshape(batch_size, n_heads, d_head)
                    freeze_attn_out_reshaped[...] = corrupt_attn_out[freeze_layer]

                    freeze_mlp = accessor.layers[freeze_layer].mlp.unwrap().output[:, -1]
                    freeze_mlp[...] = corrupt_mlp[freeze_layer]

                    # Patch the target MLP with clean activation
                    if freeze_layer == layer:
                        freeze_mlp[...] = clean_mlp[freeze_layer]

            # Get patched logits
            patched_logits = accessor.lm_head.unwrap().output[:, -1][torch.arange(batch_size), answer_tokens].save()

        # Calculate normalized intervention effect for MLP
        intervention_diff = patched_logits.to("cpu") - corrupt_answer_logits.to("cpu")
        baseline = clean_answer_logits.to("cpu") - corrupt_answer_logits.to("cpu")
        intervention_effect = intervention_diff.mean() / baseline.mean()
        results[layer, -1] = intervention_effect

    return results


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

    # Handle empty list case
    if not receiver_list:
        raise ValueError("receiver_list cannot be empty for find_earliest_receiver")
    return sorted(receiver_list, key=functools.cmp_to_key(compare_receivers))[0]


def path_patch_sender_to_logits_via_receivers_batch(
    model: LanguageModel,
    clean_prompts: list[str],
    corrupt_prompts: list[str],
    answers: list[str],
    receiver_list: list[tuple[int, int]],
    batch_size: int = 8,
    remote: bool = True,
) -> torch.Tensor:
    """
    Batched Path Patching: Sender -> Receiver Set -> Logits.
    NOTE: This only patches the activation of last token.
    NOTE: This is an extremely strict version of path patching, as it restrict the direct
    contribution of sender to receivers AND from receivers to final logits.

    Calculates the normalized effect of restoring a single sender's clean activation
    on the final logits, considering only pathways that pass collectively through
    a specified *set* of receiver components. Explicitly freezes non-patched components.

    Args:
        model: Language model to analyze.
        clean_prompts: list of original unmodified prompts.
        corrupt_prompts: list of modified/corrupted prompts.
        answers: list of expected answer strings (only first token is used).
        receiver_list: list of receiver components [(layer_idx, head_or_mlp_idx), ...]
                       defining the mediating set. Use -1 for MLP index.
        batch_size: Size of each processing batch.
        remote: Whether to run model remotely.

    Returns:
        A single tensor containing the normalized contribution scores for each
        valid sender component, mediated by the receiver set.
        Shape depends on the earliest receiver in the set.
    """
    # Input Validation
    n_samples = len(clean_prompts)
    if not (n_samples == len(corrupt_prompts) == len(answers)):
        raise ValueError("Input lists (clean_prompts, corrupt_prompts, answers) must have the same length.")
    if not receiver_list:
        raise ValueError("receiver_list cannot be empty.")

    #  Setup
    spec = get_model_specs(model)
    n_layers, n_heads = spec["n_layers"], spec["n_heads"]

    #  Tokenization
    #  NOTE: Whether the tokenizer prepends <BOS> is not specified here.
    tokenizer_kwargs = {"padding": True, "return_tensors": "pt"}
    clean_tokens = model.tokenizer(clean_prompts, padding_side="left", **tokenizer_kwargs)["input_ids"]
    corrupt_tokens = model.tokenizer(corrupt_prompts, padding_side="left", **tokenizer_kwargs)["input_ids"]
    answer_tokens = model.tokenizer(answers, padding_side="right", add_special_tokens=False, **tokenizer_kwargs)[
        "input_ids"
    ][:, 0]

    if not remote and torch.cuda.is_available():
        clean_tokens = clean_tokens.to("cuda")
        corrupt_tokens = corrupt_tokens.to("cuda")
        answer_tokens = answer_tokens.to("cuda")

    #  Determine Sender Range & Result Shape
    min_layer, min_head_or_mlp = find_earliest_receiver(receiver_list)
    if min_head_or_mlp >= 0:  # Earliest receiver is an Attention Head
        max_sender_layer = min_layer - 1
        result_shape = (min_layer, n_heads + 1)
    else:  # Earliest receiver is an MLP
        max_sender_layer = min_layer
        result_shape = (min_layer + 1, n_heads + 1)

    #  Initialize Results
    all_results = torch.zeros(result_shape, device="cpu")

    #  Batch Processing
    for i in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
        batch_start = i
        batch_end = min(i + batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        clean_batch_tokens = clean_tokens[batch_start:batch_end]
        corrupt_batch_tokens = corrupt_tokens[batch_start:batch_end]
        answer_batch_tokens = answer_tokens[batch_start:batch_end]

        # Call the core function
        batch_results_tensor = path_patch_sender_to_logits_via_receivers(
            model=model,
            clean_tokens=clean_batch_tokens,
            corrupt_tokens=corrupt_batch_tokens,
            answer_tokens=answer_batch_tokens,
            receiver_list=receiver_list,
            min_layer=min_layer,
            min_head_or_mlp=min_head_or_mlp,
            max_sender_layer=max_sender_layer,
            result_shape=result_shape,
            remote=remote,
        )
        # Accumulate results
        weight = current_batch_size / n_samples
        all_results += batch_results_tensor.cpu() * weight  # Ensure results are on CPU
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
    return all_results


def path_patch_sender_to_logits_via_receivers(
    model: LanguageModel,
    clean_tokens: Tensor,
    corrupt_tokens: Tensor,
    answer_tokens: Tensor,
    receiver_list: list[tuple[int, int]],
    min_layer: int,
    min_head_or_mlp: int,
    max_sender_layer: int,
    result_shape: tuple[int, int],
    remote: bool = True,
    # batch_index: int = 0 # TODO: double check before removing
) -> Tensor:
    """
    Core implementation: Sender -> Receiver Set -> Logits (Explicit Freeze).
    Run 1: Freezes pre-sender & intermediate layers, patches sender correctly respecting causality of nnsight.
    Run 2: Freezes non-receivers, patches receivers. Cleaned version.

    Args:
        model: Language model.
        clean_tokens: Tokenized clean prompts for the batch.
        corrupt_tokens: Tokenized corrupt prompts for the batch.
        answer_tokens: Tokenized answer tokens for the batch.
        receiver_list: list of receiver components [(layer_idx, head_or_mlp_idx), ...].
        min_layer, min_head_or_mlp: Info about the earliest receiver.
        max_sender_layer: Highest layer index containing senders.
        result_shape: Expected shape of the final result tensor.
        remote: Whether to run model remotely.

    Returns:
        A single tensor (on CPU) containing the normalized effect of each sender
        mediated through the receiver set for this batch.
    """
    #  Setup
    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]

    accessor_config = get_accessor_config(model)
    accessor = ModelAccessor(model, accessor_config)
    batch_size = len(clean_tokens)

    #  1. Calculate Baseline Logits & Difference
    with torch.no_grad():
        clean_logits_trace = model.trace(clean_tokens, trace=False, remote=remote)
        corrupt_logits_trace = model.trace(corrupt_tokens, trace=False, remote=remote)
        batch_arange = torch.arange(batch_size, device=answer_tokens.device)
        clean_answer_logits = clean_logits_trace["logits"][batch_arange, -1, answer_tokens]
        corrupt_answer_logits = corrupt_logits_trace["logits"][batch_arange, -1, answer_tokens]

    # This is usually not happenning, keep it as a warning tho.
    baseline_diff = (clean_answer_logits - corrupt_answer_logits).mean()
    epsilon = 1e-8
    if torch.abs(baseline_diff) < epsilon:
        print(f"Warning: Baseline difference {baseline_diff.item():.4f} is close to zero.")
        baseline_diff = torch.sign(baseline_diff) * max(
            torch.abs(baseline_diff), torch.tensor(epsilon, device=baseline_diff.device)
        )

    #  2. Cache Clean Sender & Corrupt Activations (as Proxies)
    clean_attn_proxies = {}  # Stores attn_input from CLEAN run for SENDERS
    clean_mlp_proxies = {}  # Stores mlp_output from CLEAN run for SENDERS
    corrupt_attn_proxies = {}  # Stores attn_input from CORRUPT run for FREEZING
    corrupt_mlp_proxies = {}  # Stores mlp_output from CORRUPT run for FREEZING

    sender_layers_to_cache = set(range(max_sender_layer + 1))
    # Determine which components are actually needed for clean cache (senders)
    valid_sender_components = defaultdict(set)
    for layer in sender_layers_to_cache:
        is_max_sender_layer_mlp_receiver = layer == max_sender_layer and min_head_or_mlp == -1
        valid_sender_components[layer].update(
            range(n_heads)
        )  # Heads are always potential senders up to max_sender_layer
        if not is_max_sender_layer_mlp_receiver:
            valid_sender_components[layer].add(n_heads)

    #  Cache Clean Activations
    with accessor.trace(remote=remote) as tracer_clean_cache:
        with tracer_clean_cache.invoke(clean_tokens):
            for layer in sender_layers_to_cache:
                is_sender_layer_attn = any(idx < n_heads for idx in valid_sender_components.get(layer, set()))
                is_sender_layer_mlp = n_heads in valid_sender_components.get(layer, set())

                # Adding a bunch of try-catches for debugging purpose.
                if is_sender_layer_attn:
                    try:
                        clean_attn_proxies[layer] = accessor.layers[layer].attention.output.unwrap().input[:, -1].save()
                    except Exception as e:
                        print(f"Clean Cache Proxy Save Warning: Attn layer {layer}. {e}")
                if is_sender_layer_mlp:
                    try:
                        clean_mlp_proxies[layer] = accessor.layers[layer].mlp.unwrap().output[:, -1].save()
                    except Exception as e:
                        print(f"Clean Cache Proxy Save Warning: MLP layer {layer}. {e}")

    #  Cache Corrupt Activations
    with accessor.trace(remote=remote) as tracer_corrupt_cache:
        with tracer_corrupt_cache.invoke(corrupt_tokens):
            for layer in range(n_layers):  # Cache ALL layers needed for freezing
                try:
                    corrupt_attn_proxies[layer] = accessor.layers[layer].attention.output.unwrap().input[:, -1].save()
                except Exception as e:
                    print(f"Corrupt Cache Proxy Save Warning: Attn layer {layer}. {e}")
                try:
                    corrupt_mlp_proxies[layer] = accessor.layers[layer].mlp.unwrap().output[:, -1].save()
                except Exception as e:
                    print(f"Corrupt Cache Proxy Save Warning: MLP layer {layer}. {e}")

    #  3. Path Patching Loop
    results_tensor = torch.zeros(result_shape, device="cpu")
    receiver_comp_indices = {
        rec: (n_heads if rec[1] == -1 else rec[1]) for rec in receiver_list
    }  # Precompute for convenience

    # TODO: double check before deleting them
    receiver_set = set(receiver_list)  # Use set for faster lookup
    max_receiver_layer = max(rec[0] for rec in receiver_list) if receiver_list else -1

    for sender_layer in tqdm(range(max_sender_layer + 1), desc="Patching Senders", leave=False):
        for sender_comp_idx in range(n_heads + 1):
            #  Check: Skip if this component is not a valid sender
            if sender_comp_idx not in valid_sender_components.get(sender_layer, set()):
                continue
            is_sender_attn = sender_comp_idx < n_heads
            # Check if clean proxy exists (should be guaranteed by above check)
            if is_sender_attn and sender_layer not in clean_attn_proxies:
                continue
            if not is_sender_attn and sender_layer not in clean_mlp_proxies:
                continue

            #  Run 1: Freeze PRE-SENDER & INTERMEDIATE layers, Patch Sender -> Collect Patched Receiver Activations
            patched_receiver_activations_proxies = {}  # Store saved PROXIES for receivers
            run1_success = False
            with accessor.trace(remote=remote) as tracer_run1:
                with tracer_run1.invoke(corrupt_tokens):
                    #  Step 1: Freeze layers BEFORE sender
                    for current_layer in range(sender_layer):
                        if current_layer in corrupt_attn_proxies:
                            accessor.layers[current_layer].attention.output.unwrap().input[:, -1][...] = (
                                corrupt_attn_proxies[current_layer].value.clone()
                            )
                        if current_layer in corrupt_mlp_proxies:
                            accessor.layers[current_layer].mlp.unwrap().output[:, -1][...] = corrupt_mlp_proxies[
                                current_layer
                            ].value.clone()

                    #  Step 2: Intervene at the SENDER layer
                    mlp_idx_sender = n_heads
                    if is_sender_attn:  # Sender is Attention Head
                        # Freeze-then-patch Attention Input
                        if sender_layer in corrupt_attn_proxies and sender_layer in clean_attn_proxies:
                            current_attn_input = accessor.layers[sender_layer].attention.output.unwrap().input[:, -1]
                            current_attn_input[...] = corrupt_attn_proxies[
                                sender_layer
                            ].value.clone()  # Freeze all heads
                            current_attn_reshaped = current_attn_input.reshape(batch_size, n_heads, d_head)
                            clean_attn_tensor = clean_attn_proxies[sender_layer].value.reshape(
                                batch_size, n_heads, d_head
                            )
                            current_attn_reshaped[:, sender_comp_idx, :] = clean_attn_tensor[
                                :, sender_comp_idx, :
                            ].clone()  # Patch sender head
                        # NOTE: Need to double check.
                        # *** DO NOT FREEZE MLP OUTPUT in sender layer if sender is ATTN ***

                    else:  # Sender is MLP
                        # Freeze Attention Input first (causally before MLP)
                        if sender_layer in corrupt_attn_proxies:
                            accessor.layers[sender_layer].attention.output.unwrap().input[:, -1][...] = (
                                corrupt_attn_proxies[sender_layer].value.clone()
                            )
                        # Freeze-then-patch MLP Output
                        if sender_layer in corrupt_mlp_proxies and sender_layer in clean_mlp_proxies:
                            current_mlp_output = accessor.layers[sender_layer].mlp.unwrap().output[:, -1]
                            current_mlp_output[...] = corrupt_mlp_proxies[sender_layer].value.clone()  # Freeze
                            current_mlp_output[...] = clean_mlp_proxies[sender_layer].value.clone()  # Patch

                    #  Step 3: Freeze INTERMEDIATE layers (after sender, before min_layer)
                    for current_layer in range(sender_layer + 1, min_layer):
                        if current_layer in corrupt_attn_proxies:
                            accessor.layers[current_layer].attention.output.unwrap().input[:, -1][...] = (
                                corrupt_attn_proxies[current_layer].value.clone()
                            )
                        if current_layer in corrupt_mlp_proxies:
                            accessor.layers[current_layer].mlp.unwrap().output[:, -1][...] = corrupt_mlp_proxies[
                                current_layer
                            ].value.clone()

                    #  Step 4: Handle freezing ATTN heads at min_layer if earliest receiver is MLP
                    # This needs to happen *only if* min_layer was not the sender layer AND earliest receiver is MLP
                    if min_head_or_mlp == -1 and min_layer != sender_layer and min_layer in corrupt_attn_proxies:
                        current_attn_input = accessor.layers[min_layer].attention.output.unwrap().input[:, -1]
                        current_attn_input[...] = corrupt_attn_proxies[min_layer].value.clone()

                    #  Step 5: Let computation flow naturally for layers >= min_layer (unless ATTN frozen just above)

                    #  Step 6: Save the activation proxy of *each receiver* component
                    for receiver_tuple in receiver_list:
                        rec_layer, rec_head_or_mlp = receiver_tuple
                        rec_comp_idx = receiver_comp_indices[receiver_tuple]
                        if rec_comp_idx < n_heads:
                            # Save the resulting attention input state
                            rec_attn_out_input = accessor.layers[rec_layer].attention.output.unwrap().input[:, -1]
                            rec_attn_out_reshaped = rec_attn_out_input.reshape(batch_size, n_heads, d_head)
                            patched_receiver_activations_proxies[receiver_tuple] = rec_attn_out_reshaped[
                                :, rec_comp_idx, :
                            ].save()
                        else:
                            # Save the resulting MLP output state
                            patched_receiver_activations_proxies[receiver_tuple] = (
                                accessor.layers[rec_layer].mlp.unwrap().output[:, -1].save()
                            )
            run1_success = True  # Assume success if trace completes without nnsight error

            # An extra check to fail early.
            if not run1_success or len(patched_receiver_activations_proxies) != len(receiver_list):
                print(f"Warning: Run 1 incomplete for sender ({sender_layer},{sender_comp_idx}). Skipping Run 2.")
                continue

            #  Run 2: Patch All Receivers, Freeze Others -> Get Final Logits
            intervened_logits_obj = None
            run2_success = False
            with accessor.trace(remote=remote) as tracer_run2:
                with tracer_run2.invoke(corrupt_tokens):
                    # Iterate through ALL layers to freeze/patch up to the end
                    for current_layer in range(n_layers):
                        mlp_idx = n_heads
                        #  Freeze/Patch Attention Heads
                        if current_layer in corrupt_attn_proxies:
                            current_attn_input = accessor.layers[current_layer].attention.output.unwrap().input[:, -1]
                            current_attn_input[...] = corrupt_attn_proxies[current_layer].value.clone()
                            current_attn_reshaped = current_attn_input.reshape(batch_size, n_heads, d_head)
                            for head_idx in range(n_heads):
                                receiver_tuple = (current_layer, head_idx)
                                if receiver_tuple in patched_receiver_activations_proxies:
                                    receiver_proxy = patched_receiver_activations_proxies[receiver_tuple]
                                    if hasattr(receiver_proxy, "value"):
                                        current_attn_reshaped[:, head_idx, :] = receiver_proxy.value.clone()

                        #  Freeze/Patch MLP
                        if current_layer in corrupt_mlp_proxies:
                            current_mlp_output = accessor.layers[current_layer].mlp.unwrap().output[:, -1]
                            current_mlp_output[...] = corrupt_mlp_proxies[current_layer].value.clone()
                            receiver_tuple = (current_layer, -1)
                            if receiver_tuple in patched_receiver_activations_proxies:
                                receiver_proxy = patched_receiver_activations_proxies[receiver_tuple]
                                if hasattr(receiver_proxy, "value"):
                                    current_mlp_output[...] = receiver_proxy.value.clone()

                    # Save the final logits
                    intervened_logits_obj = (
                        accessor.lm_head.unwrap().output[batch_arange, -1, answer_tokens].save()
                    )
            run2_success = True  # Assume success if trace completes without nnsight error

            #  Again, fail early
            if not run2_success or intervened_logits_obj is None:
                print(
                    f"Skipping calculation for sender ({sender_layer},{sender_comp_idx}) due to Run 2 failure or save error."
                )
                continue

            # Extract tensor from final logits, and I just don't get how .value works in nnsight
            if hasattr(intervened_logits_obj, "value"):
                intervened_logits_val = intervened_logits_obj.value
            elif isinstance(intervened_logits_obj, torch.Tensor):
                intervened_logits_val = intervened_logits_obj
            else:
                print(
                    f"Unexpected type for intervened_logits_obj: {type(intervened_logits_obj)}. Skipping calculation."
                )
                continue

            # Perform calculation
            corrupt_answer_logits_dev = corrupt_answer_logits.to(intervened_logits_val.device)
            baseline_diff_dev = baseline_diff.to(intervened_logits_val.device)
            intervention_diff = intervened_logits_val.mean() - corrupt_answer_logits_dev.mean()
            # Add small epsilon to denominator to prevent division by zero if baseline_diff is exactly zero
            normalized_effect = (intervention_diff / (baseline_diff_dev + 1e-12)).item()

            results_tensor[sender_layer, sender_comp_idx] = normalized_effect

    return results_tensor

def path_patch_sender_to_receivers(
    model: LanguageModel,
    clean_tokens: Tensor,
    corrupt_tokens: Tensor,
    answer_tokens: Tensor,
    receiver_list: list[tuple[int, int]],
    min_layer: int,
    min_head_or_mlp: int,
    max_sender_layer: int,
    result_shape: tuple[int, int],
    remote: bool = True,
    sender_pos: list[int] = [-1],
    receiver_pos: list[int] = [-1],
    freeze_pos: list[int] = [-1],
    # batch_index: int = 0 # TODO: double check before removing
) -> Tensor:
    """
    Core implementation: Sender -> set of Receivers
    Patch Run 1: Freezes pre-sender & intermediate layers, patches sender correctly respecting causality of nnsight. #TODO: double check
    Patch Run 2: Patches receivers. 

    Args:
        model: Language model.
        clean_tokens: Tokenized clean prompts for the batch.
        corrupt_tokens: Tokenized corrupt prompts for the batch.
        answer_tokens: Tokenized answer tokens for the batch.
        receiver_list: list of receiver components [(layer_idx, head_or_mlp_idx), ...].
        min_layer, min_head_or_mlp: Info about the earliest receiver.
        max_sender_layer: Highest layer index containing senders.
        result_shape: Expected shape of the final result tensor.
        remote: Whether to run model remotely.
        sender_pos: Position(s) to patch activations for the sender and components before the sender.
        receiver_pos: Position(s) to patch activations for the receivers.
        freeze_pos: Position(s) to freeze activations during patch run 1 
    Returns:
        A single tensor (on CPU) containing the normalized effect of each sender
        mediated through the receiver set for this batch.
    """
    #  Setup
    all_pos = list(range(len(clean_tokens[1])))
    if sender_pos == "all_pos": 
        sender_pos = all_pos
    if receiver_pos == "all_pos": 
        receiver_pos = all_pos
    if freeze_pos == "all_pos": 
        freeze_pos = all_pos

    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]
    accessor_config = get_accessor_config(model)
    accessor = ModelAccessor(model, accessor_config)
    batch_size = len(clean_tokens)

    #  1. Calculate Baseline Logits & Difference
    with torch.no_grad():
        clean_logits_trace = model.trace(clean_tokens, trace=False, remote=remote)
        corrupt_logits_trace = model.trace(corrupt_tokens, trace=False, remote=remote)
        batch_arange = torch.arange(batch_size, device=answer_tokens.device)
        clean_answer_logits = clean_logits_trace["logits"][batch_arange, -1, answer_tokens]
        corrupt_answer_logits = corrupt_logits_trace["logits"][batch_arange, -1, answer_tokens]

    # This is usually not happenning, keep it as a warning tho.
    baseline_diff = (clean_answer_logits - corrupt_answer_logits).mean()
    epsilon = 1e-8
    if torch.abs(baseline_diff) < epsilon:
        print(f"Warning: Baseline difference {baseline_diff.item():.4f} is close to zero.")
        baseline_diff = torch.sign(baseline_diff) * max(
            torch.abs(baseline_diff), torch.tensor(epsilon, device=baseline_diff.device)
        )

    #  2. Cache Clean Sender & Corrupt Activations (as Proxies)
    clean_attn_proxies = {}  # Stores attn_input from CLEAN run for SENDERS
    clean_mlp_proxies = {}  # Stores mlp_output from CLEAN run for SENDERS
    corrupt_attn_proxies = {}  # Stores attn_input from CORRUPT run for FREEZING
    corrupt_mlp_proxies = {}  # Stores mlp_output from CORRUPT run for FREEZING

    sender_layers_to_cache = set(range(max_sender_layer + 1))
    # Determine which components are actually needed for clean cache (senders)
    valid_sender_components = defaultdict(set)
    for layer in sender_layers_to_cache:
        is_max_sender_layer_mlp_receiver = layer == max_sender_layer and min_head_or_mlp == -1
        valid_sender_components[layer].update(
            range(n_heads)
        )  # Heads are always potential senders up to max_sender_layer
        if not is_max_sender_layer_mlp_receiver:
            valid_sender_components[layer].add(n_heads)

    #  Cache Clean Activations
    with accessor.trace(remote=remote) as tracer_clean_cache:
        with tracer_clean_cache.invoke(clean_tokens):
            for layer in sender_layers_to_cache:
                is_sender_layer_attn = any(idx < n_heads for idx in valid_sender_components.get(layer, set()))
                is_sender_layer_mlp = n_heads in valid_sender_components.get(layer, set())

                # Adding a bunch of try-catches for debugging purpose.
                if is_sender_layer_attn:
                    try:
                        clean_attn_proxies[layer] = accessor.layers[layer].attention.output.unwrap().input[:, sender_pos].save()
                    except Exception as e:
                        print(f"Clean Cache Proxy Save Warning: Attn layer {layer}. {e}")
                if is_sender_layer_mlp:
                    try:
                        clean_mlp_proxies[layer] = accessor.layers[layer].mlp.unwrap().output[:, sender_pos].save()
                    except Exception as e:
                        print(f"Clean Cache Proxy Save Warning: MLP layer {layer}. {e}")

    #  Cache Corrupt Activations
    with accessor.trace(remote=remote) as tracer_corrupt_cache:
        with tracer_corrupt_cache.invoke(corrupt_tokens):
            for layer in range(n_layers):  # Cache ALL layers and ALL positions needed for freezing #TODO: double check to improve efficiency 
                try:
                    corrupt_attn_proxies[layer] = accessor.layers[layer].attention.output.unwrap().input[:, :].save()
                except Exception as e:
                    print(f"Corrupt Cache Proxy Save Warning: Attn layer {layer}. {e}")
                try:
                    corrupt_mlp_proxies[layer] = accessor.layers[layer].mlp.unwrap().output[:, :].save()
                except Exception as e:
                    print(f"Corrupt Cache Proxy Save Warning: MLP layer {layer}. {e}")

    #  3. Path Patching Loop
    results_tensor = torch.zeros(result_shape, device="cpu")
    receiver_comp_indices = {
        rec: (n_heads if rec[1] == -1 else rec[1]) for rec in receiver_list
    }  # Precompute for convenience

    # TODO: double check before deleting them
    receiver_set = set(receiver_list)  # Use set for faster lookup
    max_receiver_layer = max(rec[0] for rec in receiver_list) if receiver_list else -1

    for sender_layer in tqdm(range(max_sender_layer + 1), desc="Patching Senders", leave=False):
        for sender_comp_idx in range(n_heads + 1):
            #  Check: Skip if this component is not a valid sender
            if sender_comp_idx not in valid_sender_components.get(sender_layer, set()):
                continue
            is_sender_attn = sender_comp_idx < n_heads
            # Check if clean proxy exists (should be guaranteed by above check)
            if is_sender_attn and sender_layer not in clean_attn_proxies:
                continue
            if not is_sender_attn and sender_layer not in clean_mlp_proxies:
                continue

            #  Run 1: Freeze PRE-SENDER & INTERMEDIATE layers, Patch Sender -> Collect Patched Receiver Activations
            patched_receiver_activations_proxies = {}  # Store saved PROXIES for receivers
            run1_success = False
            with accessor.trace(remote=remote) as tracer_run1:
                with tracer_run1.invoke(corrupt_tokens):
                    #  Step 1: Freeze layers BEFORE sender at freeze_pos
                    for current_layer in range(sender_layer):
                        if current_layer in corrupt_attn_proxies:
                            accessor.layers[current_layer].attention.output.unwrap().input[:, freeze_pos][...] = (
                                corrupt_attn_proxies[current_layer].value[:, freeze_pos].clone()
                            )
                        if current_layer in corrupt_mlp_proxies:
                            accessor.layers[current_layer].mlp.unwrap().output[:, freeze_pos][...] = corrupt_mlp_proxies[
                                current_layer
                            ].value[:, freeze_pos].clone()

                    #  Step 2: Intervene at the SENDER layer
                    mlp_idx_sender = n_heads
                    if is_sender_attn:  # Sender is Attention Head
                        # Freeze-then-patch Attention Input
                        if sender_layer in corrupt_attn_proxies and sender_layer in clean_attn_proxies:
                            current_attn_input = accessor.layers[sender_layer].attention.output.unwrap().input[...]
                            current_attn_input[:, freeze_pos] = corrupt_attn_proxies[
                                sender_layer
                            ].value[:, freeze_pos].clone()  # Freeze all heads @ freeze_pos
                            current_attn_reshaped = current_attn_input.reshape(batch_size, len(all_pos), n_heads, d_head)
                            clean_attn_tensor = clean_attn_proxies[sender_layer].value.reshape(
                                batch_size, len(sender_pos), n_heads, d_head
                            )
                            current_attn_reshaped[:, sender_pos, sender_comp_idx, :] = clean_attn_tensor[
                                :, :, sender_comp_idx, :
                            ].clone()  # Patch sender head
                        # TODO: Need to double check.
                        # *** DO NOT FREEZE MLP OUTPUT in sender layer if sender is ATTN ***

                    else:  # Sender is MLP
                        # Freeze Attention Input first (causally before MLP)
                        if sender_layer in corrupt_attn_proxies:
                            accessor.layers[sender_layer].attention.output.unwrap().input[:, freeze_pos][...] = (
                                corrupt_attn_proxies[sender_layer].value[:, freeze_pos].clone()
                            )
                        # Freeze-then-patch MLP Output
                        if sender_layer in corrupt_mlp_proxies and sender_layer in clean_mlp_proxies:
                            current_mlp_output = accessor.layers[sender_layer].mlp.unwrap().output[:, :]
                            current_mlp_output[:, freeze_pos] = corrupt_mlp_proxies[sender_layer].value[:, freeze_pos].clone()  # Freeze
                            current_mlp_output[:, sender_pos] = clean_mlp_proxies[sender_layer].value.clone()  # Patch

                    #  Step 3: Freeze INTERMEDIATE layers (after sender, before min_layer)
                    for current_layer in range(sender_layer + 1, min_layer):
                        if current_layer in corrupt_attn_proxies:
                            accessor.layers[current_layer].attention.output.unwrap().input[:, freeze_pos][...] = (
                                corrupt_attn_proxies[current_layer].value[:, freeze_pos].clone()
                            )
                        if current_layer in corrupt_mlp_proxies:
                            accessor.layers[current_layer].mlp.unwrap().output[:, freeze_pos][...] = corrupt_mlp_proxies[
                                current_layer
                            ].value[:, freeze_pos].clone()

                    #  Step 4: Handle freezing ATTN heads at min_layer if earliest receiver is MLP
                    # This needs to happen *only if* min_layer was not the sender layer AND earliest receiver is MLP
                    if min_head_or_mlp == -1 and min_layer != sender_layer and min_layer in corrupt_attn_proxies:
                        current_attn_input = accessor.layers[min_layer].attention.output.unwrap().input[:, freeze_pos]
                        current_attn_input[...] = corrupt_attn_proxies[min_layer].value[:, freeze_pos].clone()

                    #  Step 5: Let computation flow naturally for layers >= min_layer (unless ATTN frozen just above)

                    #  Step 6: Save the activation proxy of *each receiver* component
                    for receiver_tuple in receiver_list:
                        rec_layer, rec_head_or_mlp = receiver_tuple
                        rec_comp_idx = receiver_comp_indices[receiver_tuple]
                        if rec_comp_idx < n_heads:
                            # Save the resulting attention input state
                            rec_attn_out_input = accessor.layers[rec_layer].attention.output.unwrap().input[:, receiver_pos]
                            rec_attn_out_reshaped = rec_attn_out_input.reshape(batch_size, len(receiver_pos), n_heads, d_head)
                            patched_receiver_activations_proxies[receiver_tuple] = rec_attn_out_reshaped[
                                :, :, rec_comp_idx, :
                            ].save()
                        else:
                            # Save the resulting MLP output state
                            patched_receiver_activations_proxies[receiver_tuple] = (
                                accessor.layers[rec_layer].mlp.unwrap().output[:, receiver_pos].save()
                            )
            run1_success = True  # Assume success if trace completes without nnsight error

            # An extra check to fail early.
            if not run1_success or len(patched_receiver_activations_proxies) != len(receiver_list):
                print(f"Warning: Run 1 incomplete for sender ({sender_layer},{sender_comp_idx}). Skipping Run 2.")
                continue

            #  Run 2: Patch All Receivers, Get Final Logits
            intervened_logits_obj = None
            run2_success = False
            with accessor.trace(remote=remote) as tracer_run2:
                with tracer_run2.invoke(corrupt_tokens):
                    # Iterate through ALL layers to patch at receiver_pos
                    for current_layer in range(n_layers):
                        mlp_idx = n_heads
                        #  Patch Attention Heads at receiver_pos
                        if current_layer in corrupt_attn_proxies:
                            current_attn_input = accessor.layers[current_layer].attention.output.unwrap().input[...]
                            current_attn_reshaped = current_attn_input.reshape(batch_size, len(all_pos), n_heads, d_head)
                            for head_idx in range(n_heads):
                                receiver_tuple = (current_layer, head_idx)
                                if receiver_tuple in patched_receiver_activations_proxies:
                                    receiver_proxy = patched_receiver_activations_proxies[receiver_tuple]
                                    if hasattr(receiver_proxy, "value"):
                                        current_attn_reshaped[:, receiver_pos, head_idx, :] = receiver_proxy.value.clone()

                        #  Patch MLP
                        if current_layer in corrupt_mlp_proxies:
                            current_mlp_output = accessor.layers[current_layer].mlp.unwrap().output[...]
                            receiver_tuple = (current_layer, -1)
                            if receiver_tuple in patched_receiver_activations_proxies:
                                receiver_proxy = patched_receiver_activations_proxies[receiver_tuple]
                                if hasattr(receiver_proxy, "value"):
                                    current_mlp_output[:, receiver_pos] = receiver_proxy.value.clone()

                    # Save the final logits
                    intervened_logits_obj = (
                        accessor.lm_head.unwrap().output[batch_arange, -1, answer_tokens].save()
                    )
            run2_success = True  # Assume success if trace completes without nnsight error

            #  Again, fail early
            if not run2_success or intervened_logits_obj is None:
                print(
                    f"Skipping calculation for sender ({sender_layer},{sender_comp_idx}) due to Run 2 failure or save error."
                )
                continue

            # Extract tensor from final logits, and I just don't get how .value works in nnsight
            if hasattr(intervened_logits_obj, "value"):
                intervened_logits_val = intervened_logits_obj.value
            elif isinstance(intervened_logits_obj, torch.Tensor):
                intervened_logits_val = intervened_logits_obj
            else:
                print(
                    f"Unexpected type for intervened_logits_obj: {type(intervened_logits_obj)}. Skipping calculation."
                )
                continue

            # Perform calculation
            corrupt_answer_logits_dev = corrupt_answer_logits.to(intervened_logits_val.device)
            baseline_diff_dev = baseline_diff.to(intervened_logits_val.device)
            intervention_diff = intervened_logits_val.mean() - corrupt_answer_logits_dev.mean()
            # Add small epsilon to denominator to prevent division by zero if baseline_diff is exactly zero
            normalized_effect = (intervention_diff / (baseline_diff_dev + 1e-12)).item()

            results_tensor[sender_layer, sender_comp_idx] = normalized_effect

    return results_tensor


def path_patch_sender_to_receivers_batch(
    model: LanguageModel,
    clean_prompts: list[str],
    corrupt_prompts: list[str],
    answers: list[str],
    receiver_list: list[tuple[int, int]],
    batch_size: int = 8,
    remote: bool = True,
    sender_pos: list[int] = [-1],
    receiver_pos: list[int] = [-1],
    freeze_pos: list[int] = [-1],
) -> torch.Tensor:
    """
    Batched Path Patching: Sender -> Receiver Set
    Patch Run 1: Freezes pre-sender & intermediate layers, patches sender correctly respecting causality of nnsight. #TODO: double check
    Patch Run 2: Patches receivers. 

    Calculates the normalized effect of restoring a single sender's clean activation
    on the final logits, considering only pathways that pass collectively through
    a specified *set* of receiver components. Explicitly freezes non-patched components.

    Args:
        model: Language model to analyze.
        clean_prompts: list of original unmodified prompts.
        corrupt_prompts: list of modified/corrupted prompts.
        answers: list of expected answer strings (only first token is used).
        receiver_list: list of receiver components [(layer_idx, head_or_mlp_idx), ...]
                       defining the mediating set. Use -1 for MLP index.
        batch_size: Size of each processing batch.
        remote: Whether to run model remotely.
        sender_pos: Position(s) to patch activations for the sender and components before the sender.
        receiver_pos: Position(s) to patch activations for the receivers.
        freeze_pos: Position(s) to freeze activations during patch run 1 
    Returns:
        A single tensor containing the normalized contribution scores for each
        valid sender component, mediated by the receiver set.
        Shape depends on the earliest receiver in the set.
    """
    # Input Validation
    n_samples = len(clean_prompts)
    if not (n_samples == len(corrupt_prompts) == len(answers)):
        raise ValueError("Input lists (clean_prompts, corrupt_prompts, answers) must have the same length.")
    if not receiver_list:
        raise ValueError("receiver_list cannot be empty.")

    #  Setup
    spec = get_model_specs(model)
    n_layers, n_heads = spec["n_layers"], spec["n_heads"]

    #  Tokenization
    #  NOTE: Whether the tokenizer prepends <BOS> is not specified here.
    tokenizer_kwargs = {"padding": True, "return_tensors": "pt"}
    clean_tokens = model.tokenizer(clean_prompts, padding_side="left", **tokenizer_kwargs)["input_ids"]
    corrupt_tokens = model.tokenizer(corrupt_prompts, padding_side="left", **tokenizer_kwargs)["input_ids"]
    answer_tokens = model.tokenizer(answers, padding_side="right", add_special_tokens=False, **tokenizer_kwargs)[
        "input_ids"][:, 0]

    if not remote and torch.cuda.is_available():
        clean_tokens = clean_tokens.to("cuda")
        corrupt_tokens = corrupt_tokens.to("cuda")
        answer_tokens = answer_tokens.to("cuda")

    #  Determine Sender Range & Result Shape
    min_layer, min_head_or_mlp = find_earliest_receiver(receiver_list)
    if min_head_or_mlp >= 0:  # Earliest receiver is an Attention Head
        max_sender_layer = min_layer - 1
        result_shape = (min_layer, n_heads + 1)
    else:  # Earliest receiver is an MLP
        max_sender_layer = min_layer
        result_shape = (min_layer + 1, n_heads + 1)

    #  Initialize Results
    all_results = torch.zeros(result_shape, device="cpu")

    #  Batch Processing
    for i in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
        batch_start = i
        batch_end = min(i + batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        clean_batch_tokens = clean_tokens[batch_start:batch_end]
        corrupt_batch_tokens = corrupt_tokens[batch_start:batch_end]
        answer_batch_tokens = answer_tokens[batch_start:batch_end]

        # Call the core function
        batch_results_tensor = path_patch_sender_to_receivers(
            model=model,
            clean_tokens=clean_batch_tokens,
            corrupt_tokens=corrupt_batch_tokens,
            answer_tokens=answer_batch_tokens,
            receiver_list=receiver_list,
            min_layer=min_layer,
            min_head_or_mlp=min_head_or_mlp,
            max_sender_layer=max_sender_layer,
            result_shape=result_shape,
            remote=remote,
            sender_pos=sender_pos,
            receiver_pos=receiver_pos,
            freeze_pos=freeze_pos,
        )
        # Accumulate results
        weight = current_batch_size / n_samples
        all_results += batch_results_tensor.cpu() * weight  # Ensure results are on CPU
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
    return all_results

