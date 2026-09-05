def l2_norm_effect(clean_head_output, patched_head_output):
    """
    Relative change in a receiver head's output magnitude between the clean
    and sender-patched runs..
    """
    clean_norm = clean_head_output.norm(p=2)
    patched_norm = patched_head_output.norm(p=2)
    return ((patched_norm - clean_norm) / clean_norm).item()

def early_decode(head_output, model):
    """
    Project a single head's output (z_h @ W_O_h, shape [..., d_model]) through
    the model's final layernorm and unembedding.

    head_output: tensor, shape [..., d_model]: one head's contribution at
        one sequence position
    Returns: logits over vocab, shape [..., d_vocab].
    """
    normalized = model.ln_final(head_output)
    return model.unembed(normalized)


def topk_match_count(logits, task_relation_words, tokenizer, k=10):
    """
    Number of the top-k decoded tokens that appear in task_relation_words
    (the paper's task-descriptive term list, e.g. antonym -> ["opposite",
    "reverse", "antonym", ...]). [n]

    logits: tensor, shape [d_vocab]
    task_relation_words: set[str] -- lowercased, from task_relation_dict.json.
    """
    topk_token_ids = logits.topk(k).indices.tolist()
    topk_strs = {tokenizer.decode([tid]).strip().lower() for tid in topk_token_ids}
    return len(topk_strs & set(w.lower() for w in task_relation_words))


def n_match_effect(clean_head_output, patched_head_output, model, task_relation_words, tokenizer, k=10):
    """
    Signed change in n (top-k task-term match count) between the clean and
    sender-patched runs, for one receiver head.

    Negative = patching this sender degraded the receiver's task
    verbalization (expected if sender causally feeds the LTH).
    Zero = no effect at this (sender, receiver) pair.
    """
    clean_logits = early_decode(clean_head_output, model)
    patched_logits = early_decode(patched_head_output, model)

    n_clean = topk_match_count(clean_logits, task_relation_words, tokenizer, k=k)
    n_patched = topk_match_count(patched_logits, task_relation_words, tokenizer, k=k)

    return n_patched - n_clean