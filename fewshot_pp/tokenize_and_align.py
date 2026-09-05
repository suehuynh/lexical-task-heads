import os
import json
import argparse
from transformer_lens import HookedTransformer
from prompt_builders import create_few_shot_prompts, create_corrupt_prompts_task_shuffle

def tokenize_and_align(model, clean_prompts, corrupt_prompts, strict_check=True, qa_delimiter=":"):
    aligned = []
    mismatch_count = 0

    for clean_p, corrupt_p in zip(clean_prompts, corrupt_prompts):
        clean_tokens = model.to_tokens(clean_p)
        corrupt_tokens = model.to_tokens(corrupt_p)

        clean_pos = clean_tokens.shape[-1] - 1
        corrupt_pos = corrupt_tokens.shape[-1] - 1
        mismatch = clean_tokens.shape[-1] != corrupt_tokens.shape[-1]
        mismatch_count += int(mismatch)

        if strict_check:
            # Check the token immediately BEFORE the final position
            clean_pre_str = model.to_string(clean_tokens[0, clean_pos - 1])
            corrupt_pre_str = model.to_string(corrupt_tokens[0, corrupt_pos - 1])

            clean_has_delim = qa_delimiter in clean_pre_str
            corrupt_has_delim = qa_delimiter in corrupt_pre_str

            if clean_has_delim != corrupt_has_delim:
                raise ValueError(
                    f"Delimiter presence diverges right before the query position -- "
                    f"clean: {clean_pre_str!r} (has '{qa_delimiter}': {clean_has_delim}), "
                    f"corrupted: {corrupt_pre_str!r} (has '{qa_delimiter}': {corrupt_has_delim}). "
                    f"Formatting mismatch, not just a length difference."
                )

        aligned.append({
            "clean_tokens": clean_tokens, "corrupt_tokens": corrupt_tokens,
            "clean_pos": clean_pos, "corrupt_pos": corrupt_pos,
            "length_mismatch": mismatch,
        })

    if mismatch_count:
        print(f"[tokenize_and_align] {mismatch_count}/{len(clean_prompts)} pairs have "
              f"differing lengths -- expected with task-shuffle corruption, not a bug.")

    return aligned

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_folder", type=str, default="datasets/abstractive")
    parser.add_argument("--d_name", type=str, default="country-capital")
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_preview", type=int, default=3)
    args = parser.parse_args()

    print(f"loading {args.model_name} ...")
    model = HookedTransformer.from_pretrained(args.model_name)

    with open(os.path.join(args.dataset_folder, f"{args.d_name}.json")) as f:
        query_data = json.load(f)

    clean_prompts, answers, _ = create_few_shot_prompts(
        query_data, n_shot=args.n_shot, delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":"
    )
    corrupt_prompts, corrupt_answers, corrupt_sources = create_corrupt_prompts_task_shuffle(
        args.dataset_folder, args.d_name, args.n_shot,
        delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":",
    )

    # --- Test 1: real clean/corrupt pairs, eyeball alignment ---
    print("\n[TEST 1] real clean/corrupt pairs")
    aligned = tokenize_and_align(model=model, clean_prompts=clean_prompts[:10], corrupt_prompts=corrupt_prompts[:10])

    n_mismatched = sum(a["length_mismatch"] for a in aligned)
    print(f"  {n_mismatched}/10 pairs have differing token lengths (expected, not a bug)")

    for i in range(args.n_preview):
        a = aligned[i]
        clean_last = model.to_string(a["clean_tokens"][0, a["clean_pos"]])
        corrupt_last = model.to_string(a["corrupt_tokens"][0, a["corrupt_pos"]])
        clean_pre = model.to_string(a["clean_tokens"][0, a["clean_pos"] - 1])
        corrupt_pre = model.to_string(a["corrupt_tokens"][0, a["corrupt_pos"] - 1])
        print(f"  [{i}] clean_pos={a['clean_pos']}  token_before={clean_pre!r}  token_at_pos={clean_last!r}")
        print(f"       corrupt_pos={a['corrupt_pos']}  token_before={corrupt_pre!r}  token_at_pos={corrupt_last!r}")
        # manual check: token_before should be qa_delimiter (":") in both

    # --- Test 2: BOS handling sanity check ---
    print("\n[TEST 2] BOS token check (first 3 tokens of pair 0)")
    print("  clean:  ", aligned[0]["clean_tokens"][0, :3].tolist(),
          [model.to_string(t) for t in aligned[0]["clean_tokens"][0, :3]])
    print("  corrupt:", aligned[0]["corrupt_tokens"][0, :3].tolist(),
          [model.to_string(t) for t in aligned[0]["corrupt_tokens"][0, :3]])
    print("  -> confirm BOS appears exactly once, in the same way, in both")

    # --- Test 3: strict_check should PASS on real pairs ---
    print("\n[TEST 3] strict_check on real pairs (should not raise)")
    try:
        tokenize_and_align(model, clean_prompts[:10], corrupt_prompts[:10], strict_check=True)
        print("  [PASS] no formatting divergence detected")
    except ValueError as e:
        print(f"  [FAIL] strict_check raised unexpectedly: {e}")

    # --- Test 4: strict_check should FAIL on a deliberately broken pair ---
    print("\n[TEST 4] strict_check on a deliberately mismatched pair (SHOULD raise)")
    broken_clean = [" Vietnam: Hanoi; China:"]
    broken_corrupt = [" Messi: soccer; China."]  # "." instead of ":" before query
    try:
        tokenize_and_align(model, broken_clean, broken_corrupt, strict_check=True)
        print("  [FAIL] strict_check did NOT raise -- it should have caught this")
    except ValueError as e:
        print(f"  [PASS] strict_check correctly raised: {e}")