import torch
from torch.profiler import profile as torch_profile, ProfilerActivity, record_function
from transformers import LlamaConfig, LlamaForCausalLM
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
    PROMPT_LEN,
    MAX_NEW_TOKENS,
)

# ── Optimization flags ────────────────────────────────────────────────────────
USE_FP16 = True
USE_KV_CACHE = True
USE_COMPILE = True
USE_FLASH_ATTN = False  # uses torch SDPA (no extra package needed)


def optimized_loop(model, input_ids, n_steps):
    # TODO: fix the performance issues you found — changes may include
    # both `optimized_loop` and `generate_optimized`
    generated_tokens = []

    if USE_KV_CACHE:
        past_key_values = None
        current_input = input_ids.clone()
        for _ in range(n_steps):
            outputs = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token_id.item())
            current_input = next_token_id.unsqueeze(0)  # shape [1, 1] — only new token
        return generated_tokens

    # No KV cache — same behaviour as slow_loop
    generated_ids = input_ids.clone()
    for _ in range(n_steps):
        outputs = model(input_ids=generated_ids)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token_id.item())
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
    return generated_tokens


def profile(loop_fn, model, input_ids, trace_name: str):

    trace_path = RESULTS_DIR / trace_name

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with torch_profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function(trace_name):
            loop_fn(model, input_ids, PROFILE_STEPS)

    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"

    print(prof.key_averages().table(
        sort_by=sort_key,
        row_limit=20
    ))

    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace exported to: {trace_path}")
    # TODO: wrap loop_fn(model, input_ids, PROFILE_STEPS) with torch.profiler,
    # print the summary table, and export a Chrome trace to RESULTS_DIR / trace_name
    


def generate_optimized(optimized_trace_name: str) -> float:
    # TODO: load the model (consider dtype and other loading options),
    # then call profile() and time_generation() on optimized_loop.
    # Return the elapsed time from time_generation so main() can print a speedup.
    dtype = torch.float16 if USE_FP16 else torch.float32

    if USE_FLASH_ATTN:
        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=4096,
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=PROMPT_LEN + MAX_NEW_TOKENS + 64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config)
        model.to(device="cuda", dtype=dtype)
        model.eval()
    else:
        model = build_model(dtype)

    if USE_COMPILE:
        model = torch.compile(model)

    input_ids = get_input_ids()
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        import json, datetime
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "flags": {
                "USE_FP16": USE_FP16,
                "USE_KV_CACHE": USE_KV_CACHE,
                "USE_COMPILE": USE_COMPILE,
                "USE_FLASH_ATTN": USE_FLASH_ATTN,
            },
            "slow_s": round(slow_elapsed, 4),
            "optimized_s": round(optimized_elapsed, 4),
            "speedup": round(speedup, 4),
        }
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = RESULTS_DIR / f"summary_{ts}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
# 1. float16 (USE_FP16): switched model dtype from float32 to float16 in
#    generate_optimized(). Halves memory per parameter and per activation,
#    reducing bandwidth pressure on every forward pass. Measured speedup: ~3.8x.
#
# 2. KV cache (USE_KV_CACHE): pass use_cache=True and carry past_key_values
#    between steps. Without this the slow loop reprocesses the full growing
#    sequence each step (O(n^2) attention work); with the cache each step
#    processes only the single new token. Measured additional speedup: ~4-5x
#    on top of fp16, combined ~16x over baseline.
#
# 3. torch.compile (USE_COMPILE): wrap the model with torch.compile() before
#    the generation loop. Inductor fuses CUDA kernels, reduces launch overhead,
#    and selects faster kernel implementations. Measured additional speedup:
#    ~1.4x on top of fp16 + KV cache, combined ~22x over baseline.
#
# 4. SDPA / flash attention (USE_FLASH_ATTN): set attn_implementation="sdpa"
#    at model construction. This regressed performance (~12x vs 22x). With KV
#    cache active each decode step attends over a query of length 1, so the
#    quadratic savings of flash attention don't apply and the SDPA kernel
#    introduces overhead relative to the standard path. Left disabled.
#
# Biggest impact and why:
#
# KV cache had the largest single impact. The slow baseline reprocesses the
# entire sequence (prompt + all generated tokens) at every step, meaning
# attention cost grows quadratically with sequence length. Enabling the cache
# reduces each decode step to a single-token forward pass, eliminating the
# redundant computation entirely. float16 and torch.compile both compound on
# top of this but neither addresses as fundamental a source of waste.
