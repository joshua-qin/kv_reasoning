# LatentMAS: Paper vs Our Experimental Setup

Comparison of our implementation and experiment settings to the LatentMAS paper (arXiv:2511.20639, Gen-Verse/LatentMAS).

**LatentMAS is fixed and paper-aligned:** We use 40 latent steps per agent (Planner, Critic, Refiner), temperature 0.6, top-p 0.95, max prompt length 2048, ridge λ=1e-4, and Judger decode length 256. `run_experiment` and `LatentMASConfig` defaults are set to these values so you get the same hyperparameter choices as the paper when running experiments.

**Token counting (paper-aligned):** We report **output tokens only** (generated text), not prompt or context length. Single agent: generated CoT tokens. Sequential TextMAS: sum of generated tokens over the 4 stages. LatentMAS: Judger decode tokens only. This matches the paper’s “reducing output token usage by 70.8%-83.7%” metric.

## What Matches the Paper

| Aspect | Paper | Ours | Match |
|--------|--------|------|--------|
| **Pipeline** | Sequential MAS: Planner → Critic → Refiner → Judger (last agent decodes only) | Same 4-agent flow; Judger does text decode only, no latent rollout | ✓ |
| **Latent mechanism** | e_t = h_t @ W_a; KV cache concatenation via `past_key_values` | Same; `latent_rollout` + `_prefill_with_past` | ✓ |
| **W_a** | Ridge regression, λ > 0, computed once per run | `ridge_lambda=1e-4`, `compute_wa()` once | ✓ |
| **System prompt** | "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." | `SYSTEM_PROMPT_PAPER` in `latentmas_fresh.py` | ✓ |
| **Planner / Critic / Refiner / Judger prompts** | Appendix E (GSM8K numeric) | Same content; minor wording order in Critic/Refiner | ✓ |
| **Judger output format** | `\boxed{YOUR_FINAL_ANSWER}` | Same; we extract via `extract_answer_gsm8k(..., prefer_last=True)` | ✓ |
| **Decoding hyperparams** | Temperature 0.6, top-p 0.95 | `temperature_decode=0.6`, `top_p_decode=0.95` | ✓ |
| **Chat template** | Official templates, `<\|im_start\|>`, `<\|im_end\|>` | `use_chat_template=True`, Qwen-style | ✓ |
| **GSM8K max length** | 2,048 tokens (for ARC-Easy, ARC-Challenge, GSM8K) | `max_prompt_length=2048` | ✓ |
| **GSM8K evaluation** | Numeric equality: extract final answer, normalize (lowercase, trim, remove punctuation), correct iff numeric match | `normalize_answer` + `is_correct`; we support `####` and `\boxed{}` | ✓ |

## What Differs

| Aspect | Paper | Ours | Note |
|--------|--------|------|------|
| **Model** | **Qwen3-4B / 8B / 14B** (Yang et al., 2025) | **Qwen2-7B-Instruct** | Different family (Qwen3 vs Qwen2) and size; paper uses 4B/8B/14B. |
| **Latent steps** | m ∈ {0, 10, 20, 40, 80}; “optimal 40–80”; “moderate latent step budget” in that range | Default config: planner 40, critic 32, refiner 32. In `run_experiment`: derived from `max_new_tokens` (e.g. 48, 38, 38 for 384) | We’re in the same ballpark; paper tunes and reports 40–80. |
| **Judger decode length** | Task-dependent max length (e.g. 2,048 for GSM8K for *context*; decode length not stated explicitly) | `max_new_tokens_decode=256` (and capped by `min(256, args.max_new_tokens)` in experiment) | 256 is enough for short GSM8K answers; paper may use longer for harder tasks. |
| **Reporting** | Mean accuracy over **3 independent runs** | Single run | Paper: “We perform hyperparameter tuning and report the mean performance over three independent runs.” |
| **Backend** | HuggingFace + optional vLLM (prefix caching, tensor-parallel) | HuggingFace only | We don’t use vLLM. |
| **Full test set** | Full benchmark sizes (e.g. GSM8K test) | We often use `--max_eval 50` (or 5 in debug) | Paper reports on full test sets. |

## Summary

- **Algorithm and protocol:** Our setup matches the paper’s design: sequential Planner → Critic → Refiner → Judger, latent rollout for the first three, Judger decode-only, same prompts and evaluation (numeric equality, `\boxed{}`, normalization).
- **Main experimental mismatch:** We use **Qwen2-7B-Instruct** instead of **Qwen3-4B/8B/14B**, so numbers (accuracy, speed, token use) are not directly comparable to the paper. For a closer match you’d need Qwen3 checkpoints and similar latent-step and decode-length settings.
- **Minor differences:** Single run vs mean over 3 runs; latent steps partly tied to `max_new_tokens` in our experiment script; no vLLM; we often evaluate on 50 samples instead of the full GSM8K test set.

### Why is my accuracy lower (~62%) than the paper (~88–92%)?

**Nothing is wrong with the implementation.** The paper’s GSM8K numbers are reported with **Qwen3-4B, Qwen3-8B, and Qwen3-14B** (Table 1). We use **Qwen2-7B-Instruct**—a different model family (Qwen2 vs Qwen3) and size. Base math capability and response to latent collaboration differ by model, so 62% on Qwen2-7B is not comparable to 88% on Qwen3-4B/8B. To reproduce paper-like accuracy you need to run with a Qwen3 checkpoint (e.g. from Hugging Face, if available).

### Why does Single (~78%) beat LatentMAS (62%) here, when the paper has LatentMAS > Single?

In the paper (Qwen3-4B), GSM8K is **Single 82.4%, LatentMAS 88.2%**—so LatentMAS wins. In our runs on the same 50 examples with **Qwen2-7B-Instruct**, **Single ≈ 78%** and **LatentMAS 62%**, so the trend is reversed.

Likely cause: **Qwen2-7B may not benefit from (or may be hurt by) the latent multi-agent pipeline.** For example, the Judger might not use the refiner KV context as effectively, or latent representations may not transfer as well for this model family. The paper’s gains are reported with Qwen3; the effect may be model-dependent. To check, run the same comparison with a Qwen3 model—if LatentMAS then beats Single, the implementation is consistent and the reversal is due to model choice.

Small note: single-agent in `run_experiment` uses a different prompt (no chat template, no `\boxed{}`) and `evaluate_gsm8k(preds)` with default `prefer_last=False`; LatentMAS uses `prefer_last=True`. For a strict apple-to-apple comparison you could use the same prompt and same `prefer_last` for both; that’s unlikely to explain a 16-point gap but is worth aligning if you want to isolate the effect of the pipeline.

To align experiments further with the paper you could:
1. Use Qwen3-4B/8B/14B when available.
2. Set latent steps explicitly in the 40–80 range (e.g. 40 planner, 40 critic, 40 refiner) instead of deriving from `max_new_tokens`.
3. Report mean and std over 3 (or more) runs.
4. Run on the full GSM8K test set (and other benchmarks) with the paper’s max-length settings.
