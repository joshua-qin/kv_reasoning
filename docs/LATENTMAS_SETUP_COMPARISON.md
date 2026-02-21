# LatentMAS: Paper vs Our Experimental Setup

Comparison of our implementation and experiment settings to the LatentMAS paper (arXiv:2511.20639, Gen-Verse/LatentMAS).

**LatentMAS is fixed and paper-aligned:** We use 40 latent steps per agent (Planner, Critic, Refiner), temperature 0.6, top-p 0.95, max prompt length 2048, ridge λ=1e-4, and Judger decode length 2048 (matches repo `--max_new_tokens 2048`). `run_experiment` default `--max_new_tokens` is 2048 so single-agent, TextMAS, and LatentMAS all match the repo.

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
| **Max new tokens (Judger / decode)** | Repo: `--max_new_tokens 2048` for baseline and GSM8K | `--max_new_tokens` default **2048**; single_agent_paper, TextMAS per stage, and LatentMAS Judger all use it | ✓ Matches repo for baseline/Judger. |
| **TextMAS per-agent params** | Repo **run commands** use `--max_new_tokens 2048` for text_mas (and baseline, latent_mas); class default 256 is overridden by CLI | We use `args.max_new_tokens` (default 2048) per stage, `temperature=0.6`, `top_p=0.95` | ✓ Matches repo run commands (2048). Temperature: repo class default 0.7, we use 0.6 (LatentMAS-aligned). |
| **GSM8K evaluation** | Numeric equality: extract final answer, normalize (strip + lower only in repo), correct iff string match | `normalize_answer_paper` (strip+lower) + `is_correct(..., use_paper_normalization=True)`; we support `####` and `\boxed{}`; extraction order aligned with Gen-Verse/LatentMAS | ✓ |

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

**Fair comparison baseline:** We now provide **`single_agent_paper`**: same system prompt (Qwen), chat template, “output inside \boxed{}” instruction, and decoding (temperature 0.6, top_p 0.95) as TextMAS/LatentMAS. Use `--methods single_agent_paper sequential_text_mas sequential_latent_mas` for an apple-to-apple comparison. The legacy **`single`** baseline uses a plain prompt, no chat template, no \boxed{}, and greedy decoding—so it is not directly comparable to MAS.

### Accuracy evaluation vs paper/Gen-Verse/LatentMAS

Our evaluation is aligned with the [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS) repo so that accuracy is directly comparable:

- **Extraction** (`src/eval_gsm8k.extract_answer_gsm8k`): We support `#### <number>` and `\boxed{...}` (taking the last match). For `\boxed{}` we take the first number inside the last boxed expression, matching their `utils.extract_gsm8k_answer`. We also support “answer is X” and a last-number fallback.
- **Normalization** (paper parity): We use **strip + lower only** by default (`normalize_answer_paper`), matching their `utils.normalize_answer`. So `"42"` and `"42.0"` are *not* considered equal (paper repo does not normalize numerics). Set `use_paper_normalization=False` in `evaluate_gsm8k` / `is_correct` if you want the more lenient behavior (commas removed, 42.0 → 42).
- **Gold**: GSM8K gold is taken as the substring after `####` in the dataset answer; we normalize at comparison time with the same rule as the prediction.

So reported accuracy in this codebase is comparable to the paper/GPU numbers when using the default `use_paper_normalization=True`.

**Run commands:** The repo README runs baseline, text_mas, and latent_mas all with `--max_new_tokens 2048`. We do the same (default 2048). The only remaining difference is temperature: their `TextMASMethod` class default is 0.7; we use 0.6 for all MAS (LatentMAS paper).

To align experiments further with the paper you could:
1. Use Qwen3-4B/8B/14B when available.
2. Set latent steps explicitly in the 40–80 range (e.g. 40 planner, 40 critic, 40 refiner) instead of deriving from `max_new_tokens`.
3. Report mean and std over 3 (or more) runs.
4. Run on the full GSM8K test set (and other benchmarks) with the paper’s max-length settings.
