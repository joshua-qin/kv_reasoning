# LatentMAS Fresh: Technical Description

This document explains in technical detail how our LatentMAS implementation (`src/latentmas_fresh.py`) works. It follows the paper (arXiv:2511.20639) and the official Gen-Verse/LatentMAS repo: **intermediary agents** do **latent rollout only** (no text decode); the **last agent (Judger)** does **standard autoregressive text decode** on top of the previous agent’s KV cache.

---

## 1. High-level pipeline

We use a **sequential 4-agent** setup:

```
Planner → Critic → Refiner → Judger
```

- **Planner, Critic, Refiner**: Each gets a **text prompt** (system + role-specific text). They **prefill** that prompt (optionally on top of the previous agent’s KV cache), then run **latent autoregression** for a fixed number of steps. No tokens are decoded; only **KV caches** and **last-layer hidden states** are produced and passed on.
- **Judger**: Gets the **Refiner’s KV cache** plus the Judger prompt. It **prefills** the prompt on that cache, then performs **standard LLM autoregressive decoding** (e.g. `model.generate()`) to produce the final **text answer** (e.g. with `\boxed{...}`). No latent rollout.

So: **intermediary agents = latent rollout + KV handoff; last agent = KV + decode only.**

---

## 2. Input–output alignment (W_a)

In standard decoding, the model turns a **last-layer hidden state** \(h_t \in \mathbb{R}^{d_h}\) into logits via the LM head and then samples a token. For **latent** steps we never sample a token; we feed a **continuous vector** as the next “token” input. That vector must live in the **embedding space** the model expects (same statistics as token embeddings). The paper uses a linear map \(W_a\) so that:

\[
e_{t+1} = h_t \, W_a
\]

with \(W_a \in \mathbb{R}^{d_h \times d_h}\). So the **next input embedding** is the previous step’s last-layer hidden state, projected by \(W_a\).

**Ridge regression (our implementation):** We compute \(W_a\) once per run and reuse it for all latent steps:

\[
W_a = (W_{\text{out}}^\top W_{\text{out}} + \lambda I)^{-1} W_{\text{out}}^\top W_{\text{in}}
\]

- \(W_{\text{in}}\): input embedding matrix (embedding layer weight), shape \((V, d_h)\).
- \(W_{\text{out}}\): output embedding matrix (LM head weight), shape \((V, d_h)\).
- \(\lambda\): ridge regularisation (e.g. `ridge_lambda=1e-4`).

Implemented in `compute_wa()`. Result is cached by `(id(model), ridge_lambda)` so we don’t recompute. The idea is to map from “output” hidden space back toward “input” embedding space so that feeding \(e = h\,W_a\) into the model stays in-distribution.

---

## 3. KV cache as working memory

The model’s **past_key_values** are the layer-wise Key and Value tensors from all previous positions. We treat them as **latent working memory**: they encode the full context (prompt + prior latent steps) so we don’t re-run the transformer on earlier positions.

- **Shape (per layer):** keys and values are typically `(batch, num_heads, seq_len, head_dim)`. `seq_len` is the number of **positions** stored (prompt length + number of latent steps so far, or prompt only for the Judger before it decodes).
- **Handoff:** We pass the **full** `past_key_values` from one stage to the next. The next stage **concatenates** its new tokens (or its new single-position latent “token”) with this past, so the effective sequence is: [previous context] + [new prompt tokens] or [previous context] + [new latent position].

We use a **tuple** of `(key, value)` per layer for internal use, and convert to HuggingFace’s `DynamicCache` when calling `model(..., past_key_values=...)` or `model.generate(..., past_key_values=...)`.

---

## 4. Prefill: from scratch vs with past

**From scratch (`_prefill_from_scratch`):**  
Used for the **Planner** only.

- Tokenize the prompt (system + Planner text), truncate to `max_prompt_length`.
- One forward pass: `model(input_ids=ids, attention_mask=..., use_cache=True, output_hidden_states=True)`.
- Return:
  - `past_key_values` (tuple form),
  - last-layer hidden state at the final prompt position: `hidden_states[-1][:, -1, :]`.

**With past (`_prefill_with_past`):**  
Used for **Critic, Refiner, and (conceptually) Judger**.

- Tokenize the prompt.
- `past` is the previous stage’s full KV cache; its length is `past_len = past[0][0].shape[2]`.
- We pass **position IDs** as `[past_len, past_len+1, ..., past_len + prompt_len - 1]` so that the new tokens are seen as a continuation.
- Attention mask: shape `(1, past_len + prompt_len)`, all ones (causal masking is implied by position).
- Forward: `model(input_ids=ids, attention_mask=..., position_ids=pos_ids, past_key_values=past_dyn, use_cache=True, output_hidden_states=True)`.
- Return the **updated** `past_key_values` (now including the new prompt positions) and the last-layer hidden at the last prompt token.

So “prefill with past” = **concatenate** the new prompt on top of the previous agent’s working memory in KV space.

---

## 5. Latent rollout (Planner, Critic, Refiner only)

**Function:** `latent_rollout(model, past, h_start, wa, latent_steps)`.

- **Input:**  
  - `past`: current KV cache (after prefill for this agent),  
  - `h_start`: last-layer hidden state at the current last position (e.g. after prefill),  
  - `wa`: alignment matrix \(W_a\),  
  - `latent_steps`: number of latent steps (e.g. 40 for Planner, 32 for Critic/Refiner).

- **Loop** for `latent_steps`:
  1. **Next embedding:** \(e = h\,W_a\), then `e = e.unsqueeze(1)` → shape `(1, 1, d_h)` (one “token”, one position).
  2. **Position:** current length = `cache[0][0].shape[2]`; we use `position_ids = [[pos]]` for this single new position.
  3. **Attention mask:** `(1, pos + 1)` all ones.
  4. **Forward:** `model(inputs_embeds=e, attention_mask=..., position_ids=..., past_key_values=past_dyn, use_cache=True, output_hidden_states=True)`.
  5. **Update:**  
     - `cache = _to_past_tuple(out.past_key_values)` (cache now includes this new position).  
     - `h = out.hidden_states[-1][:, -1, :]` (new last hidden for the next latent step).  
     - `pos += 1`.

- **Output:** Updated `past` and final `h`. No token IDs are ever decoded; the “thought” is entirely in the extended KV cache and the last hidden state.

So each latent step is: **one** forward with a **single** input embedding \(e = h\,W_a\), then the cache and \(h\) are updated for the next step.

---

## 6. Judger: decode from past (text only, no latent steps)

The **Judger** must produce the final **text** answer. It does **not** run latent rollout.

**Input:**  
- Refiner’s full KV cache `cache_r`,  
- Judger prompt string (system + Judger instruction + question, with “output final answer in `\boxed{...}`”).

**Steps:**

1. **Tokenize** the Judger prompt (`add_special_tokens=False` to match the official setup).
2. **Past length:** `past_len = past[0][0].shape[2]`.
3. **Causal / position setup:**  
   - `cache_position = arange(past_len, past_len + len(prompt_ids))` (1D, no batch dim) so that the prompt tokens are at positions `past_len, ..., past_len + len(prompt_ids) - 1`.  
   - If `past_len > 0`, attention mask is `[past_mask; current_mask]` (concatenate), so the model sees full context.
4. **Convert** `past` to a `DynamicCache` (required by HuggingFace `generate` in recent Transformers).
5. **Generate:**  
   `model.generate(ids, attention_mask=..., max_new_tokens=..., temperature=..., top_p=..., do_sample=..., pad_token_id=..., eos_token_id=..., past_key_values=past_cache, cache_position=cache_position, ...)`.

So the Judger **prefills** the Judger prompt on top of the Refiner’s KV cache (via `past_key_values` and `cache_position`), then runs **standard autoregressive decoding** for up to `max_new_tokens` (e.g. 256). No \(W_a\), no latent steps.

**Output:** We take `sequences[0, prompt_len:]`, decode to text, and strip. That string is the model’s answer (we then extract the value inside `\boxed{...}` for GSM8K).

**Fallback:** If the decoded text is empty (e.g. model emitted only special tokens), we call `_decode_from_scratch` with the same Judger prompt and no past, and use that as the final answer.

---

## 7. End-to-end flow (code path)

1. **Compute \(W_a\)** once: `compute_wa(model, device, ridge_lambda)`.
2. **Planner:**  
   - Build Planner prompt (chat template: system + Planner text with question).  
   - `cache_p, h_p = _prefill_from_scratch(model, tokenizer, prompt_planner, ...)`.  
   - `cache_p, _ = latent_rollout(model, cache_p, h_p, wa, latent_steps_planner)`.
3. **Critic:**  
   - Build Critic prompt.  
   - `cache_c, h_c = _prefill_with_past(model, tokenizer, prompt_critic, past=cache_p, ...)`.  
   - `cache_c, _ = latent_rollout(model, cache_c, h_c, wa, latent_steps_critic)`.
4. **Refiner:**  
   - Build Refiner prompt.  
   - `cache_r, h_r = _prefill_with_past(model, tokenizer, prompt_refiner, past=cache_c, ...)`.  
   - `cache_r, _ = latent_rollout(model, cache_r, h_r, wa, latent_steps_refiner)`.
5. **Judger:**  
   - Build Judger prompt.  
   - `final_text, decode_tokens = _decode_from_past(model, tokenizer, cache_r, prompt=prompt_judger, max_new_tokens=..., ...)`.  
   - If `final_text` is empty, replace with `_decode_from_scratch(...)`.

Return `final_text` and a total position count (e.g. refiner cache length + Judger decode length) for logging.

---

## 8. Config and hyperparameters

- **Latent steps:** `latent_steps_planner` (e.g. 40), `latent_steps_critic`, `latent_steps_refiner` (e.g. 32). In `run_experiment` they can be derived from `max_new_tokens` (e.g. 48, 38, 38 for 384).
- **Judger decode:** `max_new_tokens_decode` (e.g. 256), `do_sample_decode`, `temperature_decode` (0.6), `top_p_decode` (0.95).
- **Alignment:** `ridge_lambda` (1e-4).
- **Prompt:** `max_prompt_length` (2048), `use_chat_template` (True). System prompt is the paper’s: “You are Qwen, created by Alibaba Cloud. You are a helpful assistant.”

---

## 9. Summary table

| Stage   | Input context              | Operation                    | Output              |
|--------|----------------------------|-----------------------------|---------------------|
| Planner| Prompt only                | Prefill from scratch → latent rollout | KV cache + hidden  |
| Critic | Planner KV + Critic prompt | Prefill with past → latent rollout    | KV cache + hidden  |
| Refiner| Critic KV + Refiner prompt | Prefill with past → latent rollout    | KV cache + hidden  |
| Judger | Refiner KV + Judger prompt | Prefill with past → **text decode**   | Final answer text   |

No intermediary stage decodes tokens; only the Judger produces text, using standard `model.generate()` with the refiner’s KV cache as context.
