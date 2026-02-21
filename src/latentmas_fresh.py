"""
Fresh LatentMAS implementation (independent of existing pipeline code).

Pipeline (aligned with paper + official repo):

  Intermediary agents (Planner, Critic, Refiner):
    - Each gets system prompt + question (and for Critic/Refiner: KV cache from the previous agent).
    - Standard latent rollout: prefill prompt on top of previous KV (KV concatenate), then
      latent autoregression (e_t = h_t @ W_a), no text decode. Output is only in latent/KV form.

  Last agent (Judger):
    - Same: system prompt + question + previous agent's KV cache (refiner cache).
    - No latent rollout. It takes the refiner KV cache, prefills the Judger prompt on top,
      then does standard LLM autoregressive decode to produce the final text answer.

  So: intermediary = latent rollout + KV concat from previous; last = KV cache + decode only.
  Paper (Section 3.2–3.3): "only the last agent decoding the final answer"; KV "layer-wise
  concatenation" via past_key_values. Official code (Gen-Verse/LatentMAS): intermediary agents
  use generate_latent_batch(..., past_key_values=past_kv); Judger uses generate_text_batch(
  judger_ids, ..., past_key_values=past_for_decoding) only — no latent steps.

Design choices:
  - Full working-memory transfer by passing full past_key_values from one stage to next.
  - Chat template (<|im_start|> / <|im_end|>) per paper Appendix C.2.
  - Latent autoregression uses e_t = h_t @ W_a where W_a is the paper's ridge solution.
"""

from __future__ import annotations


def _strip_chat_tokens(text: str) -> str:
    """Remove Qwen chat control tokens that may appear in decoded output (e.g. </think>)."""
    for tok in ("</think>", "<|im_start|>"):
        text = text.replace(tok, "")
    return text.strip()

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from src.sequential_text_mas import JUDGER_ANSWER_INSTRUCTIONS
import torch.nn.functional as F


PastKVs = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
_WA_CACHE: Dict[Tuple[int, float], torch.Tensor] = {}

# Paper Appendix C.2 + official prompts
SYSTEM_PROMPT_PAPER = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@dataclass
class LatentMASConfig:
    """Paper-aligned defaults (Section 4 + Appendix): m=40 latent steps, temp=0.6, top_p=0.95, 2048 max length."""
    latent_steps_planner: int = 40   # paper: m ∈ {0,10,20,40,80}, "optimal 40-80", "moderate budget"
    latent_steps_critic: int = 40
    latent_steps_refiner: int = 40
    max_new_tokens_decode: int = 2048  # Judger text decode; repo uses --max_new_tokens 2048 for GSM8K
    min_tokens_before_eos: int = 10   # fallback path only; not used in model.generate() path
    ridge_lambda: float = 1e-4        # paper Appendix A: λ > 0 for numerical stability
    do_sample_decode: bool = True     # paper: "temperature 0.6 and top-p 0.95"
    temperature_decode: float = 0.6
    top_p_decode: float = 0.95
    max_prompt_length: int = 2048    # paper: "2,048 tokens for ARC-Easy, ARC-Challenge, and GSM8K"
    use_chat_template: bool = True   # paper Appendix C.2: official chat templates


def _to_past_tuple(cache: Any) -> PastKVs:
    if cache is None:
        raise ValueError("cache is None")
    if isinstance(cache, tuple):
        return cache
    if hasattr(cache, "layers") and len(cache.layers) > 0:
        return tuple(
            (cache.layers[i].keys.clone(), cache.layers[i].values.clone())
            for i in range(len(cache.layers))
        )
    raise TypeError(f"Unsupported cache type: {type(cache)}")


def _tuple_to_dynamic(cache_tuple: PastKVs, config: Any) -> Any:
    from transformers.cache_utils import DynamicCache

    return DynamicCache(ddp_cache_data=cache_tuple, config=config)


def _sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
    suppress_eos_id: int | None = None,
) -> torch.Tensor:
    """Sample one next token. If suppress_eos_id is set, mask it out so we don't sample EOS (for min-tokens guard)."""
    if suppress_eos_id is not None:
        logits = logits.clone()
        logits[..., suppress_eos_id] = -float("inf")
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    temperature = max(float(temperature), 1e-5)
    probs = F.softmax(logits / temperature, dim=-1)
    if suppress_eos_id is not None:
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    top_p = float(top_p)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= top_p
        keep[..., 0] = True
        masked = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        masked = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sampled_sorted = torch.multinomial(masked, num_samples=1)
        return sorted_indices.gather(-1, sampled_sorted)
    return torch.multinomial(probs, num_samples=1)


def _get_output_embedding_weight(model: Any) -> torch.Tensor:
    out = model.get_output_embeddings()
    if out is None:
        out = getattr(model, "lm_head", None)
    if out is None or not hasattr(out, "weight"):
        raise ValueError("Could not find model output embedding weight.")
    return out.weight


def compute_wa(
    model: Any,
    device: torch.device,
    ridge_lambda: float = 1e-4,
) -> torch.Tensor:
    """
    LatentMAS alignment matrix:
      W_a = (W_out^T W_out + lambda I)^(-1) W_out^T W_in
    """
    key = (id(model), float(ridge_lambda))
    if key in _WA_CACHE:
        return _WA_CACHE[key]

    w_in = model.get_input_embeddings().weight.detach().to(torch.float32, copy=True).cpu()
    w_out = _get_output_embedding_weight(model).detach().to(torch.float32, copy=True).cpu()
    if w_in.shape != w_out.shape:
        raise ValueError(f"Shape mismatch: W_in {w_in.shape}, W_out {w_out.shape}")
    d_h = w_in.shape[1]
    eye = torch.eye(d_h, dtype=torch.float32)
    left = w_out.T @ w_out + float(ridge_lambda) * eye
    right = w_out.T @ w_in
    wa = torch.linalg.solve(left, right)
    wa = wa.to(device=device, dtype=next(model.parameters()).dtype)
    _WA_CACHE[key] = wa
    return wa


def _prefill_from_scratch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_length: int,
) -> Tuple[PastKVs, torch.Tensor]:
    """
    Prefill prompt without previous memory.
    Returns (cache, last_hidden_at_last_prompt_token).
    """
    ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)
    attn = torch.ones_like(ids, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=attn,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    cache = _to_past_tuple(out.past_key_values)
    hidden = out.hidden_states[-1][:, -1, :].detach()
    return cache, hidden


def _prefill_with_past(
    model: Any,
    tokenizer: Any,
    prompt: str,
    past: PastKVs,
    device: torch.device,
    max_length: int,
) -> Tuple[PastKVs, torch.Tensor]:
    """
    Full-memory transfer: treat `past` as latent working memory and append prompt tokens.
    Returns updated (cache, last_hidden_at_last_prompt_token).
    """
    ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)
    pos0 = past[0][0].shape[2]
    pos_ids = torch.arange(pos0, pos0 + ids.shape[1], device=device, dtype=torch.long).unsqueeze(0)
    attn = torch.ones((1, pos0 + ids.shape[1]), dtype=torch.long, device=device)
    config = getattr(model, "config", model.model.config)
    past_dyn = _tuple_to_dynamic(past, config)
    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=attn,
            position_ids=pos_ids,
            past_key_values=past_dyn,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    cache = _to_past_tuple(out.past_key_values)
    hidden = out.hidden_states[-1][:, -1, :].detach()
    return cache, hidden


def latent_rollout(
    model: Any,
    past: PastKVs,
    h_start: torch.Tensor,
    wa: torch.Tensor,
    latent_steps: int,
) -> Tuple[PastKVs, torch.Tensor]:
    """
    Latent autoregression: e_t = h_t @ W_a as next input embedding.
    """
    config = getattr(model, "config", model.model.config)
    cache = past
    h = h_start
    pos = cache[0][0].shape[2]
    bsz = h.shape[0]
    dev = h.device

    with torch.no_grad():
        for _ in range(max(0, latent_steps)):
            e = (h @ wa).unsqueeze(1)  # (1, 1, d_h)
            past_dyn = _tuple_to_dynamic(cache, config)
            pos_ids = torch.tensor([[pos]], device=dev, dtype=torch.long)
            attn = torch.ones((bsz, pos + 1), dtype=torch.long, device=dev)
            out = model(
                inputs_embeds=e,
                attention_mask=attn,
                position_ids=pos_ids,
                past_key_values=past_dyn,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            cache = _to_past_tuple(out.past_key_values)
            h = out.hidden_states[-1][:, -1, :].detach()
            pos += 1
    return cache, h


def _decode_from_past(
    model: Any,
    tokenizer: Any,
    past: PastKVs,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    min_tokens_before_eos: int = 10,
) -> Tuple[str, int]:
    """
    Last agent only: Judger decode via model.generate() with past_key_values (official
    Gen-Verse/LatentMAS generate_text_batch implementation). Takes refiner KV cache,
    prefills Judger prompt, then standard LLM decode. No latent rollout.
    """
    dev = next(model.parameters()).device
    config = getattr(model, "config", model.model.config)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    past_len = past[0][0].shape[2]
    attention_mask = torch.ones_like(ids, dtype=torch.long, device=dev)
    cache_position = torch.arange(
        past_len,
        past_len + ids.shape[-1],
        dtype=torch.long,
        device=dev,
    )
    if past_len > 0:
        past_mask = torch.ones(
            (attention_mask.shape[0], past_len),
            dtype=attention_mask.dtype,
            device=dev,
        )
        attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    past_cache = _tuple_to_dynamic(past, config)
    with torch.no_grad():
        outputs = model.generate(
            ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=max(float(temperature), 1e-5),
            top_p=float(top_p),
            do_sample=do_sample,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            past_key_values=past_cache,
            cache_position=cache_position,
            return_dict_in_generate=True,
        )
    sequences = outputs.sequences
    prompt_len = ids.shape[1]
    generated_ids = sequences[0, prompt_len:]
    text = _strip_chat_tokens(tokenizer.decode(generated_ids, skip_special_tokens=True))
    return text, int(sequences.shape[1] - prompt_len)


def _decode_from_scratch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    """Decode from scratch (no past). Used when decode-from-past returns empty (e.g. EOS immediately)."""
    dev = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids, dtype=torch.long, device=dev),
            use_cache=True,
            return_dict=True,
        )
        generated = ids.clone()
        past = _to_past_tuple(out.past_key_values)
        cur_pos = ids.shape[1]
        for _ in range(max_new_tokens - 1):
            logits = out.logits[:, -1, :]
            nxt = _sample_next_token(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            generated = torch.cat([generated, nxt], dim=1)
            config = getattr(model, "config", model.model.config)
            past_dyn = _tuple_to_dynamic(past, config)
            out = model(
                input_ids=nxt,
                attention_mask=torch.ones((1, cur_pos + 1), dtype=torch.long, device=dev),
                position_ids=torch.tensor([[cur_pos]], device=dev, dtype=torch.long),
                past_key_values=past_dyn,
                use_cache=True,
                return_dict=True,
            )
            past = _to_past_tuple(out.past_key_values)
            cur_pos += 1
            if tokenizer.eos_token_id is not None and nxt.item() == tokenizer.eos_token_id:
                break
    prompt_len = ids.shape[1]
    new_tokens = generated[0, prompt_len:]
    return _strip_chat_tokens(tokenizer.decode(new_tokens, skip_special_tokens=True))


def _apply_chat_template(
    tokenizer: Any,
    system: str,
    user_content: str,
    add_generation_prompt: bool = True,
) -> str:
    """Format using the tokenizer's chat template (paper: <|im_start|> and <|im_end|>). Requires jinja2>=3.1.0."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _planner_prompt(question: str) -> str:
    """Sequential LatentMAS Planner (paper Appendix E, GSM8K)."""
    return f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""


def _critic_prompt(question: str) -> str:
    """Sequential LatentMAS Critic (paper Appendix E)."""
    return f"""
Question: {question}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""


def _refiner_prompt(question: str) -> str:
    """Sequential LatentMAS Refiner (paper Appendix E)."""
    return f"""
Question: {question}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:
"""


def _judger_prompt(question: str) -> str:
    """Sequential LatentMAS Judger: same instructions as TextMAS Judger for consistency."""
    return f"""Target Question: {question}

{JUDGER_ANSWER_INSTRUCTIONS}
"""


def _prompt_text(
    tokenizer: Any,
    role_content: str,
    use_chat_template: bool,
) -> str:
    if use_chat_template:
        return _apply_chat_template(
            tokenizer,
            system=SYSTEM_PROMPT_PAPER,
            user_content=role_content,
            add_generation_prompt=True,
        )
    return SYSTEM_PROMPT_PAPER + "\n\n" + role_content + "\n\nResponse:"


def run_sequential_latent_mas(
    model: Any,
    tokenizer: Any,
    question: str,
    device: torch.device,
    cfg: LatentMASConfig | None = None,
) -> Tuple[str, int]:
    """
    4-agent sequential LatentMAS (paper + official repo): Planner -> Critic -> Refiner -> Judger.
    Intermediary agents: each gets prompt + previous stage KV cache, then latent rollout only (no text decode).
    Judger: gets prompt + refiner KV cache, then standard LLM decode to text only.

    1) Planner: prefill from scratch -> latent rollout -> cache_p
    2) Critic:  prefill with past=cache_p (KV concat) -> latent rollout -> cache_c
    3) Refiner: prefill with past=cache_c (KV concat) -> latent rollout -> cache_r
    4) Judger:  prefill with past=cache_r (KV concat), then decode to text (no latent steps)
    """
    if cfg is None:
        cfg = LatentMASConfig()

    use_chat = cfg.use_chat_template
    wa = compute_wa(model, device=device, ridge_lambda=cfg.ridge_lambda)

    # Stage 1: Planner (latent)
    prompt_planner = _prompt_text(tokenizer, _planner_prompt(question), use_chat)
    cache_p, h_p = _prefill_from_scratch(
        model, tokenizer, prompt_planner, device=device, max_length=cfg.max_prompt_length
    )
    cache_p, _ = latent_rollout(
        model, cache_p, h_p, wa, latent_steps=cfg.latent_steps_planner
    )

    # Stage 2: Critic (latent)
    prompt_critic = _prompt_text(tokenizer, _critic_prompt(question), use_chat)
    cache_c, h_c = _prefill_with_past(
        model, tokenizer, prompt_critic, past=cache_p, device=device, max_length=cfg.max_prompt_length
    )
    cache_c, _ = latent_rollout(
        model, cache_c, h_c, wa, latent_steps=cfg.latent_steps_critic
    )

    # Stage 3: Refiner (latent)
    prompt_refiner = _prompt_text(tokenizer, _refiner_prompt(question), use_chat)
    cache_r, h_r = _prefill_with_past(
        model, tokenizer, prompt_refiner, past=cache_c, device=device, max_length=cfg.max_prompt_length
    )
    cache_r, _ = latent_rollout(model, cache_r, h_r, wa, latent_steps=cfg.latent_steps_refiner)

    # Stage 4: Judger — KV from refiner + Judger prompt; standard LLM decode only (no latent rollout)
    prompt_judger = _prompt_text(tokenizer, _judger_prompt(question), use_chat)
    final_text, decode_tokens = _decode_from_past(
        model,
        tokenizer,
        cache_r,
        prompt=prompt_judger,
        max_new_tokens=cfg.max_new_tokens_decode,
        do_sample=cfg.do_sample_decode,
        temperature=cfg.temperature_decode,
        top_p=cfg.top_p_decode,
        min_tokens_before_eos=cfg.min_tokens_before_eos,
    )
    # Fallback only if still empty (e.g. all generated tokens were special)
    if not (final_text and final_text.strip()):
        final_text = _decode_from_scratch(
            model,
            tokenizer,
            prompt=prompt_judger,
            max_new_tokens=cfg.max_new_tokens_decode,
            do_sample=cfg.do_sample_decode,
            temperature=cfg.temperature_decode,
            top_p=cfg.top_p_decode,
        )
        decode_tokens = 0
    # Paper-aligned: report output tokens only (Judger decode), not total context positions
    return final_text, decode_tokens
