"""
Fresh LatentMAS implementation (independent of existing pipeline code).

Pipeline:
  solver (latent thoughts) -> critic (latent thoughts) -> solver (latent thoughts) -> final decode

Design choices:
  - Full working-memory transfer by passing full past_key_values from one stage to next.
  - No intermediate text decoding; only final answer is decoded.
  - Latent autoregression uses e_t = h_t @ W_a where W_a is the paper's ridge solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


PastKVs = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
_WA_CACHE: Dict[Tuple[int, float], torch.Tensor] = {}


@dataclass
class LatentMASConfig:
    latent_steps_solver_1: int = 40
    latent_steps_critic: int = 32
    latent_steps_solver_2: int = 32
    max_new_tokens_decode: int = 128
    ridge_lambda: float = 1e-4
    do_sample_decode: bool = False
    temperature_decode: float = 0.2
    top_p_decode: float = 1.0
    max_prompt_length: int = 2048
    system_prompt: str = "You are a precise reasoner."


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
) -> torch.Tensor:
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    temperature = max(float(temperature), 1e-5)
    probs = F.softmax(logits / temperature, dim=-1)
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
) -> Tuple[str, int]:
    """
    Decode final text only once from latent memory.
    """
    dev = next(model.parameters()).device
    config = getattr(model, "config", model.model.config)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    cache = past
    pos0 = cache[0][0].shape[2]
    pos_ids = torch.arange(pos0, pos0 + ids.shape[1], device=dev, dtype=torch.long).unsqueeze(0)
    attn = torch.ones((1, pos0 + ids.shape[1]), dtype=torch.long, device=dev)
    past_dyn = _tuple_to_dynamic(cache, config)
    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=attn,
            position_ids=pos_ids,
            past_key_values=past_dyn,
            use_cache=True,
            return_dict=True,
        )
        cache = _to_past_tuple(out.past_key_values)
        generated = ids.clone()
        cur_pos = pos0 + ids.shape[1]
        for _ in range(max_new_tokens - 1):
            logits = out.logits[:, -1, :]
            nxt = _sample_next_token(
                logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            generated = torch.cat([generated, nxt], dim=1)
            past_dyn = _tuple_to_dynamic(cache, config)
            out = model(
                input_ids=nxt,
                attention_mask=torch.ones((1, cur_pos + 1), dtype=torch.long, device=dev),
                position_ids=torch.tensor([[cur_pos]], device=dev, dtype=torch.long),
                past_key_values=past_dyn,
                use_cache=True,
                return_dict=True,
            )
            cache = _to_past_tuple(out.past_key_values)
            cur_pos += 1
            if tokenizer.eos_token_id is not None and nxt.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated[0], skip_special_tokens=True), int(generated.shape[1])


def _solver_prompt(question: str, system_prompt: str) -> str:
    return (
        f"{system_prompt}\n\n"
        "You are Solver. Solve the problem carefully in latent reasoning. "
        "Do not output final number yet.\n\n"
        f"Question: {question}\n\n"
        "Latent reasoning:"
    )


def _critic_prompt(question: str, system_prompt: str) -> str:
    return (
        f"{system_prompt}\n\n"
        "You are Critic. Audit prior latent reasoning for mistakes and missing constraints. "
        "Do not output final number yet.\n\n"
        f"Question: {question}\n\n"
        "Latent critique:"
    )


def _solver_refine_prompt(question: str, system_prompt: str) -> str:
    return (
        f"{system_prompt}\n\n"
        "You are Solver again. Integrate critique and finalize latent reasoning. "
        "Do not output final number yet.\n\n"
        f"Question: {question}\n\n"
        "Latent refinement:"
    )


def run_latentmas_solver_critic_solver(
    model: Any,
    tokenizer: Any,
    question: str,
    device: torch.device,
    cfg: LatentMASConfig | None = None,
) -> Tuple[str, int]:
    """
    Strict fresh LatentMAS-style flow:
      1) Solver prefill + latent rollout
      2) Critic prefill WITH full solver memory + latent rollout
      3) Solver-refine prefill WITH full critic memory + latent rollout
      4) Decode final answer once
    """
    if cfg is None:
        cfg = LatentMASConfig()

    wa = compute_wa(model, device=device, ridge_lambda=cfg.ridge_lambda)

    # Stage 1: solver latent reasoning
    cache_s1, h_s1 = _prefill_from_scratch(
        model,
        tokenizer,
        _solver_prompt(question, cfg.system_prompt),
        device=device,
        max_length=cfg.max_prompt_length,
    )
    cache_s1, _ = latent_rollout(
        model,
        cache_s1,
        h_s1,
        wa,
        latent_steps=cfg.latent_steps_solver_1,
    )

    # Stage 2: critic receives full solver working memory
    cache_c, h_c = _prefill_with_past(
        model,
        tokenizer,
        _critic_prompt(question, cfg.system_prompt),
        past=cache_s1,
        device=device,
        max_length=cfg.max_prompt_length,
    )
    cache_c, _ = latent_rollout(
        model,
        cache_c,
        h_c,
        wa,
        latent_steps=cfg.latent_steps_critic,
    )

    # Stage 3: solver receives full critic working memory
    cache_s2, h_s2 = _prefill_with_past(
        model,
        tokenizer,
        _solver_refine_prompt(question, cfg.system_prompt),
        past=cache_c,
        device=device,
        max_length=cfg.max_prompt_length,
    )
    cache_s2, _ = latent_rollout(
        model,
        cache_s2,
        h_s2,
        wa,
        latent_steps=cfg.latent_steps_solver_2,
    )

    # Final decode
    decode_prompt = (
        "\n\nNow output only the final numeric answer in this format: #### <number>\n"
        "Final answer:"
    )
    final_text, decode_tokens = _decode_from_past(
        model,
        tokenizer,
        cache_s2,
        prompt=decode_prompt,
        max_new_tokens=cfg.max_new_tokens_decode,
        do_sample=cfg.do_sample_decode,
        temperature=cfg.temperature_decode,
        top_p=cfg.top_p_decode,
    )
    total_positions = int(cache_s2[0][0].shape[2] + decode_tokens)
    return final_text, total_positions
