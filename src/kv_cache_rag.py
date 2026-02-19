"""
KV cache extraction, retrieval by key similarity, and stitching for second-round reasoning.
Training-free: two agents (two runs of same model) share selected parts of each other's KV cache.
Positional embeddings (RoPE) are re-encoded after stitching so keys use global positions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any

from .rope_utils import reencode_past_kvs_for_stitching
from .eval_gsm8k import extract_answer_gsm8k, normalize_answer

# past_key_values: tuple of (key, value) per layer
# key, value: (batch, num_heads, seq_len, head_dim)
PastKVs = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


def _sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample or greedy-pick next token from logits."""
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    temperature = max(float(temperature), 1e-5)
    probs = F.softmax(logits / temperature, dim=-1)
    top_p = float(top_p)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= top_p
        keep[..., 0] = True  # always keep at least one token
        masked = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        masked = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sampled_sorted = torch.multinomial(masked, num_samples=1)
        return sorted_indices.gather(-1, sampled_sorted)
    return torch.multinomial(probs, num_samples=1)


def _to_past_kvs_tuple(cache: Any) -> PastKVs:
    """Convert DynamicCache or tuple to (key, value) tuple per layer."""
    if cache is None:
        raise ValueError("cache is None")
    if isinstance(cache, tuple):
        return cache
    # HuggingFace DynamicCache: cache.layers[i] has .keys and .values
    if hasattr(cache, "layers") and len(cache.layers) > 0:
        return tuple(
            (cache.layers[i].keys.clone(), cache.layers[i].values.clone())
            for i in range(len(cache.layers))
        )
    raise TypeError(f"Unknown cache type: {type(cache)}")


def get_full_kv_cache(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tuple[torch.LongTensor, PastKVs]:
    """
    Run the model autoregressively and return the generated token ids and the full KV cache
    after generating `max_new_tokens` (or until EOS).
    Returns (input_ids_with_generated, past_key_values).
    """
    model.eval()
    batch_size = prompt_ids.shape[0]
    assert batch_size == 1, "Only batch_size=1 supported for now"
    past_key_values = None
    position_ids = None
    generated = prompt_ids.clone()
    attention_mask = torch.ones_like(generated, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if past_key_values is None:
                input_ids = generated
                position_ids = None  # model uses 0..seq_len-1 by default
            else:
                input_ids = generated[:, -1:]
                # Next token position = length of sequence in cache (0-indexed)
                position_ids = torch.tensor(
                    [[generated.shape[1] - 1]], device=device, dtype=torch.long
                )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = _sample_next_token(
                next_token_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=torch.long, device=device),
                ],
                dim=1,
            )

            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated, past_key_values


def last_key_from_cache(past_key_values: Any, layer: int = -1) -> torch.Tensor:
    """(batch, num_heads, 1, head_dim) — last position key at given layer."""
    kv = _to_past_kvs_tuple(past_key_values)
    k, _ = kv[layer]
    return k[:, :, -1:, :].detach()


def last_n_keys_from_cache(
    past_key_values: Any,
    n: int = 8,
    layer: int = -1,
) -> torch.Tensor:
    """
    Mean of the last n keys at the given layer. (batch, num_heads, 1, head_dim).
    More stable query than a single position for retrieval.
    """
    kv = _to_past_kvs_tuple(past_key_values)
    k, _ = kv[layer]
    seq_len = k.shape[2]
    n = min(n, seq_len)
    if n <= 0:
        return k[:, :, -1:, :].detach()
    last_n = k[:, :, -n:, :].mean(dim=2, keepdim=True).detach()
    return last_n


def all_keys_from_cache(past_key_values: Any, layer: int = -1) -> torch.Tensor:
    """(batch, num_heads, seq_len, head_dim)."""
    kv = _to_past_kvs_tuple(past_key_values)
    k, _ = kv[layer]
    return k.detach()


def _score_keys(
    query_key: torch.Tensor,
    keys: torch.Tensor,
    reduce_heads: str,
    use_cosine: bool,
) -> torch.Tensor:
    """Scores (1, seq_len). query_key (1, H, 1, D), keys (1, H, L, D)."""
    if use_cosine:
        # L2-normalize over head_dim so similarity is magnitude-invariant
        query_key = F.normalize(query_key, p=2, dim=-1)
        keys = F.normalize(keys, p=2, dim=-1)
    if reduce_heads == "mean":
        scores = (query_key * keys).sum(dim=-1).mean(dim=1)
    else:
        scores = (query_key * keys).sum(dim=(1, -1))
    return scores


def retrieve_top_k_positions(
    query_key: torch.Tensor,
    other_cache: Any,
    top_k: int,
    layer: int = -1,
    reduce_heads: str = "mean",
    use_cosine: bool = True,
    min_position: int = 0,
) -> torch.LongTensor:
    """
    query_key: (1, num_heads, 1, head_dim) — e.g. last (or mean of last n) key of agent A.
    other_cache: agent B's past_key_values.
    use_cosine: if True, L2-normalize before dot product (recommended).
    min_position: only score positions >= this (e.g. prompt length) so we retrieve from reasoning, not problem text.
    Returns indices (positions) in other_cache to keep, shape (top_k,) or fewer.
    """
    keys = all_keys_from_cache(_to_past_kvs_tuple(other_cache), layer)
    scores = _score_keys(query_key, keys, reduce_heads, use_cosine).squeeze(0)
    seq_len = scores.shape[0]
    # Restrict to [min_position, seq_len); if no range left, use full range
    if min_position >= seq_len:
        min_position = 0
    effective_len = seq_len - min_position
    k = min(top_k, effective_len)
    if k <= 0:
        return torch.arange(0, device=scores.device)
    scores_crop = scores[min_position:]
    _, rel_indices = torch.topk(scores_crop, k, dim=0)
    indices = (rel_indices.sort().values + min_position)
    return indices


def retrieve_best_contiguous_chunk(
    query_key: torch.Tensor,
    other_cache: Any,
    chunk_len: int,
    layer: int = -1,
    reduce_heads: str = "mean",
    use_cosine: bool = True,
    min_position: int = 0,
) -> torch.LongTensor:
    """
    Find the position with highest key similarity to query in the other cache (optionally
    only in positions >= min_position), then return a contiguous chunk of length chunk_len
    centered on that position.
    """
    keys = all_keys_from_cache(_to_past_kvs_tuple(other_cache), layer)
    scores = _score_keys(query_key, keys, reduce_heads, use_cosine).squeeze(0)
    seq_len = scores.shape[0]
    if min_position >= seq_len:
        min_position = 0
    effective_len = seq_len - min_position
    length = min(chunk_len, effective_len)
    if length <= 0:
        return torch.arange(0, device=scores.device, dtype=torch.long)
    scores_crop = scores[min_position:]
    best_rel = scores_crop.argmax().item()
    best_pos = min_position + best_rel
    start = max(min_position, min(best_pos - length // 2, seq_len - length))
    start = min(start, seq_len - length)
    return torch.arange(start, start + length, device=scores.device, dtype=torch.long)


def slice_cache_at_positions(
    past_key_values: Any,
    positions: torch.LongTensor,
) -> PastKVs:
    """Return new past_key_values containing only the given positions (along seq_len)."""
    kv = _to_past_kvs_tuple(past_key_values)
    result = []
    for (k, v) in kv:
        # k, v: (batch, num_heads, seq_len, head_dim)
        k_slice = k[:, :, positions, :].clone()
        v_slice = v[:, :, positions, :].clone()
        result.append((k_slice, v_slice))
    return tuple(result)


def stitch_caches(
    retrieved: PastKVs,
    own: PastKVs,
) -> PastKVs:
    """
    Prepend retrieved KV (from other agent) in front of own KV.
    So the model will "see" [retrieved; own] in that order.
    Does not re-encode positions; use stitch_and_reencode_caches for correct RoPE.
    """
    stitched = []
    for (rk, rv), (ok, ov) in zip(retrieved, own):
        new_k = torch.cat([rk, ok], dim=2)
        new_v = torch.cat([rv, ov], dim=2)
        stitched.append((new_k, new_v))
    return tuple(stitched)


def stitch_and_reencode_caches(
    first: PastKVs,
    second: PastKVs,
    first_original_positions: torch.Tensor,
    config: Any,
    device: torch.device,
) -> PastKVs:
    """
    Stitch [first; second] and re-encode RoPE so every position uses its global
    index (0, 1, ..., len(first)+len(second)-1) instead of its original position.
    first_original_positions: (len(first),) — position each first-block token had in its source run.
    Second block is assumed to have had positions 0, 1, ..., len(second)-1.
    """
    stitched = stitch_caches(first, second)
    seq_len = stitched[0][0].shape[2]
    len_second = second[0][0].shape[2]
    len_first = seq_len - len_second
    if first_original_positions.device != device:
        first_original_positions = first_original_positions.to(device)
    original_position_ids = torch.cat(
        [
            first_original_positions,
            torch.arange(len_second, device=device, dtype=first_original_positions.dtype),
        ],
        dim=0,
    )
    return reencode_past_kvs_for_stitching(stitched, original_position_ids, config, device)


def _cache_tuple_to_dynamic(cache_tuple: PastKVs, config: Any) -> Any:
    """Convert (key, value) tuple to DynamicCache so model forward accepts it."""
    from transformers.cache_utils import DynamicCache
    return DynamicCache(ddp_cache_data=cache_tuple, config=config)


def continue_with_cache(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    past_key_values: PastKVs,
    position_offset: int,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    return_cache: bool = False,
):
    """
    Run generation using existing past_key_values. prompt_ids here are the *new* tokens
    to continue with (e.g. a short string like " Let me refine my reasoning: ").
    position_offset: length of the sequence already in past_key_values (so we set position_ids correctly).
    If return_cache is True, returns (generated, final_past_kvs_tuple); else returns generated only.
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = prompt_ids.shape[0]
    assert batch_size == 1
    # Model expects DynamicCache, not tuple
    cache_for_model = _cache_tuple_to_dynamic(past_key_values, getattr(model, "config", model.model.config))
    # First forward with the continuation prompt to extend the cache.
    # attention_mask must span the FULL sequence (cache + new tokens) so the model attends to the cache.
    # See HuggingFace: (batch_size, number_of_seen_tokens + q_length).
    position_ids = torch.arange(
        position_offset,
        position_offset + prompt_ids.shape[1],
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    full_seq_len = position_offset + prompt_ids.shape[1]
    attention_mask = torch.ones((batch_size, full_seq_len), dtype=torch.long, device=device)
    outputs = model(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache_for_model,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    del cache_for_model  # free duplicate of past_key_values so we don't hold 2x cache
    generated = prompt_ids.clone()
    cur_pos = position_offset + prompt_ids.shape[1]
    # Full sequence length so causal mask includes cache
    attention_mask = torch.ones(
        (batch_size, position_offset + generated.shape[1]), dtype=torch.long, device=device
    )

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            next_token_logits = outputs.logits[:, -1, :]
            next_token = _sample_next_token(
                next_token_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.ones(
                (batch_size, position_offset + generated.shape[1]), dtype=torch.long, device=device
            )
            position_ids = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            cur_pos += 1
            if next_token.item() == tokenizer.eos_token_id:
                break

    if return_cache:
        final_tuple = _to_past_kvs_tuple(past_key_values)
        return generated, final_tuple
    return generated


def two_agent_kv_rag_round(
    model: Any,
    tokenizer: Any,
    question: str,
    device: torch.device,
    max_new_tokens_round1: int = 256,
    max_new_tokens_round2: int = 192,
    top_k: int = 24,
    use_contiguous_chunk: bool = True,
    query_last_n: int = 8,
    use_cosine: bool = True,
    retrieve_from_generated_only: bool = True,
    prompt_agent_a: str = (
        "You are agent A (precise calculator). Solve with equations, "
        "check each arithmetic step, and end with one numeric answer."
    ),
    prompt_agent_b: str = (
        "You are agent B (independent strategist). Solve with a different approach, "
        "estimate intermediate values for sanity checks, and end with one numeric answer."
    ),
    system_prompt: str = "You are a precise reasoner. Solve the following problem.",
    agent_a_do_sample: bool = False,
    agent_b_do_sample: bool = True,
    agent_a_temperature: float = 0.2,
    agent_b_temperature: float = 0.8,
    agent_a_top_p: float = 0.9,
    agent_b_top_p: float = 0.95,
    round2_do_sample: bool = False,
) -> Tuple[str, str, int]:
    """
    One round of KV-cache RAG: A and B each do CoT; then each gets top-k from the other's cache,
    stitches, and does a short second round. Returns (text_a, text_b, total_tokens).
    We use the same model for both; differentiation is only by prompt.
    """
    def make_prompt(agent_label: str, q: str) -> str:
        return f"{system_prompt}\n\n{agent_label}\n\nProblem: {q}\n\nReasoning:"

    # Round 1
    prompt_a = make_prompt(prompt_agent_a, question)
    prompt_b = make_prompt(prompt_agent_b, question)
    ids_a = tokenizer(prompt_a, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(device)
    ids_b = tokenizer(prompt_b, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(device)
    prompt_len_a = ids_a.shape[1]
    prompt_len_b = ids_b.shape[1]

    generated_a, cache_a_raw = get_full_kv_cache(
        model,
        tokenizer,
        ids_a,
        device,
        max_new_tokens=max_new_tokens_round1,
        do_sample=agent_a_do_sample,
        temperature=agent_a_temperature,
        top_p=agent_a_top_p,
    )
    generated_b, cache_b_raw = get_full_kv_cache(
        model,
        tokenizer,
        ids_b,
        device,
        max_new_tokens=max_new_tokens_round1,
        do_sample=agent_b_do_sample,
        temperature=agent_b_temperature,
        top_p=agent_b_top_p,
    )
    cache_a = _to_past_kvs_tuple(cache_a_raw)
    cache_b = _to_past_kvs_tuple(cache_b_raw)
    del cache_a_raw, cache_b_raw  # free original DynamicCaches; we only need tuple copies

    total_tokens = generated_a.shape[1] + generated_b.shape[1]

    # Query: mean of last N keys (more stable than single position); fallback to last key if N=1
    if query_last_n <= 1:
        qa_key = last_key_from_cache(cache_a, layer=-1)
        qb_key = last_key_from_cache(cache_b, layer=-1)
    else:
        qa_key = last_n_keys_from_cache(cache_a, n=query_last_n, layer=-1)
        qb_key = last_n_keys_from_cache(cache_b, n=query_last_n, layer=-1)

    # Only search in the other agent's *generated* region (skip prompt) so we retrieve reasoning, not problem text
    min_from_b = prompt_len_b if retrieve_from_generated_only else 0
    min_from_a = prompt_len_a if retrieve_from_generated_only else 0

    if use_contiguous_chunk:
        pos_from_b = retrieve_best_contiguous_chunk(
            qa_key, cache_b, chunk_len=top_k, use_cosine=use_cosine, min_position=min_from_b
        )
        pos_from_a = retrieve_best_contiguous_chunk(
            qb_key, cache_a, chunk_len=top_k, use_cosine=use_cosine, min_position=min_from_a
        )
    else:
        pos_from_b = retrieve_top_k_positions(
            qa_key, cache_b, top_k=top_k, use_cosine=use_cosine, min_position=min_from_b
        )
        pos_from_a = retrieve_top_k_positions(
            qb_key, cache_a, top_k=top_k, use_cosine=use_cosine, min_position=min_from_a
        )
    retrieved_from_b = slice_cache_at_positions(cache_b, pos_from_b)
    retrieved_from_a = slice_cache_at_positions(cache_a, pos_from_a)

    # Stitch and re-encode RoPE so positions match global [retrieved; own] indices
    config = getattr(model, "config", model.model.config)
    stitch_a = stitch_and_reencode_caches(
        retrieved_from_b, cache_a, pos_from_b, config, device
    )
    stitch_b = stitch_and_reencode_caches(
        retrieved_from_a, cache_b, pos_from_a, config, device
    )
    len_stitch_a = retrieved_from_b[0][0].shape[2] + cache_a[0][0].shape[2]
    len_stitch_b = retrieved_from_a[0][0].shape[2] + cache_b[0][0].shape[2]
    # Free large tensors before round 2 to reduce GPU memory
    del cache_a, cache_b, retrieved_from_a, retrieved_from_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Round 2: short continuation so model "refines" after seeing other's latent
    cont_prompt_a = " Refining (double-check calculations and signs): "
    cont_prompt_b = " Refining (use a different path and cross-check): "
    cont_ids_a = tokenizer(cont_prompt_a, return_tensors="pt").input_ids.to(device)
    cont_ids_b = tokenizer(cont_prompt_b, return_tensors="pt").input_ids.to(device)
    if cont_ids_a.shape[1] == 0:
        cont_ids_a = tokenizer("Refining:", return_tensors="pt").input_ids.to(device)
    if cont_ids_b.shape[1] == 0:
        cont_ids_b = tokenizer("Refining:", return_tensors="pt").input_ids.to(device)

    out_a = continue_with_cache(
        model,
        tokenizer,
        cont_ids_a,
        stitch_a,
        position_offset=len_stitch_a,
        max_new_tokens=max_new_tokens_round2,
        do_sample=round2_do_sample,
        temperature=agent_a_temperature,
        top_p=agent_a_top_p,
    )
    del stitch_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out_b = continue_with_cache(
        model,
        tokenizer,
        cont_ids_b,
        stitch_b,
        position_offset=len_stitch_b,
        max_new_tokens=max_new_tokens_round2,
        do_sample=round2_do_sample,
        temperature=agent_b_temperature,
        top_p=agent_b_top_p,
    )
    total_tokens += out_a.shape[1] + out_b.shape[1]

    # Full response = round1 + round2 so answer can appear in either
    round1_a = tokenizer.decode(generated_a[0], skip_special_tokens=True)
    round1_b = tokenizer.decode(generated_b[0], skip_special_tokens=True)
    round2_a = tokenizer.decode(out_a[0], skip_special_tokens=True)
    round2_b = tokenizer.decode(out_b[0], skip_special_tokens=True)
    text_a = round1_a + " " + round2_a
    text_b = round1_b + " " + round2_b
    return text_a, text_b, total_tokens


def two_agent_kv_full_stitch_round(
    model: Any,
    tokenizer: Any,
    question: str,
    device: torch.device,
    max_new_tokens_round1: int = 256,
    max_new_tokens_round2: int = 192,
    prompt_agent_a: str = "You are agent A. Think step by step and give your final answer.",
    prompt_agent_b: str = "You are agent B. Think step by step and give your final answer.",
    system_prompt: str = "You are a precise reasoner. Solve the following problem.",
) -> Tuple[str, str, int]:
    """
    Simple \"latent collaboration\" baseline:
    - Round 1: Agent A and B each do CoT independently, producing full KV caches.
    - Stitch: A gets [full cache of B; full cache of A], B gets [full cache of A; full cache of B]
      (no retrieval/top-k, we just share everything).
    - Round 2: short continuation from stitched caches with a \"Refining:\" prompt.

    Returns (text_a, text_b, total_tokens) where text_* = round1 + round2.
    """

    def make_prompt(agent_label: str, q: str) -> str:
        return f"{system_prompt}\n\n{agent_label}\n\nProblem: {q}\n\nReasoning:"

    # Round 1: independent CoT for A and B
    prompt_a = make_prompt(prompt_agent_a, question)
    prompt_b = make_prompt(prompt_agent_b, question)
    ids_a = tokenizer(prompt_a, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(device)
    ids_b = tokenizer(prompt_b, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(device)

    generated_a, cache_a_raw = get_full_kv_cache(
        model, tokenizer, ids_a, device, max_new_tokens=max_new_tokens_round1
    )
    generated_b, cache_b_raw = get_full_kv_cache(
        model, tokenizer, ids_b, device, max_new_tokens=max_new_tokens_round1
    )
    cache_a = _to_past_kvs_tuple(cache_a_raw)
    cache_b = _to_past_kvs_tuple(cache_b_raw)
    del cache_a_raw, cache_b_raw

    total_tokens = generated_a.shape[1] + generated_b.shape[1]

    # Full-stitch: A sees full B then full A; B sees full A then full B. Re-encode RoPE for global positions.
    config = getattr(model, "config", model.model.config)
    len_b = cache_b[0][0].shape[2]
    len_a = cache_a[0][0].shape[2]
    device = next(model.parameters()).device
    stitch_a = stitch_and_reencode_caches(
        cache_b, cache_a,
        torch.arange(len_b, device=device, dtype=torch.long),
        config, device,
    )
    stitch_b = stitch_and_reencode_caches(
        cache_a, cache_b,
        torch.arange(len_a, device=device, dtype=torch.long),
        config, device,
    )
    len_stitch_a = len_b + len_a
    len_stitch_b = len_a + len_b

    # Free originals to save memory
    del cache_a, cache_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Round 2: short continuation so model "refines" after seeing full latent of the other agent
    cont_prompt = " Refining: "
    cont_ids = tokenizer(cont_prompt, return_tensors="pt").input_ids.to(device)
    if cont_ids.shape[1] == 0:
        cont_ids = tokenizer("Refining:", return_tensors="pt").input_ids.to(device)

    out_a = continue_with_cache(
        model, tokenizer, cont_ids, stitch_a, position_offset=len_stitch_a, max_new_tokens=max_new_tokens_round2
    )
    del stitch_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_b = continue_with_cache(
        model, tokenizer, cont_ids, stitch_b, position_offset=len_stitch_b, max_new_tokens=max_new_tokens_round2
    )
    total_tokens += out_a.shape[1] + out_b.shape[1]

    # Decode full responses (round1 + round2) so evaluation can find the answer in either
    round1_a = tokenizer.decode(generated_a[0], skip_special_tokens=True)
    round1_b = tokenizer.decode(generated_b[0], skip_special_tokens=True)
    round2_a = tokenizer.decode(out_a[0], skip_special_tokens=True)
    round2_b = tokenizer.decode(out_b[0], skip_special_tokens=True)
    text_a = round1_a + " " + round2_a
    text_b = round1_b + " " + round2_b
    return text_a, text_b, total_tokens


# --- 5 heterogeneous agents, 3 rounds, full-stitch from best agent each round ---

def _default_heterogeneous_prompts() -> List[str]:
    """Five distinct agent roles for diverse reasoning."""
    return [
        "You are agent 1. You are methodical. Work step by step and show every calculation. Give your final answer.",
        "You are agent 2. You focus on key quantities first, then fill in details. Be concise but correct. Give your final answer.",
        "You are agent 3. You double-check each step and look for alternative approaches. Give your final answer.",
        "You are agent 4. You prefer setting up equations and solving them. Show your work. Give your final answer.",
        "You are agent 5. You reason in plain language first, then compute. Give your final answer.",
    ]


def _pick_best_agent_by_agreement(texts: List[str]) -> int:
    """
    Among agents that produced an answer, pick the one whose answer agrees with the most others.
    Ties: prefer lower index. Returns 0 if no one has a parseable answer.
    """
    answers = [normalize_answer(extract_answer_gsm8k(t)) for t in texts]
    n = len(answers)
    best_idx = 0
    best_count = -1
    for i in range(n):
        if not answers[i]:
            continue
        count = sum(1 for j in range(n) if answers[j] and answers[i] == answers[j])
        if count > best_count:
            best_count = count
            best_idx = i
    return best_idx


def five_agent_three_round_full_stitch(
    model: Any,
    tokenizer: Any,
    question: str,
    device: torch.device,
    num_agents: int = 5,
    num_rounds: int = 3,
    max_new_tokens_round1: int = 256,
    max_new_tokens_refine: int = 128,
    system_prompt: str = "You are a precise reasoner. Solve the following problem.",
    agent_prompts: Optional[List[str]] = None,
) -> Tuple[List[str], int]:
    """
    Five heterogeneous agents, 3 rounds. Each round after the first: each agent gets a full-stitch
    KV cache from the best agent of the previous round ( [best_cache; own_cache] ), then refines.
    Best agent = whose answer agrees with the most others (by extracted numeric answer).
    Returns (list of 5 final response texts, total_tokens).
    """
    if agent_prompts is None:
        agent_prompts = _default_heterogeneous_prompts()
    agent_prompts = agent_prompts[:num_agents]
    assert len(agent_prompts) >= num_agents, "Need at least num_agents prompts"

    config = getattr(model, "config", model.model.config)
    cont_prompt = " Refining: "
    cont_ids = tokenizer(cont_prompt, return_tensors="pt").input_ids.to(device)
    if cont_ids.shape[1] == 0:
        cont_ids = tokenizer("Refining:", return_tensors="pt").input_ids.to(device)

    total_tokens = 0
    # Round 1: each agent does independent CoT
    generated_list: List[torch.LongTensor] = []
    cache_list: List[PastKVs] = []
    for i in range(num_agents):
        prompt = f"{system_prompt}\n\n{agent_prompts[i]}\n\nProblem: {question}\n\nReasoning:"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(device)
        generated, cache_raw = get_full_kv_cache(
            model, tokenizer, ids, device, max_new_tokens=max_new_tokens_round1
        )
        generated_list.append(generated)
        cache_list.append(_to_past_kvs_tuple(cache_raw))
        total_tokens += generated.shape[1]
    texts_so_far = [tokenizer.decode(g[0], skip_special_tokens=True) for g in generated_list]
    best_idx = _pick_best_agent_by_agreement(texts_so_far)

    for round_no in range(2, num_rounds + 1):
        best_cache = cache_list[best_idx]
        len_best = best_cache[0][0].shape[2]
        new_generated_list: List[torch.LongTensor] = []
        new_cache_list: List[PastKVs] = []
        for i in range(num_agents):
            own_cache = cache_list[i]
            stitch = stitch_and_reencode_caches(
                best_cache,
                own_cache,
                torch.arange(len_best, device=device, dtype=torch.long),
                config,
                device,
            )
            len_stitch = len_best + own_cache[0][0].shape[2]
            out, new_cache = continue_with_cache(
                model,
                tokenizer,
                cont_ids,
                stitch,
                position_offset=len_stitch,
                max_new_tokens=max_new_tokens_refine,
                return_cache=True,
            )
            total_tokens += out.shape[1]
            new_generated_list.append(out)
            new_cache_list.append(new_cache)
            texts_so_far[i] = texts_so_far[i] + " " + tokenizer.decode(out[0], skip_special_tokens=True)
        generated_list = new_generated_list
        cache_list = new_cache_list
        best_idx = _pick_best_agent_by_agreement(texts_so_far)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return texts_so_far, total_tokens
