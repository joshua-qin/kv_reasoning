"""
KV cache extraction, retrieval by key similarity, and stitching for second-round reasoning.
Training-free: two agents (two runs of same model) share selected parts of each other's KV cache.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any

# past_key_values: tuple of (key, value) per layer
# key, value: (batch, num_heads, seq_len, head_dim)
PastKVs = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


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
            if do_sample:
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
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


def all_keys_from_cache(past_key_values: Any, layer: int = -1) -> torch.Tensor:
    """(batch, num_heads, seq_len, head_dim)."""
    kv = _to_past_kvs_tuple(past_key_values)
    k, _ = kv[layer]
    return k.detach()


def retrieve_top_k_positions(
    query_key: torch.Tensor,
    other_cache: Any,
    top_k: int,
    layer: int = -1,
    reduce_heads: str = "mean",
) -> torch.LongTensor:
    """
    query_key: (1, num_heads, 1, head_dim) — e.g. last key of agent A.
    other_cache: agent B's past_key_values.
    Returns indices (positions) in other_cache to keep, shape (top_k,) or fewer if cache is shorter.
    """
    keys = all_keys_from_cache(_to_past_kvs_tuple(other_cache), layer)  # (1, num_heads, seq_len, head_dim)
    # Score each position: dot product of query with that position's key, then average over heads
    if reduce_heads == "mean":
        # (1, seq_len)
        scores = (query_key * keys).sum(dim=-1).mean(dim=1)
    else:
        scores = (query_key * keys).sum(dim=(1, -1))
    scores = scores.squeeze(0)
    seq_len = scores.shape[0]
    k = min(top_k, seq_len)
    if k <= 0:
        return torch.arange(0, device=scores.device)
    _, indices = torch.topk(scores, k, dim=0)
    return indices.sort().values


def retrieve_best_contiguous_chunk(
    query_key: torch.Tensor,
    other_cache: Any,
    chunk_len: int,
    layer: int = -1,
    reduce_heads: str = "mean",
) -> torch.LongTensor:
    """
    Find the position with highest key similarity to query, then return a contiguous
    chunk of length chunk_len centered on that position. Better for first experiments:
    model sees coherent context instead of sparse reordered positions.
    """
    keys = all_keys_from_cache(_to_past_kvs_tuple(other_cache), layer)
    if reduce_heads == "mean":
        scores = (query_key * keys).sum(dim=-1).mean(dim=1).squeeze(0)
    else:
        scores = (query_key * keys).sum(dim=(1, -1)).squeeze(0)
    seq_len = scores.shape[0]
    length = min(chunk_len, seq_len)
    if length <= 0:
        return torch.arange(0, device=scores.device, dtype=torch.long)
    best_pos = scores.argmax().item()
    start = max(0, min(best_pos - length // 2, seq_len - length))
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
    """
    stitched = []
    for (rk, rv), (ok, ov) in zip(retrieved, own):
        new_k = torch.cat([rk, ok], dim=2)
        new_v = torch.cat([rv, ov], dim=2)
        stitched.append((new_k, new_v))
    return tuple(stitched)


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
) -> torch.LongTensor:
    """
    Run generation using existing past_key_values. prompt_ids here are the *new* tokens
    to continue with (e.g. a short string like " Let me refine my reasoning: ").
    position_offset: length of the sequence already in past_key_values (so we set position_ids correctly).
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = prompt_ids.shape[0]
    assert batch_size == 1
    # Model expects DynamicCache, not tuple
    cache_for_model = _cache_tuple_to_dynamic(past_key_values, getattr(model, "config", model.model.config))
    # First forward with the continuation prompt to extend the cache
    position_ids = torch.arange(
        position_offset,
        position_offset + prompt_ids.shape[1],
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)
    # If we have multiple tokens in prompt_ids, we need to run forward once to consume them
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
    attention_mask = torch.ones(
        (batch_size, generated.shape[1]), dtype=torch.long, device=device
    )

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            next_token_logits = outputs.logits[:, -1, :]
            if do_sample:
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=torch.long, device=device),
                ],
                dim=1,
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
    prompt_agent_a: str = "You are agent A. Think step by step and give your final answer.",
    prompt_agent_b: str = "You are agent B. Think step by step and give your final answer.",
    system_prompt: str = "You are a precise reasoner. Solve the following problem.",
) -> Tuple[str, str, int]:
    """
    One round of KV-cache RAG: A and B each do CoT; then each gets top-k from the other's cache,
    stitches, and does a short second round. Returns (answer_a_round2, answer_b_round2, total_tokens).
    We use the same model for both; differentiation is only by prompt.
    """
    def make_prompt(agent_label: str, q: str) -> str:
        return f"{system_prompt}\n\n{agent_label}\n\nProblem: {q}\n\nReasoning:"

    # Round 1
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
    del cache_a_raw, cache_b_raw  # free original DynamicCaches; we only need tuple copies

    total_tokens = generated_a.shape[1] + generated_b.shape[1]

    # Retrieve: A's last key -> top-k from B; B's last key -> top-k from A
    # Contiguous chunk (default) = coherent context; sparse positions = targeted but reordered
    qa_key = last_key_from_cache(cache_a, layer=-1)
    qb_key = last_key_from_cache(cache_b)
    if use_contiguous_chunk:
        pos_from_b = retrieve_best_contiguous_chunk(qa_key, cache_b, chunk_len=top_k)
        pos_from_a = retrieve_best_contiguous_chunk(qb_key, cache_a, chunk_len=top_k)
    else:
        pos_from_b = retrieve_top_k_positions(qa_key, cache_b, top_k=top_k)
        pos_from_a = retrieve_top_k_positions(qb_key, cache_a, top_k=top_k)
    retrieved_from_b = slice_cache_at_positions(cache_b, pos_from_b)
    retrieved_from_a = slice_cache_at_positions(cache_a, pos_from_a)

    # Stitch: A gets [retrieved_from_b; cache_a], B gets [retrieved_from_a; cache_b]
    stitch_a = stitch_caches(retrieved_from_b, cache_a)
    stitch_b = stitch_caches(retrieved_from_a, cache_b)
    len_stitch_a = retrieved_from_b[0][0].shape[2] + cache_a[0][0].shape[2]
    len_stitch_b = retrieved_from_a[0][0].shape[2] + cache_b[0][0].shape[2]
    # Free large tensors before round 2 to reduce GPU memory
    del cache_a, cache_b, retrieved_from_a, retrieved_from_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Round 2: short continuation so model "refines" after seeing other's latent
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

    # Full response = round1 + round2 so answer can appear in either
    round1_a = tokenizer.decode(generated_a[0], skip_special_tokens=True)
    round1_b = tokenizer.decode(generated_b[0], skip_special_tokens=True)
    round2_a = tokenizer.decode(out_a[0], skip_special_tokens=True)
    round2_b = tokenizer.decode(out_b[0], skip_special_tokens=True)
    text_a = round1_a + " " + round2_a
    text_b = round1_b + " " + round2_b
    return text_a, text_b, total_tokens
