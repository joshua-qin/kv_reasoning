"""
Baselines: single-agent CoT and two-agent text-only (vanilla debate).
"""

from __future__ import annotations

import torch
from typing import Tuple

from .kv_cache_rag import get_full_kv_cache


def single_agent_cot(
    model,
    tokenizer,
    question: str,
    device: torch.device,
    max_new_tokens: int = 512,
    system_prompt: str = "You are a precise reasoner. Solve the following problem and give your final answer.",
) -> Tuple[str, int]:
    """One model, one CoT. Returns (decoded_response, num_output_tokens). Paper-aligned: output tokens only."""
    prompt = f"{system_prompt}\n\nProblem: {question}\n\nReasoning:"
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(device)
    generated, _ = get_full_kv_cache(model, tokenizer, ids, device, max_new_tokens=max_new_tokens)
    new_ids = generated[0, ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    num_output_tokens = int(generated.shape[1] - ids.shape[1])
    return text, num_output_tokens


def two_agent_text_debate(
    model,
    tokenizer,
    question: str,
    device: torch.device,
    max_new_tokens_per_turn: int = 256,
    system_prompt: str = "You are a precise reasoner. Solve the following problem and give your final answer.",
) -> Tuple[str, str, int]:
    """
    Agent A reasons (text). Agent B sees A's full text and reasons. No KV sharing.
    Returns (response_a, response_b, total_tokens).
    """
    prompt_a = f"{system_prompt}\n\nYou are agent A. Think step by step.\n\nProblem: {question}\n\nReasoning:"
    ids_a = tokenizer(prompt_a, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(device)
    out_a, _ = get_full_kv_cache(
        model, tokenizer, ids_a, device, max_new_tokens=max_new_tokens_per_turn
    )
    text_a = tokenizer.decode(out_a[0, ids_a.shape[1]:], skip_special_tokens=True)

    # B sees question + A's response as text
    prompt_b = (
        f"{system_prompt}\n\nYou are agent B. First read agent A's reasoning below, then give your own reasoning and final answer.\n\n"
        f"Problem: {question}\n\nAgent A's reasoning:\n{text_a}\n\nYour reasoning:"
    )
    ids_b = tokenizer(prompt_b, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
    out_b, _ = get_full_kv_cache(
        model, tokenizer, ids_b, device, max_new_tokens=max_new_tokens_per_turn
    )
    text_b = tokenizer.decode(out_b[0, ids_b.shape[1]:], skip_special_tokens=True)
    # Paper-aligned: output tokens only (generated, not prompt)
    total_tokens = int((out_a.shape[1] - ids_a.shape[1]) + (out_b.shape[1] - ids_b.shape[1]))
    return text_a, text_b, total_tokens
