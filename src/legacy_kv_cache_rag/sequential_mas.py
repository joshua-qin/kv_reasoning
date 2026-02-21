"""
Sequential MAS: one round with planner → critic → refiner → solver.
Each agent sees the previous agent's output (text or KV cache).
"""

from __future__ import annotations

from typing import Any, Tuple

import torch

from .kv_cache_rag import (
    get_full_kv_cache,
    continue_with_cache,
    _to_past_kvs_tuple,
)


# --- Prompt building (shared by text and KV) ---

DEFAULT_SYSTEM = "You are a precise reasoner. Solve the following problem and give your final answer."


def _planner_prompt(question: str, system: str = DEFAULT_SYSTEM) -> str:
    return (
        f"{system}\n\n"
        "You are the **planner**. Given the problem below, produce a step-by-step plan: "
        "identify key quantities, equations, and the order of operations. "
        "Do not give the final numeric answer yet.\n\n"
        f"Problem: {question}\n\n"
        "Plan:"
    )


def _critic_prompt(question: str, planner_text: str, system: str = DEFAULT_SYSTEM) -> str:
    return (
        f"{system}\n\n"
        "You are the **critic**. Review the plan below. "
        "List any arithmetic errors, unclear steps, or missing considerations. Be concise.\n\n"
        f"Problem: {question}\n\n"
        f"Plan:\n{planner_text}\n\n"
        "Critique:"
    )


def _refiner_prompt(
    question: str, planner_text: str, critic_text: str, system: str = DEFAULT_SYSTEM
) -> str:
    return (
        f"{system}\n\n"
        "You are the **refiner**. Using the plan and the critique, produce a refined step-by-step solution. "
        "Fix any noted issues. Do not give the final numeric answer yet.\n\n"
        f"Problem: {question}\n\n"
        f"Plan:\n{planner_text}\n\n"
        f"Critique:\n{critic_text}\n\n"
        "Refined solution:"
    )


def _solver_prompt(
    question: str,
    planner_text: str,
    critic_text: str,
    refiner_text: str,
    system: str = DEFAULT_SYSTEM,
) -> str:
    return (
        f"{system}\n\n"
        "You are the **solver**. Given the refined solution below, "
        "perform the final calculation and give the answer in the format #### <number>.\n\n"
        f"Problem: {question}\n\n"
        f"Plan:\n{planner_text}\n\n"
        f"Critique:\n{critic_text}\n\n"
        f"Refined solution:\n{refiner_text}\n\n"
        "Final answer (format: #### <number>):"
    )


# --- Continuation prompts for KV (no full text; model attends to cache) ---

CONT_CRITIC = (
    "\n\nYou are the **critic**. Review the reasoning above. "
    "List any arithmetic errors, unclear steps, or improvements. Be concise.\n\nCritique:"
)

CONT_REFINER = (
    "\n\nYou are the **refiner**. Using the reasoning and critique above, "
    "produce a refined step-by-step solution. Fix any noted issues. Do not give the final number yet.\n\nRefined solution:"
)

CONT_SOLVER = (
    "\n\nYou are the **solver**. Give the final numeric answer in the format #### <number>.\n\nFinal answer:"
)


# ========== Text MAS (sequential) ==========


def sequential_text_mas(
    model: Any,
    tokenizer: Any,
    question: str,
    device: torch.device,
    max_new_tokens_planner: int = 192,
    max_new_tokens_critic: int = 128,
    max_new_tokens_refiner: int = 192,
    max_new_tokens_solver: int = 128,
    system_prompt: str = DEFAULT_SYSTEM,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_length_planner: int = 2048,
    max_length_downstream: int = 4096,
) -> Tuple[str, str, str, str, int]:
    """
    One sequential round: planner → critic → refiner → solver.
    Each agent sees the previous agent(s) as text.
    Returns (text_planner, text_critic, text_refiner, text_solver, total_tokens).
    Final answer should be extracted from text_solver.
    """
    total_tokens = 0

    # Planner
    prompt_planner = _planner_prompt(question, system_prompt)
    ids_planner = tokenizer(
        prompt_planner,
        return_tensors="pt",
        truncation=True,
        max_length=max_length_planner,
    ).input_ids.to(device)
    out_planner, _ = get_full_kv_cache(
        model,
        tokenizer,
        ids_planner,
        device,
        max_new_tokens=max_new_tokens_planner,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    text_planner = tokenizer.decode(out_planner[0], skip_special_tokens=True)
    total_tokens += out_planner.shape[1]

    # Critic (sees planner text)
    prompt_critic = _critic_prompt(question, text_planner, system_prompt)
    ids_critic = tokenizer(
        prompt_critic,
        return_tensors="pt",
        truncation=True,
        max_length=max_length_downstream,
    ).input_ids.to(device)
    out_critic, _ = get_full_kv_cache(
        model,
        tokenizer,
        ids_critic,
        device,
        max_new_tokens=max_new_tokens_critic,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    text_critic = tokenizer.decode(out_critic[0], skip_special_tokens=True)
    total_tokens += out_critic.shape[1]

    # Refiner (sees planner + critic text)
    prompt_refiner = _refiner_prompt(question, text_planner, text_critic, system_prompt)
    ids_refiner = tokenizer(
        prompt_refiner,
        return_tensors="pt",
        truncation=True,
        max_length=max_length_downstream,
    ).input_ids.to(device)
    out_refiner, _ = get_full_kv_cache(
        model,
        tokenizer,
        ids_refiner,
        device,
        max_new_tokens=max_new_tokens_refiner,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    text_refiner = tokenizer.decode(out_refiner[0], skip_special_tokens=True)
    total_tokens += out_refiner.shape[1]

    # Solver (sees planner + critic + refiner text)
    prompt_solver = _solver_prompt(
        question, text_planner, text_critic, text_refiner, system_prompt
    )
    ids_solver = tokenizer(
        prompt_solver,
        return_tensors="pt",
        truncation=True,
        max_length=max_length_downstream,
    ).input_ids.to(device)
    out_solver, _ = get_full_kv_cache(
        model,
        tokenizer,
        ids_solver,
        device,
        max_new_tokens=max_new_tokens_solver,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    text_solver = tokenizer.decode(out_solver[0], skip_special_tokens=True)
    total_tokens += out_solver.shape[1]

    return text_planner, text_critic, text_refiner, text_solver, total_tokens


# ========== KV cache RAG (sequential) ==========


def sequential_kv_mas(
    model: Any,
    tokenizer: Any,
    question: str,
    device: torch.device,
    max_new_tokens_planner: int = 192,
    max_new_tokens_critic: int = 128,
    max_new_tokens_refiner: int = 192,
    max_new_tokens_solver: int = 128,
    system_prompt: str = DEFAULT_SYSTEM,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_length_planner: int = 2048,
) -> Tuple[str, str, str, str, int]:
    """
    One sequential round: planner → critic → refiner → solver.
    Each agent sees the previous agent via KV cache (continues from previous cache + short role prompt).
    Returns (text_planner, text_critic, text_refiner, text_solver, total_tokens).
    Final answer should be extracted from text_solver.
    """
    total_tokens = 0

    # Planner: full prompt, get cache
    prompt_planner = _planner_prompt(question, system_prompt)
    ids_planner = tokenizer(
        prompt_planner,
        return_tensors="pt",
        truncation=True,
        max_length=max_length_planner,
    ).input_ids.to(device)
    generated_planner, cache_planner_raw = get_full_kv_cache(
        model,
        tokenizer,
        ids_planner,
        device,
        max_new_tokens=max_new_tokens_planner,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    text_planner = tokenizer.decode(generated_planner[0], skip_special_tokens=True)
    total_tokens += generated_planner.shape[1]
    cache_planner = _to_past_kvs_tuple(cache_planner_raw)
    del cache_planner_raw
    len_planner = generated_planner.shape[1]

    # Critic: continue from planner's cache
    cont_critic = tokenizer(
        CONT_CRITIC,
        return_tensors="pt",
    ).input_ids.to(device)
    if cont_critic.shape[1] == 0:
        cont_critic = tokenizer("Critic:", return_tensors="pt").input_ids.to(device)
    out_critic, cache_critic = continue_with_cache(
        model,
        tokenizer,
        cont_critic,
        cache_planner,
        position_offset=len_planner,
        max_new_tokens=max_new_tokens_critic,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        return_cache=True,
    )
    text_critic = tokenizer.decode(out_critic[0], skip_special_tokens=True)
    total_tokens += out_critic.shape[1]
    len_critic = len_planner + out_critic.shape[1]
    del cache_planner

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Refiner: continue from critic's cache
    cont_refiner = tokenizer(
        CONT_REFINER,
        return_tensors="pt",
    ).input_ids.to(device)
    if cont_refiner.shape[1] == 0:
        cont_refiner = tokenizer("Refiner:", return_tensors="pt").input_ids.to(device)
    out_refiner, cache_refiner = continue_with_cache(
        model,
        tokenizer,
        cont_refiner,
        cache_critic,
        position_offset=len_critic,
        max_new_tokens=max_new_tokens_refiner,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        return_cache=True,
    )
    text_refiner = tokenizer.decode(out_refiner[0], skip_special_tokens=True)
    total_tokens += out_refiner.shape[1]
    len_refiner = len_critic + out_refiner.shape[1]
    del cache_critic

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Solver: continue from refiner's cache (no need to return cache)
    cont_solver = tokenizer(
        CONT_SOLVER,
        return_tensors="pt",
    ).input_ids.to(device)
    if cont_solver.shape[1] == 0:
        cont_solver = tokenizer("Solver:", return_tensors="pt").input_ids.to(device)
    out_solver = continue_with_cache(
        model,
        tokenizer,
        cont_solver,
        cache_refiner,
        position_offset=len_refiner,
        max_new_tokens=max_new_tokens_solver,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        return_cache=False,
    )
    text_solver = tokenizer.decode(out_solver[0], skip_special_tokens=True)
    total_tokens += out_solver.shape[1]

    return text_planner, text_critic, text_refiner, text_solver, total_tokens
