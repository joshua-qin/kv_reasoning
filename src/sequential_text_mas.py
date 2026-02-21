"""
Sequential TextMAS: same 4-agent structure as LatentMAS but with text-mediated communication.
Planner -> Critic -> Refiner -> Judger; each agent decodes to text and the next sees that text.
Used as a direct text baseline to compare against LatentMAS (paper: "Sequential TextMAS").
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def _apply_chat_template(
    tokenizer: Any,
    system: str,
    user_content: str,
    add_generation_prompt: bool = True,
) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _planner_content(question: str) -> str:
    return f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""


def _critic_content(question: str, planner_text: str) -> str:
    return f"""Question: {question}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.

The Planner Agent's plan is provided below. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Planner Agent's plan:
{planner_text}

Now, output your response below:
"""


def _refiner_content(question: str, planner_text: str, critic_text: str) -> str:
    return f"""Question: {question}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) the Planner's plan and the Critic's feedback (below)
(2) the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Planner's plan:
{planner_text}

Critic's feedback:
{critic_text}

Now, output your refined plan below:
"""


def _judger_content(question: str, refiner_text: str) -> str:
    return f"""Target Question: {question}

You are a helpful assistant. You are provided with a refined plan (for reference) and the target question to solve.

Refined plan (for reference):
{refiner_text}

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""


def _decode_one_stage(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_prompt_length: int = 2048,
) -> Tuple[str, int]:
    """Run one agent: tokenize prompt, generate, return (decoded text, total new tokens)."""
    ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    ).input_ids.to(device)
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = ids.shape[1]
    new_ids = out[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, int(out.shape[1] - prompt_len)


def run_sequential_text_mas(
    model: Any,
    tokenizer: Any,
    question: str,
    device: torch.device,
    max_new_tokens_per_stage: int = 256,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_prompt_length: int = 2048,
    use_chat_template: bool = True,
) -> Tuple[str, int]:
    """
    Sequential TextMAS: Planner (text) -> Critic (text) -> Refiner (text) -> Judger (text).
    Same roles and instructions as LatentMAS but each stage produces text and the next reads it.
    Returns (final_answer_text, total_tokens_generated).
    """
    total_tokens = 0

    def prompt(user_content: str) -> str:
        if use_chat_template:
            return _apply_chat_template(tokenizer, SYSTEM_PROMPT, user_content, add_generation_prompt=True)
        return SYSTEM_PROMPT + "\n\n" + user_content + "\n\nResponse:"

    # Stage 1: Planner
    p_text, n1 = _decode_one_stage(
        model, tokenizer, prompt(_planner_content(question)), device,
        max_new_tokens=max_new_tokens_per_stage, do_sample=do_sample,
        temperature=temperature, top_p=top_p, max_prompt_length=max_prompt_length,
    )
    total_tokens += n1

    # Stage 2: Critic (sees Planner text)
    c_text, n2 = _decode_one_stage(
        model, tokenizer, prompt(_critic_content(question, p_text)), device,
        max_new_tokens=max_new_tokens_per_stage, do_sample=do_sample,
        temperature=temperature, top_p=top_p, max_prompt_length=max_prompt_length,
    )
    total_tokens += n2

    # Stage 3: Refiner (sees Planner + Critic text)
    r_text, n3 = _decode_one_stage(
        model, tokenizer, prompt(_refiner_content(question, p_text, c_text)), device,
        max_new_tokens=max_new_tokens_per_stage, do_sample=do_sample,
        temperature=temperature, top_p=top_p, max_prompt_length=max_prompt_length,
    )
    total_tokens += n3

    # Stage 4: Judger (sees Refiner text; produces final answer)
    j_text, n4 = _decode_one_stage(
        model, tokenizer, prompt(_judger_content(question, r_text)), device,
        max_new_tokens=max_new_tokens_per_stage, do_sample=do_sample,
        temperature=temperature, top_p=top_p, max_prompt_length=max_prompt_length,
    )
    total_tokens += n4

    return j_text, total_tokens
