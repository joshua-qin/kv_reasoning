"""
GSM8K evaluation: load subset, extract numeric answer, compute accuracy.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


def extract_answer_gsm8k(text: str, prefer_last: bool = False) -> Optional[str]:
    """
    GSM8K answers are usually at the end. Common patterns:
    - "#### 1234" or "#### 12.34"
    - "the answer is 42" / "Answer: 42"
    - "$42" at end
    prefer_last: if False, use first #### (original); if True, use last (standard for single-turn CoT).
    """
    text = text.strip()
    matches = list(re.finditer(r"####\s*([-+]?\d+(?:\.\d+)?)", text))
    if matches:
        m = matches[-1] if prefer_last else matches[0]
        return m.group(1).strip()
    # "answer is X" (case insensitive)
    m = re.search(r"(?:answer|final answer)\s*[:\s]+\s*([-+]?\d+(?:\.\d+)?)", text, re.I)
    if m:
        return m.group(1).strip()
    # Last number in the last line or last few lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in reversed(lines[-5:]):
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
        if nums:
            return nums[-1]
    return None


def normalize_answer(a: Optional[str]) -> str:
    if a is None:
        return ""
    a = a.strip().replace(",", "").replace(" ", "")
    if "." in a:
        try:
            return str(float(a))
        except ValueError:
            return a
    try:
        return str(int(float(a)))
    except ValueError:
        return a


def is_correct(pred: Optional[str], gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def load_gsm8k(split: str = "test", max_samples: Optional[int] = None):
    """Load GSM8K from HuggingFace datasets (real eval set)."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def evaluate_gsm8k(
    items: List[Tuple[str, str]],
) -> Tuple[float, int]:
    """
    items: list of (prediction_text, gold_answer).
    Returns (accuracy, total).
    """
    correct = 0
    for pred_text, gold in items:
        pred_ans = extract_answer_gsm8k(pred_text)
        if is_correct(pred_ans, gold):
            correct += 1
    total = len(items)
    acc = correct / total if total else 0.0
    return acc, total
