"""
GSM8K evaluation: load subset, extract numeric answer, compute accuracy.

Aligned with Gen-Verse/LatentMAS (paper arXiv:2511.20639) for comparable accuracy:
- Extraction: #### and \\boxed{} (last match); paper uses \\boxed{} first then last number.
- Normalization: use normalize_answer_paper (strip + lower only) for paper parity;
  normalize_answer is more lenient (removes commas, normalizes 42.0 -> 42).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


def _remove_commas_in_numbers(text: str) -> str:
    """Turn "130,000" into "130000" so number regex sees one value (avoids extracting "000")."""
    while True:
        new_text = re.sub(r"(\d),(\d)", r"\1\2", text)
        if new_text == text:
            break
        text = new_text
    return text


def extract_answer_gsm8k(text: str, prefer_last: bool = False) -> Optional[str]:
    """
    GSM8K answers are usually at the end. Common patterns:
    - "#### 1234" or "#### 12.34"
    - "\\boxed{42}" or "\\boxed{12.34}" (LatentMAS paper format)
    - "the answer is 42" / "Answer: 42"
    prefer_last: if False, use first #### (original); if True, use last (standard for single-turn CoT).
    """
    text = text.strip()
    text = _remove_commas_in_numbers(text)
    num_pat = r"[-+]?\d+(?:\.\d+)?"

    # #### (explicit answer marker)
    matches = list(re.finditer(r"####\s*(" + num_pat + r")", text))
    if matches:
        m = matches[-1] if prefer_last else matches[0]
        return m.group(1).strip()
    # \boxed{...}
    boxed_matches = list(re.finditer(r"\\boxed\{([^}]+)\}", text))
    if boxed_matches:
        content = boxed_matches[-1].group(1).strip()
        content = _remove_commas_in_numbers(content)
        num = re.search(num_pat, content)
        return num.group(0) if num else content
    # "answer is 42" / "Final answer: 42" — use LAST occurrence so we get the final answer
    answer_matches = list(
        re.finditer(r"(?:answer|final answer)\s*(?:is|:)\s*(" + num_pat + r")", text, re.I)
    )
    if answer_matches:
        return answer_matches[-1].group(1).strip()
    # Fallback: last number in the tail of the response (conclusion), not middle of reasoning
    tail_len = 1200
    tail = text[-tail_len:] if len(text) > tail_len else text
    numbers = re.findall(num_pat, tail)
    if numbers:
        return numbers[-1]
    numbers = re.findall(num_pat, text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer_paper(a: Optional[str]) -> str:
    """
    Normalization used by Gen-Verse/LatentMAS (utils.normalize_answer): strip + lower only.
    Use this for accuracy comparable to the paper. "42.0" and "42" are different.
    """
    if a is None:
        return ""
    return a.strip().lower()


def normalize_answer(a: Optional[str]) -> str:
    """
    Lenient normalization: strip, remove commas/spaces, normalize numeric (42.0 -> 42).
    For paper-comparable accuracy use normalize_answer_paper instead.
    """
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


def is_correct(
    pred: Optional[str],
    gold: str,
    use_paper_normalization: bool = True,
) -> bool:
    """
    Correct iff normalized pred equals normalized gold.
    use_paper_normalization=True (default): strip+lower only, matches LatentMAS paper/GPU.
    use_paper_normalization=False: lenient (commas, 42.0==42).
    """
    if use_paper_normalization:
        return normalize_answer_paper(pred) == normalize_answer_paper(gold)
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
    prefer_last: bool = False,
    use_paper_normalization: bool = True,
) -> Tuple[float, int]:
    """
    items: list of (prediction_text, gold_answer).
    prefer_last: pass to extract_answer_gsm8k (use last #### match, e.g. for LatentMAS decode).
    use_paper_normalization: if True (default), use strip+lower only for paper-comparable accuracy.
    Returns (accuracy, total).
    """
    correct = 0
    for pred_text, gold in items:
        pred_ans = extract_answer_gsm8k(pred_text, prefer_last=prefer_last)
        if is_correct(pred_ans, gold, use_paper_normalization=use_paper_normalization):
            correct += 1
    total = len(items)
    acc = correct / total if total else 0.0
    return acc, total
