#!/usr/bin/env python3
"""Run a few LatentMAS examples and print raw output + extracted answer for debugging."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval_gsm8k import extract_answer_gsm8k, normalize_answer, is_correct
from src.latentmas_fresh import LatentMASConfig, run_sequential_latent_mas


def run_one(model, tokenizer, device, cfg, ex, idx: int):
    question = ex["question"]
    gold = ex["answer"].split("####")[-1].strip()

    pred_text, n_tok = run_sequential_latent_mas(model, tokenizer, question, device, cfg)
    extracted = extract_answer_gsm8k(pred_text, prefer_last=True)
    correct = is_correct(extracted, gold)

    print(f"\n--- Ex {idx} | correct={correct} | extracted={repr(extracted)} | gold={repr(gold)} ---")
    print("PRED (last 500 chars):", repr(pred_text[-500:]) if len(pred_text) > 500 else repr(pred_text))
    return correct, extracted, gold


def main():
    from datasets import load_dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset("gsm8k", "main", split="test")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True)

    # Paper-aligned config (40 latent steps, 256 decode, temp 0.6, top_p 0.95)
    cfg = LatentMASConfig()

    n_samples = 5
    correct_count = 0
    for idx in range(n_samples):
        c, _, _ = run_one(model, tokenizer, device, cfg, ds[idx], idx)
        if c:
            correct_count += 1
    print(f"\n>>> Accuracy: {correct_count}/{n_samples} = {100.0 * correct_count / n_samples:.1f}%")


if __name__ == "__main__":
    main()
