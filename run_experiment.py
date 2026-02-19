#!/usr/bin/env python3
"""
Run the initial KV Cache RAG experiment: single-agent, text-only two-agent, KV-cache RAG, and full-stitch latent collaboration.
Reports accuracy and token usage on GSM8K (subset).

Rough timing on A100 (Qwen2-7B, 384 max_new_tokens):
  - Single agent:        ~1–2 min per 10 examples  (~5–10 min for 50)
  - Text debate:         ~2–4 min per 10 examples  (~10–20 min for 50)
  - KV RAG:              ~3–6 min per 10 examples (~15–30 min for 50)
  - Latent full-stitch:  ~4–8 min per 10 examples (longer context = more memory/time)
  Run one method at a time with --methods <name> and use the same --output to accumulate results.
  Methods: single, text_debate, kv_rag, latent_full_stitch, latent_full_stitch_random_kv (ablation),
  latent_full_stitch_random_vectors (random KV), latent_full_stitch_adverse (wrong-math KV).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# Logging: stdout, clear format, flush so progress is visible in foreground
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eval_gsm8k import (
    extract_answer_gsm8k,
    evaluate_gsm8k,
    load_gsm8k,
    normalize_answer,
)
from src.baselines import single_agent_cot, two_agent_text_debate
from src.kv_cache_rag import (
    get_full_kv_cache,
    get_unrelated_question_cache,
    get_adverse_question_cache,
    two_agent_kv_rag_round,
    two_agent_kv_full_stitch_round,
    five_agent_three_round_full_stitch,
    _pick_best_agent_by_agreement,
)


def pick_best_prediction(
    question: str,
    candidate_texts,
    model,
    tokenizer,
    device: torch.device,
    use_verifier: bool,
    verifier_max_new_tokens: int,
):
    """Disagreement-aware final answer selection with optional verifier."""
    parsed = []
    for i, text in enumerate(candidate_texts):
        ans = normalize_answer(extract_answer_gsm8k(text))
        parsed.append((i, text, ans))

    present = [(i, text, ans) for (i, text, ans) in parsed if ans]
    if not present:
        return candidate_texts[0], 0

    unique_answers = {ans for (_, _, ans) in present}
    if len(unique_answers) == 1:
        return present[0][1], 0

    if len(present) == 1:
        return present[0][1], 0

    if use_verifier and len(candidate_texts) >= 2:
        verifier_prompt = (
            "You are a strict math verifier. Two candidates solved the same problem. "
            "Choose the correct final numeric answer. "
            "Output exactly one line in this format: #### <number>\\n\\n"
            f"Problem: {question}\\n\\n"
            f"Candidate 1:\\n{candidate_texts[0]}\\n\\n"
            f"Candidate 2:\\n{candidate_texts[1]}\\n\\n"
            "Final answer (exact format: #### <number>):"
        )
        ids = tokenizer(
            verifier_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).input_ids.to(device)
        out_ids, _ = get_full_kv_cache(
            model,
            tokenizer,
            ids,
            device,
            max_new_tokens=verifier_max_new_tokens,
        )
        verifier_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        verifier_ans = normalize_answer(extract_answer_gsm8k(verifier_text))
        if verifier_ans:
            return f"#### {verifier_ans}", int(out_ids.shape[1])

    return present[0][1], 0


def load_model_and_tokenizer(model_name: str, device: torch.device, load_8bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto" if device.type == "cuda" else None,
        "trust_remote_code": True,
    }
    if load_8bit and device.type == "cuda":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs.pop("torch_dtype", None)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device.type == "cpu":
        model = model.to(device)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="KV Cache RAG reasoning experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-7B-Instruct",
        help="HuggingFace model id",
    )
    parser.add_argument(
        "--max_eval",
        type=int,
        default=50,
        help="Max number of GSM8K examples to evaluate",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=32,
        help="Top-k KV positions to retrieve from other agent",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=384,
        help="Max new tokens per CoT (round 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for results",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["single", "text_debate", "kv_rag"],
        help="Which methods to run: single, text_debate, kv_rag, latent_full_stitch, latent_full_stitch_random_kv, latent_full_stitch_random_vectors, latent_full_stitch_adverse, five_agent_three_round",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index into dataset (for running in chunks)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index into dataset (default: max_eval). Use with --start_idx to run slice.",
    )
    parser.add_argument(
        "--load_8bit",
        action="store_true",
        help="Load model in 8-bit to reduce GPU memory (for 40GB A100 + KV RAG).",
    )
    parser.add_argument(
        "--kv_rag_sparse",
        action="store_true",
        help="Use sparse top-k positions instead of one contiguous chunk (default: contiguous for first experiment).",
    )
    parser.add_argument(
        "--retrieval_score",
        type=str,
        choices=["similarity", "disagreement"],
        default="similarity",
        help="Retrieval scoring: similarity (query-key cosine) or disagreement (key-key mismatch at aligned positions).",
    )
    parser.add_argument(
        "--use_verifier",
        action="store_true",
        help="Use a verifier pass to resolve disagreements between candidate answers.",
    )
    parser.add_argument(
        "--verifier_max_new_tokens",
        type=int,
        default=96,
        help="Max new tokens for verifier pass when --use_verifier is enabled.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    # Reduce HuggingFace/tqdm noise so our logs are readable
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass

    log.info("=" * 60)
    log.info("KV Cache RAG experiment")
    log.info(
        "  model=%s  device=%s  max_eval=%s  methods=%s  retrieval_score=%s",
        args.model, args.device, args.max_eval, args.methods, args.retrieval_score,
    )
    log.info("  output=%s", args.output or "(none)")
    log.info("=" * 60)
    _flush()

    # Load existing results if we're appending to an output file
    results = {}
    if args.output and Path(args.output).exists():
        try:
            with open(args.output) as f:
                results = json.load(f)
            log.info("Loaded existing results from %s (will update only run methods).", args.output)
        except Exception as e:
            log.warning("Could not load %s: %s", args.output, e)
    _flush()

    log.info("Loading GSM8K...")
    dataset = load_gsm8k(split="test", max_samples=args.max_eval)
    start_idx = max(0, args.start_idx)
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    dataset = dataset.select(range(start_idx, end_idx))
    n_ex = len(dataset)
    log.info("Evaluating examples %d–%d (n=%d)", start_idx, start_idx + n_ex - 1, n_ex)
    _flush()

    log.info("Loading model and tokenizer (this may take a minute)...")
    model, tokenizer = load_model_and_tokenizer(args.model, device, load_8bit=args.load_8bit)
    log.info("Model loaded.")
    _flush()

    # Single agent
    if "single" in args.methods:
        log.info("")
        log.info("--- Single-agent CoT ---")
        t0 = time.perf_counter()
        preds = []
        total_tokens = 0
        for i, ex in enumerate(dataset):
            log.info("  [single] example %d/%d ...", i + 1, len(dataset))
            _flush()
            question = ex["question"]
            gold = ex["answer"].split("####")[-1].strip()
            text, n_tok = single_agent_cot(
                model, tokenizer, question, device, max_new_tokens=args.max_new_tokens
            )
            total_tokens += n_tok
            preds.append((text, gold))
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                log.info("  [single] %d/%d done in %.0fs (%.1fs/ex)", i + 1, len(dataset), elapsed, elapsed / (i + 1))
                _flush()
        acc, n = evaluate_gsm8k(preds)
        results["single_agent"] = {"accuracy": acc, "total_tokens": total_tokens, "n": n}
        elapsed = time.perf_counter() - t0
        log.info("  [single] DONE: accuracy=%.2f%%, tokens=%d, time=%.0fs", acc * 100, total_tokens, elapsed)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            log.info("  Saved to %s", args.output)
        _flush()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Text-only two-agent debate
    if "text_debate" in args.methods:
        log.info("")
        log.info("--- Two-agent text-only debate ---")
        t0 = time.perf_counter()
        preds = []
        total_tokens = 0
        for i, ex in enumerate(dataset):
            log.info("  [text_debate] example %d/%d ...", i + 1, len(dataset))
            _flush()
            question = ex["question"]
            gold = ex["answer"].split("####")[-1].strip()
            text_a, text_b, n_tok = two_agent_text_debate(
                model,
                tokenizer,
                question,
                device,
                max_new_tokens_per_turn=args.max_new_tokens,
            )
            total_tokens += n_tok
            pred_text, verifier_tok = pick_best_prediction(
                question,
                [text_a, text_b],
                model,
                tokenizer,
                device,
                args.use_verifier,
                args.verifier_max_new_tokens,
            )
            total_tokens += verifier_tok
            preds.append((pred_text, gold))
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                log.info("  [text_debate] %d/%d done in %.0fs (%.1fs/ex)", i + 1, len(dataset), elapsed, elapsed / (i + 1))
                _flush()
        acc, n = evaluate_gsm8k(preds)
        results["text_debate"] = {"accuracy": acc, "total_tokens": total_tokens, "n": n}
        elapsed = time.perf_counter() - t0
        log.info("  [text_debate] DONE: accuracy=%.2f%%, tokens=%d, time=%.0fs", acc * 100, total_tokens, elapsed)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            log.info("  Saved to %s", args.output)
        _flush()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # KV-cache RAG
    if "kv_rag" in args.methods:
        log.info("")
        log.info("--- Two-agent KV-cache RAG ---")
        t0 = time.perf_counter()
        preds = []
        total_tokens = 0
        for i, ex in enumerate(dataset):
            log.info("  [kv_rag] example %d/%d ...", i + 1, len(dataset))
            _flush()
            question = ex["question"]
            gold = ex["answer"].split("####")[-1].strip()
            # Use same round1 length as single-agent so CoT can finish and output #### answer.
            # Shorter round1 (e.g. 192) was truncating before the answer → ~2% accuracy.
            text_a, text_b, n_tok = two_agent_kv_rag_round(
                model,
                tokenizer,
                question,
                device,
                max_new_tokens_round1=args.max_new_tokens,
                max_new_tokens_round2=min(192, args.max_new_tokens),  # longer refinement
                top_k=args.top_k,
                use_contiguous_chunk=not args.kv_rag_sparse,
                agent_b_do_sample=False,  # greedy for both agents
                retrieval_score=args.retrieval_score,
            )
            total_tokens += n_tok
            pred_text, verifier_tok = pick_best_prediction(
                question,
                [text_a, text_b],
                model,
                tokenizer,
                device,
                args.use_verifier,
                args.verifier_max_new_tokens,
            )
            total_tokens += verifier_tok
            preds.append((pred_text, gold))
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                log.info("  [kv_rag] %d/%d done in %.0fs (%.1fs/ex)", i + 1, len(dataset), elapsed, elapsed / (i + 1))
                _flush()
        acc, n = evaluate_gsm8k(preds)
        kv_key = "kv_rag_" + args.retrieval_score + ("_verifier" if args.use_verifier else "")
        results[kv_key] = {"accuracy": acc, "total_tokens": total_tokens, "n": n}
        elapsed = time.perf_counter() - t0
        log.info("  [%s] DONE: accuracy=%.2f%%, tokens=%d, time=%.0fs", kv_key, acc * 100, total_tokens, elapsed)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            log.info("  Saved to %s", args.output)
        _flush()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Full-stitch latent collaboration (no retrieval: entire cache A→B and B→A)
    if "latent_full_stitch" in args.methods:
        log.info("")
        log.info("--- Two-agent full-stitch latent collaboration ---")
        t0 = time.perf_counter()
        preds = []
        total_tokens = 0
        for i, ex in enumerate(dataset):
            log.info("  [latent_full_stitch] example %d/%d ...", i + 1, len(dataset))
            _flush()
            question = ex["question"]
            gold = ex["answer"].split("####")[-1].strip()
            text_a, text_b, n_tok = two_agent_kv_full_stitch_round(
                model,
                tokenizer,
                question,
                device,
                max_new_tokens_round1=args.max_new_tokens,
                max_new_tokens_round2=min(128, args.max_new_tokens),
            )
            total_tokens += n_tok
            pred_text, verifier_tok = pick_best_prediction(
                question,
                [text_a, text_b],
                model,
                tokenizer,
                device,
                args.use_verifier,
                args.verifier_max_new_tokens,
            )
            total_tokens += verifier_tok
            preds.append((pred_text, gold))
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                log.info("  [latent_full_stitch] %d/%d done in %.0fs (%.1fs/ex)", i + 1, len(dataset), elapsed, elapsed / (i + 1))
                _flush()
        acc, n = evaluate_gsm8k(preds)
        lfs_key = "latent_full_stitch" + ("_verifier" if args.use_verifier else "")
        results[lfs_key] = {"accuracy": acc, "total_tokens": total_tokens, "n": n}
        elapsed = time.perf_counter() - t0
        log.info("  [%s] DONE: accuracy=%.2f%%, tokens=%d, time=%.0fs", lfs_key, acc * 100, total_tokens, elapsed)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            log.info("  Saved to %s", args.output)
        _flush()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Ablation: full-stitch but with KV cache from a fixed unrelated question (real but irrelevant content)
    if "latent_full_stitch_random_kv" in args.methods:
        log.info("")
        log.info("--- Two-agent full-stitch with UNRELATED-QUESTION cache (ablation) ---")
        log.info("  Precomputing unrelated-question KV cache...")
        _flush()
        unrelated_cache = get_unrelated_question_cache(
            model,
            tokenizer,
            device,
            max_new_tokens=args.max_new_tokens,
        )
        log.info("  Done. Running examples.")
        _flush()
        t0 = time.perf_counter()
        preds = []
        total_tokens = 0
        for i, ex in enumerate(dataset):
            log.info("  [latent_full_stitch_random_kv] example %d/%d ...", i + 1, len(dataset))
            _flush()
            question = ex["question"]
            gold = ex["answer"].split("####")[-1].strip()
            text_a, text_b, n_tok = two_agent_kv_full_stitch_round(
                model,
                tokenizer,
                question,
                device,
                max_new_tokens_round1=args.max_new_tokens,
                max_new_tokens_round2=min(128, args.max_new_tokens),
                unrelated_cache=unrelated_cache,
            )
            total_tokens += n_tok
            pred_text, verifier_tok = pick_best_prediction(
                question,
                [text_a, text_b],
                model,
                tokenizer,
                device,
                args.use_verifier,
                args.verifier_max_new_tokens,
            )
            total_tokens += verifier_tok
            preds.append((pred_text, gold))
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                log.info("  [latent_full_stitch_random_kv] %d/%d done in %.0fs (%.1fs/ex)", i + 1, len(dataset), elapsed, elapsed / (i + 1))
                _flush()
        acc, n = evaluate_gsm8k(preds)
        lfsr_key = "latent_full_stitch_random_kv" + ("_verifier" if args.use_verifier else "")
        results[lfsr_key] = {"accuracy": acc, "total_tokens": total_tokens, "n": n}
        elapsed = time.perf_counter() - t0
        log.info("  [%s] DONE: accuracy=%.2f%%, tokens=%d, time=%.0fs", lfsr_key, acc * 100, total_tokens, elapsed)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            log.info("  Saved to %s", args.output)
        _flush()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Ablation: full-stitch with RANDOM KV vectors (same shape/scale, no semantic content)
    if "latent_full_stitch_random_vectors" in args.methods:
        log.info("")
        log.info("--- Two-agent full-stitch with RANDOM KV vectors (ablation) ---")
        _flush()
        t0 = time.perf_counter()
        preds = []
        total_tokens = 0
        for i, ex in enumerate(dataset):
            log.info("  [latent_full_stitch_random_vectors] example %d/%d ...", i + 1, len(dataset))
            _flush()
            question = ex["question"]
            gold = ex["answer"].split("####")[-1].strip()
            text_a, text_b, n_tok = two_agent_kv_full_stitch_round(
                model,
                tokenizer,
                question,
                device,
                max_new_tokens_round1=args.max_new_tokens,
                max_new_tokens_round2=min(128, args.max_new_tokens),
                use_random_other_cache=True,
            )
            total_tokens += n_tok
            pred_text, verifier_tok = pick_best_prediction(
                question,
                [text_a, text_b],
                model,
                tokenizer,
                device,
                args.use_verifier,
                args.verifier_max_new_tokens,
            )
            total_tokens += verifier_tok
            preds.append((pred_text, gold))
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                log.info("  [latent_full_stitch_random_vectors] %d/%d done in %.0fs (%.1fs/ex)", i + 1, len(dataset), elapsed, elapsed / (i + 1))
                _flush()
        acc, n = evaluate_gsm8k(preds)
        key = "latent_full_stitch_random_vectors" + ("_verifier" if args.use_verifier else "")
        results[key] = {"accuracy": acc, "total_tokens": total_tokens, "n": n}
        elapsed = time.perf_counter() - t0
        log.info("  [%s] DONE: accuracy=%.2f%%, tokens=%d, time=%.0fs", key, acc * 100, total_tokens, elapsed)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            log.info("  Saved to %s", args.output)
        _flush()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Ablation: full-stitch with ADVERSARIAL cache (wrong math / wrong facts)
    if "latent_full_stitch_adverse" in args.methods:
        log.info("")
        log.info("--- Two-agent full-stitch with ADVERSARIAL cache (wrong-math KV) ---")
        log.info("  Precomputing adverse (wrong-math) KV cache...")
        _flush()
        adverse_cache = get_adverse_question_cache(
            model,
            tokenizer,
            device,
            max_new_tokens=args.max_new_tokens,
        )
        log.info("  Done. Running examples.")
        _flush()
        t0 = time.perf_counter()
        preds = []
        total_tokens = 0
        for i, ex in enumerate(dataset):
            log.info("  [latent_full_stitch_adverse] example %d/%d ...", i + 1, len(dataset))
            _flush()
            question = ex["question"]
            gold = ex["answer"].split("####")[-1].strip()
            text_a, text_b, n_tok = two_agent_kv_full_stitch_round(
                model,
                tokenizer,
                question,
                device,
                max_new_tokens_round1=args.max_new_tokens,
                max_new_tokens_round2=min(128, args.max_new_tokens),
                unrelated_cache=adverse_cache,
            )
            total_tokens += n_tok
            pred_text, verifier_tok = pick_best_prediction(
                question,
                [text_a, text_b],
                model,
                tokenizer,
                device,
                args.use_verifier,
                args.verifier_max_new_tokens,
            )
            total_tokens += verifier_tok
            preds.append((pred_text, gold))
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                log.info("  [latent_full_stitch_adverse] %d/%d done in %.0fs (%.1fs/ex)", i + 1, len(dataset), elapsed, elapsed / (i + 1))
                _flush()
        acc, n = evaluate_gsm8k(preds)
        key = "latent_full_stitch_adverse" + ("_verifier" if args.use_verifier else "")
        results[key] = {"accuracy": acc, "total_tokens": total_tokens, "n": n}
        elapsed = time.perf_counter() - t0
        log.info("  [%s] DONE: accuracy=%.2f%%, tokens=%d, time=%.0fs", key, acc * 100, total_tokens, elapsed)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            log.info("  Saved to %s", args.output)
        _flush()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Five heterogeneous agents, 3 rounds, full-stitch from best agent each round
    if "five_agent_three_round" in args.methods:
        log.info("")
        log.info("--- Five-agent 3-round full-stitch (best agent each round) ---")
        t0 = time.perf_counter()
        preds = []
        total_tokens = 0
        for i, ex in enumerate(dataset):
            log.info("  [five_agent_three_round] example %d/%d ...", i + 1, len(dataset))
            _flush()
            question = ex["question"]
            gold = ex["answer"].split("####")[-1].strip()
            texts, n_tok = five_agent_three_round_full_stitch(
                model,
                tokenizer,
                question,
                device,
                num_agents=5,
                num_rounds=3,
                max_new_tokens_round1=args.max_new_tokens,
                max_new_tokens_refine=min(128, args.max_new_tokens),
            )
            total_tokens += n_tok
            best_idx = _pick_best_agent_by_agreement(texts)
            pred_text = texts[best_idx]
            preds.append((pred_text, gold))
            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                log.info("  [five_agent_three_round] %d/%d done in %.0fs (%.1fs/ex)", i + 1, len(dataset), elapsed, elapsed / (i + 1))
                _flush()
        acc, n = evaluate_gsm8k(preds)
        results["five_agent_three_round"] = {"accuracy": acc, "total_tokens": total_tokens, "n": n}
        elapsed = time.perf_counter() - t0
        log.info("  [five_agent_three_round] DONE: accuracy=%.2f%%, tokens=%d, time=%.0fs", acc * 100, total_tokens, elapsed)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            log.info("  Saved to %s", args.output)
        _flush()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY (n=%d)", n_ex)
    log.info("=" * 60)
    for name, r in results.items():
        log.info("  %s: accuracy=%.2f%%, total_tokens=%d", name, r["accuracy"] * 100, r["total_tokens"])
    log.info("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Results written to %s", args.output)
    _flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
