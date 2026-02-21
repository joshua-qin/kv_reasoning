# KV Cache RAG Reasoning — Initial Experiment

## Experiment goal

We test a **training-free**, **latent-space** multi-agent reasoning setup: two "agents" (two runs of the same LLM with different prompts) each produce chain-of-thought and a **KV cache**. Instead of only sharing text, we do **RAG over the KV cache**: each agent retrieves the most relevant key-value pairs from the other agent’s cache (by attention/similarity), stitches them into its context, and does a **second round of reasoning**. The hypothesis is that this "reasoning in the language of the model" (via KV cache) improves over single-agent CoT and over text-only debate, with controlled compute (e.g. sliding window over retrieved cache).

## Design (from your notes)

- **Reasoning vs collaboration**: We want iterative refinement (multiple rounds, refer-and-rethink), not just "A does part 1, B does part 2, merge."
- **RAG on the KV cache**: Use attention/similarity to decide *which* parts of the other agent’s KV cache to reuse, then stitch those K,V pairs into the current context for a second forward pass.
- **Position embeddings**: When stitching, we keep position indices consistent so the model still sees a coherent sequence (we use a simple offset scheme for "other agent" cache).
- **Training-free**: No fine-tuning; only test-time inference and cache manipulation.
- **Baselines**:
  1. **Single agent**: One CoT, one answer.
  2. **Two agents, text-only**: A and B each reason; B sees A’s *text* and vice versa; merge answers (vanilla debate).
  3. **Two agents, KV-cache RAG (ours)**: A and B each get a first-round KV cache; we retrieve from the other’s cache by similarity, stitch, then each does a second round with stitched context.

## What we implement (minimal version)

1. **KV cache extraction**  
   Run the model with `use_cache=True`, collect `past_key_values` per layer for each agent’s first-round generation.

2. **KV cache “RAG” (retrieval)**  
   - **Query**: Mean of the **last N keys** (default N=8) at the last layer — a stable summary of “where this agent is” rather than a single token.  
   - **Keys**: Keys from the other agent’s cache at the last layer.  
   - **Search region**: By default we only score positions **after the other agent’s prompt** (`min_position = prompt_length`), so we retrieve from the other’s *reasoning* (CoT), not the shared problem text.  
   - **Scoring**: **Cosine similarity** (L2-normalize query and keys, then dot product) so magnitude doesn’t dominate; optionally raw dot product.  
   - **Selection**: Either **top-k** positions (sorted by index) or one **contiguous chunk** of length k centered on the best-scoring position (default), so the model sees coherent context.

3. **Stitching**  
   For the second round, we **prepend** the selected key-value pairs (with adjusted positions) to the current agent’s cache so that the next forward pass "sees" the retrieved latent reasoning. Position handling: we assign a contiguous range of positions to the retrieved block (e.g. 0 to k-1) and offset the agent’s own continuation accordingly.

4. **Two-agent loop**  
   - **Round 1**: Agent A generates CoT → cache A; Agent B generates CoT → cache B.  
   - **Retrieve**: From A’s last query, retrieve top-k from B’s cache; from B’s last query, retrieve top-k from A’s cache.  
   - **Round 2**: A continues with B’s retrieved cache prepended; B continues with A’s retrieved cache prepended.  
   - **Answer**: Extract final answer from the second-round outputs (e.g. last agent or majority).

5. **Evaluation**  
   Small scale on **GSM8K** (or a subset): compare accuracy and token usage across single-agent, text-only two-agent, and KV-cache RAG two-agent.

## Why this is a reasonable first step

- **Aligns with notes**: "RAG on the latents", "calculate the most related KV caches and stitch them to rethink", "use the corresponding parameters to do more rounds of thinking", training-free, two agents.  
- **Simple**: One similarity metric (query–key), top-k retrieval, one stitch and one second round.  
- **Reproducible**: Single codebase, standard HuggingFace + PyTorch, no external multi-agent frameworks required.  
- **Extensible**: Later you can add sliding window, more rounds, verifier, or learned retrieval; this gives a baseline to beat.

## Position embeddings (brief)

When we stitch "other agent" cache in front of the current agent’s continuation, we give the retrieved block positions `[0, ..., k-1]` and the current agent’s tokens positions `[k, k+1, ...]`. We use the model’s existing position embeddings (e.g. RoPE in Qwen2) so that the stitched sequence is one coherent context. No extra "which agent" embedding in this minimal version; that can be added later.

## Files

- `requirements.txt` — Python deps (torch, transformers, datasets).
- `src/legacy_kv_cache_rag/kv_cache_rag.py` — Legacy KV cache extraction/retrieval/stitch pipeline (kept for reference).
- `src/legacy_kv_cache_rag/baselines.py` — Legacy single-agent and text-only two-agent baselines.
- `src/eval_gsm8k.py` — GSM8K evaluation and answer parsing.
- `run_experiment.py` — Runs all methods and reports accuracy + token count.

## Usage

```bash
pip install -r requirements.txt
python run_experiment.py --model Qwen/Qwen2-7B-Instruct --max_eval 50 --top_k 32 --output results.json
```

Optional: `--device cuda`, `--max_new_tokens`.

### Timing on A100 (Qwen2-7B, 384 tok)

Roughly **~1–2 min per 10 examples** (single agent), **~2–4 min** (text debate), **~3–6 min** (KV RAG). For 50 examples × 3 methods, expect **~30–60 minutes** total.

### Running bit by bit

- **One method at a time** (same `--output` so results accumulate):
  ```bash
  python run_experiment.py --max_eval 50 --output results.json --methods single
  python run_experiment.py --max_eval 50 --output results.json --methods text_debate
  python run_experiment.py --max_eval 50 --output results.json --methods kv_rag
  ```
  Results are written after each method, so you can stop after any step.

- **Smaller batch first** (e.g. 10 examples to sanity-check in ~5–15 min):
  ```bash
  python run_experiment.py --max_eval 10 --output results.json
  ```

- **Dataset slice** (e.g. examples 0–9, then 10–19):
  ```bash
  python run_experiment.py --max_eval 50 --start_idx 0 --end_idx 10 --output results.json --methods single
  python run_experiment.py --max_eval 50 --start_idx 10 --end_idx 20 --output results.json --methods single
  ```
  Note: with `--start_idx`/`--end_idx`, each run overwrites that method’s entry in `results.json` for the slice only; for full 50-example stats you’d need to merge or run without slicing.
