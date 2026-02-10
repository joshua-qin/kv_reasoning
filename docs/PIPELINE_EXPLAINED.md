# KV Cache RAG Pipeline — Explained in Detail

This document walks through the **entire pipeline** for one question: what happens, what the data looks like, and why each step exists.

---

## The big picture (one sentence)

Two copies of the same model each solve the same problem (chain-of-thought); we then **copy a chunk of one model’s internal “memory” (KV cache) into the other’s**, and let each run a **second round** of generation so they can “see” the other’s reasoning in latent form.

---

## Prerequisites: what is the KV cache?

When an LLM generates text, it processes the sequence **one token at a time**. At each position, the **attention** mechanism looks at all previous positions: for each token it computes a **query** (Q), and for every previous token it has a **key** (K) and **value** (V). The model uses Q to “ask” and K,V to “answer”: it scores how relevant each past position is (Q·K) and then takes a weighted sum of the values (V).

To avoid recomputing K and V for the whole past at every new token, we **cache** them. After processing positions 0, 1, …, L−1, we have:

- **KV cache**: for each layer, one **key** tensor and one **value** tensor of shape `(batch, num_heads, seq_len, head_dim)`.

So the cache is “everything the model has already seen” in the form of keys and values. When we add position L, we only compute Q,K,V for position L and reuse the cached K,V for 0…L−1.

**Important**: Those K and V vectors depend on **position** (e.g. RoPE). So the cache is tied to a specific sequence and its positions.

---

## How KV cache stitching works

**Stitching** means: we take two KV caches (or a chunk of one + a full one) and **concatenate them along the sequence dimension** so the model sees one long “past” when we run the next forward pass.

### At the tensor level

- Each cache is a **tuple of layers**. For each layer we have:
  - **K**: shape `(batch, num_heads, seq_len, head_dim)`
  - **V**: same shape
- **`stitch_caches(retrieved, own)`** does, for **every layer**:
  - `new_K = torch.cat([retrieved_K, own_K], dim=2)`  ← concat along the sequence dimension
  - `new_V = torch.cat([retrieved_V, own_V], dim=2)`
- So the stitched cache has **sequence length = len(retrieved) + len(own)**. No mixing of layers; we only extend the sequence.

**Example (KV RAG):**

- `retrieved` = 32 positions from the other agent (each layer: 32 positions).
- `own` = full cache of the current agent (e.g. 400 positions).
- Stitched cache per layer: **32 + 400 = 432** positions. Order: **first the 32 from the other agent, then the 400 from “own”.**

### What the model sees when we continue

When we call **`continue_with_cache`** we pass:

1. **`past_key_values`** = the stitched cache (e.g. 432 positions per layer).
2. **`position_offset`** = 432 (the length of that cache).
3. **`input_ids`** = the continuation tokens (e.g. `" Refining: "`).

The model then:

- **Does not** re-run on the tokens that produced the cache. It **treats the cache as already-computed context**: positions 0 … 431.
- Runs a forward on the **new** tokens only. For those tokens it sets **`position_ids`** = 432, 433, … so they are the next positions in the sequence.
- In **attention**, for each new position the model has a **query** Q. The **keys and values** it attends to are the entire stitched cache (positions 0 … 431). So the new tokens can attend to:
  - first the “other agent” block (0 … 31),
  - then the “own” block (32 … 431).
- **Causal mask**: each position can only attend to previous positions, so the continuation tokens can attend to the full 432 positions. We pass an **attention mask** of length `position_offset + num_new_tokens` so the library builds the correct causal mask over the whole sequence.

So **stitching = one long KV sequence**. The model doesn’t know “this part is from A, this part is from B”; it just sees one contiguous past and attends over it when generating the continuation.

### Position embedding caveat

The K and V vectors in the cache were computed with **position encodings at their original positions** (e.g. RoPE(0), RoPE(1), … in each agent’s own run). When we stitch:

- We put “retrieved” at positions **0 … 31** in the new sequence, but those K,V may have been computed at positions 180 … 211 in the other agent’s run (so they encode RoPE(180), …).
- We put “own” at positions **32 … 431**, but those K,V were computed at positions 0 … 399 in the current agent’s run (so they encode RoPE(0), …).

We **do not** re-apply RoPE to the cached K,V. So the **relative positions** the model “thinks” it’s using (from the causal mask and any position logic) don’t match the positions actually baked into the vectors. That can hurt quality; it’s a known limitation of this simple stitching.

---

## Setup for one question

**Input**: One math problem string, e.g.  
`"Janet has 3 apples. She buys 2 more. How many does she have?"`

**Model**: One causal LM (e.g. Qwen2-7B). We run it **twice** with different prompts; we call those two runs “Agent A” and “Agent B.” Same weights, different text in, different text out.

**Prompts** (conceptually):

- **Agent A**:  
  `"You are a precise reasoner. You are agent A. Think step by step and give your final answer. Problem: <question> Reasoning:"`
- **Agent B**:  
  Same but “agent B” instead of “agent A.”

So both see the same problem; only the “You are agent A/B” line differs. That gives us two different chain-of-thoughts.

---

## Step 1: Round 1 — Each agent does CoT and we save the KV cache

**What we do**

- Tokenize prompt A → `ids_a` (e.g. 150 token IDs).
- Run the model **autoregressively** on `ids_a`: feed the prompt, then repeatedly take the next-token logits, pick the next token, append it, and run again with `use_cache=True`.
- Stop after `max_new_tokens_round1` tokens (e.g. 384) or when the model outputs an end-of-sequence token.

**What we get**

- **`generated_a`**: the full token sequence (prompt + generated CoT). Might look like:  
  `[prompt tokens...] [First I add 3+2...] [So 5. #### 5]`
- **`cache_a`**: the **full KV cache** at the moment we stop. For every layer it has:
  - `K`: shape `(1, num_heads, L_a, head_dim)` where `L_a = len(generated_a)`
  - `V`: same shape.

So `cache_a` is “Agent A’s internal representation” of the entire sequence (prompt + its reasoning).

We do **the same** for Agent B with prompt B:

- **`generated_b`**: prompt B + B’s CoT.
- **`cache_b`**: full KV cache for that sequence (length `L_b`).

At this point the two agents have **not** seen each other. They only have their own prompt and their own generation.

**Why we do it**

We need two independent “opinions” (two CoTs) and their internal states (caches) so we can later let each agent “peek” at the other’s reasoning in latent space.

---

## Step 2: Build a “query” from each agent’s current state

**What we do**

We want to ask: “In the **other** agent’s cache, which positions are most relevant to **this** agent’s current state?”

To represent “this agent’s current state” we use the **keys** at the **last layer** of its cache (keys encode “what this position is about” for attention). We don’t use just the very last position (that can be noisy); we take the **mean of the last N keys** (e.g. N=8):

- **`qa_key`** = mean of last 8 keys of `cache_a` at the last layer → shape `(1, num_heads, 1, head_dim)`.
- **`qb_key`** = same for `cache_b`.

So:

- **`qa_key`** ≈ “what Agent A was focusing on at the end of its reasoning.”
- **`qb_key`** ≈ “what Agent B was focusing on at the end of its reasoning.”

**Why we do it**

We need a single “query” vector per agent to compare against every position in the **other** agent’s cache. Using the last N keys makes that query more stable than using only the last token’s key.

---

## Step 3: Retrieve: find the best chunk in the other agent’s cache

**What we do**

For **Agent A** we ask: “Which part of **B’s** cache is most similar to A’s current state?”

- Take **all keys** from `cache_b` at the last layer → shape `(1, num_heads, L_b, head_dim)`.
- We **only look at positions after B’s prompt** (positions ≥ `prompt_len_b`). So we search only in B’s **generated** tokens (its reasoning), not in the shared problem text.
- For each position `i` in that range, compute a **similarity score** between `qa_key` and the key at `i` (cosine similarity: normalize both, then dot product; then average over heads). So we get one score per position in B’s “reasoning” region.
- We don’t pick 32 scattered positions; we pick **one contiguous block** of 32 positions (e.g. `top_k=32`) where the **center** of the block is the position with the **highest** score. That gives us a coherent span of B’s reasoning (e.g. positions 180–211 in B’s sequence).
- Call that list of positions **`pos_from_b`** (length 32).

We do the same the other way:

- Use **`qb_key`** to score positions in **`cache_a`** (only in A’s generated region, ≥ `prompt_len_a`).
- Pick a contiguous chunk of 32 positions in A’s cache with the best-scoring center → **`pos_from_a`**.

**What we get**

- **`pos_from_b`**: 32 indices into B’s cache (e.g. `[180, 181, …, 211]`).
- **`pos_from_a`**: 32 indices into A’s cache (e.g. `[165, 166, …, 196]`).

**Why we do it**

We want each agent to see a **relevant** part of the other’s reasoning, not random positions. Similarity in key space is used as a proxy for “relevant.” Restricting to the generated region avoids pulling the problem statement again; using a contiguous chunk keeps the snippet coherent.

---

## Step 4: Slice — keep only those positions from each cache

**What we do**

- **`retrieved_from_b`**: From `cache_b`, at **every layer**, keep only the K and V at the 32 positions in `pos_from_b`. So each layer’s K and V go from shape `(1, num_heads, L_b, head_dim)` to `(1, num_heads, 32, head_dim)`.
- **`retrieved_from_a`**: Same for `cache_a` using `pos_from_a` → 32 positions per layer.

So we have two “mini-caches”: one is 32 positions from B, one is 32 positions from A.

**Why we do it**

We can’t give each agent the **entire** other cache (too long and often not all relevant). We give only the 32 most relevant positions so the second round stays manageable and focused.

---

## Step 5: Stitch — prepend the other’s chunk in front of each agent’s own cache

**What we do**

For **Agent A**, we build a new “past” that the model will see in round 2:

- New cache for A = **[retrieved_from_b]** then **[full cache_a]** (along the sequence dimension).
  - So at each layer: `K_new = cat(retrieved_from_b.K, cache_a.K, dim=seq)` → length 32 + L_a.
  - Same for V.

So A’s context for round 2 is: “first 32 positions from B’s reasoning, then my **entire** round-1 sequence (prompt + my CoT).”

For **Agent B** we do the opposite:

- New cache for B = **[retrieved_from_a]** then **[full cache_b]**.
  - Length 32 + L_b per layer.

We compute the total lengths:

- **`len_stitch_a`** = 32 + L_a.
- **`len_stitch_b`** = 32 + L_b.

**Why we do it**

So that when A continues generating, its attention can “see” first the retrieved part of B, then its own full story. The model doesn’t get the raw text of B; it gets B’s **internal representations** (K,V) for that chunk. The hope is that this “reasoning in the language of the model” helps A refine its answer.

**Caveat (RoPE)**: The K and V in the cache were computed with **original** positions (e.g. in B’s sequence those 32 positions had positions 180–211). After stitching we use them as positions 0–31. So the **position encodings baked into K,V** no longer match the positions we assign them. That can hurt quality; it’s a known limitation of this simple stitching.

---

## Step 6: Round 2 — Each agent continues with the stitched cache

**What we do**

We don’t feed the full round-1 text again. We tell the model: “You already have this past (the stitched cache); here are a few **new** tokens to continue from.”

- **Continuation string**: e.g. `" Refining: "` → tokenize to **`cont_ids`** (a few tokens).
- For Agent A:
  - **`past_key_values`** = the stitched cache we built (32 from B + full cache_a).
  - **`position_offset`** = `len_stitch_a` (total length of that cache).
  - We run the model once to process `cont_ids` **in the context of** that past (with correct `position_ids` and full-length `attention_mask` so it attends to the whole stitched sequence).
  - Then we **autoregressively** generate more tokens (e.g. 128) as usual: one token at a time, updating the cache each time, until we hit the limit or EOS.

**What we get**

- **`out_a`**: token IDs for **only** the continuation: `" Refining: "` + the new tokens generated in round 2 (e.g. “After considering the other view, I still get 5. #### 5”). So `out_a` does **not** contain the full round-1 text; that’s already in the cache.
- **`out_b`**: same for B.

**Why we do it**

So each agent gets a **second chance** to produce an answer, having “seen” (in latent form) a relevant chunk of the other agent’s reasoning. The final answer can appear in round 1 or in round 2.

---

## Step 7: Assemble the final text and extract the answer

**What we do**

- **Round 1 text**: Decode `generated_a` and `generated_b` to strings → **`round1_a`**, **`round1_b`** (full prompt + CoT + any answer already there).
- **Round 2 text**: Decode `out_a` and `out_b` → **`round2_a`**, **`round2_b`** (only “ Refining: ” + second-round generation).
- **Full responses**:
  - **`text_a`** = `round1_a + " " + round2_a`
  - **`text_b`** = `round1_b + " " + round2_b`

For evaluation we pick one agent’s full response (e.g. B if non-empty): **`pred_text = text_b if text_b else text_a`**.

Then we run **answer extraction** on `pred_text` (e.g. look for `#### <number>`, or “answer is X”, or last number in the last few lines) and compare that to the gold answer.

**Why we do it**

We need a single string per question for the metric. Concatenating round 1 and round 2 lets the extractor find the answer in either part.

---

## End-to-end flow (summary diagram)

```
Question
    │
    ├── Prompt A ──► [Model] autoregressive with use_cache ──► generated_a, cache_a (length L_a)
    │
    └── Prompt B ──► [Model] autoregressive with use_cache ──► generated_b, cache_b (length L_b)

Query A = mean(last 8 keys of cache_a)    Query B = mean(last 8 keys of cache_b)
    │                                              │
    ▼                                              ▼
Score positions in cache_b (only ≥ prompt_len_b)   Score positions in cache_a (only ≥ prompt_len_a)
    │                                              │
    ▼                                              ▼
Best contiguous chunk of 32 in B  ──► pos_from_b   Best contiguous chunk of 32 in A  ──► pos_from_a
    │                                              │
    ▼                                              ▼
retrieved_from_b = cache_b[:, pos_from_b]           retrieved_from_a = cache_a[:, pos_from_a]
    │                                              │
    └──────────────┬───────────────────────────────┘
                   │
    stitch_a = [retrieved_from_b ; cache_a]    stitch_b = [retrieved_from_a ; cache_b]
                   │
    ┌──────────────┴──────────────┐
    ▼                              ▼
continue_with_cache(" Refining: ", stitch_a)   continue_with_cache(" Refining: ", stitch_b)
    │                              │
    ▼                              ▼
out_a (tokens)                    out_b (tokens)
    │                              │
    ▼                              ▼
text_a = round1_a + " " + round2_a    text_b = round1_b + " " + round2_b
    │                              │
    └──────────────┬───────────────┘
                   ▼
         pred_text = text_b or text_a
                   ▼
         extract_answer_gsm8k(pred_text)  vs  gold  →  correct?
```

---

## Summary table

| Step | What we have | What we do | What we get |
|------|----------------|------------|-------------|
| 1. Round 1 | Question, two prompts | Run model twice (A and B), save full token sequence + KV cache each time | `generated_a`, `cache_a`, `generated_b`, `cache_b` |
| 2. Query | Both caches | Mean of last 8 keys at last layer for each | `qa_key`, `qb_key` |
| 3. Retrieve | Queries + other cache | Score positions (cosine) in *generated* region only; take best contiguous chunk of 32 | `pos_from_b`, `pos_from_a` |
| 4. Slice | Full caches + position lists | Keep K,V only at those positions, every layer | `retrieved_from_b`, `retrieved_from_a` |
| 5. Stitch | Retrieved chunks + own full cache | Prepend other’s chunk in front of own cache | `stitch_a`, `stitch_b`, and their lengths |
| 6. Round 2 | Stitched cache + “ Refining: ” | Run model to process continuation then generate more tokens | `out_a`, `out_b` |
| 7. Answer | Round 1 + round 2 token IDs | Decode, concatenate, pick one agent, extract number | `pred_text`, extracted answer, correctness |

---

## Full-stitch latent collaboration (no retrieval)

We also implement a **simpler** variant called **full-stitch** (`latent_full_stitch` in `run_experiment.py`). It’s the same idea as KV cache RAG, but we **don’t do retrieval**: we share the **entire** KV cache of each agent with the other.

### How it works

1. **Round 1** — Same as KV RAG: Agent A and Agent B each do independent CoT on the same question. We get `generated_a`, `cache_a`, `generated_b`, `cache_b` (full caches, no slicing).

2. **Stitch (no retrieval)**  
   - **Agent A’s context for round 2** = **[entire cache of B]** then **[entire cache of A]**.
     - So A “sees” first every position of B’s run (prompt + CoT), then every position of its own run.
   - **Agent B’s context for round 2** = **[entire cache of A]** then **[entire cache of B]**.
     - So B sees first all of A, then all of itself.

3. **Round 2** — Same as KV RAG: we run the model with the stitched cache and a short continuation prompt (`" Refining: "`), then generate more tokens. Each agent’s round-2 output is decoded and appended to its round-1 text for answer extraction.

### Difference from KV RAG

| | KV RAG | Full-stitch |
|---|--------|-------------|
| **What we share** | A **chunk** of the other’s cache (e.g. 32 positions chosen by key similarity) | The **entire** other cache (all positions) |
| **Retrieval** | Yes: query = last-N keys, score by cosine, take best contiguous block | No: no scoring, no top-k |
| **Context length in round 2** | `top_k + own_length` (bounded) | `other_length + own_length` (can be 2× round-1 length) |
| **Use case** | Controlled cost; only “relevant” part of the other agent | Baseline: does “see everything latent” help at all? |

So full-stitch is a **latent collaboration** where each agent literally gets the **full** internal state (KV cache) of the other before refining. It uses more memory and context length; the tradeoff is no retrieval logic and a direct comparison: “does sharing everything (latent) beat sharing a retrieved subset (KV RAG) or only text (text debate)?”

### How to run it

```bash
python run_experiment.py --max_eval 50 --output results.json --methods latent_full_stitch
```

Or together with other methods (e.g. run single, text_debate, kv_rag, then latent_full_stitch and let results accumulate in `results.json`).

---

## One-line recap

**KV RAG:** Two agents each do CoT and fill a KV cache; we use key similarity to pull a 32-position “reasoning” chunk from the other’s cache, prepend it to each agent’s own cache, then run a short second round (“Refining:” + more tokens) and take the final answer from the combined round-1 + round-2 text.

**Full-stitch:** Same two-round structure, but we prepend the **entire** other agent’s KV cache (no retrieval). Each agent refines after “seeing” the full latent state of the other.
