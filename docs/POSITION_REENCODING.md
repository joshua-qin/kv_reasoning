# Position embedding fix: implementation and pipeline

## Is it correct?

**Yes.** The implementation correctly re-encodes RoPE on stitched KV caches so that each key has the rotary embedding for its **global** position in the stitched sequence (0, 1, …, L−1). That matches what the model expects when it attends to the cache and uses `position_ids` for new tokens.

---

## Why stitching breaks positions

1. **Normal run:** Token at position `i` gets RoPE with angle θ_i. Cached key is `K_i = R(θ_i) @ (W_k x_i)`.
2. **After naive stitch:** We concatenate two caches, e.g. `[cache_B_chunk; cache_A]`.  
   - Positions 0..R−1 in the stitched sequence still hold keys that were computed with **B’s** positions (e.g. 180..180+R−1), so they have RoPE(180), …, RoPE(180+R−1).  
   - Positions R..L−1 hold A’s keys with RoPE(0), …, RoPE(L−R−1).
3. **What the model does:** When continuing, it treats the cache as positions 0..L−1 and uses **no extra RoPE** on the cached keys (they are used as-is). So it implicitly assumes key at index `i` has RoPE(i). That is wrong after a naive stitch.

So we must **re-encode**: replace each cached key’s RoPE with the one for its global index.

---

## RoPE math we use

Same as Qwen2/LLaMA:

- **Forward (apply RoPE at position `pos`):**  
  `k_embed = k * cos(θ_pos) + rotate_half(k) * sin(θ_pos)`
- **Inverse (remove RoPE):**  
  `k_raw = k_embed * cos(θ_pos) - rotate_half(k_embed) * sin(θ_pos)`  
  (rotation by −θ; cos(−θ)=cos(θ), sin(−θ)=−sin(θ).)

So:

- `apply_rotary(x, cos, sin)` → forward.
- `apply_rotary_inverse(x, cos, sin)` → same cos/sin but with minus sign in the sin term in the formula.

We use the same `inv_freq` and θ = `position * inv_freq` as the model (from `config.rope_parameters["rope_theta"]` and `head_dim`).

---

## Pipeline (step by step)

### 1. Build stitched cache and original positions

- **Stitch:** `stitched = [first; second]` (concat on seq dimension, per layer).
- **Original positions:** For every position in `stitched` we need the position it had in its **source** run:
  - First block: `first_original_positions` (length = len(first)).
  - Second block: `0, 1, …, len(second)-1` (second was always 0-indexed in its own run).

So:

```text
original_position_ids = [first_original_positions[0], ..., first_original_positions[R-1],
                        0, 1, ..., len(second)-1]
```

Length = R + len(second) = total stitched length L.

### 2. Re-encode keys (rope_utils)

- **Global positions:** `global_positions = [0, 1, …, L-1]`.
- **Cos/sin:**  
  - `cos_old, sin_old` = RoPE cos/sin for `original_position_ids`.  
  - `cos_new, sin_new` = RoPE cos/sin for `global_positions`.
- **Per layer, per position:**  
  - `k_raw = apply_rotary_inverse(k_cached, cos_old, sin_old)` → remove original RoPE.  
  - `k_new = apply_rotary(k_raw, cos_new, sin_new)` → apply RoPE for 0..L−1.
- **Values:** Left unchanged (RoPE is only on Q/K in these models).

Result: stitched cache where key at index `i` has RoPE(i). Values unchanged.

### 3. Continue generation

- `continue_with_cache(..., past_key_values=reencoded_cache, position_offset=L, ...)`.
- Model receives cache of length L and uses it as “positions 0..L−1”.
- New tokens get `position_ids` L, L+1, …; their Q’s get RoPE(L), RoPE(L+1), …
- Attention uses cached K as-is (no second RoPE), so we **must** have stored K_i with RoPE(i). That is exactly what re-encoding does.

So the full pipeline is consistent and correct.

---

## Where it’s used

### KV cache RAG

- **Stitch for A:** `[retrieved_from_B; cache_A]`.  
  - First block: positions in B were `pos_from_b` (the retrieved indices).  
  - Second block: A’s own cache was 0..len(cache_A)−1.  
- **Stitch for B:** `[retrieved_from_A; cache_B]` with `pos_from_a` and 0..len(cache_B)−1.

So we call:

- `stitch_and_reencode_caches(retrieved_from_b, cache_a, pos_from_b, config, device)` for A.
- `stitch_and_reencode_caches(retrieved_from_a, cache_b, pos_from_a, config, device)` for B.

### Full stitch

- **Stitch for A:** `[cache_B; cache_A]`.  
  - First block: B’s positions 0..len(cache_B)−1.  
  - Second block: A’s positions 0..len(cache_A)−1.  
- Same for B with `[cache_A; cache_B]`.

So:

- `stitch_and_reencode_caches(cache_b, cache_a, arange(len_b), config, device)` for A.
- `stitch_and_reencode_caches(cache_a, cache_b, arange(len_a), config, device)` for B.

---

## Summary

- **Correctness:** Cached keys are transformed so that key at global index `i` has RoPE(i), matching the model’s assumption when it uses the cache without reapplying RoPE.
- **Implementation:** `rope_utils.py` does inverse RoPE then forward RoPE with the right cos/sin; `stitch_and_reencode_caches` in `kv_cache_rag.py` builds `original_position_ids` and calls `reencode_past_kvs_for_stitching`.
- **Pipeline:** Stitch → build original positions → re-encode keys (V unchanged) → pass re-encoded cache and `position_offset=L` into `continue_with_cache`.
