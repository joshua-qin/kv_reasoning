# Improving the MAS Setup and Evaluation

**Problem:** Text MAS (text debate) underperforms single-agent (70% vs 78%). Latent MAS is meant to be “text MAS but more efficient,” so if the underlying multi-agent setup isn’t good, latent won’t show a real advantage. We need to fix the MAS setup and evaluation so that (1) multi-agent can actually help, and (2) we can fairly compare text vs latent.

---

## 1. Fix the text debate setup

### 1.1 Asymmetry and “non-debate”

- **Current:** A reasons once; B sees A’s full text and reasons once. A never sees B. So it’s “B reacts to A,” not a debate.
- **Ideas:**
  - **True 2-turn debate:** A → B (critique/correct) → A (revise). Or: A → B → pick between A’s final vs B’s. This gives a clear “last word” and forces revision.
  - **Symmetric “best of two”:** Run two **independent** single-agent CoTs (same prompt, different seed/temp), no cross-reading. Then pick the answer that matches the other, or use a verifier, or oracle “best of two.” This isolates “diversity of runs” from “interaction.”

### 1.2 Role and prompt design

- **Current:** “Agent A / Agent B. Think step by step.” B: “First read agent A’s reasoning below, then give your own.” No explicit critique or error-checking.
- **Ideas:**
  - **Explicit roles:** “Agent A (solver): solve step by step.” “Agent B (checker): check A’s reasoning for arithmetic or logic errors; if you find one, give the correct answer; if not, confirm A’s answer.” So B’s job is to verify/correct, not just re-solve.
  - **Critic-style B:** “Review the solution below. List any errors. Then give the correct final answer (#### number).”
  - **Diverse strategies in prompt:** A: “Use equations and exact arithmetic.” B: “Use estimation and sanity checks, then refine.” So the two runs are less correlated (different failure modes).

### 1.3 Answer aggregation

- **Current:** If both agree → use it. If they disagree and verifier on → verifier picks (often no gain). If they disagree and no verifier → **return first agent’s answer** (`present[0][1]`), which is arbitrary.
- **Ideas:**
  - **Report multiple aggregation strategies:** (1) **First** (current), (2) **Second** (B only), (3) **Verifier**, (4) **Best-of-two oracle** (pick the answer that matches gold; upper bound for “diversity only”). Then you see how much headroom aggregation has.
  - **Confidence / self-consistency:** If the model can output a confidence or you run a short “is this answer correct?” pass, use that to break ties.
  - **Stronger verifier:** Dedicated verifier prompt that only outputs “1” or “2” (which candidate) plus #### number; or chain “re-solve and compare” so the verifier actually recomputes.

---

## 2. Evaluation improvements

### 2.1 Best-of-k baselines (critical)

- **Best-of-2 single:** Run single-agent CoT **twice** (e.g. different temperature or seed), no sharing. Pick the answer that matches the other; if they disagree, pick at random or with verifier. This tells you: “How much does **diversity of two runs** alone help?”
- **Compare:** single (1 run) vs best-of-2 single vs text debate vs latent. If best-of-2 single already beats single and matches text debate, then text debate’s gain is just diversity; if text debate beats best-of-2 single, then **interaction** (B reading A) adds something. Only then does “latent vs text” (efficiency of that interaction) matter.

### 2.2 Disaggregated metrics

- **By agreement:** Bucket examples into (A and B agree) vs (disagree). Report accuracy in each bucket. Multi-agent should help most when they disagree and one is right (verifier or aggregation picks the right one).
- **By difficulty:** If you have difficulty labels or length/complexity proxy, report accuracy by stratum. Multi-agent might help more on hard examples.
- **When does MAS help vs hurt:** Count “both wrong,” “one right,” “both right,” “debate flips to wrong.” If text debate often “flips” a correct single-agent answer to wrong (e.g. B convinces the pipeline to pick B’s wrong answer), that explains 70% vs 78%.

### 2.3 Task and dataset choice

- **Current:** One slice of GSM8K (n=50), single already 78%. Little headroom.
- **Ideas:**
  - **Harder math:** MATH (or GSM8K hard subset) where single-agent is lower (e.g. 45–60%). More room for collaboration to correct errors.
  - **Same setup across tasks:** Run single, text debate, latent full-stitch (and best-of-2 single) on the **same** n and seeds. If multi-agent beats single on the harder task, then compare text vs latent there (accuracy and token cost).

---

## 3. Making latent vs text a fair comparison

- **Token parity (optional):** Cap text debate so total tokens ≈ latent (e.g. shorter B context, or truncate A’s text when feeding to B). Then you compare “same budget: text sharing vs latent sharing.”
- **Same aggregation:** Use the same `pick_best_prediction` (and verifier setting) for text and latent so the only variable is how B “sees” A (text vs KV).
- **Same roles/prompts:** Use the same agent A / B role text in both text and latent runs so the only difference is the medium (text vs latent), not the task design.

---

## 4. Concrete implementation checklist

| Priority | Change | Purpose |
|----------|--------|--------|
| 1 | Add **best-of-2 single** (two independent CoTs, same prompt, pick on agreement or verifier) | See if any gain is diversity vs interaction |
| 2 | Fix **aggregation when disagree**: report “first” / “second” / “verifier” / “oracle best-of-2”; avoid silent “first only” | Understand headroom and fairness |
| 3 | Give B an **explicit checker/critic** role and optional **A→B→A** second round | Make text debate a real “debate” that can correct errors |
| 4 | Add **disaggregated metrics**: accuracy when A&B agree vs disagree; optionally by difficulty | See where MAS helps or hurts |
| 5 | Run same methods on a **harder setting** (e.g. MATH or GSM8K subset where single &lt; 65%) | Get a regime where multi-agent can beat single |
| 6 | (Later) Token-parity text vs latent, same prompts and aggregation | Fair “efficiency” comparison when MAS helps |

---

## 5. One-sentence takeaway

**Improve the MAS setup (roles, turns, aggregation) and add best-of-k + disaggregated evals so that multi-agent can actually beat single somewhere; only then is comparing text vs latent (and claiming latent is “text but more efficient”) meaningful.**
