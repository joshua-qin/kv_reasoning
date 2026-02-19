#!/usr/bin/env bash
# DG-KV experiment matrix (2 agents, 50 examples). Writes to results_dgkv.json.
# Estimated total: ~2–2.5 h on A100 (single ~7m, text_debate ~12m, kv_rag x4 ~1h, latent x2 ~35m).

set -e
cd "$(dirname "$0")"
OUT=results_dgkv.json
N=50

echo "[$(date)] Starting DG-KV experiments (n=$N), output=$OUT"

# Baselines
python run_experiment.py --methods single text_debate --max_eval $N --output $OUT
python run_experiment.py --methods kv_rag --retrieval_score similarity --max_eval $N --output $OUT
python run_experiment.py --methods kv_rag --retrieval_score disagreement --max_eval $N --output $OUT
python run_experiment.py --methods kv_rag --retrieval_score similarity --use_verifier --max_eval $N --output $OUT
python run_experiment.py --methods kv_rag --retrieval_score disagreement --use_verifier --max_eval $N --output $OUT
python run_experiment.py --methods latent_full_stitch --max_eval $N --output $OUT
python run_experiment.py --methods latent_full_stitch --use_verifier --max_eval $N --output $OUT

echo "[$(date)] Done. Results in $OUT"
