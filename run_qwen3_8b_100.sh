#!/usr/bin/env bash
# Run single + sequential_text_mas + sequential_latent_mas on Qwen3-8B, 100 examples.
# Use with tmux (recommended) or nohup so it survives SSH disconnect.
#
# With tmux:
#   tmux new -s run100
#   ./run_qwen3_8b_100.sh
#   Ctrl+B then D to detach
#
# With nohup:
#   nohup ./run_qwen3_8b_100.sh >> results/run_qwen3_8b_100.log 2>&1 &
#   disown -h %1

set -e
cd "$(dirname "$0")"
mkdir -p results
python run_experiment.py \
  --model Qwen/Qwen3-8B \
  --methods single sequential_text_mas sequential_latent_mas \
  --max_eval 100 \
  --output results/results_qwen3_8b_100.json
echo "Done. Results in results/results_qwen3_8b_100.json"
