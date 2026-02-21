#!/usr/bin/env bash
# Run single_agent_paper (fair baseline) + sequential_text_mas + sequential_latent_mas on Qwen3-4B, 50 examples.
# Use single_agent_paper for fair comparison: same prompt, chat template, \boxed{}, and decoding as MAS.
#
# With tmux:
#   tmux new -s run4b50
#   ./run_qwen3_4b_50.sh
#   Ctrl+B then D to detach
#
# With nohup:
#   nohup ./run_qwen3_4b_50.sh 2>&1 | tee results/run_qwen3_4b_50.log &
#   disown -h %1

set -e
cd "$(dirname "$0")"
mkdir -p results
python run_experiment.py \
  --model Qwen/Qwen3-4B \
  --methods single_agent_paper sequential_text_mas sequential_latent_mas \
  --max_eval 50 \
  --output results/results_qwen3_4b_50.json
echo "Done. Results in results/results_qwen3_4b_50.json"
