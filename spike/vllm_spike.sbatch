#!/bin/bash
#SBATCH -A pn_wosc091525s
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J mr-vllm-spike
#SBATCH -o /gpfs/scratch/klow/model-runner/logs/vllm-spike-%j.log
# Phase 0 spike. Submit from /gpfs/scratch/klow/model-runner/spike:  sbatch vllm_spike.sbatch
B=/gpfs/scratch/klow/model-runner
export HF_HOME=/gpfs/scratch/klow/hf/cache        # the 9B is already downloaded here
export HF_HUB_OFFLINE=1
export TMPDIR=$B/tmp
export MODEL_ID=${MODEL_ID:-xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic}
export INPUT=$B/spike/round1_anchors.json
export TEMPLATE=$B/spike/prompt.txt
export OUT=$B/spike/vllm-spike-$SLURM_JOB_ID.jsonl
echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
curl -sS -m 10 -o /dev/null -w "gpu node -> huggingface.co HTTP %{http_code}\n" https://huggingface.co || echo "gpu node -> huggingface.co UNREACHABLE"
$B/envs/vllm/bin/python -I $B/spike/vllm_spike.py
echo "python exit status: $?"
