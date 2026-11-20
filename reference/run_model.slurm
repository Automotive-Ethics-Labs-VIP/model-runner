#!/bin/bash
#SBATCH -A pn_wosc091525s
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -J hf-run
#SBATCH -o run-%j.log
# Submit from /gpfs/scratch/$USER/hf :  sbatch run_model.slurm
# Other model:  sbatch --export=ALL,MODEL_ID=Nanbeige/Nanbeige4.3-33B run_model.slurm
BASE=/gpfs/scratch/$USER/hf
export HF_HOME=$BASE/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODEL_ID=${MODEL_ID:-xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic}
export OUT_PATH=$BASE/responses-${SLURM_JOB_ID}.txt

echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi
$BASE/env/bin/python -I "$BASE/run_model.py"
echo "python exit status: $?"
nvidia-smi
echo "output: $OUT_PATH"
