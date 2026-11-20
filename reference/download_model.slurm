#!/bin/bash
#SBATCH -A pn_wosc091525s
#SBATCH -p short-40core-shared
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J hf-download
#SBATCH -o dl-%j.log
# Downloads the model weights to scratch so the GPU job can run fully offline.
# Submit from /gpfs/scratch/$USER/hf :  sbatch download_model.slurm
# Other model:  sbatch --export=ALL,MODEL_ID=Nanbeige/Nanbeige4.3-33B download_model.slurm
BASE=/gpfs/scratch/$USER/hf
export HF_HOME=$BASE/cache
export TMPDIR=$BASE/tmp
export MODEL_ID=${MODEL_ID:-xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic}
echo "downloading $MODEL_ID to $HF_HOME"
$BASE/env/bin/python -I - <<'EOF'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(os.environ["MODEL_ID"])
print("snapshot at:", path)
EOF
echo "python exit status: $?"
du -shL "$HF_HOME"/hub/models--*/snapshots/* 2>/dev/null
echo done
