#!/bin/bash
# One-time env build. Run on the login / OnDemand shell:  bash setup_env.sh
# Everything lives under scratch because the home quota is only 20G.
set -e
BASE=/gpfs/scratch/$USER/hf
ENV=$BASE/env
mkdir -p "$BASE/cache" "$BASE/pip-cache" "$BASE/tmp"
export PIP_CACHE_DIR=$BASE/pip-cache
export TMPDIR=$BASE/tmp
export HF_HOME=$BASE/cache

module load anaconda/3

if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -p "$ENV" --override-channels -c conda-forge python=3.11 pip
fi

"$ENV/bin/pip" install -U pip
# Latest transformers on purpose: the model is tagged qwen3_5 and the HF page gives no minimum version.
"$ENV/bin/pip" install -U torch torchvision transformers accelerate huggingface_hub pillow

echo "=== versions ==="
"$ENV/bin/python" -I - <<'EOF'
import torch, transformers, accelerate, huggingface_hub
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| accelerate", accelerate.__version__, "| hub", huggingface_hub.__version__)
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic")
print("config OK, model_type:", cfg.model_type)
EOF
echo ">> Setup done"
