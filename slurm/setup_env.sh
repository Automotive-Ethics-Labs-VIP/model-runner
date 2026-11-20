#!/bin/bash
# One-time setup on the SeaWulf login node:  bash slurm/setup_env.sh [vllm|hf]
# Builds a conda env under scratch (home quota is 20 GB) and installs model-runner into it.
#   vllm: vLLM brings its own torch (0.34.0 -> torch 2.14 cu130, verified 2026-11-20). Default engine.
#   hf:   the stack proven by job 2417039 (torch 2.15 cu132, transformers 5.22).
# Keep them separate: installing vLLM into the hf env would replace the torch that is known to work.
set -e
KIND=${1:-vllm}
REPO=$(cd "$(dirname "$0")/.." && pwd)
HOME_DIR=${MODELRUNNER_HOME:-/gpfs/scratch/$USER/model-runner}
ENV=$HOME_DIR/envs/$KIND
export PIP_CACHE_DIR=$HOME_DIR/pip-cache TMPDIR=$HOME_DIR/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HOME_DIR/logs"
module load anaconda/3
[ -x "$ENV/bin/python" ] || conda create -y -p "$ENV" --override-channels -c conda-forge python=3.12 pip
"$ENV/bin/pip" install -U pip
case "$KIND" in
  vllm) "$ENV/bin/pip" install -U vllm ;;
  hf)   "$ENV/bin/pip" install -U torch torchvision transformers accelerate ;;
  *)    echo "usage: $0 [vllm|hf]"; exit 1 ;;
esac
"$ENV/bin/pip" install -e "$REPO"
"$ENV/bin/python" -I -c "import modelrunner, torch, transformers; print('modelrunner', modelrunner.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
echo "env ready: $ENV/bin/modelrunner"
