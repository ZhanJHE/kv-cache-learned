#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "===== Creating Conda Environment ====="
conda create -n kv_cache python=3.10 -y

# 激活环境（注意：在脚本中 source conda 需要初始化）
# 以下命令适用于 bash；若使用其他 shell，请手动激活
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kv_cache

echo "===== Installing Dependencies ====="
pip install -r autodl_cloud/requirements.txt

echo "===== Verifying Installation ====="
python -c "import vllm; print('vLLM version:', vllm.__version__)"
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"

echo "===== Initializing Directories ====="
mkdir -p data/traces data/features results/figures results/logs

echo "===== Setup Complete ====="
echo "Please activate the environment with: conda activate kv_cache"
