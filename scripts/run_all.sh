#!/bin/bash
set -e

# 获取脚本所在目录，并切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "===== Project Root: $PROJECT_DIR ====="

# 检查 conda 环境（可选）
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Current conda env: $CONDA_DEFAULT_ENV"
else
    echo "[Warning] 未检测到 conda 环境，建议先执行: conda activate kv_cache"
fi

echo "===== Step 1: Collecting Trace ====="
python scripts/collect_trace.py

echo "===== Step 2: Training Predictor ====="
cd predictor
python train.py
cd ..

echo "===== Step 3: Running Simulation ====="
cd simulator
python evaluate.py
cd ..

echo "===== Done ====="
echo "Results:"
echo "  - Trace: data/traces/"
echo "  - Model: predictor/reuse_predictor.pt"
echo "  - Figures: results/figures/"
echo "  - Logs: results/logs/"
