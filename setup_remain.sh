#!/bin/bash
# ================================================================
#  kv-cache-learned 剩余环境配置脚本
#  在 conda create -n kv_cache python=3.10 -y 之后运行
# ================================================================

set -e
cd /root/autodl-tmp

ENV_PYTHON="/root/miniconda3/envs/kv_cache/bin/python"
ENV_PIP="/root/miniconda3/envs/kv_cache/bin/pip"

echo "=========================================="
echo "  kv-cache-learned 环境配置 (后续步骤)"
echo "  目标环境: kv_cache (Python 3.10)"
echo "=========================================="

# 1. 安装 PyTorch 2.3.0（~779MB，需要较长时间）
echo ""
echo "===== [1/4] 安装 PyTorch 2.3.0 ====="
echo "      大小约 779MB，包含约 2.5GB CUDA 库"
echo "      如中断可重新运行，pip 会从断点继续"
echo ""
$ENV_PIP install "torch==2.3.0"

# 2. 安装剩余依赖
echo ""
echo "===== [2/4] 安装剩余 Python 依赖 ====="
echo ""
$ENV_PIP install "matplotlib>=3.7.0" "seaborn>=0.12.0"
$ENV_PIP install "transformers>=4.40.0" "datasets>=2.14.0" "accelerate>=0.25.0"
$ENV_PIP install "vllm==0.5.0"

# 3. 验证安装
echo ""
echo "===== [3/4] 验证环境 ====="
echo ""
$ENV_PYTHON -c "
import vllm; print('vLLM:', vllm.__version__)
import torch; print('PyTorch:', torch.__version__)
import transformers; print('Transformers:', transformers.__version__)
import numpy; print('NumPy:', numpy.__version__)
import pandas; print('Pandas:', pandas.__version__)
import sklearn; print('scikit-learn:', sklearn.__version__)
import matplotlib; print('Matplotlib:', matplotlib.__version__)
import seaborn; print('Seaborn:', seaborn.__version__)
import datasets; print('Datasets:', datasets.__version__)
import accelerate; print('Accelerate:', accelerate.__version__)
"

# 4. 确认目录结构
echo ""
echo "===== [4/4] 确认目录结构 ====="
echo ""
for d in data/traces data/features results/figures results/logs; do
    if [ -d "$d" ]; then
        echo "  [OK] $d"
    else
        mkdir -p "$d"
        echo "  [创建] $d"
    fi
done

echo ""
echo "=========================================="
echo "  环境配置完成！"
echo "=========================================="
echo ""
echo "后续运行:"
echo "  conda activate kv_cache"
echo "  python scripts/collect_trace.py     # 阶段1: 收集Trace（需GPU）"
echo "  cd predictor && python train.py && cd ..  # 阶段2: 训练预测器"
echo "  cd simulator && python evaluate.py && cd .. # 阶段3: 模拟器评估"
echo "  bash scripts/run_all.sh             # 一键运行 1→2→3"
echo ""
