#!/bin/bash
set -e

# KV Cache Learned - AutoDL 一键部署与运行脚本
# 用法：上传代码到 AutoDL 后，在项目根目录执行
#   bash autodl_cloud/deploy.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  KV Cache Learned - AutoDL 一键部署"
echo "  Project: $PROJECT_DIR"
echo "=========================================="

# ---------- 1. 检查 conda ----------
if ! command -v conda &> /dev/null; then
    echo "[Error] conda 未找到，请先安装 Miniconda/Anaconda"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME="kv_cache"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "===== 创建 conda 环境: $ENV_NAME ====="
    conda create -n $ENV_NAME python=3.10 -y
else
    echo "===== conda 环境已存在: $ENV_NAME ====="
fi

conda activate $ENV_NAME

# ---------- 2. 安装依赖 ----------
echo "===== 安装依赖 ====="
pip install -r autodl_cloud/requirements.txt

# ---------- 3. 验证环境 ----------
echo "===== 验证环境 ====="
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"

# ---------- 4. 初始化目录 ----------
echo "===== 初始化目录 ====="
mkdir -p data/traces data/features results/figures results/logs

# ---------- 5. 下载数据集 ----------
DATASET_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

if [ ! -f data/sharegpt.json ]; then
    echo "===== 下载 ShareGPT 数据集 ====="
    if command -v wget &> /dev/null; then
        wget -O data/sharegpt.json "$DATASET_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o data/sharegpt.json "$DATASET_URL"
    else
        echo "[Error] 未找到 wget 或 curl，请手动下载数据集"
        exit 1
    fi
    echo "数据集下载完成: data/sharegpt.json"
else
    echo "===== 数据集已存在，跳过下载 ====="
fi

# ---------- 6. 模型路径提示 ----------
MODEL_PATH="/root/autodl-tmp/models/llama-2-7b-chat"
if [ ! -d "$MODEL_PATH" ]; then
    echo "[Info] 模型未找到: $MODEL_PATH"
    echo "       实验脚本将尝试从 HuggingFace 自动下载（需网络通畅）"
    echo "       如使用 AutoDL 模型库，请创建软链接："
    echo "         ln -s /root/autodl-tmp/models/<模型目录> $MODEL_PATH"
fi

# ---------- 7. 运行全链路实验 ----------
echo "===== 运行实验（阶段 1→3） ====="
bash scripts/run_all.sh

# ---------- 8. 结果报告 ----------
echo ""
echo "=========================================="
echo "           部署与实验全部完成"
echo "=========================================="
echo "结果文件:"
echo "  - Trace:       data/traces/sharegpt_trace.jsonl"
echo "  - 预测器模型:   predictor/reuse_predictor.pt"
echo "  - 命中率图表:   results/figures/hit_rate_comparison.png"
echo "  - 模拟器数据:   results/logs/simulation_results.json"
echo "  - vLLM 日志:    results/logs/vllm_server.log"
echo "=========================================="

# 简单验证
if [ -f results/figures/hit_rate_comparison.png ]; then
    echo "[✓] 图表已生成，实验链路完整"
else
    echo "[Warning] 图表未生成，请检查日志排查问题"
fi
