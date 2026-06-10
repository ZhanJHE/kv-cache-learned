#!/bin/bash
set -e

# ================================================================
#  KV Cache Learned - 环境一键配置脚本
#  执行: bash scripts/setup_env.sh
#  说明: 安装 Python 依赖 + 下载数据集 + 创建目录 + 验证
#  (Python 版本沿用当前环境，不再强制 3.10)
#  (GPU 本身和模型文件除外，需有卡后手动准备)
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  KV Cache Learned 环境配置"
echo "  项目路径: $PROJECT_DIR"
echo "  当前 Python: $(python --version 2>&1)"
echo "=========================================="

# ─── 1. 清理 pip 缓存（避免之前半截下载的坏包干扰） ──────────
echo ""
echo "[信息] 清理 pip 缓存..."
pip cache purge 2>/dev/null || true
rm -rf ~/.cache/pip/ 2>/dev/null || true
echo "[OK] pip 缓存已清理"

# ─── 2. 安装 Python 依赖 ──────────────────────────────────────
echo ""
echo "===== 安装 Python 依赖 ====="
echo "     依赖清单: autodl_cloud/requirements.txt"
echo "     注意: 首次安装需下载约 2.5GB CUDA 库，需耐心等待"
echo ""

# 先升级 pip
pip install --upgrade pip

echo ""
echo "--- 2.1 安装科学计算基础库 ---"
pip install "numpy>=1.24.0" "pandas>=2.0.0" "scikit-learn>=1.3.0" \
    "matplotlib>=3.7.0" "seaborn>=0.12.0" "tqdm>=4.65.0" "requests>=2.31.0"

echo ""
echo "--- 2.2 安装深度学习框架 ---"
pip install "torch==2.3.0"

echo ""
echo "--- 2.3 安装 ML 库 ---"
pip install "transformers>=4.40.0" "datasets>=2.14.0" "accelerate>=0.25.0"

echo ""
echo "--- 2.4 安装 xformers（vLLM 依赖，需从源码编译） ---"
echo "     注意: 编译约需 5~30 分钟，取决于机器性能"
echo "     如果不装 vLLM（只跑阶段 2/3），跳过也没关系"
echo "     跳过方法: export SKIP_VLLM=1  或  Ctrl+C 中断此步"
echo ""
if [ "${SKIP_VLLM:-0}" != "1" ]; then
    pip install "xformers==0.0.26.post1" --no-build-isolation
    echo ""
    echo "--- 2.5 安装 vLLM ---"
    pip install "vllm==0.5.0"
else
    echo "[跳过] SKIP_VLLM=1，跳过 xformers 和 vLLM 安装"
fi

echo ""
echo "[OK] 所有 Python 依赖安装完成"

# ─── 3. 创建运行时目录（幂等） ────────────────────────────────
echo ""
echo "===== 创建运行时目录 ====="
mkdir -p data/traces data/features results/figures results/logs
echo "[OK] 目录就绪:"
echo "      data/traces/    - Trace 文件"
echo "      data/features/  - 训练特征"
echo "      results/figures/ - 图表产出"
echo "      results/logs/   - 实验日志"

# ─── 4. 下载 ShareGPT 数据集 ──────────────────────────────────
echo ""
echo "===== 下载 ShareGPT 数据集 ====="
DATASET_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

if [ -f "data/sharegpt.json" ]; then
    SIZE=$(stat --printf="%s" data/sharegpt.json 2>/dev/null || stat -f%z data/sharegpt.json 2>/dev/null)
    if [ "$SIZE" -gt 1000000 ]; then
        echo "[OK] 数据集已存在 (data/sharegpt.json, 约 $((SIZE/1024/1024)) MB)"
    else
        echo "[信息] 数据集文件过小 ($SIZE 字节)，重新下载..."
        rm -f data/sharegpt.json
        download_dataset
    fi
else
    echo "[下载] 从 HuggingFace 下载 ShareGPT 数据集..."
    echo "       链接: $DATASET_URL"
    
    if command -v wget &>/dev/null; then
        wget -O data/sharegpt.json "$DATASET_URL" --progress=bar:force 2>&1
    elif command -v curl &>/dev/null; then
        curl -L -o data/sharegpt.json "$DATASET_URL"
    else
        echo "[Error] 未找到 wget 或 curl，请手动下载:"
        echo "   wget -O data/sharegpt.json $DATASET_URL"
        exit 1
    fi
    
    # 验证下载
    if [ -f "data/sharegpt.json" ]; then
        SIZE=$(stat --printf="%s" data/sharegpt.json 2>/dev/null || stat -f%z data/sharegpt.json 2>/dev/null)
        echo "[OK] 数据集下载完成: data/sharegpt.json (约 $((SIZE/1024/1024)) MB)"
    else
        echo "[Error] 数据集下载失败"
        exit 1
    fi
fi

# ─── 5. 验证安装 ──────────────────────────────────────────────
echo ""
echo "===== 验证环境 ====="
echo ""

check_version() {
    local pkg=$1
    local cmd=$2
    if python -c "$cmd" 2>/dev/null; then
        echo "  [OK] $pkg"
    else
        echo "  [FAIL] $pkg"
    fi
}

check_version "PyTorch"        "import torch; print(f'  torch {torch.__version__}')"
if python -c "import xformers" 2>/dev/null; then
    check_version "xformers"       "import xformers; print(f'  xformers {xformers.__version__}')"
else
    echo "  [跳过] xformers (未安装，阶段 2/3 不需)"
fi
if python -c "import vllm" 2>/dev/null; then
    check_version "vLLM"           "import vllm; print(f'  vllm {vllm.__version__}')"
else
    echo "  [跳过] vLLM (未安装，阶段 2/3 不需)"
fi
check_version "Transformers"   "import transformers; print(f'  transformers {transformers.__version__}')"
check_version "Datasets"       "import datasets; print(f'  datasets {datasets.__version__}')"
check_version "NumPy"          "import numpy; print(f'  numpy {numpy.__version__}')"
check_version "Pandas"         "import pandas; print(f'  pandas {pandas.__version__}')"
check_version "scikit-learn"   "import sklearn; print(f'  scikit-learn {sklearn.__version__}')"
check_version "Matplotlib"     "import matplotlib; print(f'  matplotlib {matplotlib.__version__}')"
check_version "Seaborn"        "import seaborn; print(f'  seaborn {seaborn.__version__}')"
check_version "tqdm"           "import tqdm; print(f'  tqdm {tqdm.__version__}')"
check_version "requests"       "import requests; print(f'  requests {requests.__version__}')"

# ─── 6. 显示后续步骤 ──────────────────────────────────────────
echo ""
echo "=========================================="
echo "  环境配置完成！"
echo "=========================================="
echo ""
echo "📦 已安装:"
echo "   - Python 依赖 (torch, vllm, transformers ...)"
echo "   - ShareGPT 数据集 (data/sharegpt.json)"
echo ""
echo "📁 已创建:"
echo "   - data/traces/   data/features/"
echo "   - results/figures/  results/logs/"
echo ""
echo "⚠️  还需要准备:"
echo "   1. 确保已开启 GPU 实例"
echo "   2. 下载模型 (collect_trace.py 会自动下载，或:)"
echo "       export KV_CACHE_MODEL=/path/to/your/model"
echo ""
echo "🚀 运行实验:"
echo "   bash scripts/run_all.sh"
echo ""
echo "=========================================="
