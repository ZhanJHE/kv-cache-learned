#!/bin/bash
# AutoDL 离线安装脚本
# 用法：在服务器上执行 bash autodl_cloud/offline_install.sh
# 如果 pip_packages/ 目录存在（在本地提前下载好打包上传的），就离线安装
# 否则在线安装（从清华镜像源下载）

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

PACKAGES_DIR="$PROJECT_DIR/pip_packages"
REQUIREMENTS="$PROJECT_DIR/autodl_cloud/requirements.txt"

echo "========================================"
echo "  KV Cache 学习项目 - 环境安装"
echo "========================================"

# 检查 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_NAME="kv_cache"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[✓] conda 环境 '$ENV_NAME' 已存在"
else
    echo "[...] 创建 conda 环境 '$ENV_NAME' ..."
    conda create -n $ENV_NAME python=3.10 -y
fi

conda activate $ENV_NAME

# 检查是否有离线包
if [ -d "$PACKAGES_DIR" ] && [ "$(ls -A $PACKAGES_DIR 2>/dev/null)" ]; then
    echo ""
    echo "========================================"
    echo "  检测到离线包目录: $PACKAGES_DIR"
    echo "  使用离线安装模式"
    echo "========================================"

    # 先装 torch（需要指定 CUDA 索引）
    if ls $PACKAGES_DIR/torch-2.3.0*cu121*.whl 2>/dev/null; then
        echo "安装 torch (CUDA 12.1)..."
        pip install --no-index --find-links=$PACKAGES_DIR torch==2.3.0+cu121
    elif ls $PACKAGES_DIR/torch-2.3.0*.whl 2>/dev/null; then
        echo "安装 torch (CPU 版)..."
        pip install --no-index --find-links=$PACKAGES_DIR torch==2.3.0
    else
        echo "[!] 未找到 torch 离线包，尝试在线安装..."
        pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
    fi

    # 安装其余离线包
    echo "安装其余依赖..."
    pip install --no-index --find-links=$PACKAGES_DIR -r $REQUIREMENTS
else
    echo ""
    echo "========================================"
    echo "  未检测到离线包，使用在线安装模式"
    echo "  镜像源: 清华 TUNA"
    echo "========================================"
    echo ""
    echo "[提示] 下载约 5 GB，网速 10 MB/s 时约需 8-10 分钟"
    echo ""

    # 在线安装
    pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
    pip install -r $REQUIREMENTS
fi

echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "验证安装："
python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import vllm; print('vllm', vllm.__version__)"

echo ""
echo "启动实验："
echo "  bash scripts/run_all.sh"
echo "========================================"
