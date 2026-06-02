# 部署与环境配置指南

本文档说明如何在 AutoDL 云端环境（或其他 CUDA Linux 服务器）部署本项目并配置运行所需依赖。

---

## 一、运行环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Linux (Ubuntu 20.04+) | Ubuntu 22.04 |
| Python | 3.10 | 3.10 |
| CUDA | 11.8 | 12.1 |
| GPU 显存 | 16 GB | 24 GB (RTX 3090 / A100 40G) |
| 硬盘空间 | 50 GB | 100 GB（含模型与数据集） |
| 网络 | 可访问 HuggingFace / ModelScope | 可访问 HuggingFace |

> **注意**：本项目当前阶段主要在 **AutoDL** 平台验证，若在其他环境运行，请将脚本中的绝对路径 `/root/autodl-tmp/kv_cache_project/` 替换为实际路径。

---

## 二、快速开始（一键部署）

在终端执行以下命令：

```bash
# 1. 克隆仓库（若在 AutoDL 已挂载数据盘，建议放到 /root/autodl-tmp/）
cd /root/autodl-tmp/
git clone <your-repo-url> kv_cache_project
cd kv_cache_project

# 2. 创建 Conda 环境
conda create -n kv_cache python=3.10 -y
conda activate kv_cache

# 3. 安装依赖
pip install -r autodl_cloud/requirements.txt

# 4. 初始化目录结构
mkdir -p data/traces data/features results/figures results/logs
```

---

## 三、详细步骤

### 3.1 创建 Conda 环境

```bash
conda create -n kv_cache python=3.10 -y
conda activate kv_cache
```

> 建议使用 Conda 隔离环境，避免与系统 Python 或其他项目冲突。

### 3.2 安装 Python 依赖

依赖清单已整理在 `autodl_cloud/requirements.txt` 中：

```text
vllm==0.5.0
torch==2.1.2
transformers
datasets
accelerate
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
```

安装命令：

```bash
pip install -r autodl_cloud/requirements.txt
```

> **版本锁定说明**：`vllm==0.5.0` 与 `torch==2.1.2` 为硬编码版本。Trace Hook 逻辑依赖 vLLM 特定版本的内部 API（`BlockManager`、`PhysicalTokenBlock`），升级可能导致 Trace 收集失效。

### 3.3 初始化目录结构

项目运行时需要以下目录存放数据、模型与结果：

```bash
mkdir -p data/traces      # KV Cache 访问 Trace（JSONL）
mkdir -p data/features    # 预测器训练数据（CSV / Parquet）
mkdir -p results/figures  # 实验图表产出
mkdir -p results/logs     # 训练与模拟日志
```

若后续按 `exp_plan.md` 补充源码，完整目录结构应为：

```
kv_cache_project/
├── autodl_cloud/
│   └── requirements.txt
├── data/
│   ├── sharegpt.json
│   ├── traces/
│   └── features/
├── vllm_patch/
├── predictor/
├── simulator/
├── real_system/
├── scripts/
└── results/
    ├── figures/
    └── logs/
```

### 3.4 下载数据集与模型（按需）

#### 数据集：ShareGPT

```bash
# 方式一：通过 HuggingFace
cd data
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O sharegpt.json

# 方式二：通过 ModelScope（国内镜像）
wget https://www.modelscope.cn/datasets/baicai/ShareGPT/resolve/master/ShareGPT_V3_unfiltered_cleaned_split.json -O sharegpt.json
```

#### 模型：Llama-2-7B-chat / Qwen2-7B-Instruct

首次运行时，vLLM 会自动从 HuggingFace 缓存模型到 `~/.cache/huggingface/`。若网络受限，可提前通过 `huggingface-cli` 或 `modelscope` 下载：

```bash
# HuggingFace
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./models/llama-2-7b-chat

# ModelScope
pip install modelscope
modelscope download --model qwen/Qwen2-7B-Instruct --local_dir ./models/qwen2-7b-instruct
```

---

## 四、验证安装

依次执行以下检查，确保环境就绪：

```bash
# 1. Python 版本
python --version  # 期望：Python 3.10.x

# 2. vLLM 安装
python -c "import vllm; print(vllm.__version__)"  # 期望：0.5.0

# 3. PyTorch + CUDA 可用
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
# 期望：True, 11.8 或 12.1

# 4. GPU 显存识别
nvidia-smi
```

---

## 五、各阶段运行命令

环境配置完成后，可按如下顺序执行实验：

| 阶段 | 命令 | 说明 |
|------|------|------|
| 1. 收集 Trace | `python scripts/collect_trace.py` | Hook vLLM BlockManager，生成 `data/traces/sharegpt_trace.jsonl` |
| 2. 训练预测器 | `cd predictor && python train.py && cd ..` | 训练 `ReusePredictor`，保存权重到 `results/models/` |
| 3. 模拟器评估 | `cd simulator && python evaluate.py && cd ..` | 对比 LRU / FIFO / Learned / Belady 命中率 |
| 一键执行 | `bash scripts/run_all.sh` | 顺序执行 1→2→3（脚本需提前创建） |

> **提示**：当前仓库处于实验规划阶段，上述脚本需根据 `exp_plan.md` 中的代码蓝图实际编写后才能运行。

---

## 六、配置文件与路径说明

本项目在脚本顶部使用**全大写常量**管理路径，默认指向 `/root/autodl-tmp/kv_cache_project/`。若需迁移到其他机器，请全局替换以下前缀：

```python
# 示例（各脚本中常见）
PROJECT_ROOT = "/root/autodl-tmp/kv_cache_project"
TRACE_PATH = f"{PROJECT_ROOT}/data/traces/sharegpt_trace.jsonl"
MODEL_SAVE = f"{PROJECT_ROOT}/results/models/reuse_predictor.pt"
```

建议替换为相对路径或环境变量形式：

```python
import os
PROJECT_ROOT = os.environ.get("KV_CACHE_PROJECT", "/root/autodl-tmp/kv_cache_project")
```

---

## 七、故障排查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| `pip install vllm==0.5.0` 失败 | CUDA 版本不匹配 | 检查 `nvcc -V`，使用与 CUDA 对应的 torch 轮子，或升级 pip |
| `import vllm` 报错 | GCC 版本过低 | Ubuntu 22.04 默认 GCC 11 足够；20.04 需升级 GCC |
| HuggingFace 下载超时 | 网络限制 | 使用 ModelScope 镜像，或配置 HF-Mirror |
| GPU OOM | 显存不足 | 降低模型规模（如改用 Qwen2-1.8B），或减少并发请求数 |
| Trace 文件为空 | Hook 未生效 | 确认 vLLM 版本严格为 0.5.0，检查日志中是否有 Monkey Patch 提示 |

---

## 八、许可证

本项目采用 GNU General Public License v3（GPL v3）。部署、修改或集成到 vLLM 源码时，请遵守相应的源码开放与许可证传递义务。
