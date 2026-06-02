# kv-cache-learned

> AI 编码助手阅读指南。本文件面向对项目零了解的智能体，请优先阅读本文件再操作代码库。

---

## 项目概述

本项目是**面向大模型推理的 KV Cache 学习型替换策略**研究仓库，目标是通过轻量预测器估计缓存块未来重用概率，在显存受限场景下优于传统 LRU 策略。

**当前状态**：本项目目前为**实验规划阶段**，尚未包含实际可运行的源代码。所有设计细节、模块划分和代码蓝图均记录在 `exp_plan.md` 中。

- **目标平台**：AutoDL（Linux, CUDA, PyTorch 预装环境）
- **推理框架**：vLLM ≥ 0.5.0
- **实验模型**：Llama-2-7B-chat / Qwen2-7B-Instruct
- **数据集**：ShareGPT（用于模拟多租户负载）
- **许可证**：GNU General Public License v3（见 `LICENSE`）

---

## 仓库结构

当前仓库仅包含以下文件，**不存在 `pyproject.toml`、`requirements.txt`、`setup.py` 等常规构建配置文件**：

```
kv-cache-learned/
├── AGENTS.md          # 本文件
├── README.md          # 项目简介（中文）
├── exp_plan.md        # 完整实验规划文档（含代码蓝图、执行步骤、风险应对）
└── LICENSE            # GPL v3
```

### 关键文件说明

- **`exp_plan.md`**（~35 KB）：唯一的核心文档。内含四阶段实验的完整设计：
  1. **Trace 收集**：通过 Hook vLLM `BlockManager` 记录 `allocate`/`access`/`evict`/`free` 事件；
  2. **预测器训练**：基于 Trace 构建 8 维特征，训练轻量 2 层 MLP（`ReusePredictor`）做二分类（未来是否被重用）；
  3. **Trace-driven 模拟器**：实现 LRU、FIFO、Learned、Belady 四种驱逐策略，对比命中率；
  4. **真实系统集成**：将 Learned Evictor 手动集成到 vLLM 源码（Tier 2 / 可选加分项）。

  文档中还包含预设目录结构、环境安装命令、`scripts/run_all.sh` 一键脚本、实验检查清单与风险应对表。

- **`README.md`**：一句话概括项目目标与技术路线。

- **`LICENSE`**：GPL v3 全文。

---

## 技术栈与运行时架构（规划）

根据 `exp_plan.md`，项目计划采用如下技术栈：

| 层级 | 技术/工具 | 说明 |
|------|-----------|------|
| 语言 | Python 3.10 | 全部脚本与模型代码 |
| 深度学习 | PyTorch 2.1.2 | 预测器训练与推理 |
| 推理框架 | vLLM 0.5.0 | 提供 LLM  serving 与 BlockManager Hook 点 |
| 模型生态 | transformers, datasets, accelerate | 模型下载与数据加载 |
| 数据科学 | numpy, pandas, scikit-learn | 特征工程与指标计算 |
| 可视化 | matplotlib, seaborn | 实验图表产出 |
| 其他 | tqdm, requests | 进度条与 HTTP 请求 |

### 计划中的模块划分

```
/root/autodl-tmp/kv_cache_project/        # 文档中假设的部署根目录
├── data/
│   ├── sharegpt.json                     # 原始数据集
│   ├── traces/                           # KV Cache 访问 Trace（JSONL）
│   └── features/                         # 预测器训练数据
├── vllm_patch/
│   ├── trace_logger.py                   # 无侵入式 Hook vLLM BlockManager
│   └── block_manager_v2.py               # 扩展 BlockManager（预留）
├── predictor/
│   ├── model.py                          # ReusePredictor（2 层 MLP）
│   ├── dataset.py                        # KVCacheDataset（8 维特征构造）
│   └── train.py                          # 训练脚本（含早停）
├── simulator/
│   ├── policies.py                       # LRU / FIFO / Learned / Belady
│   ├── simulator.py                      # BlockManagerSimulator
│   └── evaluate.py                       # 命中率对比与绘图
├── real_system/
│   └── learned_evictor.py              # vLLM 自定义 Evictor 插件
├── scripts/
│   ├── setup_env.sh
│   ├── collect_trace.sh
│   ├── run_simulation.sh
│   ├── run_benchmark.sh
│   └── run_all.sh                        # 一键执行三阶段实验
└── results/
    ├── figures/                          # 命中率对比图、ROC 曲线等
    └── logs/                             # 实验日志
```

> **注意**：以上目录和文件目前**均不存在于本仓库**中，仅在 `exp_plan.md` 中以代码块形式提供蓝图。若后续实际创建，应同步更新本文件。

---

## 构建与运行命令（规划）

由于目前无实际源码，以下命令来自 `exp_plan.md` 的设计，供未来实施参考：

### 环境准备

```bash
conda create -n kv_cache python=3.10 -y
conda activate kv_cache
pip install vllm==0.5.0 torch==2.1.2 transformers datasets accelerate \
            numpy pandas matplotlib seaborn scikit-learn tqdm
```

### 各阶段执行

```bash
# 阶段 1：收集 Trace
python scripts/collect_trace.py

# 阶段 2：训练预测器
cd predictor && python train.py && cd ..

# 阶段 3：模拟器评估
cd simulator && python evaluate.py && cd ..

# 一键执行（规划中）
bash scripts/run_all.sh
```

---

## 代码风格与开发约定

根据 `exp_plan.md` 中的代码蓝图，项目遵循以下风格：

- **语言**：所有注释、文档字符串、变量命名均采用**中文语境**（如 `event_type="allocate"`，类名 `KVTraceLogger`，注释使用中文）。
- **类型注解**：广泛使用 Python `typing`（`Dict`, `List`, `Optional` 等）。
- **配置硬编码**：脚本顶部使用全大写常量（`TRACE_PATH`, `MODEL_SAVE`, `BATCH_SIZE` 等），路径默认指向 `/root/autodl-tmp/kv_cache_project/`。
- ** Hook 风格**：通过动态替换类方法（Monkey Patch）实现无侵入式 Trace 收集，避免直接修改 vLLM 源码（Tier 1）。Tier 2 才需要手动修改 vLLM 源码。
- **模型保存**：使用 `torch.save(model.state_dict(), ...)` 只存权重，不存完整模型。

---

## 测试策略

当前仓库**无测试代码**。`exp_plan.md` 中提供了实验检查清单（Checklist）作为人工验证标准：

| 检查项 | 通过标准 |
|--------|---------|
| vLLM 安装 | `python -c "import vllm; print(vllm.__version__)"` 无报错 |
| GPU 可识别 | `nvidia-smi` 显示显存 |
| Trace 收集 | `sharegpt_trace.jsonl` 行数 > 1000 |
| 事件类型完整 | 包含 allocate / access / evict / free |
| 预测器收敛 | Val AUC > 0.65 |
| 模拟器 LRU | 不同 Block 数下命中率单调递增 |
| Learned 优于 LRU | 至少一个配置下命中率提升 > 5% |
| 真实系统（可选） | vLLM 能成功响应请求 |

---

## 安全与合规注意事项

1. **GPL v3 许可证**：本项目采用 GPL v3。若后续集成到 vLLM 源码或分发二进制，需遵守相应源码开放与许可证传递义务。
2. **路径硬编码风险**：`exp_plan.md` 中的大量脚本使用绝对路径 `/root/autodl-tmp/...`，在 AutoDL 之外的环境运行时需要批量替换。
3. **vLLM 版本锁定**：明确锁定 `vllm==0.5.0`，因为 Hook 逻辑依赖特定版本的内部 API（`BlockManager`, `PhysicalTokenBlock`）。升级 vLLM 可能导致 Hook 失效。
4. **无秘密管理**：当前设计中未涉及 API Key、数据库密码等敏感信息；若未来扩展需注意避免将 HuggingFace Token 等硬编码入仓库。

---

## 给 AI 助手的操作建议

- **若用户要求实现代码**：请优先阅读 `exp_plan.md`，其中每个模块都有可直接转写的 Python 代码块。按阶段 1→2→3→4 顺序实施最符合原设计意图。
- **若用户要求运行实验**：当前仓库无实际代码，需先根据 `exp_plan.md` 生成源码和目录结构，并确保运行在具备 CUDA + vLLM 的 Linux 环境（如 AutoDL）。
- **若用户要求修改设计**：任何对模块划分、特征维度、模型结构的改动都应同步更新 `exp_plan.md`，以保持文档与实现一致。
