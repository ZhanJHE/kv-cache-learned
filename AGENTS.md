# kv-cache-learned

> AI 编码助手阅读指南。本文件面向对项目零了解的智能体，请优先阅读本文件再操作代码库。

---

## 项目概述

本项目是**面向大模型推理的 KV Cache 学习型替换策略**研究仓库，目标是通过轻量预测器估计缓存块未来重用概率，在显存受限场景下优于传统 LRU 策略。

**当前状态**：代码已实现（阶段 1→3 完整源码 + 阶段 4 预留），可直接上传至 AutoDL 等 Linux CUDA 环境运行。原始蓝图仍保留在 `exp_plan.md` 中供对照。

- **目标平台**：AutoDL（Linux, CUDA, PyTorch 预装环境）
- **推理框架**：vLLM ≥ 0.5.0
- **实验模型**：Llama-2-7B-chat / Qwen2-7B-Instruct
- **数据集**：ShareGPT（用于模拟多租户负载）
- **许可证**：GNU General Public License v3（见 `LICENSE`）

---

## 仓库结构

```
kv-cache-learned/
├── AGENTS.md                # 本文件
├── README.md                # 项目简介（中文）
├── exp_plan.md              # 完整实验规划文档（~35 KB，含原始蓝图）
├── DEPLOYMENT.md            # AutoDL 云端部署与环境配置指南
├── LICENSE                  # GPL v3
├── autodl_cloud/
│   └── requirements.txt     # Python 依赖清单
├── data/
│   ├── traces/              # KV Cache 访问 Trace（JSONL，运行后生成）
│   └── features/            # 预测器训练数据（预留）
├── vllm_patch/
│   ├── __init__.py
│   ├── trace_logger.py      # 无侵入式 Hook vLLM BlockManager
│   └── api_server_wrapper.py # 带 Hook 启动 vLLM API Server（子进程包装器）
├── predictor/
│   ├── __init__.py
│   ├── model.py             # ReusePredictor（2 层 MLP）
│   ├── dataset.py           # KVCacheDataset（8 维特征构造）
│   └── train.py             # 训练脚本（含早停）
├── simulator/
│   ├── __init__.py
│   ├── policies.py          # LRU / FIFO / Learned / Belady
│   ├── simulator.py         # BlockManagerSimulator
│   └── evaluate.py          # 命中率对比与绘图
├── real_system/
│   ├── __init__.py
│   └── learned_evictor.py   # vLLM 自定义 Evictor 插件（Tier 2 / 可选）
├── scripts/
│   ├── setup_env.sh         # 环境初始化脚本
│   ├── collect_trace.py     # Trace 收集入口（阶段 1）
│   └── run_all.sh           # 一键执行 1→2→3
└── results/
    ├── figures/             # 实验图表产出
    └── logs/                # 实验日志
```

> **注意**：`data/traces/` 与 `results/` 下的具体文件需在 AutoDL 环境运行后生成。

---

## 技术栈与运行时架构

| 层级 | 技术/工具 | 说明 |
|------|-----------|------|
| 语言 | Python 3.10 | 全部脚本与模型代码 |
| 深度学习 | PyTorch 2.3.0 | 预测器训练与推理 |
| 推理框架 | vLLM 0.5.0 | 提供 LLM serving 与 BlockManager Hook 点 |
| 模型生态 | transformers, datasets, accelerate | 模型下载与数据加载 |
| 数据科学 | numpy, pandas, scikit-learn | 特征工程与指标计算 |
| 可视化 | matplotlib, seaborn | 实验图表产出 |
| 其他 | tqdm, requests | 进度条与 HTTP 请求 |

---

## 构建与运行命令

### 环境准备（AutoDL）

```bash
conda create -n kv_cache python=3.10 -y
conda activate kv_cache
pip install -r autodl_cloud/requirements.txt
```

### 路径说明

各脚本已改用相对路径或 `pathlib` 自动推导项目根目录，不再强制依赖 `/root/autodl-tmp/`：
- `PROJECT_ROOT = Path(__file__).resolve().parent.parent`
- 模型路径可通过环境变量 `KV_CACHE_MODEL` 覆盖。

### 各阶段执行

```bash
# 阶段 1：收集 Trace
python scripts/collect_trace.py

# 阶段 2：训练预测器
cd predictor && python train.py && cd ..

# 阶段 3：模拟器评估
cd simulator && python evaluate.py && cd ..

# 一键执行
bash scripts/run_all.sh
```

---

## 代码风格与开发约定

- **语言**：所有注释、文档字符串、变量命名均采用**中文语境**（如 `event_type="allocate"`，类名 `KVTraceLogger`，注释使用中文）。
- **类型注解**：广泛使用 Python `typing`（`Dict`, `List`, `Optional` 等）。
- **配置**：脚本顶部使用全大写常量（`TRACE_PATH`, `MODEL_SAVE`, `BATCH_SIZE` 等），但优先通过 `pathlib` 推导或环境变量覆盖。
- **Hook 风格**：通过动态替换类方法（Monkey Patch）实现无侵入式 Trace 收集，避免直接修改 vLLM 源码（Tier 1）。Tier 2 才需要手动修改 vLLM 源码。
- **模型保存**：使用 `torch.save(model.state_dict(), ...)` 只存权重，不存完整模型。

---

## 测试策略

当前仓库**无自动化单元测试**。`exp_plan.md` 中提供了实验检查清单（Checklist）作为人工验证标准：

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
2. **vLLM 版本锁定**：明确锁定 `vllm==0.5.0`，因为 Hook 逻辑依赖特定版本的内部 API（`BlockManager`, `PhysicalTokenBlock`）。升级 vLLM 可能导致 Hook 失效。
3. **无秘密管理**：当前设计中未涉及 API Key、数据库密码等敏感信息；若未来扩展需注意避免将 HuggingFace Token 等硬编码入仓库。

---

## 给 AI 助手的操作建议

- **若用户要求实现代码**：核心源码已全部生成，请按 `exp_plan.md` 检查是否遗漏；如需修改特征维度或模型结构，请同步更新 `predictor/` 与 `simulator/` 中的对应实现，并保持 `exp_plan.md` 一致。
- **若用户要求运行实验**：当前环境为 Windows，vLLM 相关代码（Trace 收集、真实系统集成）需在 Linux+CUDA 环境（如 AutoDL）运行。模拟器与预测器训练脚本理论上可在 CPU 环境测试，但建议上传至目标环境后统一执行 `bash scripts/run_all.sh`。
- **若用户要求修改设计**：任何对模块划分、特征维度、模型结构的改动都应同步更新 `exp_plan.md`，以保持文档与实现一致。
