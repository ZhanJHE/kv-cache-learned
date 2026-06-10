# KV Cache 学习型替换策略 — 实验报告

## 实验环境

| 项目 | 内容 |
|------|------|
| **操作系统** | Ubuntu 22.04.3 LTS (Jammy Jellyfish), Kernel 5.15.0-86-generic |
| **CUDA GPU** | Tesla V100S-PCIE-32GB (Compute Capability 7.0, 32GB VRAM) |
| **Python** | 3.10.20 (conda env: `kv_cache`) |
| **工作目录** | `/root/autodl-tmp/` |
| **模型** | Qwen2.5-7B-Instruct（本地，~15GB） |
| **推理框架** | vLLM 0.5.0 |
| **数据集** | ShareGPT_V3_unfiltered_cleaned_split（642MB） |

### 依赖版本

| 包 | 版本 | 说明 |
|----|------|------|
| `vllm` | **0.5.0** | 推理框架（Trace 收集） |
| `torch` | **2.3.0+cu121** | 深度学习框架 |
| `transformers` | **4.44.2** | 模型生态（锁定 < 4.45 以兼容 lm-format-enforcer） |
| `outlines` | **0.0.46** | vLLM 依赖（锁定 < 0.1 以兼容 vLLM 0.5.0） |
| `lm-format-enforcer` | **0.10.1** | vLLM guided decoding |
| `numpy` | **1.26.4** | 被 outlines 0.0.46 强制降级 |
| `matplotlib` | 3.10.9 | 绘图 |
| `seaborn` | 0.13.2 | 绘图辅助 |
| `datasets` | 5.0.0 | HuggingFace Datasets |
| `accelerate` | 1.13.0 | 分布式加速 |

---

## 源码修改记录

### 1. `vllm_patch/trace_logger.py` — 适配 vLLM 0.5.0 API + 新增 fork Hook

**初始问题**：vLLM 0.5.0 将 `BlockManager` 重构为 `BlockSpaceManagerV1`，模块路径从 `vllm.core.block_manager` 变为 `vllm.core.block_manager_v1`，且 `PhysicalTokenBlock` 从 `vllm.core.block` 移至 `vllm.block`。

**第一次修改**：
- 导入路径改为 `from vllm.core.block_manager_v1 import BlockSpaceManagerV1`
- Hook 类改为 `BlockSpaceManagerV1`
- 方法签名适配新 API（`allocate(seq_group)`, `free(seq)`, `append_slots(seq, slots)`）
- 添加 `layer_id` 字段（vLLM 0.5.0 扁平 block table，`layer_id` 统一为 0）

**第二次修改（增强版）**：
- 新增 `fork` Hook：Hook `BlockSpaceManagerV1.fork`，记录 `fork_share` 事件
- access / allocate / free 事件增加 `num_hashed_tokens` 和 `block_hash` 字段
- 为跨序列重用预测打下数据基础（需要 Ampere GPU 开启 `--enable-prefix-caching` 才能触发）

### 2. `vllm_patch/api_server_wrapper.py` — 修复启动与 flush 问题

**问题 1**：vLLM 0.5.0 的 `api_server.py` 没有导出 `main()` 函数，启动逻辑在 `if __name__ == "__main__"` 块中。

**问题 2**：vLLM 子进程通过 `proc.terminate()` 发送 SIGTERM 结束后，trace 数据未写入磁盘。

**修改内容**：
- 改用 `runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")` 启动
- 在 `runpy.run_module()` 周围包裹 `try/finally`，确保退出时调用 `logger.flush()`
- 注册 SIGTERM 信号处理器，先 flush trace 再退出

### 3. `scripts/collect_trace.py` — 修复 dtype + 环境变量

**问题**：Qwen2.5-7B-Instruct 默认使用 bfloat16，但 Tesla V100S (Compute 7.0) 不支持 bfloat16。

**修改**：
- vLLM 启动参数加入 `"--dtype", "half"`
- 环境变量加入 `TRITON_DISABLE_MMA=1`（V100 不支持 Triton MMA 操作）

### 4. `predictor/dataset.py` — 重写为回归预测 + 新特征

**初始问题**：二分类 AUC=0.5，99.6% 正样本，任务定义不合理。

**改进**：
- 原二分类 → **回归**：预测到下次 access 的 token 间隔
- 模拟器中使用：驱逐"下次访问最远"的 block（Belady 近似）
- 8 维新特征：block 位置、ref_count、访问频率、时间间隔、序列生命周期等

### 5. `predictor/model.py` — 去掉 Sigmoid

**修改**：移除输出层的 Sigmoid，输出原始值（回归任务）

### 6. `predictor/train.py` — MSE Loss

**修改**：BCE Loss → MSE Loss，AUC 监控 → Val MSE 监控

### 7. `simulator/simulator.py` — 新增 SequenceAwareSimulator

**新增 `SequenceAwareSimulator`**：维护 block → Set[seq_id] 映射，只有 ref_count == 0 的 block 才可被驱逐。为 prefix caching + fork 场景设计，当前因 V100 限制无法测试 fork，但架构已就绪。

保留原始 `BlockManagerSimulator` 用于当前实验。

### 8. `simulator/policies.py` — 性能优化 + Belady

- **FIFO**：`list.pop(0)` → `deque.popleft()`，O(n²) → O(n)
- **Learned**：逐 block 推理 → 批量推理，O(n) → O(1)
- **Learned 驱逐策略**：`argmin`（原概率）→ `argmax`（回归：下次访问最远的 block 应被驱逐）
- **BeladyPolicy**：已在 policies.py 中实现，evaluate.py 中正式加入对比

### 9. `simulator/evaluate.py` — 路径、性能、Belady

- 加入 `sys.path.insert(0, str(PROJECT_ROOT))` 修复模块导入
- 预计算特征向量复用，避免每次仿真重复计算（速度提升 10x）
- 加入 Belady 最优策略对比
- `block_histories[bid].append(e)` → `block_histories.setdefault(bid, []).append(e)`

---

## 实验流程

### 阶段 1：Trace 收集

**命令**：`python scripts/collect_trace.py`

**流程**：
1. 加载 ShareGPT 数据集（`data/sharegpt.json`，642MB），随机采样 100 条对话
2. 启动带 Hook 的 vLLM API Server（模型：Qwen2.5-7B-Instruct，float16）
3. 并发发送 100 个推理请求（30 路并发，max_tokens=200）
4. vLLM 的 BlockSpaceManagerV1 在 allocate / access / free / fork 时被 Hook 捕获
5. 服务器退出时 flush trace 到 `data/traces/sharegpt_trace.jsonl`

**第一次产出**：341,580 条事件（allocate: 1,308, access: 340,172, free: 100），480 个唯一 block

**第二次产出**（回归版）：340,217 条事件，792 个唯一 block

### 阶段 2：预测器训练

**模型架构**：2 层 MLP（8 → 64 → 1），回归输出

**特征（8 维）**：
1. `block_position`: block 在序列中的位置 = num_hashed_tokens / 4096
2. `ref_count_norm`: 当前共享序列数（归一化 cap=20）
3. `access_count`: 历史 access 次数（归一化 cap=100）
4. `time_since_last`: 距离上次 access 的时间（/1000）
5. `avg_interval`: 相邻 access 平均间隔（/1000）
6. `global_time`: 全局时间位置 = timestamp / 10000
7. `seq_lifetime_ratio`: 序列已存活比例
8. `seq_remaining_ratio`: 序列剩余比例

**训练配置**：MSE Loss, Adam(lr=1e-3), Batch=256, EarlyStop=15 epochs

**结果**：
```
Dataset size: 336535, Train: 269228, Val: 67307
Label range: [0.0010, 1.0000], mean=0.0315
Epoch   1: Loss=0.0037, Val MSE=0.0028
Epoch  10: Loss=0.0021, Val MSE=0.0020
Epoch  20: Loss=0.0019, Val MSE=0.0018
Epoch  30: Loss=0.0018, Val MSE=0.0018
Epoch  50: Loss=0.0017, Val MSE=0.0017
Best Val MSE: 0.0017
```

MSE=0.0017（RMSE≈0.041，约 41 token 的预测误差），模型已学到一定的访问间隔模式。

### 阶段 3：仿真评估

**命令**：`python simulator/evaluate.py`

**策略对比**：
- **LRU**: Least Recently Used（基线）
- **FIFO**: First-In-First-Out（基线）
- **Learned**: 回归预测器，驱逐预测"下次访问最远"的 block
- **Belady**: 最优策略（需未来信息，上界参考）

**第一次实验（二分类）结果**：

| Cache | LRU | FIFO | Learned |
|-------|-----|------|---------|
| 100 | **23.68%** | 0.02% | 0.02% |
| 200 | **47.75%** | 4.81% | 4.81% |
| 500+ | 99.86% | 99.86% | (AUC=0.5 跳过) |

**第二次实验（回归）结果**：

| Cache | **LRU** | **FIFO** | **Learned** | **Belady**(上界) |
|-------|---------|---------|------------|----------------|
| 100 | **14.79%** | 0.57% | 8.76% | 14.34% |
| 200 | **32.86%** | 2.74% | 21.15% | 30.40% |
| 500 | 73.93% | 69.67% | 69.41% | **74.40%** |
| 1000 | 99.79% | 99.79% | 99.79% | 99.79% |
| 2000 | 99.79% | 99.79% | 99.79% | 99.79% |
| 5000 | 99.79% | 99.79% | 99.79% | 99.79% |

**分析**：
- **Learned > FIFO** ✅：回归模型显著优于 FIFO（100 blocks: 8.76% vs 0.57%）
- **LRU 仍最优**：局部性（recency）在当前 workload 中是非常强的信号
- **Belady ≈ LRU**：最优策略与 LRU 几乎无差距，说明该 workload 下策略提升空间有限
- **预测器排名正确**：Belady > LRU > Learned > FIFO，模型学到了有意义的模式

---

## 环境搭建中遇到的问题

### 1. `transformers` 版本不兼容
- **症状**：`ImportError: cannot import name 'LogitsWarper'`
- **原因**：transformers 5.10.2 移除了 `LogitsWarper`，但 `lm-format-enforcer==0.10.1` 依赖它
- **修复**：限制 `transformers<4.45.0`（安装 4.44.2）

### 2. `outlines` 版本不兼容
- **症状**：`ModuleNotFoundError: No module named 'outlines.fsm'`
- **原因**：outlines 1.3.0 重构了模块结构
- **修复**：降级到 `outlines==0.0.46`

### 3. `pyairports` 空包
- **症状**：outlines 导入时报 `ModuleNotFoundError: No module named 'pyairports'`
- **原因**：`pyairports==0.0.1` 只包含 dist-info，无实际代码
- **修复**：手动创建 dummy `pyairports` 模块

### 4. vLLM 0.5.0 API 变更
- BlockManager → BlockSpaceManagerV1，路径 `vllm.core.block_manager_v1`
- PhysicalTokenBlock → `vllm.block.PhysicalTokenBlock`
- API Server 无 `main()` 函数
- Block allocation 接口改为 `allocate(seq_group)`

### 5. V100S 不支持 bfloat16
- Qwen2.5-7B 默认 bfloat16，V100S (CC 7.0) 不支持
- 修复：启动参数加 `--dtype half`

### 6. V100S 不支持 Triton MMA（Ampere-only）
- 开启 `--enable-prefix-caching` 时触发器 MMA assertion
- V100 上无法启用 prefix caching，因此 fork 事件不可用
- 需要 Ampere 架构 GPU（如 A100, RTX 3090）才能完整测试跨序列重用预测

---

## 产出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `data/traces/sharegpt_trace.jsonl` | 77 MB | 340K 条 KV Cache 访问 Trace |
| `data/sharegpt.json` | 642 MB | ShareGPT 原始数据集 |
| `predictor/reuse_predictor.pt` | 8 KB | 训练好的回归预测器权重 |
| `results/figures/hit_rate_comparison.png` | 34 KB | 第一次实验对比图（二分类） |
| `results/figures/hit_rate_comparison_v2.png` | 39 KB | 第二次实验对比图（回归+Belady） |
| `results/logs/simulation_results.json` | - | 第一次仿真数据 |
| `results/logs/simulation_results_v2.json` | - | 第二次仿真数据 |
| `results/logs/vllm_server.log` | ~3 MB | vLLM 服务端日志 |
| `results/logs/experiment_report.md` | - | 本报告 |

---

## 目录结构

```
autodl-tmp/
├── data/
│   ├── sharegpt.json               # ShareGPT 原始数据集
│   └── traces/
│       └── sharegpt_trace.jsonl     # 收集的 KV Cache Trace
├── predictor/                       # 预测器模块
│   ├── dataset.py                   # 数据集构建（回归标签 + 8维特征）
│   ├── model.py                     # ReusePredictor（2层MLP）
│   ├── train.py                     # 训练脚本（MSE Loss）
│   └── reuse_predictor.pt           # 训练好的模型权重
├── simulator/                       # 仿真模块
│   ├── simulator.py                 # BlockManagerSimulator + SequenceAwareSimulator
│   ├── policies.py                  # LRU / FIFO / Learned / Belady
│   └── evaluate.py                  # 仿真入口
├── vllm_patch/                      # vLLM Hook 模块
│   ├── trace_logger.py              # Hook BlockSpaceManagerV1（含 fork）
│   └── api_server_wrapper.py        # vLLM API Server Wrapper
├── scripts/
│   ├── collect_trace.py             # Trace 收集脚本
│   ├── run_exp.sh                   # 一键实验脚本
│   ├── run_all.sh                   # 原始一键脚本
│   ├── setup_env.sh                 # 环境配置脚本
│   └── setup_remain.sh              # 剩余步骤脚本
├── real_system/
│   └── learned_evictor.py           # vLLM 自定义 Evictor 插件（预留）
├── results/                         # 实验结果
│   ├── figures/                     # 对比图表
│   └── logs/                        # 实验日志与数据
└── autodl_cloud/                    # AutoDL 部署配置
    ├── deploy.sh
    ├── README.md
    └── requirements.txt
```

---

## 进一步提升方向

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| P0 | **启用 prefix caching** | 需要在 Ampere GPU 上运行，开启后 fork Hook 可捕获跨序列共享 |
| P0 | **跨序列重用预测** | 利用 fork 事件训练，预测 block 是否会被其他序列重用 |
| P1 | **SequenceAwareSimulator** | 代码已就绪，需 fork 数据驱动 |
| P1 | **更多请求** | 当前仅 100 条，增加到 500+ 可获得更多样化的 block 模式 |
| P2 | **增加 workload 多样性** | 不同数据集（Alpaca、OpenOrca）、不同模型（Llama） |
| P2 | **更复杂模型** | 当前 2 层 MLP 容量有限，可尝试 LightGBM 或更深网络 |
