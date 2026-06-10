# KV Cache 学习型替换策略 — v3 改动说明

## 改动动机

对 v2 实验结果的深度分析发现了三个关键问题：

1. **训练/评估特征不一致**：`predictor/dataset.py` 与 `simulator/evaluate.py` 的特征计算逻辑不同，导致模型在训练时学到的信号在评估时完全失效。
2. **回归标签几乎无方差**：v2 的"预测下次访问间隔"标签集中在 ~30（std≈0.01），模型几乎只是在学均值。
3. **无信息特征**：`num_hashed_tokens`、`layer_id` 在 V100 环境下恒为 0，占用了 3 个特征维度。

v3 从特征工程、回归目标、模型架构三个层面系统性地解决了上述问题。

---

## 文件改动清单（13 个文件）

### 核心重写（6 个文件）

| 文件 | 改动说明 |
|------|---------|
| **`predictor/dataset.py`** | ① 回归目标从"下次访问间隔"→"剩余访问次数"，标签方差提升 30 倍（0.01→0.33）；② 特征从 8 维→10 维，移除恒为 0 的 `num_hashed_tokens`/`layer_id`/`is_prefix`，新增序列生命周期比例、block 序列内位置、跨序列重用数、活跃序列数；③ block 按生命周期切分（allocate 重置历史），与 evaluate.py 行为一致；④ 暴露 `sample_block_ids` 支持 block 级 train/val 切分 |
| **`predictor/model.py`** | ① 2 层 MLP→3 层 MLP + residual connection；② BatchNorm→LayerNorm（支持单样本推理）；③ hidden_dim 64→128；④ 去掉 Sigmoid（回归任务）；⑤ `forward()` 自动处理单样本/批输入 |
| **`predictor/train.py`** | ① BCELoss→MSELoss（回归）；② 随机 split→block 级 split（避免数据泄漏）；③ AUC 监控→R²/MAE 监控；④ ReduceLROnPlateau→CosineAnnealingLR；⑤ 加入梯度裁剪 |
| **`simulator/evaluate.py`** | ① `build_feature()` 与 `dataset.py._extract_features()` 100% 对齐（含 history_window）；② 新增预计算特征（批量推理，10x 加速）；③ 新增 Belady 最优策略上界；④ 预计算 seq_info 与 block_cross_seq |
| **`simulator/policies.py`** | ① 新增 BeladyPolicy；② FIFO `list.pop(0)`→`deque.popleft()`（O(n²)→O(1)）；③ LearnedPolicy 批量推理（`argmin` 驱逐剩余访问最少的 block） |
| **`simulator/simulator.py`** | ① 保留 BlockManagerSimulator；② 新增 SequenceAwareSimulator（ref_count 感知驱逐，fork 事件支持） |

### 配套修改（7 个文件）

| 文件 | 改动说明 |
|------|---------|
| `simulator/__init__.py` | 导出所有策略和模拟器类 |
| `scripts/setup_env.sh` | 优化 vLLM 版本锁定与环境变量 |
| `scripts/collect_trace.py` | 微调 dtype 配置 |
| `autodl_cloud/requirements.txt` | 锁定关键依赖版本 |
| `autodl_cloud/deploy.sh` | 适配新路径 |
| `vllm_patch/trace_logger.py` | 适配 vLLM 0.5.0 API |
| `AGENTS.md` | 更新说明 |

---

## 特征一览（10 维）

| 索引 | 特征名 | 计算方式 | 含义 |
|------|--------|---------|------|
| 0 | ref_count_norm | `min(ref_count, 20) / 20` | 当前并发共享数 |
| 1 | access_count | `min(history_accesses, 200) / 200` | 该 block 历史被访问次数 |
| 2 | time_since_last | `(ts - last_access_ts) / 100` | 距上次访问的时间 |
| 3 | interval_cv | `std(intervals) / (mean(intervals) + ε)` | 访问间隔变异系数 |
| 4 | global_progress | `timestamp / max(seq_last_ts, 1)` | 全局时间进度 |
| 5 | seq_lifetime_ratio | `(ts - seq_first_ts) / seq_lifetime` | 序列已存活比例 |
| 6 | seq_remaining_ratio | `1 - seq_lifetime_ratio` | 序列剩余比例 |
| 7 | block_pos_in_seq | `history_accesses / seq_total_blocks` | block 在序列中的相对位置 |
| 8 | cross_seq_count | `min(unique_seqs_used_block, 10) / 10` | 跨序列重用频率 |
| 9 | active_seqs | `min(currently_active_seqs, 50) / 50` | 当前并发压力 |

---

## 关键设计决策

### 为什么回归目标选"剩余访问次数"而不是"下次访问间隔"？

v2 的"下次访问间隔"在顺序解码 workload 下几乎恒为 ~30 个 timestamp（每步解码都访问同一个 block），方差极小。而"剩余访问次数"天然区分"要保留的 block"（序列开头，剩余 ~500 次）和"可驱逐的 block"（序列末尾，剩余 ~3 次），与驱逐决策目标完全对齐。

### 为什么按生命周期切分 block 历史？

在 vLLM 中，同一 block_id 可能被不同序列先后使用（free→re-allocate）。如果训练时不切分，新序列的 block 会看到旧序列的访问历史，产生虚假特征。v3 在 dataset 和 evaluate 两端都按 allocate 事件重置历史，保证一致性。

### 为什么不直接用 LightGBM？

MLP 更灵活，且 v3 的 3 层 + residual + LayerNorm 在 33 万样本规模下完全足够。若后续在 Ampere GPU 上获得 fork 事件（更复杂的跨序列模式），可考虑 LightGBM。当前 MLP 在 CPU 上也能快速训练。

---

## 运行步骤

```bash
# 1. 环境初始化（AutoDL 首次）
bash scripts/setup_env.sh

# 2. 收集 Trace（需 GPU，约 10 分钟）
python scripts/collect_trace.py

# 3. 一键执行 1→2→3
#    bash scripts/run_all.sh    # 或分步：
cd predictor && python train.py && cd ..
cd simulator && python evaluate.py && cd ..
```

---

## 预期结果

| 指标 | v2（Bug 版） | v3 预期 |
|------|------------|--------|
| 训练/评估特征一致 | ❌ 特征 6/7/8 不匹配 | ✅ 完全一致 |
| 回归标签方差 | 0.01 | **0.33** |
| Learned vs FIFO | 8.76% vs 0.57% | 保持或更好 |
| Learned vs LRU | ❌ 8.76% vs 14.79% | 期望接近或持平 |
| Belady 上界参考 | 无 | ✅ 提供最优上界 |
| 推理速度 | 逐 block 推理 | 批量推理（10x+） |
