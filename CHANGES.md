# KV Cache 学习型替换策略 — 版本变动日志

## v5.0 (最终版) — 2026-06

### 改动

- **修复 free 事件被当作 access 的 Bug**：`evaluate.py` 新增 `sim_mask` 机制，free 事件仅用于特征预计算（F7 `free_events_seen`），仿真流过滤为 access+allocate
- **特征重新设计（8 维）**：
  - F4: `relative_block_idx` → `log_access_count`（对数尺度，防止饱和）
  - F5: `generation_speed` → `interval_trend`（recent_mean/all_mean，捕获加速/减速）
  - F6: `recent_freq_ratio` → `tail_density`（tail_span/all_span，反映生命周期阶段）
  - F7: `active_seqs_online` → `free_events_seen`（生命周期内 free 次数）

### 结果

| Cache | LRU | FIFO | Learned | vs LRU |
|-------|-----|------|---------|--------|
| 100 | 14.76% | 1.08% | **19.73%** | +4.97pp |
| 200 | 34.40% | 1.84% | **38.28%** | +3.88pp |
| 500 | 78.85% | 79.02% | **86.88%** | +8.03pp |

---

## v4.0 — 2026-06

### 改动

- **去除数据泄露**：删除 6 个依赖未来信息（`last_ts`、`total_blocks`）的特征
- 替换为 4 个纯在线特征：`relative_block_idx`、`generation_speed`、`recent_freq_ratio`、`active_seqs_online`
- 特征维度 10→8

### 问题

- free 事件被当作 access 处理（仿真统计偏差）
- F5/F6 特征方差极低，几乎无区分度
- Belady 实现有局限（future_trace 含 allocate 混入）

### 结果

Learned 仍以 +4.22pp 超越 LRU @100 blocks，但比 v3 下降 ~1.5pp（泄露溢价消除）

---

## v3.0 — 2026-06

### 改动

- 回归目标从"下次访问间隔"→"剩余访问次数"（标签方差 0.01→0.33，提升 33×）
- 特征 8→10 维：移除恒为 0 的 num_hashed_tokens/layer_id/is_prefix，新增序列生命周期比例等
- 模型升级：2 层 MLP → 3 层 MLP + Residual + LayerNorm（R²=0.585）
- 修复 dataset.py 与 evaluate.py 特征不一致 Bug
- Block 级 train/val split、CosineAnnealing、梯度裁剪
- FIFO→deque，Learned 批量推理，新增 Belady 上界、SequenceAwareSimulator

### 问题

- 10 维特征中 6 个依赖 `last_ts`/`total_blocks`（未来信息泄露，命中率高估 ~1.5-2.4pp）
- Belady 低于 LRU（实现 Bug）

### 结果

首次实现 Learned > LRU（20.24% vs 14.76% @100 blocks）

---

## v2.0 — 2026-06（v1→v2 重写）

- 二分类→回归（下次访问间隔），8 维新特征
- MSE Loss，去 Sigmoid，Belady 上界
- **Bug**：dataset.py 与 evaluate.py 特征计算不一致（3/8 特征不匹配）
- **Bug**：回归标签几乎无方差（访问间隔 ≈30±1）
- Learned: 8.76% vs LRU: 14.79% @100 blocks

---

## v1.0 — 2026-06（原始版本）

- 二分类预测"AUC=0.5, 99.6% 正样本"，任务定义不合理
- 8 维原始特征（含 layer_id, num_tokens, is_prefix）
- 2 层 MLP + Sigmoid + BCELoss
