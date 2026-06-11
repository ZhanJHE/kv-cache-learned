# KV Cache 学习型替换策略 — 最终实验报告

## 实验目标

验证学习型缓存替换策略（Learned）相比传统 LRU/FIFO 在 KV Cache 场景下的命中率优势。

## 实验环境

| 项目 | 内容 |
|------|------|
| GPU | Tesla V100S-PCIE-32GB (CC 7.0) |
| 模型 | Qwen2.5-7B-Instruct (float16) |
| 推理框架 | vLLM 0.5.0 |
| 数据集 | ShareGPT (642MB, 100 conversations) |
| Trace | 317,380 条 KV Cache 访问事件 |
| Python | 3.10.20, conda env `kv_cache` |

## 版本演化

| 版本 | 特征 | 预测目标 | 模型 | Learned(100) | 超越LRU? | 说明 |
|------|------|---------|------|-------------|---------|------|
| v1 | 8维(含layer_id) | 二分类 | 2层MLP | 0.02% | ❌ | 基线，AUC=0.5 |
| v2 | 8维(新) | 回归(间隔) | 2层MLP | 8.76% | ❌ | 标签方差0.01 |
| v3 | **10维**(含last_ts) | 回归(剩余次数) | 3层MLP+Res | **20.24%** | ✅ | **有数据泄露** |
| v4 | 8维(去泄露) | 回归(剩余次数) | 3层MLP+Res | 18.72% | ✅ | 去未来泄露 |
| **v5** | **8维(对数+趋势)** | 回归(剩余次数) | 3层MLP+Res | **19.73%** | ✅ | **最终版** |

## 最终结果 (v5)

| Cache | LRU | FIFO | **Learned** | Belady |
|-------|-----|------|------------|--------|
| 100 | 14.76% | 1.08% | **19.73%** | 15.82% |
| 200 | 34.40% | 1.84% | **38.28%** | 32.32% |
| 500 | 78.85% | 79.02% | **86.88%** | 75.17% |
| 1000 | 99.77% | 99.77% | 99.77% | 99.77% |

## 核心结论

1. **学习型策略真实有效** — v5 Learned 以 +4.97pp 显著超越 LRU，无数据泄露
2. **特征演化**：从 8维→10维→8维，最终 8 维特征全部为纯在线可计算信息
3. **Belady 实现有局限** — future_trace 含 allocate 事件，导致其不是纯最优上界
4. **完整版本链验证了每一步改进的效果**

## 特征设计 (v5 最终版)

| # | 名称 | 计算 | 说明 |
|---|------|------|------|
| F0 | ref_count_norm | min(ref_count,20)/20 | 当前共享数 |
| F1 | access_count_norm | min(hist_access,200)/200 | 历史访问量 |
| F2 | time_since_last | (ts-last)/100 | 距上次访问 |
| F3 | interval_cv | std/mean | 间隔变异系数 |
| F4 | log_access_count | log2(cnt+1)/log2(201) | 对数访问量 |
| F5 | interval_trend | recent_mean/all_mean | 加速/减速趋势 |
| F6 | tail_density | tail_span/all_span | 尾部集中度 |
| F7 | free_events_seen | min(free_count,5)/5 | 生命周期释放数 |

## 文件结构

```
├── predictor/
│   ├── dataset.py          # 数据集 (8维特征，回归标签)
│   ├── model.py            # 3层MLP+Residual+LayerNorm
│   ├── train.py            # 训练脚本
│   └── reuse_predictor.pt  # 训练好的模型权重
├── simulator/
│   ├── simulator.py        # BlockManagerSimulator
│   ├── policies.py         # LRU/FIFO/Learned/Belady
│   └── evaluate.py         # 仿真评估入口
├── vllm_patch/
│   ├── trace_logger.py     # vLLM Hook (含 fork)
│   └── api_server_wrapper.py
├── scripts/
│   ├── collect_trace.py    # Trace 收集
│   ├── run_v5.sh           # v5 实验脚本
│   └── run_v4.sh           # v4 实验脚本
├── data/traces/sharegpt_trace.jsonl  # 317K Trace
├── results/
│   ├── figures/hit_rate_comparison.png  # 对比图
│   └── logs/
│       ├── simulation_results.json    # v5 最终数据
│       ├── v5_summary.md              # 本报告
│       ├── v4_summary.md
│       ├── v3_summary.md
│       ├── experiment_report.md       # 完整报告
│       └── requirements_frozen.txt    # 环境依赖
└── autodl_cloud/
    └── requirements.txt    # 关键依赖
```
