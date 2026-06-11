import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import torch

# 将项目根目录加入 sys.path（在脚本目录之前），确保 simulator 包优先
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(1, str(_SCRIPT_DIR))

from simulator import BlockManagerSimulator
from simulator.policies import LRUPolicy, FIFOPolicy, LearnedPolicy, BeladyPolicy

# 配置
PROJECT_ROOT = _PROJECT_ROOT
TRACE_PATH = PROJECT_ROOT / "data" / "traces" / "sharegpt_trace.jsonl"
PREDICTOR_PATH = PROJECT_ROOT / "predictor" / "reuse_predictor.pt"
RESULT_DIR = PROJECT_ROOT / "results"
HISTORY_WINDOW = 200


def build_feature(event, history):
    """
    从 trace 事件和 block 历史构建特征。

    **必须与 predictor/dataset.py 的 _extract_features 完全一致！**
    **所有特征仅使用当前及过去信息，无未来泄露**
    """
    hist = history[-HISTORY_WINDOW:]
    hist_access = [h for h in hist if h["event_type"] == "access"]

    # F0: ref_count
    f0 = min(event.get("ref_count", 1), 20) / 20.0

    # F1: access count
    f1 = min(len(hist_access), 200) / 200.0

    # F2: time since last access
    if hist_access:
        f2 = (event["timestamp"] - hist_access[-1]["timestamp"]) / 100.0
    else:
        f2 = 1.0

    # F3: interval cv
    if len(hist_access) >= 2:
        intervals = [
            hist_access[j + 1]["timestamp"] - hist_access[j]["timestamp"]
            for j in range(len(hist_access) - 1)
        ]
        f3 = min(np.std(intervals) / (np.mean(intervals) + 1e-6), 2.0)
    else:
        f3 = 0.0

    # F4: log access count
    f4 = np.log2(len(hist_access) + 1) / np.log2(201)

    # F5: interval trend (recent vs historical)
    if len(hist_access) >= 4:
        intervals = [
            hist_access[j + 1]["timestamp"] - hist_access[j]["timestamp"]
            for j in range(len(hist_access) - 1)
        ]
        recent_mean = np.mean(intervals[-3:])
        all_mean = np.mean(intervals)
        f5 = min(recent_mean / max(all_mean, 1e-6), 2.0) / 2.0
    else:
        f5 = 0.5

    # F6: tail density
    if len(hist_access) >= 10:
        tail_span = hist_access[-1]["timestamp"] - hist_access[-10]["timestamp"]
        all_span = hist_access[-1]["timestamp"] - hist_access[0]["timestamp"]
        f6 = min(tail_span / max(all_span, 1), 1.0)
    elif len(hist_access) >= 2:
        all_span = hist_access[-1]["timestamp"] - hist_access[0]["timestamp"]
        f6 = min(1.0, all_span / max(all_span, 1))
    else:
        f6 = 0.5

    # F7: free events seen in this lifecycle
    free_count = sum(1 for h in hist if h["event_type"] == "free")
    f7 = min(free_count, 5) / 5.0

    return np.array([f0, f1, f2, f3, f4, f5, f6, f7], dtype=np.float32)


def precompute_features(stream):
    """
    预计算所有事件的特征向量。
    返回 (feats, sim_mask)，其中 sim_mask 标识应送入仿真的事件 (access/allocate)。
    """
    block_histories = {}
    feats = []
    sim_mask = []

    for e in stream:
        bid = e["block_id"]

        if e["event_type"] == "allocate":
            block_histories[bid] = []

        history = block_histories.get(bid, [])
        feat = build_feature(e, history)
        feats.append(feat)

        # 标记可仿真的事件
        sim_mask.append(e["event_type"] in ("access", "allocate"))

        block_histories.setdefault(bid, []).append(e)

    return feats, sim_mask


def run_simulation(policy, total_blocks, stream, precomputed_feats, sim_mask):
    sim = BlockManagerSimulator(total_blocks, policy)

    for idx, e in enumerate(stream):
        if not sim_mask[idx]:
            continue  # 跳过 free 事件
        bid = e["block_id"]
        feat = precomputed_feats[idx]
        sim.access(bid, feat, e["timestamp"])

    return sim.get_hit_rate(), sim.get_stats()


def main():
    # ========== 1. 加载 Trace ==========
    print("Loading trace...", flush=True)
    with open(TRACE_PATH, 'r', encoding='utf-8') as f:
        trace_events = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(trace_events)} events")

    # 过滤：保留所有事件类型（含 free，用于活跃序列追踪）
    stream = [
        e for e in trace_events
        if e["event_type"] in ("access", "allocate", "free")
    ]
    print(f"Filtered to {len(stream)} events (access + allocate + free)")

    if not stream:
        print("Trace 为空。")
        return

    # ========== 2. 预计算特征 ==========
    print("Precomputing features (v5: log_scale + trend + free_fix)...", flush=True)
    precomputed_feats, sim_mask = precompute_features(stream)
    print(f"Precomputed {len(precomputed_feats)} features, {sum(sim_mask)} simulatable")

    # ========== 3. 容量配置 ==========
    block_configs = [100, 200, 500, 1000, 2000, 5000]
    results = {name: [] for name in ["LRU", "FIFO", "Learned", "Belady"]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}, Configs: {block_configs}\n")

    # Belady 预构建 future trace (只包含 access/allocate)
    future_trace = [
        (e["timestamp"], e["block_id"]) for e in stream
        if e["event_type"] in ("access", "allocate")
    ]

    for num_blocks in block_configs:
        print(f"=== Total Blocks: {num_blocks} ===")

        # LRU
        hr, st = run_simulation(LRUPolicy(), num_blocks, stream, precomputed_feats, sim_mask)
        results["LRU"].append(hr)
        print(f"LRU     Hit: {hr:.4f}  (evicts={st['evictions']})")

        # FIFO
        hr, st = run_simulation(FIFOPolicy(), num_blocks, stream, precomputed_feats, sim_mask)
        results["FIFO"].append(hr)
        print(f"FIFO    Hit: {hr:.4f}  (evicts={st['evictions']})")

        # Learned
        if PREDICTOR_PATH.exists():
            learned = LearnedPolicy(PREDICTOR_PATH, device=device)
            hr, st = run_simulation(learned, num_blocks, stream, precomputed_feats, sim_mask)
            results["Learned"].append(hr)
            print(f"Learned Hit: {hr:.4f}  (evicts={st['evictions']})")
        else:
            print(f"[Warn] 预测器未找到: {PREDICTOR_PATH}")
            results["Learned"].append(None)

        # Belady (上界)
        belady = BeladyPolicy(future_trace)
        hr, st = run_simulation(belady, num_blocks, stream, precomputed_feats, sim_mask)
        results["Belady"].append(hr)
        print(f"Belady  Hit: {hr:.4f}  (evicts={st['evictions']})")
        print()

    # ========== 4. 绘图 ==========
    (RESULT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 7))
    colors = {
        'LRU': '#2196F3',
        'FIFO': '#FF9800',
        'Learned': '#4CAF50',
        'Belady': '#9C27B0',
    }

    for name, hrs in results.items():
        valid = [(b, h) for b, h in zip(block_configs, hrs) if h is not None]
        if valid:
            xs, ys = zip(*valid)
            plt.plot(xs, ys, marker='o', label=name, color=colors.get(name),
                     linewidth=2, markersize=8)

    plt.xlabel("Cache Capacity (Blocks)", fontsize=12)
    plt.ylabel("Hit Rate", fontsize=12)
    plt.title("KV Cache Block Hit Rate vs Capacity (no future leak)", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(block_configs, [str(b) for b in block_configs])
    plt.ylim(0, 1.05)

    save_path = RESULT_DIR / "figures" / "hit_rate_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

    # ========== 5. 保存数据 ==========
    (RESULT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    data_path = RESULT_DIR / "logs" / "simulation_results.json"
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(
            {"block_configs": block_configs, "results": results},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Results saved to {data_path}")

    # ========== 6. 简要分析 ==========
    print("\n--- Summary ---")
    for b in block_configs:
        vals = []
        for name in ["LRU", "FIFO", "Learned", "Belady"]:
            if results[name]:
                idx = block_configs.index(b)
                if idx < len(results[name]) and results[name][idx] is not None:
                    vals.append(f"{name}={results[name][idx]:.3f}")
        if vals:
            print(f"  {b:5d} blocks: " + ", ".join(vals))


if __name__ == "__main__":
    main()
