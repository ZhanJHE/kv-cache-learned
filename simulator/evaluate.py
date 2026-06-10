import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import torch

from simulator import BlockManagerSimulator
from simulator.policies import LRUPolicy, FIFOPolicy, LearnedPolicy, BeladyPolicy

# 配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_PATH = PROJECT_ROOT / "data" / "traces" / "sharegpt_trace.jsonl"
PREDICTOR_PATH = PROJECT_ROOT / "predictor" / "reuse_predictor.pt"
RESULT_DIR = PROJECT_ROOT / "results"


def precompute_seq_info(trace_events):
    """
    预计算序列信息，与 dataset.py._build_samples 的逻辑一致。

    返回:
        seq_info: {seq_id: {"first_ts", "last_ts", "total_blocks"}}
        block_seq_users: {block_id: cross_seq_count}
    """
    seq_info = {}
    for t in trace_events:
        sid = t["seq_id"]
        ts = t["timestamp"]
        if sid not in seq_info:
            seq_info[sid] = {
                "first_ts": ts,
                "last_ts": ts,
                "blocks": set(),
            }
        info = seq_info[sid]
        info["first_ts"] = min(info["first_ts"], ts)
        info["last_ts"] = max(info["last_ts"], ts)
        info["blocks"].add(t["block_id"])
    for sid in seq_info:
        seq_info[sid]["total_blocks"] = max(len(seq_info[sid]["blocks"]), 1)

    block_seq_users = defaultdict(set)
    for t in trace_events:
        block_seq_users[t["block_id"]].add(t["seq_id"])
    block_cross_seq = {
        bid: len(users) for bid, users in block_seq_users.items()
    }

    return seq_info, block_cross_seq


def build_feature(event, history, seq_info, block_cross_seq, history_window=200):
    """
    从 trace 事件和 block 历史构建特征。

    **必须与 predictor/dataset.py 的 _extract_features 完全一致！**
    """
    hist = history[-history_window:]   # 截断到 history_window，与 dataset 一致
    hist_access = [h for h in hist if h["event_type"] == "access"]

    sid = event["seq_id"]
    seq = seq_info.get(
        sid,
        {"first_ts": event["timestamp"], "last_ts": event["timestamp"], "total_blocks": 1},
    )
    seq_lifetime = max(seq["last_ts"] - seq["first_ts"], 1)
    seq_elapsed = event["timestamp"] - seq["first_ts"]

    block_pos_in_seq = len(hist_access) / max(seq["total_blocks"], 1)

    if len(hist_access) >= 2:
        intervals = [
            hist_access[j + 1]["timestamp"] - hist_access[j]["timestamp"]
            for j in range(len(hist_access) - 1)
        ]
        interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
    else:
        interval_cv = 0.0

    active_seqs = sum(
        1
        for sid2, si in seq_info.items()
        if si["first_ts"] <= event["timestamp"] <= si["last_ts"]
    )

    cross_seq = block_cross_seq.get(event.get("block_id", -1), 1)

    features = [
        min(event.get("ref_count", 1), 20) / 20.0,
        min(len(hist_access), 200) / 200.0,
        (
            (event["timestamp"] - hist_access[-1]["timestamp"]) / 100.0
            if hist_access
            else 1.0
        ),
        min(interval_cv, 2.0),
        event["timestamp"] / max(seq["last_ts"], 1),
        seq_elapsed / seq_lifetime,
        1.0 - seq_elapsed / seq_lifetime,
        block_pos_in_seq,
        min(cross_seq, 10) / 10.0,
        min(active_seqs, 50) / 50.0,
    ]
    return np.array(features, dtype=np.float32)


def precompute_features(stream, seq_info, block_cross_seq):
    """
    预计算所有事件的特征向量，避免每次仿真重复计算（10x+ 加速）。
    返回与 stream 等长的 feature 数组。
    """
    block_histories = {}
    feats = []
    for e in stream:
        bid = e["block_id"]
        if e["event_type"] == "allocate":
            block_histories[bid] = []
        history = block_histories.get(bid, [])
        feat = build_feature(e, history, seq_info, block_cross_seq)
        feats.append(feat)
        block_histories.setdefault(bid, []).append(e)
    return feats


def run_simulation(policy, total_blocks, stream, precomputed_feats):
    sim = BlockManagerSimulator(total_blocks, policy)
    block_histories = {}

    for idx, e in enumerate(stream):
        bid = e["block_id"]

        if e["event_type"] == "allocate":
            block_histories[bid] = []

        # 预计算特征
        feat = precomputed_feats[idx]

        sim.access(bid, feat, e["timestamp"])
        block_histories.setdefault(bid, []).append(e)

    return sim.get_hit_rate(), sim.get_stats()


def main():
    # ========== 1. 加载 Trace ==========
    print("Loading trace...", flush=True)
    with open(TRACE_PATH, 'r', encoding='utf-8') as f:
        trace_events = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(trace_events)} events")

    # 过滤：只保留 access 和 allocate
    stream = [e for e in trace_events if e["event_type"] in ("access", "allocate")]
    print(f"Filtered to {len(stream)} events (access + allocate)")

    if not stream:
        print("Trace 为空或没有 access/allocate 事件。")
        return

    # ========== 2. 预计算序列信息 ==========
    print("Precomputing sequence info...", flush=True)
    seq_info, block_cross_seq = precompute_seq_info(trace_events)
    print(f"Sequences: {len(seq_info)}, Unique blocks: {len(block_cross_seq)}")

    # ========== 3. 预计算特征 ==========
    print("Precomputing features...", flush=True)
    precomputed_feats = precompute_features(stream, seq_info, block_cross_seq)
    print(f"Precomputed {len(precomputed_feats)} features")

    # ========== 4. 容量配置 ==========
    block_configs = [100, 200, 500, 1000, 2000, 5000]
    results = {name: [] for name in ["LRU", "FIFO", "Learned", "Belady"]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}, Configs: {block_configs}\n")

    # Belady 预构建 future trace
    future_trace = [(e["timestamp"], e["block_id"]) for e in stream]

    for num_blocks in block_configs:
        print(f"=== Total Blocks: {num_blocks} ===")

        # LRU
        hr, st = run_simulation(LRUPolicy(), num_blocks, stream, precomputed_feats)
        results["LRU"].append(hr)
        print(f"LRU     Hit: {hr:.4f}  (evicts={st['evictions']})")

        # FIFO
        hr, st = run_simulation(FIFOPolicy(), num_blocks, stream, precomputed_feats)
        results["FIFO"].append(hr)
        print(f"FIFO    Hit: {hr:.4f}  (evicts={st['evictions']})")

        # Learned
        if PREDICTOR_PATH.exists():
            learned = LearnedPolicy(PREDICTOR_PATH, device=device)
            hr, st = run_simulation(learned, num_blocks, stream, precomputed_feats)
            results["Learned"].append(hr)
            print(f"Learned Hit: {hr:.4f}  (evicts={st['evictions']})")
        else:
            print(f"[Warn] 预测器未找到: {PREDICTOR_PATH}")
            results["Learned"].append(None)

        # Belady (上界)
        belady = BeladyPolicy(future_trace)
        hr, st = run_simulation(belady, num_blocks, stream, precomputed_feats)
        results["Belady"].append(hr)
        print(f"Belady  Hit: {hr:.4f}  (evicts={st['evictions']})")
        print()

    # ========== 5. 绘图 ==========
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
    plt.title("KV Cache Block Hit Rate vs Capacity", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(block_configs, [str(b) for b in block_configs])
    plt.ylim(0, 1.05)

    save_path = RESULT_DIR / "figures" / "hit_rate_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

    # ========== 6. 保存数据 ==========
    (RESULT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    data_path = RESULT_DIR / "logs" / "simulation_results.json"
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(
            {"block_configs": block_configs, "results": results},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Results saved to {data_path}")

    # ========== 7. 简要分析 ==========
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
