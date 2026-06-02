import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from simulator import BlockManagerSimulator
from policies import LRUPolicy, FIFOPolicy, LearnedPolicy, BeladyPolicy
from predictor.dataset import KVCacheDataset

# 配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_PATH = PROJECT_ROOT / "data" / "traces" / "sharegpt_trace.jsonl"
PREDICTOR_PATH = PROJECT_ROOT / "predictor" / "reuse_predictor.pt"
RESULT_DIR = PROJECT_ROOT / "results"


def build_feature(event, num_layers=32):
    """从单个 trace 事件构建特征（与 dataset 一致）"""
    return np.array([
        event["layer_id"] / num_layers,
        event["num_tokens"] / 16.0,
        1.0 if event.get("is_prefix", False) else 0.0,
        0.0,  # 历史访问次数（简化：单事件无法获得历史）
        0.0,
        0.0,
        0.0,
        event["timestamp"] / 10000.0,
    ], dtype=np.float32)


def run_simulation(policy_name, policy, total_blocks, trace_events):
    sim = BlockManagerSimulator(total_blocks, policy)

    for e in trace_events:
        feat = build_feature(e)
        sim.access(e["block_id"], feat, e["timestamp"])

    return sim.get_hit_rate(), sim.get_stats()


def main():
    # 加载 trace
    with open(TRACE_PATH, 'r', encoding='utf-8') as f:
        trace_events = [json.loads(line) for line in f if line.strip()]

    # 过滤只保留 access 和 allocate（allocate 视为 miss）
    stream = [e for e in trace_events if e["event_type"] in ["access", "allocate"]]

    if not stream:
        print("Trace 为空或没有 access/allocate 事件，请先执行 Trace 收集。")
        return

    # 不同显存容量配置（Block 数）
    block_configs = [100, 200, 500, 1000, 2000, 5000]

    results = {name: [] for name in ["LRU", "FIFO", "Learned"]}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for num_blocks in block_configs:
        print(f"\n=== Total Blocks: {num_blocks} ===")

        # LRU
        hr, st = run_simulation("LRU", LRUPolicy(), num_blocks, stream)
        results["LRU"].append(hr)
        print(f"LRU  Hit Rate: {hr:.4f}")

        # FIFO
        hr, st = run_simulation("FIFO", FIFOPolicy(), num_blocks, stream)
        results["FIFO"].append(hr)
        print(f"FIFO Hit Rate: {hr:.4f}")

        # Learned
        if PREDICTOR_PATH.exists():
            learned = LearnedPolicy(PREDICTOR_PATH, device=device)
            hr, st = run_simulation("Learned", learned, num_blocks, stream)
            results["Learned"].append(hr)
            print(f"Learned Hit Rate: {hr:.4f}")
        else:
            print(f"[Warning] 预测器模型未找到: {PREDICTOR_PATH}，跳过 Learned 策略")
            results["Learned"].append(None)

    # 绘图
    (RESULT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, hrs in results.items():
        valid = [(b, h) for b, h in zip(block_configs, hrs) if h is not None]
        if valid:
            xs, ys = zip(*valid)
            plt.plot(xs, ys, marker='o', label=name)
    plt.xlabel("Total Blocks (Cache Capacity)")
    plt.ylabel("Hit Rate")
    plt.title("KV Cache Block Hit Rate vs Capacity")
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULT_DIR / "figures" / "hit_rate_comparison.png")
    print(f"\nFigure saved to {RESULT_DIR / 'figures' / 'hit_rate_comparison.png'}")

    # 保存数据
    (RESULT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / "logs" / "simulation_results.json", 'w', encoding='utf-8') as f:
        json.dump({"block_configs": block_configs, "results": results}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
