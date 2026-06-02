import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path


class KVCacheDataset(Dataset):
    """
    从 Trace 构建预测器训练数据
    标签：未来 N 个 token 内是否被重用（二分类）
    """
    def __init__(self, trace_path, history_window=50, future_window=100, num_layers=32):
        self.history_window = history_window
        self.future_window = future_window
        self.num_layers = num_layers

        # 加载并解析 Trace
        self.traces = self._load_traces(trace_path)
        self.samples = self._build_samples()

    def _load_traces(self, path):
        traces = []
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                traces.append(json.loads(line))
        return traces

    def _build_samples(self):
        # 按 block_id 分组，构建时间线
        block_events = defaultdict(list)
        for t in self.traces:
            block_events[t["block_id"]].append(t)

        samples = []
        for block_id, events in block_events.items():
            events.sort(key=lambda x: x["timestamp"])

            for i, evt in enumerate(events):
                if evt["event_type"] != "access":
                    continue

                # 特征
                feat = self._extract_features(events, i)

                # 标签：未来 future_window 个 token 内是否有 access
                future_ts = evt["timestamp"] + self.future_window
                has_future = any(
                    e["timestamp"] > evt["timestamp"] and e["timestamp"] <= future_ts
                    and e["event_type"] == "access"
                    for e in events[i + 1:]
                )

                samples.append((feat, int(has_future)))

        return samples

    def _extract_features(self, events, idx):
        evt = events[idx]
        hist = events[max(0, idx - self.history_window):idx]

        # 8 维特征向量
        features = [
            evt["layer_id"] / self.num_layers,           # 层深度（归一化）
            evt["num_tokens"] / 16.0,                     # block 占用率（block size=16）
            1.0 if evt["is_prefix"] else 0.0,             # 是否前缀
            len(hist),                                    # 历史访问次数
            np.mean([e["timestamp"] for e in hist]) if hist else 0,  # 平均访问时间
            (evt["timestamp"] - hist[-1]["timestamp"]) if hist else 1000,  # 距离上次访问
            len([e for e in hist if e["is_prefix"]]) / max(len(hist), 1),  # 历史前缀比例
            evt["timestamp"] / 10000.0,                   # 全局时间（归一化）
        ]
        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        return torch.FloatTensor(feat), torch.FloatTensor([label])
