import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path


class KVCacheDataset(Dataset):
    """
    从 Trace 构建预测器训练数据（回归，无未来泄露）

    目标：给定一个 block 在某次 access 时的状态，
    预测该 block 还会被访问多少次（剩余访问次数）。

    所有特征仅依赖当前及过去的信息，不依赖 last_ts / total_blocks 等未来信息。

    特征（8 维）：
        0. ref_count_norm:      当前 ref_count（归一化 cap=20）
        1. access_count_norm:   该 block 历史 access 次数（归一化 cap=200）
        2. time_since_last:     距离该 block 上一次 access 的时间（归一化 /100）
        3. interval_cv:         access 间隔的变异系数（std/mean）
        4. log_access_count:    log2(cnt+1)/log2(201)，对数尺度下的访问量
        5. interval_trend:      最近3次间隔均值 / 全部间隔均值（<0.5加速, >0.5减速）
        6. tail_density:        最近 10 次访问的时间跨度 / 总跨度（尾部集中度）
        7. free_events_seen:    生命周期内该 block 被 free 的次数（归一化 cap=5）
    """
    def __init__(self, trace_path, history_window=200):
        self.history_window = history_window
        self.traces = self._load_traces(trace_path)
        self.samples, self.sample_block_ids = self._build_samples()

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
        # ========== 1. 按 block_id 分组 ==========
        block_events = defaultdict(list)
        for t in self.traces:
            block_events[t["block_id"]].append(t)

        # ========== 2. 构建训练样本 ==========
        samples = []
        sample_block_ids = []
        for block_id, events in block_events.items():
            events.sort(key=lambda x: x["timestamp"])

            # 按 allocate 事件切分生命周期
            lifetimes = []
            current_life = []
            for e in events:
                if e["event_type"] == "allocate" and current_life:
                    lifetimes.append(current_life)
                    current_life = [e]
                else:
                    current_life.append(e)
            if current_life:
                lifetimes.append(current_life)

            for life_events in lifetimes:
                for i, evt in enumerate(life_events):
                    if evt["event_type"] != "access":
                        continue

                    feat = self._extract_features(evt, life_events, i)

                    # 标签：当前生命周期内的剩余访问次数
                    remaining = sum(
                        1 for e in life_events[i + 1:] if e["event_type"] == "access"
                    )
                    label = min(remaining, 200) / 200.0

                    samples.append((feat, label))
                    sample_block_ids.append(block_id)

        return samples, sample_block_ids

    def _extract_features(self, evt, all_events, idx):
        """提取 8 维特征（与 evaluate.py 的 build_feature 必须一致！）"""
        hist = all_events[max(0, idx - self.history_window):idx]
        hist_access = [e for e in hist if e["event_type"] == "access"]

        # --- F0: ref_count ---
        f0 = min(evt.get("ref_count", 1), 20) / 20.0

        # --- F1: access count ---
        f1 = min(len(hist_access), 200) / 200.0

        # --- F2: time since last access ---
        if hist_access:
            f2 = (evt["timestamp"] - hist_access[-1]["timestamp"]) / 100.0
        else:
            f2 = 1.0

        # --- F3: interval cv ---
        if len(hist_access) >= 2:
            intervals = [
                hist_access[j + 1]["timestamp"] - hist_access[j]["timestamp"]
                for j in range(len(hist_access) - 1)
            ]
            f3 = min(np.std(intervals) / (np.mean(intervals) + 1e-6), 2.0)
        else:
            f3 = 0.0

        # --- F4: log access count ---
        f4 = np.log2(len(hist_access) + 1) / np.log2(201)

        # --- F5: interval trend (recent vs historical) ---
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

        # --- F6: tail density ---
        if len(hist_access) >= 10:
            tail_span = hist_access[-1]["timestamp"] - hist_access[-10]["timestamp"]
            all_span = hist_access[-1]["timestamp"] - hist_access[0]["timestamp"]
            f6 = min(tail_span / max(all_span, 1), 1.0)
        elif len(hist_access) >= 2:
            all_span = hist_access[-1]["timestamp"] - hist_access[0]["timestamp"]
            f6 = min(1.0, all_span / max(all_span, 1))
        else:
            f6 = 0.5

        # --- F7: free events seen in this lifecycle ---
        f7 = min(sum(1 for e in hist if e["event_type"] == "free"), 5) / 5.0

        features = [f0, f1, f2, f3, f4, f5, f6, f7]
        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        return torch.FloatTensor(feat), torch.FloatTensor([label])
