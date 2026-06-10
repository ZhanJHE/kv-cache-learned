import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path


class KVCacheDataset(Dataset):
    """
    从 Trace 构建预测器训练数据（回归）

    目标：给定一个 block 在某次 access 时的状态，
    预测该 block 还会被访问多少次（剩余访问次数）。

    驱逐决策：保留剩余访问次数多的 block，驱逐剩余访问次数少的。

    特征（10 维）：
        0. ref_count_norm:      当前 ref_count（归一化 cap=20）
        1. access_count:        该 block 历史 access 次数（归一化 cap=200）
        2. time_since_last:     距离该 block 上一次 access 的时间（归一化 /100）
        3. interval_cv:         access 间隔的变异系数（std/mean）
        4. global_progress:     全局时间进度
        5. seq_lifetime_ratio:  所属序列已存活时间比例
        6. seq_remaining_ratio: 所属序列剩余时间比例
        7. block_pos_in_seq:    block 在所属序列中的相对位置（0=最晚, 1=最早）
        8. cross_seq_count:     该 block 被多少不同序列使用过（归一化 cap=10）
        9. active_seqs:         当前活跃序列数（归一化 cap=50）
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
        # ========== 1. 预计算每个序列的活跃区间 ==========
        seq_info = {}
        for t in self.traces:
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

        # ========== 2. 按 block_id 分组 ==========
        block_events = defaultdict(list)
        for t in self.traces:
            block_events[t["block_id"]].append(t)

        # ========== 3. 统计每个 block 被多少不同序列使用 ==========
        block_seq_users = defaultdict(set)
        for t in self.traces:
            block_seq_users[t["block_id"]].add(t["seq_id"])

        # ========== 4. 构建训练样本 ==========
        # 每个 block 可能有多个生命周期（被 free 后又 allocate）
        # 每个生命周期独立构建样本，与 evaluate.py 的 precompute 行为一致
        samples = []
        sample_block_ids = []
        for block_id, events in block_events.items():
            events.sort(key=lambda x: x["timestamp"])
            cross_seq_count = len(block_seq_users[block_id])

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
                # 在这个生命周期内构建样本
                for i, evt in enumerate(life_events):
                    if evt["event_type"] != "access":
                        continue

                    # 注入跨序列信息
                    evt = dict(evt)
                    evt["cross_seq_count"] = cross_seq_count

                    # 特征：使用当前生命周期内的历史
                    feat = self._extract_features(evt, life_events, i, seq_info)

                    # 标签：当前生命周期内的剩余访问次数
                    remaining = sum(
                        1 for e in life_events[i + 1 :] if e["event_type"] == "access"
                    )
                    label = min(remaining, 200) / 200.0

                    samples.append((feat, label))
                    sample_block_ids.append(block_id)

        return samples, sample_block_ids

    def _extract_features(self, evt, all_events, idx, seq_info):
        """提取 10 维特征（与 evaluate.py 的 build_feature 必须一致！）"""
        hist = all_events[max(0, idx - self.history_window) : idx]
        hist_access = [e for e in hist if e["event_type"] == "access"]

        sid = evt["seq_id"]
        seq = seq_info.get(
            sid,
            {"first_ts": evt["timestamp"], "last_ts": evt["timestamp"], "total_blocks": 1},
        )
        seq_lifetime = max(seq["last_ts"] - seq["first_ts"], 1)
        seq_elapsed = evt["timestamp"] - seq["first_ts"]

        # block 在该序列中的相对位置
        block_pos_in_seq = len(hist_access) / max(seq["total_blocks"], 1)

        # access 间隔的变异系数
        if len(hist_access) >= 2:
            intervals = [
                hist_access[j + 1]["timestamp"] - hist_access[j]["timestamp"]
                for j in range(len(hist_access) - 1)
            ]
            interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
        else:
            interval_cv = 0.0

        # 当前活跃序列数
        active_seqs = sum(
            1
            for sid2, si in seq_info.items()
            if si["first_ts"] <= evt["timestamp"] <= si["last_ts"]
        )

        features = [
            # 0. ref_count
            min(evt.get("ref_count", 1), 20) / 20.0,

            # 1. 历史 access 次数
            min(len(hist_access), 200) / 200.0,

            # 2. 距离上次 access
            (
                (evt["timestamp"] - hist_access[-1]["timestamp"]) / 100.0
                if hist_access
                else 1.0
            ),

            # 3. access 间隔变异系数
            min(interval_cv, 2.0),

            # 4. 全局时间进度
            evt["timestamp"] / max(seq["last_ts"], 1),

            # 5. 序列已存活比例
            seq_elapsed / seq_lifetime,

            # 6. 序列剩余比例
            1.0 - seq_elapsed / seq_lifetime,

            # 7. block 在序列中的相对位置
            block_pos_in_seq,

            # 8. 跨序列使用数
            min(evt.get("cross_seq_count", 1), 10) / 10.0,

            # 9. 活跃序列数
            min(active_seqs, 50) / 50.0,
        ]
        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        return torch.FloatTensor(feat), torch.FloatTensor([label])
