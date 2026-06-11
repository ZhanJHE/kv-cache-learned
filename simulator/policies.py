import numpy as np
from collections import OrderedDict, defaultdict, deque
from typing import Dict, List, Optional
import torch


class BlockMetadata:
    def __init__(self, block_id, features, timestamp):
        self.block_id = block_id
        self.features = features      # numpy array
        self.last_access = timestamp
        self.first_access = timestamp
        self.access_count = 1


class EvictionPolicy:
    def on_access(self, block_id, timestamp):
        pass

    def on_insert(self, block_id, metadata):
        pass

    def select_victim(self, candidates: Dict[int, BlockMetadata]) -> int:
        raise NotImplementedError


class LRUPolicy(EvictionPolicy):
    """LRU 基线"""
    def __init__(self):
        self.access_order = OrderedDict()

    def on_access(self, block_id, timestamp):
        if block_id in self.access_order:
            self.access_order.move_to_end(block_id)

    def on_insert(self, block_id, metadata):
        self.access_order[block_id] = metadata

    def select_victim(self, candidates):
        for bid in self.access_order:
            if bid in candidates:
                return bid
        return next(iter(candidates))


class FIFOPolicy(EvictionPolicy):
    """FIFO 基线（使用 deque，O(1) 出队）"""
    def __init__(self):
        self.insert_order = deque()

    def on_insert(self, block_id, metadata):
        self.insert_order.append(block_id)

    def select_victim(self, candidates):
        while self.insert_order:
            bid = self.insert_order.popleft()
            if bid in candidates:
                return bid
        return next(iter(candidates))


class LearnedPolicy(EvictionPolicy):
    """
    学习型策略：预测 block 剩余访问次数，驱逐剩余最少的。

    批量推理（高效）：每次驱逐时对所有候选 block 一次性推理，
    选择模型预测分数最低的驱逐。
    """
    def __init__(self, model_path, device='cpu'):
        from predictor.model import ReusePredictor
        self.model = ReusePredictor(input_dim=8).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device

    def on_access(self, block_id, timestamp):
        pass

    def on_insert(self, block_id, metadata):
        pass

    def select_victim(self, candidates: Dict[int, BlockMetadata]):
        if not candidates:
            return None

        bids = list(candidates.keys())
        feats = torch.from_numpy(
            np.array([candidates[bid].features for bid in bids], dtype=np.float32)
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(feats).squeeze(-1).cpu().numpy()

        # 回归：分数越低 = 剩余访问越少 = 优先驱逐
        return bids[int(scores.argmin())]


class BeladyPolicy(EvictionPolicy):
    """
    Belady 最优策略（需要未来信息，仅作上界参考）

    驱逐"下一次访问最远"的 block（包括永不再访问的）。
    """
    def __init__(self, future_trace):
        """
        future_trace: list of (timestamp, block_id) 按时间排序的未来访问序列
        """
        self.future_access = defaultdict(list)
        for ts, bid in future_trace:
            self.future_access[bid].append(ts)
        # 转为迭代器，每次 on_access 时推进
        self.future_iters = {
            bid: iter(ts_list) for bid, ts_list in self.future_access.items()
        }
        self.next_access = {}

    def on_access(self, block_id, timestamp):
        it = self.future_iters.get(block_id)
        if it:
            try:
                self.next_access[block_id] = next(it)
            except StopIteration:
                self.next_access[block_id] = float('inf')
        else:
            self.next_access[block_id] = float('inf')

    def on_insert(self, block_id, metadata):
        self.on_access(block_id, metadata.first_access)

    def select_victim(self, candidates):
        farthest = -1
        victim = None
        for bid in candidates:
            nxt = self.next_access.get(bid, float('inf'))
            if nxt > farthest:
                farthest = nxt
                victim = bid
        return victim
