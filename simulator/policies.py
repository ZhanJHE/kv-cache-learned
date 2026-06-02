import numpy as np
from collections import OrderedDict, defaultdict
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
        # 返回最久未访问的
        for bid in self.access_order:
            if bid in candidates:
                return bid
        return next(iter(candidates))


class FIFOPolicy(EvictionPolicy):
    """FIFO 基线"""
    def __init__(self):
        self.insert_order = []

    def on_insert(self, block_id, metadata):
        self.insert_order.append(block_id)

    def select_victim(self, candidates):
        while self.insert_order:
            bid = self.insert_order.pop(0)
            if bid in candidates:
                return bid
        return list(candidates.keys())[0]


class LearnedPolicy(EvictionPolicy):
    """学习型策略：驱逐预测重用概率最低的块"""
    def __init__(self, model_path, device='cpu'):
        from predictor.model import ReusePredictor
        self.model = ReusePredictor().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device

    def on_access(self, block_id, timestamp):
        pass  # 不依赖访问历史

    def on_insert(self, block_id, metadata):
        pass

    def select_victim(self, candidates: Dict[int, BlockMetadata]):
        # 对所有候选块打分，选概率最低的驱逐
        scores = {}
        with torch.no_grad():
            for bid, meta in candidates.items():
                feat = torch.FloatTensor(meta.features).unsqueeze(0).to(self.device)
                score = self.model(feat).item()  # 重用概率
                scores[bid] = score

        return min(scores, key=scores.get)


class BeladyPolicy(EvictionPolicy):
    """Belady 最优策略（需要未来信息，仅作上界参考）"""
    def __init__(self, future_trace):
        """
        future_trace: list of (timestamp, block_id) 未来访问序列
        """
        self.future_access = defaultdict(list)
        for ts, bid in future_trace:
            self.future_access[bid].append(ts)
        # 转为迭代器
        self.future_iters = {bid: iter(ts_list) for bid, ts_list in self.future_access.items()}
        self.next_access = {}

    def on_access(self, block_id, timestamp):
        # 更新该 block 的下一次访问时间
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
        # 驱逐下一次访问最远的（或不再访问的）
        farthest = -1
        victim = None
        for bid in candidates:
            nxt = self.next_access.get(bid, float('inf'))
            if nxt > farthest:
                farthest = nxt
                victim = bid
        return victim
