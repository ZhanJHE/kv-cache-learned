from typing import Dict
import numpy as np
from policies import BlockMetadata, EvictionPolicy


class BlockManagerSimulator:
    """
    KV Cache Block 管理模拟器
    """
    def __init__(self, total_blocks: int, policy: EvictionPolicy):
        self.total_blocks = total_blocks
        self.policy = policy
        self.cache: Dict[int, BlockMetadata] = {}  # block_id -> BlockMetadata
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_access": 0
        }

    def access(self, block_id: int, features: np.ndarray, timestamp: int):
        self.stats["total_access"] += 1

        if block_id in self.cache:
            # Hit
            self.stats["hits"] += 1
            self.cache[block_id].last_access = timestamp
            self.cache[block_id].access_count += 1
            self.policy.on_access(block_id, timestamp)
            return True
        else:
            # Miss
            self.stats["misses"] += 1

            if len(self.cache) >= self.total_blocks:
                # 需要驱逐
                victim_id = self.policy.select_victim(self.cache)
                del self.cache[victim_id]
                self.stats["evictions"] += 1

            meta = BlockMetadata(block_id, features, timestamp)
            self.cache[block_id] = meta
            self.policy.on_insert(block_id, meta)
            return False

    def get_hit_rate(self):
        total = self.stats["total_access"]
        return self.stats["hits"] / total if total > 0 else 0

    def get_stats(self):
        return self.stats.copy()
