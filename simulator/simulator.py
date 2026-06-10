from typing import Dict, Set
import numpy as np
from simulator.policies import BlockMetadata, EvictionPolicy


class BlockManagerSimulator:
    """
    KV Cache Block 管理模拟器（基础版）

    - 所有 cache 中的 block 均可被驱逐（无 ref_count 约束）
    - 适用于无 prefix caching 的 workload
    """
    def __init__(self, total_blocks: int, policy: EvictionPolicy):
        self.total_blocks = total_blocks
        self.policy = policy
        self.cache: Dict[int, BlockMetadata] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_access": 0,
        }

    def access(self, block_id: int, features: np.ndarray, timestamp: int):
        self.stats["total_access"] += 1

        if block_id in self.cache:
            # Hit
            self.stats["hits"] += 1
            meta = self.cache[block_id]
            meta.features = features
            meta.last_access = timestamp
            meta.access_count += 1
            self.policy.on_access(block_id, timestamp)
            return True
        else:
            # Miss
            self.stats["misses"] += 1

            if len(self.cache) >= self.total_blocks:
                victim_id = self.policy.select_victim(self.cache)
                if victim_id is not None:
                    del self.cache[victim_id]
                    self.stats["evictions"] += 1

            meta = BlockMetadata(block_id, features, timestamp)
            self.cache[block_id] = meta
            self.policy.on_insert(block_id, meta)
            return False

    def get_hit_rate(self):
        total = self.stats["total_access"]
        return self.stats["hits"] / total if total > 0 else 0.0

    def get_stats(self):
        return self.stats.copy()


class SequenceAwareSimulator:
    """
    序列感知的 KV Cache Block 管理模拟器

    与 BlockManagerSimulator 的核心区别：
    - 维护 block -> Set[seq_id] 映射，模拟真实的 ref_count
    - 只有 ref_count == 0 的 block 才可被驱逐
    - 支持 fork 事件：新序列共享已有 block

    这在有 prefix caching 的场景下更真实，当前 V100 无 fork 事件但架构已就绪。
    """
    def __init__(self, total_blocks: int, policy: EvictionPolicy):
        self.total_blocks = total_blocks
        self.policy = policy
        self.cache: Dict[int, BlockMetadata] = {}
        self.block_users: Dict[int, Set[int]] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_access": 0,
            "forks": 0,
            "frees": 0,
        }

    def access(self, block_id: int, features: np.ndarray, timestamp: int,
               seq_id: int = -1, event_type: str = "access"):
        """处理一个 trace 事件"""
        self.stats["total_access"] += 1

        if event_type == "allocate":
            return self._handle_allocate(block_id, features, timestamp, seq_id)
        elif event_type == "fork_share":
            return self._handle_fork(block_id, features, timestamp, seq_id)
        elif event_type == "free":
            return self._handle_free(block_id, seq_id)
        else:
            return self._handle_access(block_id, features, timestamp, seq_id)

    def _handle_allocate(self, block_id, features, timestamp, seq_id):
        if block_id not in self.block_users:
            self.block_users[block_id] = set()

        if block_id in self.cache:
            self.block_users[block_id].add(seq_id)
            self.stats["hits"] += 1
            meta = self.cache[block_id]
            meta.features = features
            meta.last_access = timestamp
            meta.access_count += 1
            return True

        self.stats["misses"] += 1
        self.block_users[block_id].add(seq_id)

        if len(self.cache) >= self.total_blocks:
            evictable = {
                bid: self.cache[bid]
                for bid in self.cache
                if len(self.block_users.get(bid, set())) == 0
            }
            if evictable:
                victim_id = self.policy.select_victim(evictable)
                if victim_id is not None:
                    del self.cache[victim_id]
                    self.block_users.pop(victim_id, None)
                    self.stats["evictions"] += 1

        meta = BlockMetadata(block_id, features, timestamp)
        self.cache[block_id] = meta
        self.policy.on_insert(block_id, meta)
        return False

    def _handle_fork(self, block_id, features, timestamp, seq_id):
        self.block_users.setdefault(block_id, set()).add(seq_id)

        if block_id in self.cache:
            self.stats["hits"] += 1
            self.stats["forks"] += 1
            meta = self.cache[block_id]
            meta.features = features
            meta.last_access = timestamp
            meta.access_count += 1
            self.policy.on_access(block_id, timestamp)
            return True

        self.stats["misses"] += 1
        if len(self.cache) >= self.total_blocks:
            evictable = {
                bid: self.cache[bid]
                for bid in self.cache
                if len(self.block_users.get(bid, set())) == 0
            }
            if evictable:
                victim_id = self.policy.select_victim(evictable)
                if victim_id is not None:
                    del self.cache[victim_id]
                    self.block_users.pop(victim_id, None)
                    self.stats["evictions"] += 1
        meta = BlockMetadata(block_id, features, timestamp)
        self.cache[block_id] = meta
        self.policy.on_insert(block_id, meta)
        return False

    def _handle_free(self, block_id, seq_id):
        self.stats["frees"] += 1
        if block_id in self.block_users:
            self.block_users[block_id].discard(seq_id)
        return True

    def _handle_access(self, block_id, features, timestamp, seq_id):
        if block_id in self.cache:
            self.stats["hits"] += 1
            meta = self.cache[block_id]
            meta.features = features
            meta.last_access = timestamp
            meta.access_count += 1
            self.policy.on_access(block_id, timestamp)
            return True
        else:
            self.stats["misses"] += 1
            if len(self.cache) >= self.total_blocks:
                evictable = {
                    bid: self.cache[bid]
                    for bid in self.cache
                    if len(self.block_users.get(bid, set())) == 0
                }
                if evictable:
                    victim_id = self.policy.select_victim(evictable)
                    if victim_id is not None:
                        del self.cache[victim_id]
                        self.block_users.pop(victim_id, None)
                        self.stats["evictions"] += 1
            meta = BlockMetadata(block_id, features, timestamp)
            self.cache[block_id] = meta
            self.policy.on_insert(block_id, meta)
            return False

    def get_hit_rate(self):
        total = self.stats["total_access"]
        return self.stats["hits"] / total if total > 0 else 0.0

    def get_stats(self):
        return self.stats.copy()
