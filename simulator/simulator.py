from typing import Dict, Set
import numpy as np
from simulator.policies import BlockMetadata, EvictionPolicy


class BlockManagerSimulator:
    """
    KV Cache Block 管理模拟器（原始版本）
    - 所有 block 均可被驱逐
    """
    def __init__(self, total_blocks: int, policy: EvictionPolicy):
        self.total_blocks = total_blocks
        self.policy = policy
        self.cache: Dict[int, BlockMetadata] = {}
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "total_access": 0}

    def access(self, block_id: int, features: np.ndarray, timestamp: int):
        self.stats["total_access"] += 1
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


class SequenceAwareSimulator:
    """
    序列感知的 KV Cache Block 管理模拟器

    与 BlockManagerSimulator 的核心区别：
    - 维护 `block -> Set[seq_id]` 映射，模拟真实的 ref_count
    - 只有 ref_count == 0（不被任何序列使用）的 block 才可被驱逐
    - 支持 fork 事件：新序列共享已有 block

    这意味着驱逐决策只在 "所有使用此 block 的序列都已释放它" 之后才发生。
    """
    def __init__(self, total_blocks: int, policy: EvictionPolicy):
        self.total_blocks = total_blocks
        self.policy = policy
        # block_id -> BlockMetadata
        self.cache: Dict[int, BlockMetadata] = {}
        # block_id -> Set[seq_id]（正在使用该 block 的序列）
        self.block_users: Dict[int, Set[int]] = {}
        # 统计
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
        """
        处理一个 trace 事件。

        参数：
            event_type: "allocate" / "access" / "fork_share" / "free"
        """
        self.stats["total_access"] += 1

        if event_type == "allocate":
            return self._handle_allocate(block_id, features, timestamp, seq_id)

        elif event_type == "fork_share":
            return self._handle_fork(block_id, features, timestamp, seq_id)

        elif event_type == "free":
            return self._handle_free(block_id, seq_id)

        elif event_type == "access":
            return self._handle_access(block_id, features, timestamp, seq_id)

        return False

    def _handle_allocate(self, block_id, features, timestamp, seq_id):
        """新 block 分配"""
        # 确保 block 在 block_users 中
        if block_id not in self.block_users:
            self.block_users[block_id] = set()

        # 如果 block 已在 cache 中（之前被其他序列分配，via fork 共享），复用
        if block_id in self.cache:
            self.block_users[block_id].add(seq_id)
            self.stats["hits"] += 1
            meta = self.cache[block_id]
            meta.features = features
            meta.last_access = timestamp
            meta.access_count += 1
            return True

        # 不在 cache 中，需要分配
        self.stats["misses"] += 1
        self.block_users[block_id].add(seq_id)

        if len(self.cache) >= self.total_blocks:
            # 需要驱逐：只考虑 ref_count == 0 的 block
            evictable = {
                bid: self.cache[bid]
                for bid in self.cache
                if len(self.block_users.get(bid, set())) == 0
            }
            if evictable:
                victim_id = self.policy.select_victim(evictable)
                del self.cache[victim_id]
                self.block_users.pop(victim_id, None)
                self.stats["evictions"] += 1
            # 如果没有可驱逐的 block，说明 cache 满了但所有 block 都活跃
            # 这种情况在真实系统中不会发生（OOM），模拟时直接扩容
            # 但这里为了公平对比，仍然插入（总统计会异常，标记走 Miss）

        meta = BlockMetadata(block_id, features, timestamp)
        self.cache[block_id] = meta
        self.policy.on_insert(block_id, meta)
        return False

    def _handle_fork(self, block_id, features, timestamp, seq_id):
        """新序列 fork 共享已有 block"""
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

        # block 不在 cache 中：之前被完全驱逐了，重新分配 (cache miss)
        self.stats["misses"] += 1
        if len(self.cache) >= self.total_blocks:
            evictable = {
                bid: self.cache[bid]
                for bid in self.cache
                if len(self.block_users.get(bid, set())) == 0
            }
            if evictable:
                victim_id = self.policy.select_victim(evictable)
                del self.cache[victim_id]
                self.block_users.pop(victim_id, None)
                self.stats["evictions"] += 1
        meta = BlockMetadata(block_id, features, timestamp)
        self.cache[block_id] = meta
        self.policy.on_insert(block_id, meta)
        return False

    def _handle_free(self, block_id, seq_id):
        """序列释放 block：从 block_users 移除"""
        self.stats["frees"] += 1
        if block_id in self.block_users:
            self.block_users[block_id].discard(seq_id)
        # 如果 ref_count 降到 0 且 block 在 cache 中，记一次 "逻辑驱逐机会"
        return True

    def _handle_access(self, block_id, features, timestamp, seq_id):
        """普通 access"""
        if block_id in self.cache:
            self.stats["hits"] += 1
            meta = self.cache[block_id]
            meta.features = features
            meta.last_access = timestamp
            meta.access_count += 1
            self.policy.on_access(block_id, timestamp)
            return True
        else:
            # block 不在 cache 中（被驱逐后又被 access，不太可能但处理）
            self.stats["misses"] += 1
            if len(self.cache) >= self.total_blocks:
                evictable = {
                    bid: self.cache[bid]
                    for bid in self.cache
                    if len(self.block_users.get(bid, set())) == 0
                }
                if evictable:
                    victim_id = self.policy.select_victim(evictable)
                    del self.cache[victim_id]
                    self.block_users.pop(victim_id, None)
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
