"""
Trace Logger: 无侵入式 Hook vLLM BlockManager
使用方法：在启动 vLLM 前导入此模块，自动替换 BlockManager 方法
"""
import json
import time
from typing import Dict, List, Optional
from collections import defaultdict


class KVTraceLogger:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.traces: List[Dict] = []
        self.global_token_counter = 0  # 全局 token 计数器，模拟时间戳
        self._hooked = False

    def log(self, event_type: str, block_id: int, seq_id: int,
            layer_id: int, num_tokens: int, is_prefix: bool = False,
            metadata: Optional[Dict] = None):
        entry = {
            "timestamp": self.global_token_counter,
            "event_type": event_type,      # allocate / access / evict / free
            "block_id": block_id,
            "seq_id": seq_id,
            "layer_id": layer_id,
            "num_tokens": num_tokens,
            "is_prefix": is_prefix,
            "metadata": metadata or {},
            "wall_time": time.time()
        }
        self.traces.append(entry)

    def increment_token(self):
        """每生成一个 token 调用一次，作为逻辑时间"""
        self.global_token_counter += 1

    def flush(self):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for t in self.traces:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')
        print(f"[TraceLogger] Flushed {len(self.traces)} events to {self.output_path}")

    def hook_vllm(self):
        """动态 Hook vLLM 的 BlockManager"""
        if self._hooked:
            return

        try:
            from vllm.core.block_manager import BlockManager
            from vllm.core.block import PhysicalTokenBlock

            original_allocate = BlockManager.allocate
            original_free = BlockManager.free
            original_append_slots = BlockManager.append_slots

            # 获取层数（从模型配置推断，这里简化处理）
            num_layers = 32  # Llama-2-7B 默认 32 层，实际应从模型获取

            # 在闭包中捕获 logger 实例，避免依赖全局变量
            logger = self

            def hooked_allocate(block_manager_self, seq_id, prompt_token_ids, seq_len):
                # 调用原方法
                result = original_allocate(block_manager_self, seq_id, prompt_token_ids, seq_len)

                # 记录新分配的 blocks
                seq = block_manager_self.block_tables[seq_id]
                for layer_id, blocks in enumerate(seq):
                    for block in blocks:
                        if block.ref_count == 1:  # 新分配
                            logger.log(
                                event_type="allocate",
                                block_id=block.block_number,
                                seq_id=seq_id,
                                layer_id=layer_id,
                                num_tokens=block.num_tokens,
                                is_prefix=(seq_len < 50)  # 简化：前 50 token 视为前缀
                            )
                return result

            def hooked_append_slots(block_manager_self, seq, num_tokens):
                # 记录访问（append_slots 意味着需要读取已有 block 并写入新 block）
                seq_id = id(seq)  # 简化标识
                for layer_id, blocks in enumerate(seq):
                    for block in blocks:
                        logger.log(
                            event_type="access",
                            block_id=block.block_number,
                            seq_id=seq_id,
                            layer_id=layer_id,
                            num_tokens=block.num_tokens,
                            is_prefix=False
                        )
                logger.increment_token()
                return original_append_slots(block_manager_self, seq, num_tokens)

            def hooked_free(block_manager_self, seq):
                seq_id = id(seq)
                for layer_id, blocks in enumerate(seq):
                    for block in blocks:
                        logger.log(
                            event_type="free",
                            block_id=block.block_number,
                            seq_id=seq_id,
                            layer_id=layer_id,
                            num_tokens=block.num_tokens
                        )
                return original_free(block_manager_self, seq)

            BlockManager.allocate = hooked_allocate
            BlockManager.append_slots = hooked_append_slots
            BlockManager.free = hooked_free

            self._hooked = True
            print("[TraceLogger] Hooked vLLM BlockManager successfully")

        except Exception as e:
            print(f"[TraceLogger] Hook failed: {e}")
