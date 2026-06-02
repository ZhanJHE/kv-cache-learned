"""
vLLM 自定义 Evictor 插件（需手动集成到 vLLM 源码）
目标文件：vllm/core/evictor.py
"""
import torch
import numpy as np
from typing import List


class LearnedEvictor:
    """
    基于预测器的驱逐策略
    使用方式：修改 vLLM 配置，指定 eviction_policy="learned"

    注意：本类继承自 vllm.core.evictor.Evictor，
    实际集成时需要根据 vLLM 0.5.0 的 Evictor 基类做适配。
    """
    def __init__(self, model_path, device='cuda'):
        from predictor.model import ReusePredictor
        self.model = ReusePredictor().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device

    def get_block_to_evict(self, blocks: List) -> object:
        if not blocks:
            raise ValueError("No blocks to evict")

        # 提取特征（需与 BlockManager 协调，在 Block 中预存特征）
        scores = []
        with torch.no_grad():
            for block in blocks:
                # 假设 block 有 cached_features 属性
                feat = getattr(block, 'cached_features', None)
                if feat is None:
                    # 无特征时退化为 LRU（给低分）
                    scores.append(0.0)
                    continue

                feat_t = torch.FloatTensor(feat).unsqueeze(0).to(self.device)
                score = self.model(feat_t).item()
                scores.append(score)

        # 驱逐重用概率最低的
        victim_idx = int(np.argmin(scores))
        return blocks[victim_idx]
