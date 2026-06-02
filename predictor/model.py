import numpy as np
import torch
import torch.nn as nn


class ReusePredictor(nn.Module):
    """轻量 2 层 MLP，预测 KV Block 未来重用概率"""
    def __init__(self, input_dim=8, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出概率
        )

    def forward(self, x):
        return self.net(x)

    def predict_score(self, features):
        """给定 numpy 特征，返回重用概率（用于模拟器）"""
        self.eval()
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            if features.dim() == 1:
                features = features.unsqueeze(0)
            return self.forward(features).squeeze().item()
