import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReusePredictor(nn.Module):
    """3 层 MLP + LayerNorm + Residual，预测 block 剩余访问次数（回归）"""
    def __init__(self, input_dim=8, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        # 不用 Sigmoid: 回归任务，输出原始值

    def forward(self, x):
        # 处理单样本输入 (无 batch 维度)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single = True
        else:
            single = False

        h1 = F.relu(self.ln1(self.fc1(x)))
        h1 = self.dropout(h1)
        h2 = F.relu(self.ln2(self.fc2(h1)))
        # Residual connection (需要维度匹配: hidden_dim == hidden_dim)
        h2 = h1 + h2
        h2 = self.dropout(h2)
        h3 = F.relu(self.ln3(self.fc3(h2)))
        h3 = self.dropout(h3)
        out = self.out(h3)

        if single:
            out = out.squeeze(0)
        return out

    def predict_score(self, features):
        """给定 numpy 特征，返回剩余访问次数预测值（用于模拟器）"""
        self.eval()
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            if features.dim() == 1:
                features = features.unsqueeze(0)
            return self.forward(features).squeeze(-1).item()
