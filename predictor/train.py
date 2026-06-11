import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import r2_score
from model import ReusePredictor
from dataset import KVCacheDataset

# 配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_PATH = PROJECT_ROOT / "data" / "traces" / "sharegpt_trace.jsonl"
MODEL_SAVE = PROJECT_ROOT / "predictor" / "reuse_predictor.pt"
BATCH_SIZE = 256
EPOCHS = 80
LR = 1e-3
PATIENCE = 20
VAL_RATIO = 0.2


def block_level_split(dataset, val_ratio=VAL_RATIO):
    """按 block_id 分组切分 train/val，避免数据泄漏"""
    block_ids = list(set(dataset.sample_block_ids))
    random.shuffle(block_ids)
    split_point = int(len(block_ids) * (1 - val_ratio))
    train_blocks = set(block_ids[:split_point])
    val_blocks = set(block_ids[split_point:])

    train_idx = [
        i for i, bid in enumerate(dataset.sample_block_ids) if bid in train_blocks
    ]
    val_idx = [
        i for i, bid in enumerate(dataset.sample_block_ids) if bid in val_blocks
    ]
    return train_idx, val_idx


def train():
    # ========== 1. 加载数据 ==========
    full_dataset = KVCacheDataset(TRACE_PATH)
    train_idx, val_idx = block_level_split(full_dataset)

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"Dataset size: {len(full_dataset)}, Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 标签分布
    all_labels = np.array([full_dataset[i][1].item() for i in range(len(full_dataset))])
    print(f"Label range: [{all_labels.min():.4f}, {all_labels.max():.4f}], "
          f"mean={all_labels.mean():.4f}, std={all_labels.std():.4f}")

    # ========== 2. 模型 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReusePredictor(input_dim=8).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_mse = float("inf")
    patience_counter = 0

    # ========== 3. 训练循环 ==========
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # --- Val ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                preds = model(feats).cpu().numpy()
                val_preds.extend(preds.flatten())
                val_labels.extend(labels.numpy().flatten())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_mse = np.mean((val_preds - val_labels) ** 2)
        val_mae = np.mean(np.abs(val_preds - val_labels))
        val_r2 = r2_score(val_labels, val_preds)

        print(f"Epoch {epoch + 1:3d}: Loss={train_loss / len(train_loader):.4f}, "
              f"Val MSE={val_mse:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}, "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

        # --- Early Stopping ---
        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(model.state_dict(), MODEL_SAVE)
            patience_counter = 0
            print(f"  -> Best model saved (MSE={best_mse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} (best Val MSE={best_mse:.4f})")
                break

    # ========== 4. 结果 ==========
    print(f"\nTraining complete. Best Val MSE: {best_mse:.4f}, "
          f"Model saved to {MODEL_SAVE}")

    # 最终评估
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.to(device)
            preds = model(feats).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    final_mse = np.mean((all_preds - all_labels) ** 2)
    final_mae = np.mean(np.abs(all_preds - all_labels))
    final_r2 = r2_score(all_labels, all_preds)
    print(f"Final Val: MSE={final_mse:.4f}, MAE={final_mae:.4f}, R²={final_r2:.4f}")


if __name__ == "__main__":
    train()
