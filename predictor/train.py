import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from model import ReusePredictor
from dataset import KVCacheDataset

# 配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_PATH = PROJECT_ROOT / "data" / "traces" / "sharegpt_trace.jsonl"
MODEL_SAVE = PROJECT_ROOT / "predictor" / "reuse_predictor.pt"
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
PATIENCE = 10


def train():
    # 加载数据
    full_dataset = KVCacheDataset(TRACE_PATH)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"Dataset size: {len(full_dataset)}, Train: {train_size}, Val: {val_size}")

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReusePredictor().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_auc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Val
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                preds = model(feats).cpu().numpy()
                val_preds.extend(preds.flatten())
                val_labels.extend(labels.numpy().flatten())

        val_preds_bin = [1 if p > 0.5 else 0 for p in val_preds]
        auc = roc_auc_score(val_labels, val_preds)
        acc = accuracy_score(val_labels, val_preds_bin)

        print(f"Epoch {epoch + 1}: Loss={train_loss / len(train_loader):.4f}, AUC={auc:.4f}, Acc={acc:.4f}")

        scheduler.step(1 - auc)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), MODEL_SAVE)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    print(f"Best AUC: {best_auc:.4f}, Model saved to {MODEL_SAVE}")


if __name__ == "__main__":
    train()
