```markdown
# 选题5：大语言模型推理中的KV Cache管理策略 — 实验规划文档

> **目标**：在AutoDL平台完成从Trace收集、预测器训练到策略对比的完整实验链路，验证学习型缓存替换策略相比LRU的命中优势。
> **平台**：AutoDL（Linux, CUDA, PyTorch预装环境）
> **模型**：Llama-2-7B-chat / Qwen2-7B-Instruct
> **框架**：vLLM ≥ 0.5.0

---

## 一、项目目录结构（AutoDL数据盘）

在AutoDL实例中，数据盘通常为 `/root/autodl-tmp`。所有项目文件存放于此。

```bash
/root/autodl-tmp/
├── kv_cache_project/
│   ├── data/                    # 数据集与Trace文件
│   │   ├── sharegpt.json      # ShareGPT原始数据
│   │   ├── traces/            # 收集的KV Cache访问Trace
│   │   │   ├── short_seq_trace.jsonl
│   │   │   └── long_seq_trace.jsonl
│   │   └── features/          # 预测器训练数据
│   ├── vllm_patch/            # 对vLLM的补丁文件（Hook代码）
│   │   ├── block_manager_v2.py
│   │   └── trace_logger.py
│   ├── simulator/             # Trace-driven模拟器
│   │   ├── simulator.py
│   │   ├── policies.py        # LRU, FIFO, Learned, Belady
│   │   └── evaluate.py
│   ├── predictor/             # 轻量预测器
│   │   ├── train.py
│   │   ├── model.py
│   │   └── dataset.py
│   ├── real_system/           # 真实系统集成（Tier 2）
│   │   └── learned_evictor.py
│   ├── scripts/               # 一键执行脚本
│   │   ├── setup_env.sh
│   │   ├── collect_trace.sh
│   │   ├── run_simulation.sh
│   │   └── run_benchmark.sh
│   └── results/               # 实验结果与图表
│       ├── figures/
│       └── logs/
└── models/                    # 模型权重缓存（建议用软链接或AutoDL模型库）
    └── llama-2-7b-chat/
```

---

## 二、环境准备（AutoDL初始化）

### 2.1 进入实例后执行

```bash
# 进入数据盘（AutoDL数据盘，持久化存储）
cd /root/autodl-tmp

# 创建项目目录
mkdir -p kv_cache_project/{data/traces,data/features,vllm_patch,simulator,predictor,real_system,scripts,results/{figures,logs}}

# 创建Python环境（如果AutoDL未提供合适环境）
conda create -n kv_cache python=3.10 -y
conda activate kv_cache

# 安装依赖
pip install vllm==0.5.0 torch==2.1.2 transformers datasets accelerate numpy pandas matplotlib seaborn scikit-learn tqdm

# 验证vLLM安装
python -c "import vllm; print(vllm.__version__)"
```

### 2.2 下载数据集

```bash
cd /root/autodl-tmp/kv_cache_project/data

# 下载ShareGPT（用于模拟多租户负载）
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O sharegpt.json

# 或者使用HuggingFace datasets库（更稳定）
python -c "
from datasets import load_dataset
ds = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered', 'default')
ds['train'].to_json('sharegpt.json')
"
```

---

## 三、第一阶段：Trace收集（Tier 1核心）

### 3.1 原理

在vLLM的`BlockManager`中Hook以下事件：
- `allocate`：新分配Block时记录
- `access`：读取Block时记录（推理时KV Cache读取）
- `free/evict`：Block被驱逐时记录

### 3.2 文件：`vllm_patch/trace_logger.py`

```python
"""
Trace Logger: 无侵入式Hook vLLM BlockManager
使用方法：在启动vLLM前导入此模块，自动替换BlockManager方法
"""
import json
import time
import torch
from typing import Dict, List, Optional
from collections import defaultdict

class KVTraceLogger:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.traces = []
        self.global_token_counter = 0  # 全局token计数器，模拟时间戳
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
        """每生成一个token调用一次，作为逻辑时间"""
        self.global_token_counter += 1
    
    def flush(self):
        with open(self.output_path, 'w') as f:
            for t in self.traces:
                f.write(json.dumps(t) + '\n')
        print(f"[TraceLogger] Flushed {len(self.traces)} events to {self.output_path}")
    
    def hook_vllm(self):
        """动态Hook vLLM的BlockManager"""
        if self._hooked:
            return
        
        try:
            from vllm.core.block_manager import BlockManager
            from vllm.core.block import PhysicalTokenBlock
            
            original_allocate = BlockManager.allocate
            original_free = BlockManager.free
            original_append_slots = BlockManager.append_slots
            
            # 获取层数（从模型配置推断，这里简化处理）
            num_layers = 32  # Llama-2-7B默认32层，实际应从模型获取
            
            def hooked_allocate(self, seq_id, prompt_token_ids, seq_len):
                # 调用原方法
                result = original_allocate(self, seq_id, prompt_token_ids, seq_len)
                
                # 记录新分配的blocks
                seq = self.block_tables[seq_id]
                for layer_id, blocks in enumerate(seq):
                    for block in blocks:
                        if block.ref_count == 1:  # 新分配
                            logger.log(
                                event_type="allocate",
                                block_id=block.block_number,
                                seq_id=seq_id,
                                layer_id=layer_id,
                                num_tokens=block.num_tokens,
                                is_prefix=(seq_len < 50)  # 简化：前50token视为前缀
                            )
                return result
            
            def hooked_append_slots(self, seq, num_tokens):
                # 记录访问（append_slots意味着需要读取已有block并写入新block）
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
                return original_append_slots(self, seq, num_tokens)
            
            def hooked_free(self, seq):
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
                return original_free(self, seq)
            
            BlockManager.allocate = hooked_allocate
            BlockManager.append_slots = hooked_append_slots
            BlockManager.free = hooked_free
            
            self._hooked = True
            print("[TraceLogger] Hooked vLLM BlockManager successfully")
            
        except Exception as e:
            print(f"[TraceLogger] Hook failed: {e}")

# 全局单例
logger = KVTraceLogger("/root/autodl-tmp/kv_cache_project/data/traces/trace.jsonl")
```

### 3.3 文件：`scripts/collect_trace.py`

```python
#!/usr/bin/env python3
"""
Trace收集脚本：启动vLLM服务并发送请求，收集KV Cache访问Trace
"""
import os
import sys
import json
import random
import requests
import threading
import time

# 先导入Hook
sys.path.insert(0, "/root/autodl-tmp/kv_cache_project/vllm_patch")
from trace_logger import logger

# 配置
MODEL_PATH = "/root/autodl-tmp/models/llama-2-7b-chat"  # 或从HF下载
TRACE_OUTPUT = "/root/autodl-tmp/kv_cache_project/data/traces/sharegpt_trace.jsonl"
GPU_UTIL = 0.6  # 限制显存，强制产生竞争
MAX_SEQ_LEN = 4096

def prepare_model():
    """如果本地没有模型，从HF下载"""
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from HuggingFace...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

def load_sharegpt_samples(num_samples=100, max_len=2000):
    """加载ShareGPT数据，过滤过长样本"""
    with open("/root/autodl-tmp/kv_cache_project/data/sharegpt.json", 'r') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        conv = item.get("conversations", [])
        if not conv:
            continue
        text = " ".join([c["value"] for c in conv if c["from"] in ["human", "gpt"]])
        if 50 < len(text) < max_len:
            samples.append(text)
        if len(samples) >= num_samples:
            break
    return samples

def start_vllm_server():
    """启动vLLM API服务（子进程）"""
    import subprocess
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--gpu-memory-utilization", str(GPU_UTIL),
        "--max-model-len", str(MAX_SEQ_LEN),
        "--port", "8000",
        "--block-size", "16",  # 明确指定block size
        "--swap-space", "4",   # 启用CPU swap，增加竞争
    ]
    # 在后台运行
    proc = subprocess.Popen(cmd, stdout=open("/root/autodl-tmp/kv_cache_project/results/logs/vllm_server.log", 'w'),
                           stderr=subprocess.STDOUT)
    time.sleep(60)  # 等待模型加载
    return proc

def send_requests(samples, concurrency=10, total_requests=50):
    """并发发送请求"""
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    
    def send_one(text):
        payload = {
            "model": MODEL_PATH,
            "prompt": text[:1000],  # 截断避免OOM
            "max_tokens": 200,
            "temperature": 0.7
        }
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            return r.status_code == 200
        except:
            return False
    
    # 并发发送
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_one, random.choice(samples)) for _ in range(total_requests)]
        results = [f.result() for f in futures]
    
    print(f"Completed {sum(results)}/{total_requests} requests")
    return results

if __name__ == "__main__":
    # 准备
    logger.output_path = TRACE_OUTPUT
    prepare_model()
    samples = load_sharegpt_samples()
    
    # Hook vLLM
    logger.hook_vllm()
    
    # 启动服务
    proc = start_vllm_server()
    
    try:
        # 发送请求
        send_requests(samples, concurrency=20, total_requests=100)
        
        # 等待队列清空
        time.sleep(30)
        
        # 保存Trace
        logger.flush()
        print(f"Trace saved to {TRACE_OUTPUT}")
        print(f"Total events: {len(logger.traces)}")
        
    finally:
        proc.terminate()
```

### 3.4 执行命令

```bash
cd /root/autodl-tmp/kv_cache_project
conda activate kv_cache

# 首次运行：收集Trace
python scripts/collect_trace.py

# 验证Trace
head -5 data/traces/sharegpt_trace.jsonl
```

---

## 四、第二阶段：特征工程与预测器训练

### 4.1 文件：`predictor/dataset.py`

```python
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class KVCacheDataset(Dataset):
    """
    从Trace构建预测器训练数据
    标签：未来N个token内是否被重用（二分类）
    """
    def __init__(self, trace_path, history_window=50, future_window=100, num_layers=32):
        self.history_window = history_window
        self.future_window = future_window
        self.num_layers = num_layers
        
        # 加载并解析Trace
        self.traces = self._load_traces(trace_path)
        self.samples = self._build_samples()
    
    def _load_traces(self, path):
        traces = []
        with open(path, 'r') as f:
            for line in f:
                traces.append(json.loads(line))
        return traces
    
    def _build_samples(self):
        # 按block_id分组，构建时间线
        block_events = defaultdict(list)
        for t in self.traces:
            block_events[t["block_id"]].append(t)
        
        samples = []
        for block_id, events in block_events.items():
            events.sort(key=lambda x: x["timestamp"])
            
            for i, evt in enumerate(events):
                if evt["event_type"] != "access":
                    continue
                
                # 特征
                feat = self._extract_features(events, i)
                
                # 标签：未来future_window个token内是否有access
                future_ts = evt["timestamp"] + self.future_window
                has_future = any(
                    e["timestamp"] > evt["timestamp"] and e["timestamp"] <= future_ts 
                    and e["event_type"] == "access"
                    for e in events[i+1:]
                )
                
                samples.append((feat, int(has_future)))
        
        return samples
    
    def _extract_features(self, events, idx):
        evt = events[idx]
        hist = events[max(0, idx-self.history_window):idx]
        
        # 8维特征向量
        features = [
            evt["layer_id"] / self.num_layers,           # 层深度（归一化）
            evt["num_tokens"] / 16.0,                     # block占用率（block size=16）
            1.0 if evt["is_prefix"] else 0.0,             # 是否前缀
            len(hist),                                    # 历史访问次数
            np.mean([e["timestamp"] for e in hist]) if hist else 0,  # 平均访问时间
            (evt["timestamp"] - hist[-1]["timestamp"]) if hist else 1000,  # 距离上次访问
            len([e for e in hist if e["is_prefix"]]) / max(len(hist),1),  # 历史前缀比例
            evt["timestamp"] / 10000.0,                   # 全局时间（归一化）
        ]
        return np.array(features, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        return torch.FloatTensor(feat), torch.FloatTensor([label])
```

### 4.2 文件：`predictor/model.py`

```python
import torch
import torch.nn as nn

class ReusePredictor(nn.Module):
    """轻量2层MLP，预测KV Block未来重用概率"""
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
        """给定numpy特征，返回重用概率（用于模拟器）"""
        self.eval()
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            if features.dim() == 1:
                features = features.unsqueeze(0)
            return self.net(features).squeeze().item()
```

### 4.3 文件：`predictor/train.py`

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from model import ReusePredictor
from dataset import KVCacheDataset

# 配置
TRACE_PATH = "/root/autodl-tmp/kv_cache_project/data/traces/sharegpt_trace.jsonl"
MODEL_SAVE = "/root/autodl-tmp/kv_cache_project/predictor/reuse_predictor.pt"
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
        
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, AUC={auc:.4f}, Acc={acc:.4f}")
        
        scheduler.step(1-auc)
        
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
```

### 4.4 执行命令

```bash
cd /root/autodl-tmp/kv_cache_project/predictor
python train.py
```

---

## 五、第三阶段：Trace-driven模拟器

### 5.1 文件：`simulator/policies.py`

```python
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
    """LRU基线"""
    def __init__(self):
        self.access_order = OrderedDict()
    
    def on_access(self, block_id, timestamp):
        if block_id in self.access_order:
            self.access_order.move_to_end(block_id)
    
    def on_insert(self, block_id, metadata):
        self.access_order[block_id] = metadata
    
    def select_victim(self, candidates):
        # 返回最久未访问的
        return next(iter(self.access_order))

class FIFOPolicy(EvictionPolicy):
    """FIFO基线"""
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
    """Belady最优策略（需要未来信息，仅作上界参考）"""
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
        # 更新该block的下一次访问时间
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
```

### 5.2 文件：`simulator/simulator.py`

```python
from typing import Dict, List
from policies import BlockMetadata, EvictionPolicy

class BlockManagerSimulator:
    """
    KV Cache Block管理模拟器
    """
    def __init__(self, total_blocks: int, policy: EvictionPolicy):
        self.total_blocks = total_blocks
        self.policy = policy
        self.cache = {}  # block_id -> BlockMetadata
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
```

### 5.3 文件：`simulator/evaluate.py`

```python
import json
import numpy as np
import matplotlib.pyplot as plt
from simulator import BlockManagerSimulator
from policies import LRUPolicy, FIFOPolicy, LearnedPolicy, BeladyPolicy, BlockMetadata
from predictor.dataset import KVCacheDataset
import sys

TRACE_PATH = "/root/autodl-tmp/kv_cache_project/data/traces/sharegpt_trace.jsonl"
PREDICTOR_PATH = "/root/autodl-tmp/kv_cache_project/predictor/reuse_predictor.pt"
RESULT_DIR = "/root/autodl-tmp/kv_cache_project/results"

def load_trace_as_stream(trace_path):
    """将Trace转为模拟器输入流"""
    events = []
    with open(trace_path, 'r') as f:
        for line in f:
            e = json.loads(line)
            if e["event_type"] in ["access", "allocate"]:
                # 为每个事件构造特征（简化：使用与预测器一致的特征）
                # 注意：实际应从trace中恢复完整特征，这里用dataset的构建逻辑
                events.append(e)
    return events

def build_feature(event, num_layers=32):
    """从单个trace事件构建特征（与dataset一致）"""
    return np.array([
        event["layer_id"] / num_layers,
        event["num_tokens"] / 16.0,
        1.0 if event.get("is_prefix", False) else 0.0,
        0.0,  # 历史访问次数（简化）
        0.0,
        0.0,
        0.0,
        event["timestamp"] / 10000.0,
    ], dtype=np.float32)

def run_simulation(policy_name, policy, total_blocks, trace_events):
    sim = BlockManagerSimulator(total_blocks, policy)
    
    for e in trace_events:
        feat = build_feature(e)
        sim.access(e["block_id"], feat, e["timestamp"])
    
    return sim.get_hit_rate(), sim.get_stats()

def main():
    # 加载trace
    with open(TRACE_PATH, 'r') as f:
        trace_events = [json.loads(line) for line in f]
    
    # 过滤只保留access和allocate（allocate视为miss）
    stream = [e for e in trace_events if e["event_type"] in ["access", "allocate"]]
    
    # 不同显存容量配置（Block数）
    block_configs = [100, 200, 500, 1000, 2000, 5000]
    
    results = {name: [] for name in ["LRU", "FIFO", "Learned"]}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for num_blocks in block_configs:
        print(f"\n=== Total Blocks: {num_blocks} ===")
        
        # LRU
        hr, st = run_simulation("LRU", LRUPolicy(), num_blocks, stream)
        results["LRU"].append(hr)
        print(f"LRU  Hit Rate: {hr:.4f}")
        
        # FIFO
        hr, st = run_simulation("FIFO", FIFOPolicy(), num_blocks, stream)
        results["FIFO"].append(hr)
        print(f"FIFO Hit Rate: {hr:.4f}")
        
        # Learned
        learned = LearnedPolicy(PREDICTOR_PATH, device=device)
        hr, st = run_simulation("Learned", learned, num_blocks, stream)
        results["Learned"].append(hr)
        print(f"Learned Hit Rate: {hr:.4f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    for name, hrs in results.items():
        plt.plot(block_configs, hrs, marker='o', label=name)
    plt.xlabel("Total Blocks (Cache Capacity)")
    plt.ylabel("Hit Rate")
    plt.title("KV Cache Block Hit Rate vs Capacity")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{RESULT_DIR}/figures/hit_rate_comparison.png")
    print(f"\nFigure saved to {RESULT_DIR}/figures/hit_rate_comparison.png")
    
    # 保存数据
    import json
    with open(f"{RESULT_DIR}/logs/simulation_results.json", 'w') as f:
        json.dump({"block_configs": block_configs, "results": results}, f, indent=2)

if __name__ == "__main__":
    import torch
    main()
```

### 5.4 执行命令

```bash
cd /root/autodl-tmp/kv_cache_project/simulator
python evaluate.py
```

---

## 六、第四阶段：真实系统集成（Tier 2，加分项）

### 6.1 文件：`real_system/learned_evictor.py`

```python
"""
vLLM自定义Evictor插件（需手动集成到vLLM源码）
目标文件：vllm/core/evictor.py
"""
import torch
import numpy as np
from typing import List
from vllm.core.evictor import Evictor, Block

class LearnedEvictor(Evictor):
    """
    基于预测器的驱逐策略
    使用方式：修改vLLM配置，指定eviction_policy="learned"
    """
    def __init__(self, model_path, device='cuda'):
        from predictor.model import ReusePredictor
        self.model = ReusePredictor().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device
    
    def get_block_to_evict(self, blocks: List[Block]) -> Block:
        if not blocks:
            raise ValueError("No blocks to evict")
        
        # 提取特征（需与BlockManager协调，在Block中预存特征）
        scores = []
        with torch.no_grad():
            for block in blocks:
                # 假设block有cached_features属性
                feat = getattr(block, 'cached_features', None)
                if feat is None:
                    # 无特征时退化为LRU（给低分）
                    scores.append(0.0)
                    continue
                
                feat_t = torch.FloatTensor(feat).unsqueeze(0).to(self.device)
                score = self.model(feat_t).item()
                scores.append(score)
        
        # 驱逐重用概率最低的
        victim_idx = int(np.argmin(scores))
        return blocks[victim_idx]
```

### 6.2 集成步骤（手动）

1. 找到AutoDL环境中vLLM的安装路径：`python -c "import vllm; print(vllm.__path__)"`
2. 备份并修改 `vllm/core/evictor.py`，添加 `LearnedEvictor` 类
3. 在 `vllm/core/block_manager.py` 中为 `PhysicalTokenBlock` 增加特征缓存
4. 启动vLLM时通过环境变量或参数启用

---

## 七、一键执行脚本

### 文件：`scripts/run_all.sh`

```bash
#!/bin/bash
set -e

PROJECT_DIR="/root/autodl-tmp/kv_cache_project"
cd $PROJECT_DIR

echo "===== Step 1: Collecting Trace ====="
python scripts/collect_trace.py

echo "===== Step 2: Training Predictor ====="
cd predictor
python train.py
cd ..

echo "===== Step 3: Running Simulation ====="
cd simulator
python evaluate.py
cd ..

echo "===== Done ====="
echo "Results:"
echo "  - Trace: data/traces/"
echo "  - Model: predictor/reuse_predictor.pt"
echo "  - Figures: results/figures/"
echo "  - Logs: results/logs/"
```

赋予执行权限：
```bash
chmod +x /root/autodl-tmp/kv_cache_project/scripts/run_all.sh
```

---

## 八、实验检查清单

| 阶段 | 检查项 | 通过标准 |
|------|--------|---------|
| **环境** | vLLM安装成功 | `python -c "import vllm; print(vllm.__version__)"` 无报错 |
| **环境** | GPU可识别 | `nvidia-smi` 显示显存 |
| **Trace** | 收集到事件 | `sharegpt_trace.jsonl` 行数 > 1000 |
| **Trace** | 事件类型完整 | 包含allocate/access/evict/free |
| **预测器** | 训练收敛 | Val AUC > 0.65（基线要求） |
| **模拟器** | LRU可运行 | 不同Block数下命中率单调递增 |
| **模拟器** | Learned优于LRU | 在至少一个配置下命中率提升 > 5% |
| **真实系统** | vLLM可启动 | 能成功响应请求（可选） |

---

## 九、论文图表预期产出

1. **Trace特征分析图**：KV Block访问的时间分布热力图、层间访问频率
2. **预测器ROC曲线**：二分类性能
3. **命中率对比图**：LRU vs FIFO vs Learned vs Belady（不同容量）
4. **显存周转图**：驱逐次数对比
5. **端到端延迟图**（Tier 2）：TTFT/TPOT对比（如有）

---

## 十、风险应对

| 风险 | 应对 |
|------|------|
| AutoDL实例无预装vLLM | 执行 `pip install vllm==0.5.0` |
| 模型下载慢/失败 | 改用ModelScope镜像或AutoDL模型库 |
| Trace文件过大 | 按请求分片存储，定期flush |
| 预测器AUC过低 | 增加特征（如attention score）、调整future_window |
| 模拟器运行慢 | 减少block_configs测试点，先用1000 events验证 |
| vLLM版本API变动 | 锁定版本 `vllm==0.5.0`，不追新 |

---

## 附录：AutoDL常用命令

```bash
# 查看数据盘
df -h | grep autodl-tmp

# 后台运行（收集Trace时）
nohup python scripts/collect_trace.py > results/logs/collect.log 2>&1 &

# 实时监控GPU
watch -n 1 nvidia-smi

# 压缩结果下载
tar -czvf results.tar.gz results/
# 通过AutoDL控制台下载
```

---

**文档版本**：v1.0  
**适用场景**：AutoDL单卡实验（RTX 3090/4090/A10/A100）  
**预期周期**：2-3周完成Tier 1，第4周尝试Tier 2
```

这份文档可以直接复制保存为 `experiment_plan.md`，丢给你的Agent执行。每个代码块都有明确路径和依赖说明，Agent可以按顺序逐个文件生成并执行。