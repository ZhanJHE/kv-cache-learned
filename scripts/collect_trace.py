#!/usr/bin/env python3
"""
Trace 收集脚本：启动带 Hook 的 vLLM 服务并发送请求，收集 KV Cache 访问 Trace
"""
import os
import sys
import json
import random
import requests
import time
import subprocess
from pathlib import Path

# 计算项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 配置
MODEL_PATH = os.environ.get("KV_CACHE_MODEL", "/root/autodl-tmp/models/qwen2.5-7b-instruct")
TRACE_OUTPUT = PROJECT_ROOT / "data" / "traces" / "sharegpt_trace.jsonl"
GPU_UTIL = 0.6
MAX_SEQ_LEN = 4096
SHAREGPT_PATH = PROJECT_ROOT / "data" / "sharegpt.json"


def prepare_model():
    """如果本地没有模型，从 HuggingFace 下载"""
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from HuggingFace...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        os.makedirs(MODEL_PATH, exist_ok=True)
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")


def load_sharegpt_samples(num_samples=100, max_len=2000):
    """加载 ShareGPT 数据，过滤过长样本"""
    with open(SHAREGPT_PATH, 'r', encoding='utf-8') as f:
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
    """启动带 Hook 的 vLLM API 服务（子进程）"""
    log_path = PROJECT_ROOT / "results" / "logs" / "vllm_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper_path = PROJECT_ROOT / "vllm_patch" / "api_server_wrapper.py"
    cmd = [
        sys.executable, str(wrapper_path),
        "--model", MODEL_PATH,
        "--gpu-memory-utilization", str(GPU_UTIL),
        "--max-model-len", str(MAX_SEQ_LEN),
        "--port", "8000",
        "--block-size", "16",
        "--swap-space", "4",
    ]

    env = os.environ.copy()
    env["KV_TRACE_OUTPUT"] = str(TRACE_OUTPUT)
    project_root_str = str(PROJECT_ROOT)
    pythonpath = env.get("PYTHONPATH", "")
    if project_root_str not in pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = f"{project_root_str}{os.pathsep}{pythonpath}" if pythonpath else project_root_str

    with open(log_path, 'w', encoding='utf-8') as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    time.sleep(60)  # 等待模型加载
    return proc


def send_requests(samples, concurrency=10, total_requests=50):
    """并发发送请求"""
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}

    def send_one(text):
        payload = {
            "model": MODEL_PATH,
            "prompt": text[:1000],
            "max_tokens": 200,
            "temperature": 0.7
        }
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            return r.status_code == 200
        except Exception:
            return False

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_one, random.choice(samples)) for _ in range(total_requests)]
        results = [f.result() for f in futures]

    print(f"Completed {sum(results)}/{total_requests} requests")
    return results


if __name__ == "__main__":
    # 准备
    TRACE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prepare_model()
    samples = load_sharegpt_samples()

    # 启动服务（Hook 在子进程 wrapper 内自动完成）
    proc = start_vllm_server()

    try:
        send_requests(samples, concurrency=20, total_requests=100)
        # 等待队列清空，并给 wrapper 留时间 flush
        time.sleep(30)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # 检查并报告 Trace 结果
    if TRACE_OUTPUT.exists():
        with open(TRACE_OUTPUT, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f if _.strip())
        print(f"Trace saved to {TRACE_OUTPUT}")
        print(f"Total events: {line_count}")
        if line_count < 1000:
            print(f"[Warning] Trace 事件数较少 ({line_count})，可能 Hook 未生效或请求未完成")
    else:
        print(f"[Error] Trace file not found: {TRACE_OUTPUT}")
