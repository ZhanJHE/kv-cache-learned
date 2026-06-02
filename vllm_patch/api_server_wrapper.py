#!/usr/bin/env python3
"""
vLLM API Server Wrapper: 在启动 vLLM 服务前自动 Hook BlockManager 以收集 Trace

用法：
    python vllm_patch/api_server_wrapper.py [标准 vLLM 参数...]

环境变量：
    KV_TRACE_OUTPUT: trace 输出文件路径（默认: data/traces/sharegpt_trace.jsonl）
"""
import os
import sys

# 将项目根目录加入 Python 路径，确保能导入 trace_logger
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from vllm_patch.trace_logger import KVTraceLogger

# 初始化 Logger 并 Hook
TRACE_OUTPUT = os.environ.get("KV_TRACE_OUTPUT", "data/traces/sharegpt_trace.jsonl")
os.makedirs(os.path.dirname(TRACE_OUTPUT), exist_ok=True)

logger = KVTraceLogger(TRACE_OUTPUT)
logger.hook_vllm()

# 启动 vLLM 入口
from vllm.entrypoints.openai.api_server import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        logger.flush()
