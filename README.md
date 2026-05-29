# kv-cache-learned
面向大模型推理的 KV Cache 学习型替换策略。通过轻量预测器估计缓存块未来重用概率，在显存受限场景下优于传统 LRU。包含 Trace 收集、预测器训练、Trace-driven 模拟器及 vLLM 集成验证。
