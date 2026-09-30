# LLM 实验整理目录

本目录只收录六项任务的**代码、实验数据、实验结果与重写的[实验报告](实验报告.md)**。这是原项目的整理副本；原目录未移动或删除。

| 目录 | 内容 | 主要结果 |
| --- | --- | --- |
| `task-1-transformer/` | 手写 Transformer、情感分类数据、热图 | `eval/result.json`、`figures/` |
| `task-2-mini-gpt/` | BPE/MiniGPT、训练/验证语料、生成样例 | `eval/result.json`、`samples.txt`、`kv_benchmark.json` |
| `task-3-sft-dpo/` | LoRA/SFT/DPO、偏好数据、训练配置 | `eval/result.json`、`compare_output.txt` |
| `task-4-rag/` | PDF、问答数据、FAISS 索引、RAG 代码 | `eval/result.json`、`eval/ablation_results.json` |
| `task-5-tool-agent/` | ReAct Agent、题集、逐题轨迹 | `eval/result.json`、`eval/*traces.jsonl` |
| `task-6-coding-agent/` | MCP、Skills、toy-repo | `eval/result.json`、`eval/toy_repo_trace.json` |

- `requirements.txt` 各任务单独保留，`_eval_harness.py` 为评测共用代码；`poetryFromTang.txt` 是任务二下载脚本使用的原始小语料。
- **不收录模型权重**：`best.pt`、`lora.pt`、各模型目录里的 `.safetensors` 等均不在这里。`task-2-mini-gpt/ckpt/tokenizer.json` 仅为分词器词表；任务三的 `ckpt/*/config.json` 仅为实验配置；任务四的 `data/index/` 是文档检索索引而非模型参数。
- 任务二只保留切分后的 `train.txt` / `dev.txt`，不重复拷贝生成它们的原始 parquet；未收录临时缓存、`__pycache__`、toy-repo 内嵌 `.git`。任务三的 MOSS SFT 原始语料未在原项目中保留；按需运行 `data/download.py` 重新准备。历史 `dataset_info.json` / 配置 / trace 中的本机绝对路径用于记录原始实验位置，迁移后不能当作新目录路径。
- **阅读已有结果无需权重**。如需重新训练/评测，先在目标任务安装其 `requirements.txt`，按对应代码中 `models/`、`ckpt/` 的约定准备外部基础模型或重新训练 checkpoint，再从该任务目录运行对应脚本。评测会写入 `eval/result.json`；本整理中的原结果即当前报告的依据。部分脚本使用本地模型或网络服务，所需资源按各任务代码配置。
