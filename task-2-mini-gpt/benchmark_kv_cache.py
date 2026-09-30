"""Task 2 S3: KV cache 开/关推理速度对比（同样词元数计时）。"""
import time
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "task-2-mini-gpt" else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "task-2-mini-gpt"))

from src.model import load_for_eval


@torch.no_grad()
def benchmark(model, tokenizer, prompt_ids, num_new_tokens=100, use_cache=True, warmup=3):
    """计时生成 num_new_tokens 个 token 的耗时。"""
    # warmup
    for _ in range(warmup):
        if use_cache:
            model.generate(prompt_ids, max_new_tokens=10, temperature=0)
        else:
            # 逐步重算全序列
            ids = prompt_ids.clone()
            for _ in range(10):
                logits = model(ids[:, -model.block_size:])
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=-1)

    if prompt_ids.device.type == "cuda":
        torch.cuda.synchronize(prompt_ids.device)
    start = time.perf_counter()
    
    if use_cache:
        model.generate(prompt_ids, max_new_tokens=num_new_tokens, temperature=0)
    else:
        ids = prompt_ids.clone()
        for _ in range(num_new_tokens):
            logits = model(ids[:, -model.block_size:])
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=-1)
    
    if prompt_ids.device.type == "cuda":
        torch.cuda.synchronize(prompt_ids.device)
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_for_eval(str(ROOT / "task-2-mini-gpt" / "ckpt" / "best.pt"))
    model.eval()
    model.to(device)
    
    prompt_text = "春风得意马蹄疾"
    prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], device=device)
    print(f"Prompt: {prompt_text} ({len(prompt_ids[0])} tokens)")
    print(f"Block size: {model.block_size}")
    
    results = []
    for n_tokens in [50, 100, 108]:
        t_cache = benchmark(model, tokenizer, prompt_ids, n_tokens, use_cache=True, warmup=2)
        t_no_cache = benchmark(model, tokenizer, prompt_ids, n_tokens, use_cache=False, warmup=2)
        speedup = t_no_cache / t_cache if t_cache > 0 else 0
        row = {"tokens": n_tokens, "cache_seconds": t_cache,
               "no_cache_seconds": t_no_cache, "speedup": speedup}
        results.append(row)
        print(f"\n{'='*40}")
        print(f"生成 {n_tokens} tokens:")
        print(f"  KV cache:    {t_cache:.3f}s ({n_tokens/t_cache:.1f} tok/s)")
        print(f"  No cache:    {t_no_cache:.3f}s ({n_tokens/t_no_cache:.1f} tok/s)")
        print(f"  Speedup:     {speedup:.1f}x")
    import json
    (ROOT / "task-2-mini-gpt" / "kv_benchmark.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
