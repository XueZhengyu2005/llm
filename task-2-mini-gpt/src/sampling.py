"""采样策略：greedy / top-k / top-p / temperature。"""
import torch
import torch.nn.functional as F


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """logits: (B, V) -> (B,) 采样的下一个 token id。"""
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k is not None:
        k = min(max(int(top_k), 0), logits.size(-1))
        if k > 0:
            values, indices = logits.topk(k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(dim=-1, index=indices, src=values)
            logits = filtered

    if top_p is not None:
        top_p = float(top_p)
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p 必须位于 (0, 1]")
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # 保留累计概率首次超过 top_p 的位置及其之前的所有 token
        remove = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
