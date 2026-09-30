"""MiniGPT：decoder-only 语言模型（RoPE + KV cache + weight tying）。"""
import codecs
import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .sampling import sample_next_token
from .tokenizer import BPETokenizer


class Block(nn.Module):
    def __init__(self, d_model, n_heads, block_size, ff_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, new_cache


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_heads=6, n_layers=4,
                 block_size=128, ff_mult=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.max_seq_len = block_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, block_size, ff_mult, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        # weight tying 时 N(0,1) 的 embedding 会让初始 logits 尺度爆炸，必须用小方差初始化
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, ids, kv_cache=None, return_cache=False):
        """ids: (B, T)。返回 logits (B, T, V)；return_cache 时返回 (logits, cache)。

        kv_cache 是每层 (K, V) 组成的 list（或 None），增量解码时原样传回。
        """
        x = self.embed(ids)
        new_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, cache = block(x, layer_cache)
            new_cache.append(cache)
        logits = self.lm_head(self.ln_f(x))
        if return_cache:
            return logits, new_cache
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, top_k=None, top_p=None, temperature=1.0, repetition_penalty=1.0, token_bytes=None, utf8_safe=False):
        """prompt_ids: (B, T) LongTensor -> (B, T + max_new_tokens)。内部使用 KV cache。"""
        if not torch.is_tensor(prompt_ids):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids 必须是形状 (B, T) 的二维张量")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 不能为负数")

        # 超长 prompt 采用左侧滑动窗口，保留最新上下文。
        ids = prompt_ids[:, -self.block_size:]
        logits, cache = self(ids, return_cache=True)
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :].clone()
            if utf8_safe and token_bytes is not None:
                # byte-level BPE 可能生成无法组成 UTF-8 的字节；只保留仍是合法
                # UTF-8 前缀的候选 token，避免样例中出现替换字符。
                for batch in range(next_logits.size(0)):
                    prefix = b"".join(token_bytes[int(token)] for token in ids[batch].tolist())
                    allowed = []
                    for token, raw in enumerate(token_bytes):
                        try:
                            codecs.getincrementaldecoder("utf-8")().decode(prefix + raw, final=False)
                            allowed.append(token)
                        except UnicodeDecodeError:
                            pass
                    if allowed:
                        allowed_mask = torch.zeros_like(next_logits[batch], dtype=torch.bool)
                        allowed_mask[allowed] = True
                        next_logits[batch] = next_logits[batch].masked_fill(~allowed_mask, float("-inf"))
            if repetition_penalty is not None and repetition_penalty > 1.0:
                for batch in range(next_logits.size(0)):
                    for token_id in set(ids[batch].tolist()):
                        if next_logits[batch, token_id] >= 0:
                            next_logits[batch, token_id] /= repetition_penalty
                        else:
                            next_logits[batch, token_id] *= repetition_penalty
            next_id = sample_next_token(next_logits, temperature, top_k, top_p)
            ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)
            if ids.size(1) >= self.block_size:
                # cache 到达窗口上限后重新 prefill，避免 RoPE/cache 越界。
                context = ids[:, -self.block_size:]
                logits, cache = self(context, return_cache=True)
            else:
                logits, cache = self(next_id.unsqueeze(1), kv_cache=cache, return_cache=True)
        return ids


def load_for_eval(ckpt_path):
    """按自检契约返回 (model, tokenizer)。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = MiniGPT(
        vocab_size=ckpt["vocab_size"],
        d_model=config["d_model"], n_heads=config["n_heads"],
        n_layers=config["n_layers"], block_size=config["block_size"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tokenizer = BPETokenizer()
    for left, right in ckpt["merges"]:
        tokenizer._add_merge(left, right)
    return model, tokenizer
