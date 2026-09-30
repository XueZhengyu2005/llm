"""Causal multi-head attention，支持 KV cache 增量解码。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope, precompute_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.0, base=10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        cos, sin = precompute_rope(self.d_k, block_size, base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, kv_cache=None):
        """x: (B, T_new, D)。kv_cache: 该层 (K, V)，形状 (B, H, T_past, Dk)。

        返回 (out, new_cache)。cache 中保存的 K 已经加过 RoPE，增量时不再重复旋转。
        """
        B, T, D = x.shape
        H, Dk = self.n_heads, self.d_k

        q = self.q_proj(x).view(B, T, H, Dk).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Dk).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dk).transpose(1, 2)

        past_len = 0
        if kv_cache is not None:
            past_len = kv_cache[0].size(2)

        if T == 0:
            return x, kv_cache
        if past_len + T > self.cos.size(0):
            raise ValueError(
                f"序列长度超过 block_size={self.cos.size(0)}: past={past_len}, new={T}"
            )

        # RoPE 只加 Q/K；位置从 past_len 续接，保证增量解码与全量前向一致
        pos = slice(past_len, past_len + T)
        q = apply_rope(q, self.cos[pos], self.sin[pos])
        k = apply_rope(k, self.cos[pos], self.sin[pos])

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        S = k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dk)

        if T > 1:
            q_pos = torch.arange(past_len, past_len + T, device=x.device)
            k_pos = torch.arange(S, device=x.device)
            mask = k_pos[None, :] > q_pos[:, None]  # (T, S) True=屏蔽未来
            scores = scores.masked_fill(mask, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v)  # (B,H,T,Dk)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context), (k, v)
