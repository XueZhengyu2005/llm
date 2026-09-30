"""手写 scaled dot-product attention 与 multi-head attention。

约定：mask 为布尔张量，True 表示该位置被屏蔽（不参与注意力）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None, return_attention=False):
    """缩放点积注意力，True mask 表示屏蔽。"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    all_masked = None
    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(mask, float("-inf"))
        # 全 PAD query/key 时 softmax(-inf) 会产生 NaN；该行应返回零注意力。
        all_masked = mask.expand_as(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_masked, 0.0)
    attn = F.softmax(scores, dim=-1)
    if all_masked is not None:
        attn = attn.masked_fill(all_masked, 0.0)
    output = torch.matmul(attn, V)
    return (output, attn) if return_attention else output


class MultiHeadAttention(nn.Module):
    """多头自注意力：Q/K/V 投影 + 拼分 head + 输出投影。"""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """x: (B, T, D) -> out: (B, T, D), attn: (B, H, T, T)"""
        B, T, _ = x.shape
        H, Dk = self.n_heads, self.d_k

        q = self.q_proj(x).view(B, T, H, Dk).transpose(1, 2)  # (B,H,T,Dk)
        k = self.k_proj(x).view(B, T, H, Dk).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dk).transpose(1, 2)

        context, attn = scaled_dot_product_attention(q, k, v, mask, return_attention=True)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)                       # (B,H,T,Dk)

        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(context)
        return out, attn


def causal_mask(T, device=None):
    """返回 (T, T) 上三角布尔 mask，True = 屏蔽未来位置。"""
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
