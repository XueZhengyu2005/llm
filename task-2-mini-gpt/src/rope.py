"""RoPE 旋转位置编码（前后折半配对，GPT-NeoX 风格），只作用于 Q/K。"""
import torch


def precompute_rope(dim, max_len, base=10000.0):
    """返回 (max_len, dim/2) 的 cos / sin 表。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(max_len).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, D)；cos/sin: (T, D/2)。把 x 的前后两半视为旋转对。"""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
