"""Transformer encoder block：Pre-LN 结构（attention + FFN + 双残差）。"""
import torch.nn as nn

from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """Pre-LN：先归一再进子层，残差加在归一前后的原表示上。"""
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, attn_weights
