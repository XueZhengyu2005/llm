"""TransformerClassifier：字符 embedding + N 层 block + masked mean pooling + 分类头。"""
import torch
import torch.nn as nn

from .block import TransformerBlock
from .tokenizer import CharTokenizer


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=2, d_model=128, n_heads=4,
                 n_layers=4, ff_mult=4, max_len=256, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_mult, dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, ids, padding_mask=None, return_attention=False):
        """ids: (B, T)；padding_mask: (B, T) True=pad。返回 (B, num_classes)。"""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.embedding(ids) + self.pos_embedding(pos)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask[:, None, None, :]  # (B,1,1,T) 广播到所有 head/query

        all_attn = []
        for block in self.blocks:
            x, attn = block(x, attn_mask)
            if return_attention:
                all_attn.append(attn)

        if padding_mask is None:
            pooled = x.mean(dim=1)
        else:
            keep = (~padding_mask).unsqueeze(-1).float()          # (B,T,1)
            pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1e-9)

        logits = self.classifier(pooled)
        if return_attention:
            return logits, torch.stack(all_attn, dim=1)  # (B, layers, heads, T, T)
        return logits


def load_for_eval(ckpt_path):
    """按自检契约返回 (model, tokenize_fn)。tokenize_fn(text) -> LongTensor (T,)。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = TransformerClassifier(
        vocab_size=ckpt["vocab_size"],
        num_classes=config["num_classes"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_len=config["max_len"],
        dropout=0.0,  # 评估时关闭 dropout
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tokenizer = CharTokenizer.from_vocab(ckpt["vocab"])
    max_len = config["max_len"]

    def tokenize_fn(text):
        ids = tokenizer.encode(text)[:max_len]
        if not ids:
            ids = [tokenizer.stoi["<unk>"]]
        return torch.tensor(ids, dtype=torch.long)

    return model, tokenize_fn
