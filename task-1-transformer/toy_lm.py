"""M4：同一个手写 attention + causal mask 跑一个 toy 字符级语言模型。"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from src.attention import MultiHeadAttention, causal_mask


ROOT = Path(__file__).parent
TEXT = (ROOT.parent / "poetryFromTang.txt").read_text(encoding="utf-8")
BLOCK = 64
VOCAB = sorted(set(TEXT))
STOI = {ch: i for i, ch in enumerate(VOCAB)}


class ToyLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(BLOCK, d_model)
        self.attns = nn.ModuleList([MultiHeadAttention(d_model, n_heads) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        ) for _ in range(n_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2 * n_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        B, T = ids.shape
        mask = causal_mask(T, device=ids.device)              # (T,T) True=屏蔽未来
        x = self.embed(ids) + self.pos(torch.arange(T, device=ids.device))
        for i, (attn, ffn) in enumerate(zip(self.attns, self.ffns)):
            a, _ = attn(self.lns[2 * i](x), mask)
            x = x + a
            x = x + ffn(self.lns[2 * i + 1](x))
        return self.head(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ids = torch.tensor([STOI[c] for c in TEXT if c in STOI], dtype=torch.long)
    model = ToyLM(len(VOCAB)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    step = 0
    for epoch in range(3):
        perm = torch.randperm(len(ids) - BLOCK - 1)
        for i in range(0, len(perm), 32):
            batch_idx = perm[i:i + 32]
            batch = torch.stack([ids[j:j + BLOCK + 1] for j in batch_idx]).to(device)
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), batch[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
        print(f"epoch {epoch + 1} loss={loss.item():.4f}")

    model.eval()
    prompt = "巴山上峡"
    ctx = torch.tensor([[STOI[c] for c in prompt]], dtype=torch.long, device=device)
    out = []
    with torch.no_grad():
        for _ in range(32):
            logits = model(ctx[:, -BLOCK:])
            nxt = logits[0, -1].argmax()
            ctx = torch.cat([ctx, nxt.view(1, 1)], dim=1)
            out.append(VOCAB[nxt])
    print("生成样例：", prompt + "".join(out))


if __name__ == "__main__":
    main()
