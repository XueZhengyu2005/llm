"""M5：对正/负/长句各画一张注意力热图（选第 4 层、head 0）。"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent))
from src.model import load_for_eval


ROOT = Path(__file__).parent


def plot_attention(text, label, model, tokenizer, out_name, layer=-1, head=0, max_tokens=40):
    ids = tokenizer(text)[:max_tokens].unsqueeze(0)
    with torch.no_grad():
        _, attn = model(ids, padding_mask=None, return_attention=True)
    weights = attn[0, layer, head].cpu().numpy()  # (T, T)
    tokens = list(text[:max_tokens])

    plt.figure(figsize=(8, 7))
    plt.imshow(weights, cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, fontsize=9)
    plt.yticks(range(len(tokens)), tokens, fontsize=9)
    plt.colorbar(label="attention weight")
    plt.xlabel("Key 位置")
    plt.ylabel("Query 位置")
    plt.title(f"Attention（{out_name.stem}，label={label}）")
    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close()
    print(f"saved {out_name}")


def main():
    model, tokenize = load_for_eval(str(ROOT / "ckpt" / "best.pt"))
    dev = pd.read_parquet(ROOT / "data" / "validation.parquet")

    pos = dev[dev.label == 1].iloc[0]["text"]
    neg = dev[dev.label == 0].iloc[0]["text"]
    longest = dev.loc[dev.text.str.len().idxmax()]

    plot_attention(pos, 1, model, tokenize, ROOT / "figures" / "attention_positive.png")
    plot_attention(neg, 0, model, tokenize, ROOT / "figures" / "attention_negative.png")
    plot_attention(longest["text"], int(longest["label"]), model, tokenize,
                   ROOT / "figures" / "attention_long.png")


if __name__ == "__main__":
    main()
