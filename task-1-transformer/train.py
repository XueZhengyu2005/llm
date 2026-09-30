"""任务一训练入口：ChnSentiCorp 情感分类。"""
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

from src.model import TransformerClassifier
from src.tokenizer import CharTokenizer


ROOT = Path(__file__).parent
MAX_LEN = 256
BATCH = 32
EPOCHS = 10
LR = 3e-4
PATIENCE = 3
CONFIG = dict(num_classes=2, d_model=128, n_heads=4, n_layers=4,
              ff_mult=4, max_len=MAX_LEN, dropout=0.2)


class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.items = [(tokenizer.encode(t)[:MAX_LEN], int(l))
                      for t, l in zip(df["text"], df["label"])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    ids_list, labels = zip(*batch)
    max_len = max(len(ids) for ids in ids_list)
    ids = torch.zeros(len(ids_list), max_len, dtype=torch.long)
    mask = torch.ones(len(ids_list), max_len, dtype=torch.bool)
    for i, seq in enumerate(ids_list):
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = False
    labels = torch.tensor(labels, dtype=torch.long)
    return ids, mask, labels


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        pred = model(ids, padding_mask=mask).argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / total


def lr_lambda(step, warmup=300, total=None):
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    train_df = pd.read_parquet(ROOT / "data" / "train.parquet")
    dev_df = pd.read_parquet(ROOT / "data" / "validation.parquet")
    tokenizer = CharTokenizer(train_df["text"].tolist())
    print(f"vocab_size={tokenizer.vocab_size} train={len(train_df)} dev={len(dev_df)} device={device}")

    train_loader = DataLoader(TextDataset(train_df, tokenizer), batch_size=BATCH,
                              shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(TextDataset(dev_df, tokenizer), batch_size=128,
                            shuffle=False, collate_fn=collate)

    model = TransformerClassifier(tokenizer.vocab_size, **CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, total=total_steps))
    criterion = torch.nn.CrossEntropyLoss()

    (ROOT / "ckpt").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    best_acc, bad_epochs = 0.0, 0
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = n_batches = 0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(ids, padding_mask=mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
            loss_history.append(loss.item())

        acc = evaluate(model, dev_loader, device)
        print(f"epoch {epoch:02d} loss={total_loss/n_batches:.4f} dev_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            bad_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "vocab": tokenizer.stoi,
                "vocab_size": tokenizer.vocab_size,
                "config": CONFIG,
            }, ROOT / "ckpt" / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"early stop at epoch {epoch} (best dev_acc={best_acc:.4f})")
                break

    print(f"best dev_acc={best_acc:.4f} -> ckpt/best.pt")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history)
        plt.xlabel("step")
        plt.ylabel("cross-entropy loss")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.savefig(ROOT / "figures" / "loss_curve.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
