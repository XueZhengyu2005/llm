"""任务二训练入口：唐诗 quick-start 档，目标 dev PPL < 50。"""
import math
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from src.model import MiniGPT
from src.tokenizer import BPETokenizer


def decode_display(tokenizer, ids):
    data = b"".join(tokenizer.vocab[int(i)] for i in ids)
    return data.decode("utf-8", errors="ignore")


ROOT = Path(__file__).parent
BLOCK = 128
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
MERGES = 64
STEPS = 3000
BATCH = 16
LR = 3e-4
DROPOUT = 0.1
EVAL_EVERY = 200


@torch.no_grad()
def dev_ppl(model, dev_ids, device):
    """与 eval/run.py 相同口径：按 block_size 非重叠分块累加 NLL。"""
    model.eval()
    nll, n_tok = 0.0, 0
    for i in range(0, max(1, len(dev_ids) - 1), model.block_size):
        window = dev_ids[i:i + model.block_size + 1]
        if len(window) < 2:
            break
        chunk = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(chunk[:, :-1])
        nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1), reduction="sum").item()
        n_tok += chunk.size(1) - 1
    return math.exp(nll / n_tok) if n_tok else float("inf")


def _limit_text_by_mb(text, max_mb):
    if max_mb <= 0 or len(text.encode("utf-8")) <= max_mb * 1024 * 1024:
        return text
    target = max_mb * 1024 * 1024
    byte_len = len(text.encode("utf-8"))
    char_target = max(1, int(len(text) * target / byte_len))
    cut = text.rfind("\n\n", 0, char_target)
    if cut < 0:
        cut = target
    return text[:cut]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=int, default=MERGES)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--train-mb", type=int, default=50,
                    help="最多使用多少 MB 的 train.txt；0 表示全部")
    ap.add_argument("--tokenizer-mb", type=int, default=20,
                    help="用于训练 BPE 的文本大小；0 表示全部 train 文本")
    ap.add_argument("--reuse-tokenizer", action="store_true",
                    help="复用 ckpt/tokenizer.json，跳过大语料 BPE 重训")
    ap.add_argument("--dev-mb", type=int, default=1,
                    help="训练过程最多编码多少 MB 的 dev 文本；0 表示全部")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    full_train_text = (ROOT / "data" / "train.txt").read_text(encoding="utf-8")
    full_dev_text = (ROOT / "data" / "dev.txt").read_text(encoding="utf-8")
    train_text = _limit_text_by_mb(full_train_text, args.train_mb)
    dev_text = _limit_text_by_mb(full_dev_text, args.dev_mb)

    print(f"训练文本：{len(train_text.encode('utf-8')) / 1024 / 1024:.1f} MB")
    print("训练 BPE tokenizer ...")
    (ROOT / "ckpt").mkdir(parents=True, exist_ok=True)
    tokenizer_path = ROOT / "ckpt" / "tokenizer.json"
    if args.reuse_tokenizer and tokenizer_path.exists():
        print(f"复用 tokenizer：{tokenizer_path}")
        tok = BPETokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer_text = _limit_text_by_mb(train_text, args.tokenizer_mb)
        tok = BPETokenizer().train(tokenizer_text, num_merges=args.merges)
        tok.save(tokenizer_path)
    train_ids = tok.encode(train_text)
    dev_ids = tok.encode(dev_text)
    print(f"vocab={tok.vocab_size} train_tokens={len(train_ids)} dev_tokens={len(dev_ids)}")

    model = MiniGPT(tok.vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                    n_layers=N_LAYERS, block_size=BLOCK, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params/1e6:.2f}M device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=args.steps, pct_start=0.1)

    best_ppl = float("inf")
    ids_tensor = torch.tensor(train_ids, dtype=torch.long)
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, len(train_ids) - BLOCK - 1, (BATCH,))
        batch = torch.stack([ids_tensor[i:i + BLOCK + 1] for i in idx]).to(device)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        if step % EVAL_EVERY == 0:
            ppl = dev_ppl(model, dev_ids, device)
            marker = ""
            if ppl < best_ppl:
                best_ppl = ppl
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                                   block_size=BLOCK),
                    "merges": [list(p) for p in tok.merge_rules],
                    "vocab_size": tok.vocab_size,
                }, ROOT / "ckpt" / "best.pt")
                marker = " *"
            print(f"step {step:4d} loss={loss.item():.4f} dev_ppl={ppl:.2f}{marker}")

    print(f"best dev_ppl={best_ppl:.2f} -> ckpt/best.pt")

    # 四种采样策略对比样例
    model.eval()
    prompt = "从前,有一个小男孩,"
    prompt_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    strategies = [
        ("greedy", dict(temperature=0)),
        ("top_k=5,t=0.3", dict(top_k=5, temperature=0.3)),
        ("top_p=0.9,t=0.4", dict(top_p=0.9, temperature=0.4)),
        ("t=0.6", dict(temperature=0.6)),
    ]
    print("\n生成样例：")
    for name, kwargs in strategies:
        torch.manual_seed(7)
        out = model.generate(prompt_ids, max_new_tokens=28, repetition_penalty=1.1,
                             token_bytes=tok.vocab, utf8_safe=True, **kwargs)
        print(f"[{name}] {decode_display(tok, out[0].tolist())}")


if __name__ == "__main__":
    main()
