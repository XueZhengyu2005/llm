"""从 ckpt/best.pt 加载模型，生成四种采样策略对比样例。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.model import load_for_eval


def decode_display(tokenizer, ids):
    data = b"".join(tokenizer.vocab[int(i)] for i in ids)
    return data.decode("utf-8", errors="ignore")


ROOT = Path(__file__).parent


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_for_eval(str(ROOT / "ckpt" / "best.pt"))
    model.to(device)

    prompt = "从前,有一个小男孩,"
    prompt_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    strategies = [
        ("greedy", dict(temperature=0.0)),
        ("top_k=5, temperature=0.3", dict(top_k=5, temperature=0.3)),
        ("top_p=0.9, temperature=0.4", dict(top_p=0.9, temperature=0.4)),
        ("temperature=0.6", dict(temperature=0.6)),
    ]

    lines = [f"prompt: {prompt}", ""]
    for name, kwargs in strategies:
        torch.manual_seed(7)
        out = model.generate(prompt_ids, max_new_tokens=28, repetition_penalty=1.1,
                             token_bytes=tok.vocab, utf8_safe=True, **kwargs)
        text = decode_display(tok, out[0].tolist())
        lines.append(f"[{name}]\n{text}\n")

    out_path = ROOT / "samples.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
