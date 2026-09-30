"""任务三 SFT：MOSS 多轮对话 + 手写 LoRA + assistant-only loss。"""
import json
import sys
import zipfile
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

from src.chat import build_labels, format_messages
from src.lora import inject_lora, save_lora


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-0.5B"
MAX_LENGTH = 512
BATCH = 8
EPOCHS = 2
LR = 3e-4
N_EXAMPLES = 50000
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "lm_head",
]
LORA_R = 32
LORA_ALPHA = 64
EOS_WEIGHT = 50


def load_moss_examples(n_examples):
    """解压并读取 MOSS jsonl，转换为 messages 列表。"""
    data_dir = ROOT / "data" / "moss-sft"
    zip_path = data_dir / "moss-003-sft-no-tools.jsonl.zip"
    jsonl_path = data_dir / "moss-003-sft-no-tools.jsonl"

    if not jsonl_path.exists():
        if not zip_path.exists():
            sys.exit(f"[错误] 缺少 {zip_path}")
        print("解压 MOSS jsonl.zip ...")
        with zipfile.ZipFile(zip_path) as zf:
            member = [n for n in zf.namelist() if n.endswith(".jsonl")][0]
            zf.extract(member, data_dir)
            extracted = data_dir / member
            if extracted != jsonl_path:
                extracted.rename(jsonl_path)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    examples = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if len(examples) >= n_examples:
                break
            item = json.loads(line)
            # 按完整轮次贪心装入 MAX_LENGTH：保证样本以 assistant + <|im_end|> 结尾，
            # 避免 512 截断把「回答结束→停止」信号全部切掉（MOSS 全对话中位数 ~1300 token）。
            messages = []
            def turn_number(key):
                import re
                match = re.search(r"(\d+)$", str(key))
                return (0, int(match.group(1))) if match else (1, str(key))

            for key in sorted(item.get("chat", {}).keys(), key=turn_number):
                turn = item["chat"][key]
                human = _clean_tag(turn.get("Human", ""), "Human", "eoh")
                moss = _clean_tag(turn.get("MOSS", ""), "MOSS", "eom")
                if not human or not moss:
                    continue
                candidate = messages + [
                    {"role": "user", "content": human},
                    {"role": "assistant", "content": moss},
                ]
                if len(tok(format_messages(candidate), add_special_tokens=False).input_ids) > MAX_LENGTH:
                    break
                messages = candidate
            if messages and messages[-1]["role"] == "assistant":
                examples.append(messages)
    print(f"加载 MOSS 样例：{len(examples)} 条")
    return examples


def _clean_tag(text, tag, end_tag):
    """去掉 MOSS 原始格式里的 <|Tag|>: ... <eox> 包裹。"""
    s = str(text).strip()
    prefix = f"<|{tag}|>: "
    if s.startswith(prefix):
        s = s[len(prefix):]
    end_marker = f"<{end_tag}>"
    if s.endswith(end_marker):
        s = s[: -len(end_marker)]
    return s.strip()


class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.items = []
        for messages in examples:
            text = format_messages(messages)
            enc = tokenizer(text, add_special_tokens=False, truncation=True,
                            max_length=MAX_LENGTH)
            ids = torch.tensor(enc.input_ids, dtype=torch.long)
            labels = build_labels(ids, messages)
            labels[ids == tokenizer.pad_token_id] = -100  # 防御：正文出现 pad 字符
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch, pad_id):
    max_len = max(ids.size(0) for ids, _ in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, :ids.size(0)] = ids
        labels[i, :lab.size(0)] = lab
        attention_mask[i, :ids.size(0)] = 1
    return input_ids, labels, attention_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=N_EXAMPLES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--r", type=int, default=LORA_R)
    parser.add_argument("--alpha", type=float, default=LORA_ALPHA)
    parser.add_argument("--eos-weight", type=float, default=EOS_WEIGHT)
    parser.add_argument("--output", type=Path, default=ROOT / "ckpt" / "sft")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    examples = load_moss_examples(args.examples)
    dataset = SFTDataset(examples, tok)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16)
    inject_lora(model, target_modules=TARGET_MODULES, r=args.r, alpha=args.alpha)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        total_loss, n = 0.0, 0
        for input_ids, labels, attention_mask in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            shifted_labels = labels[:, 1:].reshape(-1)
            token_loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                shifted_labels, ignore_index=-100, reduction="none")
            weights = (shifted_labels != -100).float()
            weights = torch.where(
                shifted_labels == tok.convert_tokens_to_ids("<|im_end|>"),
                torch.full_like(weights, args.eos_weight), weights)
            loss = (token_loss * weights).sum() / weights.sum()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += loss.item(); n += 1; step += 1
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step} loss={total_loss/n:.4f}")

    save_lora(model, out_dir / "lora.pt")
    (out_dir / "config.json").write_text(json.dumps({
        "base_model": str(MODEL_PATH), "r": args.r, "alpha": args.alpha,
        "target_modules": TARGET_MODULES, "examples": len(dataset),
        "epochs": args.epochs, "max_length": MAX_LENGTH,
        "eos_weight": args.eos_weight, "learning_rate": args.lr,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SFT 完成，LoRA 权重保存到 {out_dir}")


if __name__ == "__main__":
    main()
