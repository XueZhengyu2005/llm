"""任务三 DPO：在 SFT 之上做偏好对齐，reference model 只 forward。"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from src.chat import format_messages
from src.lora import inject_lora, load_lora, save_lora


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-0.5B"
SFT_PATH = ROOT / "ckpt" / "sft" / "lora.pt"
SFT_CONFIG = ROOT / "ckpt" / "sft" / "config.json"
BETA = 0.1
BATCH = 2
EPOCHS = 1
LR = 5e-5
N_PAIRS = 800
MAX_LENGTH = 512


def load_dpo_pairs():
    path = ROOT / "data" / "dpo" / "dpo_zh.json"
    if not path.exists():
        sys.exit(f"[错误] 缺少 {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for item in data:
        if len(pairs) >= N_PAIRS:
            break
        convs = item.get("conversations", [])
        if not convs or "chosen" not in item or "rejected" not in item:
            continue
        prompt = next(
            (turn.get("value", "") for turn in convs
             if str(turn.get("from", turn.get("role", ""))).lower()
             in {"human", "user", "prompt"}),
            convs[0].get("value", ""),
        )
        chosen = str(item["chosen"].get("value", "")).strip()
        rejected = str(item["rejected"].get("value", "")).strip()
        if not str(prompt).strip() or not chosen or not rejected:
            continue
        pairs.append((str(prompt), chosen, rejected))
    print(f"加载 DPO 偏好对：{len(pairs)} 条")
    return pairs


def build_pair_ids(pair, tok):
    prompt, chosen, rejected = pair
    prompt_text = format_messages([{"role": "user", "content": prompt}])
    prompt_text += "<|im_start|>assistant\n"
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids

    seqs = []
    for completion in (chosen + "<|im_end|>", rejected + "<|im_end|>"):
        comp_ids = tok(completion, add_special_tokens=False).input_ids
        if len(comp_ids) >= MAX_LENGTH:
            prompt_part = []
            comp_part = comp_ids[:MAX_LENGTH]
        else:
            prompt_part = prompt_ids[:MAX_LENGTH - len(comp_ids)]
            comp_part = comp_ids
        ids = prompt_part + comp_part
        labels = [-100] * len(prompt_part) + comp_part
        if not any(label != -100 for label in labels):
            return None
        seqs.append((ids, labels))
    return seqs


def collate(pairs, tok):
    flat = [seq for pair in pairs for seq in pair]
    max_len = max(len(ids) for ids, _ in flat)
    input_ids = torch.full((len(flat), max_len), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(flat), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(flat), max_len), dtype=torch.long)
    for i, (ids, lab) in enumerate(flat):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        attention_mask[i, :len(ids)] = 1
    return input_ids, labels, attention_mask


def seq_logprobs(model, input_ids, labels, attention_mask):
    """返回每个序列 completion 部分的 sum logprob (B,)。"""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logprobs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    token_lp = logprobs.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (token_lp * mask).sum(dim=1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pairs = load_dpo_pairs()
    sft_config = json.loads(SFT_CONFIG.read_text(encoding="utf-8"))

    policy = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16)
    inject_lora(
        policy, sft_config["target_modules"],
        r=sft_config["r"], alpha=sft_config["alpha"])
    load_lora(policy, SFT_PATH)
    policy.to(device)
    policy.train()

    ref = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16)
    inject_lora(
        ref, sft_config["target_modules"],
        r=sft_config["r"], alpha=sft_config["alpha"])
    load_lora(ref, SFT_PATH)
    ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=LR)

    out_dir = ROOT / "ckpt" / "dpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    margins = []
    for epoch in range(EPOCHS):
        for i in range(0, len(pairs), BATCH):
            batch_pairs = [build_pair_ids(pair, tok) for pair in pairs[i:i + BATCH]]
            batch_pairs = [pair for pair in batch_pairs if pair is not None]
            if not batch_pairs:
                continue
            input_ids, labels, attention_mask = collate(batch_pairs, tok)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            with torch.no_grad():
                ref_lp = seq_logprobs(ref, input_ids, labels, attention_mask)
            pol_lp = seq_logprobs(policy, input_ids, labels, attention_mask)

            chosen_lp, rejected_lp = pol_lp[0::2], pol_lp[1::2]
            ref_chosen, ref_rejected = ref_lp[0::2], ref_lp[1::2]
            logits = BETA * ((chosen_lp - ref_chosen) - (rejected_lp - ref_rejected))
            loss = -F.logsigmoid(logits).mean()

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            margins.extend((chosen_lp - rejected_lp).detach().cpu().tolist())
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch + 1} step {step} dpo_loss={loss.item():.4f} "
                      f"margin_mean={sum(margins)/len(margins):.3f}")

    save_lora(policy, out_dir / "lora.pt")
    (out_dir / "config.json").write_text(json.dumps({
        "base_model": str(MODEL_PATH), "sft_lora": str(SFT_PATH),
        "r": sft_config["r"], "alpha": sft_config["alpha"],
        "target_modules": sft_config["target_modules"],
        "beta": BETA, "pairs": len(pairs), "epochs": EPOCHS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DPO 完成，LoRA 权重保存到 {out_dir}")


if __name__ == "__main__":
    main()
