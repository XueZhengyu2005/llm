"""对比 base / SFT / DPO 三个模型在同一指令上的输出。"""
import sys
import json
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat import format_messages
from src.lora import inject_lora, load_lora


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Qwen2.5-0.5B"

PROMPTS = [
    "请用一句话介绍什么是深度学习。",
    "帮我写一条关于春天的微博，要求 30 字以内。",
    "把这句话翻译成英文：今天天气很好。",
]


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=96):
    messages = [{"role": "user", "content": prompt}]
    text = format_messages(messages) + "<|im_start|>assistant\n"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
    )
    return tok.decode(out[0][ids.size(1):], skip_special_tokens=True).strip()


def load_variant(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16)
    if name != "base":
        config = json.loads(
            (ROOT / "ckpt" / name / "config.json").read_text(encoding="utf-8"))
        inject_lora(
            model, config["target_modules"],
            r=config["r"], alpha=config["alpha"])
        load_lora(model, ROOT / "ckpt" / name / "lora.pt")
    model.to(device)
    model.eval()
    return model, tok


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lines = []
    for prompt in PROMPTS:
        lines.append(f"Q: {prompt}")
        for name in ("base", "sft", "dpo"):
            model, tok = load_variant(name, device)
            answer = generate(model, tok, prompt)
            lines.append(f"[{name.upper()}] {answer}")
            del model
            torch.cuda.empty_cache()
        lines.append("")
    output = "\n".join(lines)
    print(output)
    (ROOT / "compare_output.txt").write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
