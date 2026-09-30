"""Qwen ChatML 模板与 assistant-only loss masking。"""
import torch

_TOK = None


def _segments(messages):
    """把多轮对话拆成 (text, trainable) 片段。

    模板控制符（<|im_start|>role、换行）全部不训练；
    assistant 正文与其结尾 <|im_end|> 参与训练（学习何时停止）。
    """
    parts = []
    for msg in messages:
        role, content = msg["role"], str(msg["content"])
        parts.append((f"<|im_start|>{role}\n", False))
        parts.append((content, role == "assistant"))
        parts.append(("<|im_end|>", role == "assistant"))
        parts.append(("\n", False))
    return parts


def format_messages(messages):
    """应用 Qwen 官方 ChatML 模板：<|im_start|>role\\ncontent<|im_end|>\\n。"""
    return "".join(text for text, _ in _segments(messages))


def build_labels(input_ids, messages):
    """返回与 input_ids 同形状的 labels；非 assistant 部分填 -100。

    用 offset mapping 把每个 token 映射回原字符串的字符区间，
    落在 assistant 正文（含结尾 <|im_end|>）内的 token 才参与 loss。
    """
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids)

    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        from pathlib import Path
        model_path = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B"
        _TOK = AutoTokenizer.from_pretrained(str(model_path))
    tok = _TOK

    # 构造完整文本，同时记录 assistant 可训练的字符区间
    labels = torch.full_like(input_ids, -100)
    text, spans = "", []
    for msg in messages:
        role, content = msg["role"], str(msg["content"])
        text += f"<|im_start|>{role}\n"
        content_start = len(text)
        text += content
        text += "<|im_end|>"
        trainable_end = len(text)  # assistant 的 <|im_end|> 也训练（学会停止）
        text += "\n"
        if role == "assistant":
            spans.append((content_start, trainable_end))

    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    for i, (start, end) in enumerate(enc.offset_mapping):
        if i >= len(labels):
            break
        if any(s <= start and end <= e for s, e in spans):
            labels[i] = input_ids[i]
    return labels
