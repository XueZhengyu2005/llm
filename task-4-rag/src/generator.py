"""Qwen2.5-7B-Instruct-AWQ 生成器。"""
import torch
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct-AWQ"

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("加载 Qwen2.5-7B-Instruct-AWQ ...")
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    _model.eval()
    print("模型加载完成")
    return _model, _tokenizer


def generate(prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    """用 chat template 生成回答。"""
    model, tokenizer = load_model()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip()


def context_excerpt(text: str, max_chars: int = 500) -> str:
    """返回实际送入生成 prompt 的 source 片段。"""
    return str(text).strip()[:max_chars]


def build_rag_prompt(query: str, contexts: List[Dict]) -> str:
    """把检索结果拼成 RAG prompt。"""
    context_texts = []
    for i, ctx in enumerate(contexts, 1):
        text = context_excerpt(ctx["text"])
        context_texts.append(f"[{i}] {text}")
    
    context_str = "\n\n".join(context_texts)
    
    return f"""请根据以下参考资料回答问题。只能使用参考资料中包含的信息，如果参考资料中没有相关信息，请回答"根据提供的资料，我无法回答这个问题。"

参考资料：
{context_str}

问题：{query}

回答："""
