"""手写 LoRA：低秩注入、前向叠加、可训练参数隔离与权重合并。"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B(A x)；A 用 kaiming 初始化，B 零初始化。"""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r

        for p in self.base.parameters():
            p.requires_grad_(False)

        device = base.weight.device
        lora_a = torch.empty(r, base.in_features)
        nn.init.kaiming_uniform_(lora_a, a=5 ** 0.5)
        self.lora_A = nn.Parameter(lora_a.to(device=device))
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, r, device=device))

    def forward(self, x):
        lora_out = (x.to(self.lora_A.dtype) @ self.lora_A.T) @ self.lora_B.T
        return self.base(x) + self.scaling * lora_out.to(x.dtype)

    def merge(self):
        """把 scaling * B @ A 合并回 base 权重，返回原 Linear。"""
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.data += delta.to(self.base.weight.dtype)
        return self.base


def inject_lora(model, target_modules, r=8, alpha=16):
    """把名字匹配 target_modules 的 nn.Linear 替换为 LoRALinear。

    其余参数全部 requires_grad=False，只有 LoRA 的 A / B 可训练。
    """
    target_modules = set(target_modules)
    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in target_modules:
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, LoRALinear(module, r, alpha))
            replaced += 1
    if replaced == 0:
        raise ValueError(f"没有匹配到任何目标线性层：{target_modules}")
    return model


def merge_lora(model):
    """把所有 LoRALinear 的增量合并回原权重并移除 LoRA 分支。"""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, module.merge())
    return model


def save_lora(model, path):
    """只保存 LoRA 增量权重（A / B），体积小。"""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[name] = {
                "lora_A": module.lora_A.detach().cpu(),
                "lora_B": module.lora_B.detach().cpu(),
                "r": module.r,
                "scaling": module.scaling,
            }
    torch.save(state, path)
    return path


def load_lora(model, path):
    """把 save_lora 保存的增量权重加载回已注入 LoRA 的模型。"""
    state = torch.load(path, map_location="cpu", weights_only=False)
    for name, module in model.named_modules():
        if name in state:
            module.lora_A.data.copy_(state[name]["lora_A"])
            module.lora_B.data.copy_(state[name]["lora_B"])
    return model
