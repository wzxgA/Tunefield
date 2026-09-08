"""T0 · 基座定版与加载冒烟。

提供：基座登记（pin）、加载器、最小推理冒烟。基座覆盖 Qwen 系列（默认 Qwen2.5-0.5B-Instruct），
8GB 显存目标。定版信息写入 models/bases/_pinned.json，前端首页据此展示所选基座。

依赖（torch/transformers）由 _load_backend 惰性导入：仅执行加载/冒烟时才重签名，避免 CLI
其他命令（serve 等）在未安装训练依赖时启动即报错。

冒烟只做一次短生成验证「权重可加载 + 显存够 + 推理通」，不做效果评测。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tunefield import config

# 默认基座：Qwen2.5-0.5B-Instruct（轻量，8GB 显存充裕）
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
PINNED_FILE_NAME = "_pinned.json"


def _pinned_path() -> Path:
    return config.BASES_DIR / PINNED_FILE_NAME


def pin_base(model_id: str = DEFAULT_BASE) -> dict[str, Any]:
    """登记定版基座并落盘 _pinned.json。"""
    config.ensure_dirs()
    info = {
        "base": model_id,
        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": None,
        "vram_gb": None,
    }
    _pinned_path().write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return info


def pinned_info() -> dict[str, Any]:
    """读取当前定版基座信息；未定版时返回默认基座占位。"""
    if _pinned_path().exists():
        return json.loads(_pinned_path().read_text(encoding="utf-8"))
    return {"base": DEFAULT_BASE, "pinned_at": None, "device": None, "vram_gb": None}


def _load_backend() -> tuple[Any, Any, Any]:
    """惰性导入 torch / AutoModel / AutoTokenizer。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def load_base(
    model_id: str | None = None,
    *,
    device: str | None = None,
) -> tuple[Any, Any, str]:
    """加载定版基座（模型 + tokenizer），返回 (model, tokenizer, device)。

    device 依次取：参数指定 → cuda(若可用) → mps → cpu。
    """
    model_id = model_id or pinned_info()["base"]
    torch, AutoModel, AutoTokenizer = _load_backend()

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    _strict_device = device  # 供冒烟写回定版信息

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        torch_dtype="auto",
    )
    if device != "cuda":
        model.to(device)
    model.eval()
    return model, tokenizer, _strict_device


def smoke(
    model_id: str | None = None,
    *,
    prompt: str = "你好，请简单介绍一下你自己。",
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """加载基座并做一次短生成，返回耗时/显存/输出；同时把 device/显存写回定版记录。

    副作用：更新 _pinned.json 的 device 与 vram_gb 字段。
    """
    torch, _, _ = _load_backend()
    model, tokenizer, device = load_base(model_id, device="cuda")

    start = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    vram_gb = None
    if device == "cuda":
        alloc = torch.cuda.max_memory_allocated() / 1024**3
        vram_gb = round(alloc, 2)

    info = pinned_info()
    info.update(device=device, vram_gb=vram_gb, last_smoke_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _pinned_path().write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "base": model_id or info["base"],
        "device": device,
        "elapsed_s": round(elapsed, 2),
        "vram_gb": vram_gb,
        "prompt": prompt,
        "output": text,
    }