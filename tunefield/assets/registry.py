"""T9 · 资产登记：目录即模型（adapters / gguf 的登记、检索与指纹承载）。

B3.2 约定：`fingerprint.json` 是资产层唯一事实来源——任何一份 GGUF 都能反查它由
哪份数据、哪组超参、哪个基座产出。本模块只做「登记 + 检索」的薄封装，
文件落盘由 engine/exporter 负责。
"""

from __future__ import annotations

import secrets
from pathlib import Path

from tunefield.serve import db


def slugify(text: str) -> str:
    """产物/Ollama 模型名专用:非 ASCII 转 '-',纯中文回退 'model'。

    C 工具链(llama-quantize)与 Ollama 模型名对非 ASCII 敏感,统一 ASCII 化;
    领域原名仍完整保存在指纹与数据库中用于展示。
    """
    import re

    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return s or "model"


def register_adapter(*, job_id: str, domain: str, path: Path, fingerprint: dict) -> str:
    """把训练产物目录登记为 adapter 资产，返回 adapter_id（= job_id）。"""
    adapter_id = job_id
    db.insert_adapter(
        id=adapter_id,
        job_id=job_id,
        domain=domain,
        version=job_id[:8],
        path=str(path),
        fingerprint_json=_dumps(fingerprint),
    )
    return adapter_id


def register_quantized(
    *, adapter_id: str, quant: str, path: Path
) -> dict:
    """登记一份 GGUF 产物，返回 quantized_models 行。"""
    size = path.stat().st_size if path.exists() else 0
    return db.insert_quantized_model(
        id=secrets.token_hex(8),
        adapter_id=adapter_id,
        quant=quant,
        path=str(path),
        size_bytes=size,
    )


def list_gguf_models() -> list[dict]:
    """GGUF 清单（供 GET /api/models），附带 domain/version 便于前端展示。"""
    adapters = {a["id"]: a for a in db.list_adapters()}
    out: list[dict] = []
    for m in db.list_quantized_models():
        adapter = adapters.get(m["adapter_id"]) or {}
        domain = adapter.get("domain") or ""
        version = adapter.get("version") or ""
        slug = slugify(domain) or "model"
        out.append(
            {
                "id": m["id"],
                "adapter_id": m["adapter_id"],
                "job_id": adapter.get("job_id") or m["adapter_id"],
                "domain": domain,
                "version": version,
                "slug": slug,
                "ollama_name": f"tunefield-{slug}-{version}",
                "quant": m["quant"],
                "path": m["path"],
                "size_bytes": m["size_bytes"],
                "created_at": m["created_at"],
            }
        )
    return out


def get_gguf_model(model_id: str) -> dict | None:
    for m in list_gguf_models():
        if m["id"] == model_id:
            return m
    return None


def _dumps(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
