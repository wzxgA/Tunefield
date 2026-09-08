"""T2 · 纯文本解析器：字节 → 段落级中间文本。

编码探测：UTF-8(含 BOM) → GB18030(覆盖 GBK/中文系) → Latin-1 保底。
段落切分：按空行分隔（空行不落地）；无空行时整篇作为单段，供后续滑窗切片。
"""

from __future__ import annotations

import re


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return data.decode("utf-8", errors="replace")


def _split_paragraphs(text: str) -> list[str]:
    raw = re.split(r"\r\n|\r|\n", text)
    paras: list[str] = []
    buf: list[str] = []
    for line in raw:
        if line.strip():
            buf.append(line.rstrip())
        elif buf:
            paras.append("\n".join(buf))
            buf = []
    if buf:
        paras.append("\n".join(buf))
    return paras


def parse(name: str, data: bytes) -> dict:
    """txt / 附加文本类（json/csv/yaml/html 等）：按段落产出中间文本段。"""
    text = _decode(data)
    # 空文件 / 全是空白
    if not text.strip():
        return {"status": "empty", "chars": 0, "segments": []}
    paras = _split_paragraphs(text)
    segments = [
        {"index": i, "text": p, "meta": {"kind": "paragraph"}}
        for i, p in enumerate(paras)
    ]
    return {
        "status": "ok",
        "chars": len(text),
        "segments": segments,
    }
