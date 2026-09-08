"""T2 · DOCX 解析器（python-docx）：段落与表格 → 文本段。

每个非空段落为一文本段（meta.block=paragraph）；表格按行拼接为文本段
（meta.block=table），行内单元格以「 | 」连接，保留表格的横向结构信息。
"""

from __future__ import annotations

import io


def parse(name: str, data: bytes) -> dict:
    try:
        import docx
    except ImportError:
        return {
            "status": "error",
            "error": "缺少 python-docx（pip install python-docx）",
            "chars": 0,
            "segments": [],
        }

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        return {
            "status": "error",
            "error": f"DOCX 打开失败：{exc}",
            "chars": 0,
            "segments": [],
        }

    segments: list[dict] = []

    for para in document.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        segments.append(
            {
                "index": len(segments),
                "text": t,
                "meta": {
                    "kind": "paragraph",
                    "style": para.style.name if para.style is not None else "",
                },
            }
        )

    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if not any(cells):
                continue
            t = " | ".join(c for c in cells if c)
            segments.append(
                {
                    "index": len(segments),
                    "text": t,
                    "meta": {"kind": "table"},
                }
            )

    chars = sum(len(s["text"]) for s in segments)
    if not segments:
        return {
            "status": "empty",
            "error": "文档为空（无可抽取的段落或表格）",
            "chars": 0,
            "segments": [],
        }
    return {"status": "ok", "chars": chars, "segments": segments}
