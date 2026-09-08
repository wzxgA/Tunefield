"""T2 · 结构化解析统一入口。

对外提供两级接口：
- ``parse_bytes(name, data)``：单文件字节 → 统一解析结果（中间文本段 + 元信息）；
- ``scan_dataset_files(dataset)``：扫描 data/raw/<hash> 逐文件解析，
  产出「逐文件状态 + 中间文本预览」，供后端 API 与上传屏使用。

统一解析结果结构：:

    {
      "name": "相对路径",
      "kind": "txt | md | pdf | docx | code | other",
      "language": "python | ... | None",
      "status": "ok | empty | error | unsupported",
      "error": 失败原因或 None,
      "chars": 抽取文本总字符数,
      "blocks": 文本段数量,
      "segments": [{"index": int, "text": str, "meta": {...}}],
    }

解析层只做「字节 → 中间文本 + 元信息」，乱码/页眉等清洗交给 T3，切片交给 T4。
"""

from __future__ import annotations

from pathlib import Path

from tunefield import config

from . import code, docx, md, pdf, txt

# 扩展名 → kind（优先纯文本与文档；其余落入 code 语言表 / 不支持）
_TXT_EXTS = {
    ".txt", ".text", ".log",
    ".json", ".jsonl", ".ndjson",
    ".csv", ".tsv",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".html", ".htm",
    ".tex", ".rst", ".sql",
}
_MD_EXTS = {".md", ".markdown", ".mdown"}

_KIND_MODULE = {
    "txt": txt,
    "md": md,
    "pdf": pdf,
    "docx": docx,
    "code": code,
}


def detect(name: str) -> tuple[str | None, str | None]:
    """按扩展名识别 (kind, language)。language 仅 code 非空。"""
    ext = Path(name).suffix.lower()
    if ext in _TXT_EXTS:
        return "txt", None
    if ext in _MD_EXTS:
        return "md", None
    if ext == ".pdf":
        return "pdf", None
    if ext == ".docx":
        return "docx", None
    lang = code.language_for(ext)
    if lang is not None:
        return "code", lang
    return None, None  # 不支持的类型


def parse_bytes(name: str, data: bytes) -> dict:
    """解析单个文件字节；未知类型返回 unsupported，不抛异常。"""
    kind, language = detect(name)
    result: dict = {
        "name": name,
        "kind": kind or "other",
        "language": language,
        "status": "unsupported" if kind is None else "ok",
        "error": None,
        "chars": 0,
        "blocks": 0,
        "segments": [],
    }
    if kind is None:
        result["error"] = "不支持的文件类型（支持 txt/md/pdf/docx 及常见源代码）"
        return result
    if not data or not data.strip():
        result["status"] = "empty"
        result["error"] = "空文件"
        return result

    module = _KIND_MODULE[kind]
    try:
        parsed = (
            module.parse(name, data, language)
            if kind == "code"
            else module.parse(name, data)
        )
    except Exception as exc:  # 解析器内部未知异常：标记失败，不中断整体
        result["status"] = "error"
        result["error"] = f"解析异常：{exc}"
        return result

    result.update(parsed)
    result["blocks"] = len(parsed.get("segments") or [])
    return result


def _preview(result: dict, limit: int = 600) -> str:
    """把解析结果的中间文本拼成短预览（供逐文件状态卡片展示）。"""
    parts: list[str] = []
    used = 0
    for seg in (result.get("segments") or [])[:3]:
        t = (seg.get("text") or "").strip()
        if not t:
            continue
        parts.append(t)
        used += len(t)
        if used >= limit:
            break
    text = "\n".join(parts).strip()
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def scan_dataset_files(
    dataset: dict, *, preview_chars: int = 600, with_clean: bool = True
) -> list[dict]:
    """对 data/raw/<content_hash> 逐文件解析，返回逐文件状态与预览列表。

    dataset 需含 content_hash（tunefield.serve.db 返回的记录即满足）。
    with_clean=True 时附加 T3 清洗摘要：
    {before_chars, after_chars, removed_chars, rules, preview}。
    """
    root = config.RAW_DIR / dataset["content_hash"]
    if not root.exists():
        return []

    from tunefield.pipeline.chunking import chunk_summary
    from tunefield.pipeline.cleaning import clean_segments

    files: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        r = parse_bytes(rel, data)
        clean = None
        chunks = None
        if with_clean and r["status"] == "ok" and r["segments"]:
            c = clean_segments(r["segments"])
            clean = {
                "before_chars": c["before_chars"],
                "after_chars": c["after_chars"],
                "removed_chars": c["before_chars"] - c["after_chars"],
                "rules": {k: v for k, v in c["stats"].items() if v > 0},
                "preview": _preview({"segments": c["segments"]}, preview_chars),
            }
            # T4：清洗后段 → 切片（默认块长/重叠档位，供预览与分布展示）
            chunks = chunk_summary(c["segments"])
        files.append(
            {
                "name": rel,
                "kind": r["kind"],
                "language": r["language"],
                "status": r["status"],
                "error": r["error"],
                "chars": r["chars"],
                "blocks": r["blocks"],
                "preview": _preview(r, preview_chars),
                "clean": clean,
                "chunks": chunks,
            }
        )
    return files
