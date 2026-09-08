"""T2 · PDF 解析器：逐页文本抽取（PyMuPDF），段落保留页码元信息。

扫描件 / 图片型 PDF 没有文本层，get_text 结果为空 —— 归类为 empty
（提示「疑似扫描件」），由用户决定是否换源，不在此静默产出空语料。
"""

from __future__ import annotations


def parse(name: str, data: bytes) -> dict:
    try:
        import pymupdf as fitz  # PyMuPDF（新版包名）
    except ImportError:  # pragma: no cover - 旧版以 fitz 命名
        try:
            import fitz
        except ImportError:
            return {
                "status": "error",
                "error": "缺少 PyMuPDF（pip install pymupdf）",
                "chars": 0,
                "segments": [],
            }

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        return {
            "status": "error",
            "error": f"PDF 打开失败：{exc}",
            "chars": 0,
            "segments": [],
        }

    if doc.needs_pass:
        doc.close()
        return {
            "status": "error",
            "error": "PDF 已加密，无法抽取文本",
            "chars": 0,
            "segments": [],
        }

    segments: list[dict] = []
    try:
        for page_no in range(doc.page_count):
            page = doc.load_page(page_no)
            page_text = page.get_text("text")
            page_text = "\n".join(
                ln.rstrip() for ln in page_text.splitlines() if ln.strip()
            )
            if not page_text.strip():
                continue
            segments.append(
                {
                    "index": len(segments),
                    "text": page_text,
                    "meta": {"kind": "page", "page": page_no + 1},
                }
            )
    finally:
        doc.close()

    chars = sum(len(s["text"]) for s in segments)
    if not segments:
        return {
            "status": "empty",
            "error": "未抽取到文本（疑似扫描件 / 图片型 PDF）",
            "chars": 0,
            "segments": [],
        }
    return {"status": "ok", "chars": chars, "segments": segments}
