"""T2 · Markdown 解析器：标题结构感知 → 带层级元信息的中间文本块。

保留三件事，供 T4 结构感知切片与前端预览使用：
- 标题行按 # 层级单独成块，携带 level / title / 祖先标题链；
- 正文按空行分段落；
- 围栏代码块整体成一块（附围栏声明语言），不被空行打断。
"""

from __future__ import annotations

import re

from .txt import _decode

_FENCE_OPEN = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*(.*)$")


def parse(name: str, data: bytes) -> dict:
    text = _decode(data)
    if not text.strip():
        return {"status": "empty", "chars": 0, "segments": []}

    lines = text.splitlines()
    segments: list[dict] = []
    ancestors: list[tuple[int, str]] = []  # 当前标题链 [(level, title)]
    buf: list[str] = []
    fence_marker: str | None = None
    fence_lang: str = ""

    def flush_text(meta: dict | None = None) -> None:
        nonlocal buf
        if not buf:
            return
        text_ = "\n".join(buf).strip()
        buf = []
        if not text_:
            return
        m = {
            "kind": "paragraph",
            "ancestors": [t for _, t in ancestors],
            **(meta or {}),
        }
        segments.append({"index": len(segments), "text": text_, "meta": m})

    def emit_code() -> None:
        nonlocal fence_lang
        lang = fence_lang
        fence_lang = ""
        text_ = "\n".join(buf).strip()
        buf.clear()
        if text_:
            segments.append(
                {
                    "index": len(segments),
                    "text": text_,
                    "meta": {
                        "kind": "code",
                        "fence_lang": lang,
                        "ancestors": [t for _, t in ancestors],
                    },
                }
            )

    for raw in lines:
        stripped = raw.strip()
        if fence_marker is not None:
            # 整行由同一种围栏字符构成 → 结束
            if stripped.startswith(fence_marker) and all(
                ch == fence_marker[0] for ch in stripped
            ):
                emit_code()
                fence_marker = None
            else:
                buf.append(raw)
            continue

        fm = _FENCE_OPEN.match(raw)
        if fm and fm.group(1) in ("```", "~~~"):
            flush_text()
            fence_marker = fm.group(1)
            fence_lang = fm.group(2).strip()
            continue

        hm = re.match(r"^#{1,6}[ \t]+", raw)
        if hm:
            flush_text()
            level = len(hm.group(0).strip())
            title = raw[hm.end():].strip()
            while ancestors and ancestors[-1][0] >= level:
                ancestors.pop()
            ancestors.append((level, title))
            segments.append(
                {
                    "index": len(segments),
                    "text": raw.rstrip(),
                    "meta": {
                        "kind": "heading",
                        "level": level,
                        "title": title,
                        "ancestors": [t for _, t in ancestors[:-1]],
                    },
                }
            )
            continue

        if not stripped:
            flush_text()
        else:
            buf.append(raw)

    if fence_marker is not None:
        # 文件以未闭合围栏收尾：仍保留内容块
        emit_code()
    else:
        flush_text()

    if not segments:
        return {"status": "empty", "chars": 0, "segments": []}
    return {"status": "ok", "chars": len(text), "segments": segments}
