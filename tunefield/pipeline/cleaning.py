"""T3 · 清洗与归一（数据管线第 2 步）。

输入：T2 解析产出的中间文本段（含 meta.kind）；输出：清洗后的段 + 逐规则命中统计。
对齐 B4.2-2：剔除乱码/控制字符、HTML 残留、独立页码与重复页眉/页脚行，统一全半角与换行。

设计约定：
- 每条规则是独立纯函数，返回 (new_text, events)，events 为该规则处理的字符/行/片段数，
  汇总进 stats，清洗前后字数差异 = before_chars - after_chars（供质检报告/预览展示）；
- 代码块（meta.kind in {"code","source"}）只做字符级归一，跳过 HTML 剔除与重复行识别，
  避免误伤源码里的字符串字面量、缩进与空行结构；
- 重复执行近似幂等（首轮转换后无残留，stats 归零），不改变段落边界（切片交给 T4）。
"""

from __future__ import annotations

import html as _html
import re

# 规则键 → 展示说明（前端也据此映射标签，键名保持稳定）
RULE_LABELS = {
    "invisible": "控制字符/乱码",
    "space": "全角空格",
    "fullwidth": "全半角",
    "html": "HTML 残留",
    "page": "页码行",
    "boiler": "重复页眉/页脚",
}

# 不可见/格式字符：C0(除 \t\n\r)、C1、零宽与双向控制、软连字符、BOM 等
_INVISIBLE_RE = re.compile(
    "[\u0000-\u0008\u000b\u000c\u000e-\u001f"
    "\u007f-\u009f"
    "\u00ad"
    "\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060\ufeff]"
)

# 特殊空白：全角空格、NBSP、窄 NBSP、图元空格 → 半角空格
_SPECIAL_SPACES = str.maketrans(
    {"\u3000": " ", "\u00a0": " ", "\u202f": " ", "\u2007": " ", "\u2009": " "}
)

# 全角 ASCII（0xFF01–0xFF5E）→ 半角（统一全半角）。其中在中文本地常用作标点的
# 码点（！  （  ） ， ． ： ； ？）保留全角，避免中文语料可读性损伤。
_FW_START, _FW_END = 0xFF01, 0xFF5E
_FW_SHIFT = 0xFEE0
_FW_KEEP = {0xFF01, 0xFF08, 0xFF09, 0xFF0C, 0xFF0E, 0xFF1A, 0xFF1B, 0xFF1F}

_CR_RE = re.compile(r"\r\n?|\r")

# HTML 片段
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
_ENTITY_RE = re.compile(
    r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|"
    r"amp|lt|gt|quot|apos|nbsp|mdash|ndash|hellip|#39);"
)

# 独立页码行（页脚常见形态）
_PAGE_LINE_RE = re.compile(
    r"^\s*[-–—·.]?\s*("
    r"\d{1,4}\s*[-–—·]\s*\d{1,4}"        # 12 - 5 / 第 3 / 5
    r"|page\s*\d{1,4}"
    r"|p\.?\s*\d{1,4}"
    r"|第\s*\d{1,4}\s*页"
    r"|\d{1,4}\s*/\s*\d{1,4}"
    r"|\d{1,4})"
    r"\s*[-–—·.]?\s*$",
    re.IGNORECASE,
)

# 段级文本结束标点（重复行识别时据此排除完整句子）
_ENDING_PUNCT = set("。.！？!?…；;：:、）】》\"'”’")


def _add(stats: dict[str, int], key: str, n: int) -> None:
    if n:
        stats[key] = stats.get(key, 0) + n


def _is_code(meta: dict) -> bool:
    return meta.get("kind") in {"code", "source"}


# ---------------------------------------------------------------------------
# 字符级规则（单段文本）
# ---------------------------------------------------------------------------

def _step_invisible(text: str) -> tuple[str, int]:
    n = len(_INVISIBLE_RE.findall(text))
    return _INVISIBLE_RE.sub("", text), n


def _step_spaces(text: str) -> tuple[str, int]:
    # 先统计命中再翻译（maketrans 表以码点为键）
    n = sum(1 for ch in text if ord(ch) in _SPECIAL_SPACES)
    return text.translate(_SPECIAL_SPACES), n


def _step_fullwidth(text: str) -> tuple[str, int]:
    n = 0
    out: list[str] = []
    for ch in text:
        o = ord(ch)
        if _FW_START <= o <= _FW_END and o not in _FW_KEEP:
            out.append(chr(o - _FW_SHIFT))
            n += 1
        else:
            out.append(ch)
    return "".join(out), n


def _step_html(text: str) -> tuple[str, int]:
    """剔除 HTML 残留：先解实体（含二次产生的尖括号）再删标签，保证幂等。"""
    entities = len(_ENTITY_RE.findall(text))
    text = _html.unescape(text)
    comments = len(_HTML_COMMENT_RE.findall(text))
    text = _HTML_COMMENT_RE.sub("", text)
    text = _HTML_BR_RE.sub("\n", text)
    tags = len(_HTML_TAG_RE.findall(text))
    text = _HTML_TAG_RE.sub("", text)
    return text, entities + comments + tags


def clean_text(text: str) -> tuple[str, dict]:
    """单段字符级清洗（无文档级重复行识别）。返回 (清洗后文本, {规则: 命中数})。"""
    stats: dict[str, int] = {}
    t = _CR_RE.sub("\n", text)
    for key, fn in (
        ("invisible", _step_invisible),
        ("space", _step_spaces),
        ("fullwidth", _step_fullwidth),
        ("html", _step_html),
    ):
        t, n = fn(t)
        _add(stats, key, n)
    return t, stats


# ---------------------------------------------------------------------------
# 文档级：独立页码行 / 跨段重复页眉页脚
# ---------------------------------------------------------------------------

def _collect_boiler(segments: list[dict]) -> set[str]:
    """跨段统计完全相同行；出现 ≥3 次、长度适中且非完整句子 → 疑似页眉/页脚。"""
    counts: dict[str, int] = {}
    for seg in segments:
        if _is_code(seg.get("meta") or {}):
            continue
        for line in seg["text"].splitlines():
            s = line.strip()
            if s:
                counts[s] = counts.get(s, 0) + 1
    return {
        line
        for line, c in counts.items()
        if 4 <= len(line) <= 64
        and c >= 3
        and not line.endswith(tuple(_ENDING_PUNCT))
        and not _PAGE_LINE_RE.match(line)
    }


def _drop_lines(text: str, drop: set[str]) -> tuple[str, int, int]:
    """删除独立页码行与重复页眉行；返回 (文本, 页数页码行, 页眉行)。"""
    page_n = 0
    boil_n = 0
    keep: list[str] = []
    for line in text.splitlines():
        if _PAGE_LINE_RE.match(line.strip()):
            page_n += 1
            continue
        s = line.strip()
        if s in drop:
            boil_n += 1
            continue
        keep.append(line)
    return "\n".join(keep), page_n, boil_n


def clean_segments(segments: list[dict]) -> dict:
    """清洗整份文档（段序列），含文档级页眉/页码行识别。

    返回 {segments, stats, before_chars, after_chars}：
    - segments: 与入参同序的新段（text 已清洗，index/meta 保留）；
    - stats: {规则键: 命中数}（仅 >0）；
    - before_chars/after_chars: 清洗前后总字符数，差值即字数差异（入报告）。
    """
    stats: dict[str, int] = {}
    before = sum(len(s.get("text", "")) for s in segments)
    if not segments:
        return {"segments": [], "stats": {}, "before_chars": 0, "after_chars": 0}

    boiler = _collect_boiler(segments)
    new_segments: list[dict] = []
    for seg in segments:
        text = seg.get("text", "")
        meta = seg.get("meta") or {}
        is_code = _is_code(meta)

        t = _CR_RE.sub("\n", text)
        if not is_code:
            t, pn, bn = _drop_lines(t, boiler)
            _add(stats, "page", pn)
            _add(stats, "boiler", bn)
        for key, fn in (
            ("invisible", _step_invisible),
            ("space", _step_spaces),
            ("fullwidth", _step_fullwidth),
        ):
            t, n = fn(t)
            _add(stats, key, n)
        if not is_code:
            t, n = _step_html(t)
            _add(stats, "html", n)

        new_segments.append({"index": seg.get("index"), "text": t, "meta": meta})

    after = sum(len(s["text"]) for s in new_segments)
    return {"segments": new_segments, "stats": stats,
            "before_chars": before, "after_chars": after}
