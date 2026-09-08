"""T4 · 结构感知切片（数据管线第 3 步）。

输入：T3 清洗后的段序列（含 meta）；输出：可训练的文本块序列 + 长度分布。
对齐 B2.2 切片策略：
- 结构化文本（md/标题）：标题层级树 → 段落聚合，硬边界优先；
- 纯文本 / PDF / DOCX：段落为原子聚合，超长段在句子边界滑窗切（句子不腰斩）；
- 源代码：tree-sitter 已按函数/类给出定义段，**整块保留、函数不切半**
  （超上限块原样保留并标记 oversized，由质检环节提示，不在此切破函数）；
- 相邻滑窗窗口间保留 overlap 句子，防语义断裂。

计量：训练前不加载重型 tokenizer，块长/重叠参数以「近似 token」计
（中文等 CJK 每字 ≈1 token，拉丁/数字 ≈每 4 字符 1 token），窗口语义对用户可见，
正式落库前可另行以真 tokenizer 复核（T5/T6 环节）。
"""

from __future__ import annotations

import math
import re

# 分布分桶（近似 token），用于直方图与 P50/P90 展示
BUCKETS = [
    (0, 256),
    (256, 512),
    (512, 768),
    (768, 1024),
    (1024, 1536),
    (1536, 2048),
    (2048, 4096),
    (4096, None),  # 超长块（多为不腰斩的巨型代码块）
]

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")
_LATIN_RE = re.compile(r"[A-Za-z0-9_]+")
_WS_RE = re.compile(r"\s")

# 弱句边界：句末/问叹/分号/换行/逗号（中文长句的逗号分句也在内），保证切点不落词中
_SENT_END = "。！？!?；;，,\n"


def approx_tokens(text: str) -> int:
    """近似 token 估算（仅用于切片窗口计量，非训练精度）。"""
    cjk = len(_CJK_RE.findall(text))
    latin_chars = sum(len(m) for m in _LATIN_RE.findall(text))
    whitespace = len(_WS_RE.findall(text))
    others = len(text) - cjk - latin_chars - whitespace  # 符号/标点
    return max(1, math.ceil(cjk + latin_chars / 4 + others / 1.5))


def _split_sentences(text: str) -> list[str]:
    """按弱句边界切为完整片段（保留标点；片段内部不再被切）。"""
    parts: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in _SENT_END:
            parts.append("".join(buf).strip())
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _slide(text: str, size: int, overlap: int) -> list[str]:
    """句子粒度滑窗：超长文本按句聚成 ≤size 的窗口，跨窗保留 overlap 尾句。"""
    sents = _split_sentences(text)
    if not sents:
        return []

    windows: list[str] = []
    current: list[str] = []

    def emit() -> None:
        nonlocal current
        if current:
            windows.append("".join(current))
            # 取窗口尾部若干完整句作为下一窗口的重叠（累计 ≤ overlap，至少 1 句）
            acc = 0
            tail_take: list[str] = []
            for s in reversed(current):
                st = approx_tokens(s)
                if tail_take and acc + st > overlap:
                    break
                tail_take.insert(0, s)
                acc += st
            if not tail_take:
                tail_take = [current[-1]]
            current = list(tail_take)

    for sent in sents:
        st = approx_tokens(sent)
        if st >= size:  # 单句超窗：不切句，独立成块（宁超不破句）
            emit()
            windows.append(sent)
            current = []
            continue
        if current and approx_tokens("".join(current)) + st > size:
            emit()
        current.append(sent)
    emit()
    return windows


def _chunk_prose(segments: list[dict], *, size: int, overlap: int) -> list[dict]:
    """散文切分：段落原子聚合 ≤size；超长段句子滑窗（含重叠）。

    产出块字典：{text, meta:{kind:'prose', label, headers, ...}}
    """
    blocks: list[dict] = []

    def push(text: str, meta: dict, label: str) -> None:
        blocks.append(
            {
                "text": text,
                "meta": {"kind": "prose", "label": label, **meta},
            }
        )

    # 收集为不可再分的原子单元：(type, payload)
    #   ("para", text, meta) —— 普通段落，可聚合
    #   ("win",  text, meta) —— 滑窗产物，自成一块（不再二次聚合）
    units: list[tuple[str, str, dict]] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        meta = dict(seg.get("meta") or {})
        if approx_tokens(text) > size:
            for w in _slide(text, size, overlap):
                units.append(("win", w, meta))
        else:
            units.append(("para", text, meta))

    cur_texts: list[str] = []
    cur_tokens = 0
    cur_label = "正文"
    cur_meta: dict = {}

    def flush() -> None:
        nonlocal cur_texts, cur_tokens, cur_label, cur_meta
        if cur_texts:
            text = "\n".join(cur_texts)
            blocks.append(
                {
                    "text": text,
                    "meta": {"kind": "prose", "label": cur_label, **cur_meta},
                }
            )
        cur_texts, cur_tokens = [], 0
        cur_label, cur_meta = "正文", {}

    for utype, text, meta in units:
        tok = approx_tokens(text)
        # 标题段：作为后续正文的标签追踪（聚合语义下标题并入正文块）
        if meta.get("kind") == "heading":
            title = meta.get("title") or text[:40]
            cur_label = title
            cur_meta["headers"] = [*(meta.get("ancestors") or []), title]
        if utype == "win":
            flush()
            push(text, meta, cur_label or "正文")
            continue
        if cur_texts and cur_tokens + tok > size:
            flush()
            cur_label = meta.get("title") if meta.get("kind") == "heading" else "正文"
        cur_texts.append(text)
        cur_tokens += tok
    flush()
    return blocks


def _code_blocks(segments: list[dict]) -> list[dict]:
    """代码定义段即块：tree-sitter 已给出边界，函数/类不切半。"""
    blocks: list[dict] = []
    for seg in segments:
        text = seg.get("text") or ""
        if not text.strip():
            continue
        blocks.append({"text": text, "meta": dict(seg.get("meta") or {})})
    return blocks


def chunk_segments(
    segments: list[dict], *, size: int = 768, overlap: int = 96
) -> list[dict]:
    """清洗后的段序列 → 文本块。

    源码段（meta.kind in {"code","source"}）整体成块不切半；其余按散文切分。
    返回 [{"index","text","chars","tokens","oversized","meta"}]。
    """
    if not segments:
        return []
    code = [s for s in segments if (s.get("meta") or {}).get("kind") in {"code", "source"}]
    prose = [s for s in segments if (s.get("meta") or {}).get("kind") not in {"code", "source"}]

    blocks = _chunk_prose(prose, size=size, overlap=overlap) + _code_blocks(code)
    for i, b in enumerate(blocks):
        b["index"] = i
        b["chars"] = len(b["text"])
        b["tokens"] = approx_tokens(b["text"])
        b["oversized"] = b["tokens"] > 2048
    return blocks


def _bucket_index(tokens: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if hi is None or tokens < hi:
            return i
    return len(BUCKETS) - 1


def chunk_summary(
    segments: list[dict], *, size: int = 768, overlap: int = 96, sample_limit: int = 6
) -> dict:
    """块长分布汇总：分桶直方图 / min·max·mean·p50·p90 / 少量样本块。

    供 files API 与前端切片预览（直方图 + 样本），避免整份块文本全量下发。
    """
    blocks = chunk_segments(segments, size=size, overlap=overlap)
    tokens = sorted(b["tokens"] for b in blocks)
    buckets = [0] * len(BUCKETS)
    for t in tokens:
        buckets[_bucket_index(t)] += 1

    def pct(p: float) -> int:
        if not tokens:
            return 0
        return tokens[min(len(tokens) - 1, int(math.ceil(p * (len(tokens) - 1))))]

    n = len(tokens)
    samples = []
    for b in blocks[:sample_limit]:
        meta = b.get("meta") or {}
        label = meta.get("symbol") or meta.get("title") or meta.get("label") or ""
        text = b["text"]
        samples.append(
            {
                "index": b["index"],
                "tokens": b["tokens"],
                "oversized": b["oversized"],
                "label": label,
                "text": text[:200] + ("…" if len(text) > 200 else ""),
            }
        )
    return {
        "count": n,
        "buckets": [
            {"min": lo, "max": hi, "count": buckets[i]}
            for i, (lo, hi) in enumerate(BUCKETS)
        ],
        "stats": {
            "min": tokens[0] if tokens else 0,
            "max": tokens[-1] if tokens else 0,
            "mean": round(sum(tokens) / n, 1) if tokens else 0,
            "p50": pct(0.5),
            "p90": pct(0.9),
        },
        "params": {"size": size, "overlap": overlap},
        "samples": samples,
    }
