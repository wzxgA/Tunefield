"""T6 · 质检报告（数据管线第 5 步）：判断一批数据是否可训练，并给出行动建议。

指标与阈值对齐 B4.2 表：

| 指标 | 阈值 | 动作 |
|---|---|---|
| 样本数 | < 200 | 阻断（error） |
| 原始语料总量 | < 5 MB 纯文本 | 警告（warn） |
| 块级重复率 | > 20% | 警告 |
| 乱码/低质块占比 | > 5% | 警告（附样例） |
| 长度分布 | P50/P90 展示 | 偏离 [256, 2048] 提示调参 |

低质块启发式（不引入重型模型）：块内含替换符/异常控制字符，或有效字符
（中文/拉丁）占比过低（<60%）视为低质块。
"""

from __future__ import annotations

import re
import statistics

from .chunking import approx_tokens

# 阈值（对齐方案 B4.2）
MIN_SAMPLES = 200          # 阻断：样本数低于此值无法训练
MIN_CORPUS_MB = 5.0        # 警告：语料总量低于此值有过拟合风险
MAX_DUP_RATE = 0.20        # 警告：块级重复率上限
MAX_LOWQ_RATE = 0.05       # 警告：低质块占比上限
LEN_RANGE = (256, 2048)    # 长度分布参考区间（近似 token）

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]")
_CJK_LATIN_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ffa-zA-Z0-9]")


def is_low_quality(text: str) -> bool:
    """低质块启发式：乱码/控制字符占比高，或有效字符占比过低。"""
    if not text.strip():
        return True
    bad = _CONTROL_RE.findall(text)
    if bad and len(bad) / max(len(text), 1) > 0.01:
        return True
    effective = len(_CJK_LATIN_RE.findall(text))
    return effective / max(len(text), 1) < 0.6


def block_duplicate_rate(blocks: list[dict]) -> float:
    """块级重复率：去空白/小写归一后完全相同的块占比（第一版启发式）。"""
    if not blocks:
        return 0.0
    keys = [re.sub(r"\s+", "", (b.get("text") or "")).lower() for b in blocks]
    return 1.0 - len(set(keys)) / len(keys)


def _percent(x: float) -> str:
    return f"{x * 100:.1f}%"


def evaluate(
    *,
    dataset_name: str,
    corpus_bytes: int,
    cleaned_texts: list[str],
    blocks: list[dict],
    records: list[dict],
    file_status: dict[str, int] | None = None,
    template: str = "continuation",
    chunk_size: int = 768,
    overlap: int = 96,
    lowq_limit: int = 3,
) -> dict:
    """按阈值产出质检报告（checks 每项含 metric/value/level/advice）。

    level ∈ info | warn | error；存在 error 即阻断训练（overall=blocked）。
    """
    sample_count = len(records)
    corpus_mb = round(corpus_bytes / (1024 * 1024), 2)
    dup_rate = block_duplicate_rate(blocks)

    lowq_texts = [t for t in cleaned_texts if t and is_low_quality(t)]
    lowq_rate = len(lowq_texts) / max(len(cleaned_texts), 1)

    token_values = sorted(approx_tokens((b.get("text") or "")) for b in blocks)
    p50 = statistics.median(token_values) if token_values else 0
    p90 = token_values[min(len(token_values) - 1, int(0.9 * (len(token_values) - 1)))] if token_values else 0
    max_tokens = token_values[-1] if token_values else 0

    checks: list[dict] = []
    level_error = False
    level_warn = False

    def add(metric, value, level, advice):
        nonlocal level_error, level_warn
        if level == "error":
            level_error = True
        elif level == "warn":
            level_warn = True
        checks.append({"metric": metric, "value": value, "level": level, "advice": advice})

    add(
        "样本数",
        str(sample_count),
        "error" if sample_count < MIN_SAMPLES else "info",
        (
            f"样本仅 {sample_count} 条，低于 {MIN_SAMPLES} 条门槛，不足以微调。"
            "建议补充数据，或调小块长/减少重叠以产出更多切片。"
            if sample_count < MIN_SAMPLES
            else "样本量充足。"
        ),
    )
    add(
        "语料总量",
        f"{corpus_mb} MB",
        "warn" if corpus_mb < MIN_CORPUS_MB else "info",
        (
            f"纯文本语料约 {corpus_mb} MB，低于 {MIN_CORPUS_MB} MB，存在过拟合风险，"
            "建议补充语料或增加早停策略。"
            if corpus_mb < MIN_CORPUS_MB
            else "语料规模在推荐区间内。"
        ),
    )
    add(
        "块级重复率",
        _percent(dup_rate),
        "warn" if dup_rate > MAX_DUP_RATE else "info",
        (
            f"重复率 {_percent(dup_rate)} 高于 {_percent(MAX_DUP_RATE)}，"
            "重复内容会放大过拟合，建议去重后重建。"
            if dup_rate > MAX_DUP_RATE
            else "块级重复率在正常范围。"
        ),
    )
    add(
        "低质块占比",
        _percent(lowq_rate),
        "warn" if lowq_rate > MAX_LOWQ_RATE else "info",
        (
            f"低质/乱码块占比 {_percent(lowq_rate)} 高于 {_percent(MAX_LOWQ_RATE)}，"
            "将影响训练质量，建议人工清洗或剔除低质来源后重建。"
            if lowq_rate > MAX_LOWQ_RATE
            else "低质块占比在正常范围。"
        ),
    )
    add(
        "长度分布",
        f"P50 {p50} / P90 {p90}",
        "warn" if (token_values and (p90 < LEN_RANGE[0] or p90 > LEN_RANGE[1])) else "info",
        (
            "块长明显偏离推荐区间 [256, 2048] token，建议调整块长/重叠参数后重建。"
            if token_values and (p90 < LEN_RANGE[0] or p90 > LEN_RANGE[1])
            else "块长分布接近推荐区间。"
        ),
    )

    overall = "error" if level_error else ("warn" if level_warn else "ok")
    lowq_examples = [t[:120] + ("…" if len(t) > 120 else "") for t in lowq_texts[:lowq_limit]]

    return {
        "dataset": dataset_name,
        "overall": overall,
        "summary": {
            "sample_count": sample_count,
            "corpus_mb": corpus_mb,
            "block_count": len(blocks),
            "dup_rate": round(dup_rate, 4),
            "lowq_rate": round(lowq_rate, 4),
            "length": {"p50": p50, "p90": p90, "max": max_tokens},
            "files": file_status or {},
            "params": {"template": template, "chunk_size": chunk_size, "overlap": overlap},
        },
        "checks": checks,
        "lowq_examples": lowq_examples,
    }
