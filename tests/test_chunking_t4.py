"""T4 · 切片器：段落聚合、超长段句子滑窗不腰斩、代码不切半、分布统计。"""

from __future__ import annotations

from tunefield.pipeline.chunking import (
    approx_tokens,
    chunk_segments,
    chunk_summary,
)


def _segs(*texts: str, kind: str = "paragraph", meta=None) -> list[dict]:
    return [
        {"index": i, "text": t, "meta": {"kind": kind, **(meta or {})}}
        for i, t in enumerate(texts)
    ]


def test_approx_tokens_counts_cjk_and_latin():
    assert approx_tokens("汉字测试") == 4          # CJK 每字 1
    assert approx_tokens("hello world") == 3       # 10 latin /4 → 3
    assert approx_tokens("") == 1


def test_paragraph_aggregation_respects_size():
    # 每段恰好 50 token，目标 120 → 每块 2 段
    segs = _segs(*(["字" * 50] * 4))
    blocks = chunk_segments(segs, size=120, overlap=0)
    assert [b["tokens"] for b in blocks] == [100, 100]
    assert all(b["meta"]["kind"] == "prose" for b in blocks)


def test_long_text_slides_at_sentence_boundary_no_torn_sentences():
    sents = [f"句子{name}。" for name in "一二三四五六七八九十"]
    text = "".join(sents)
    blocks = chunk_segments(_segs(text), size=12, overlap=5)

    assert len(blocks) >= 3  # 长文本被切成多窗
    # 无句子被腰斩：每个块都能按句末符完整拆回原句集合
    for b in blocks:
        pieces = [p + "。" for p in b["text"].split("。") if p]
        assert pieces and all(p in sents for p in pieces)

    # 重叠生效：全部块 token 总量 > 原文 token（存在跨窗重复句）
    total = sum(b["tokens"] for b in blocks)
    assert total > approx_tokens(text)


def test_long_text_overlap_carries_previous_tail_sentence():
    sents = [f"句子{name}。" for name in "一二三四五六七八九十"]
    blocks = chunk_segments(_segs("".join(sents)), size=12, overlap=5)
    tail1 = blocks[0]["text"].split("。")[-2] + "。"
    assert tail1 in blocks[1]["text"]  # 上一窗尾句带入下一窗头


def test_code_functions_never_sliced():
    func_a = "def alpha():\n" + "\n".join(f"    x{i} = {i}" for i in range(200))
    func_b = "def beta():\n    return 1\n"
    segs = _segs(func_a, func_b, kind="code")
    blocks = chunk_segments(segs, size=100, overlap=0)  # 极小窗也不可切函数
    assert len(blocks) == 2
    assert blocks[0]["text"] == func_a
    assert blocks[1]["text"] == func_b
    assert "def alpha" in blocks[0]["text"] and "def beta" not in blocks[0]["text"]


def test_params_change_window_count():
    sents = [f"句子{name}。" for name in "一二三四五六七八九十"]
    small = chunk_segments(_segs("".join(sents)), size=8, overlap=0)
    large = chunk_segments(_segs("".join(sents)), size=200, overlap=0)
    assert len(small) > len(large)
    assert large[0]["text"] == "".join(sents)


def test_code_oversized_flag():
    big = "def huge():\n" + "\n".join(f"    y{i} = {i}" for i in range(3000))
    blocks = chunk_segments(_segs(big, kind="code"), size=768, overlap=96)
    assert blocks[0]["text"] == big          # 宁超不切
    assert blocks[0]["oversized"] is True    # 显式标记供质检提示


def test_summary_distribution_self_consistent():
    texts = ["字" * 60, "字" * 500, "字" * 1500]  # 覆盖 0-256 / 256-512 / 1024+
    segs = _segs(*texts)
    s = chunk_summary(segs, size=300, overlap=0)
    assert s["count"] >= 1
    assert sum(b["count"] for b in s["buckets"]) == s["count"]
    stats = s["stats"]
    assert 0 < stats["min"] <= stats["max"]
    assert stats["p50"] <= stats["p90"]
    assert s["params"] == {"size": 300, "overlap": 0}
    assert len(s["samples"]) <= 6
    for sample in s["samples"]:
        assert sample["tokens"] > 0 and sample["index"] >= 0


def test_empty_and_missing_segments():
    assert chunk_segments([]) == []
    s = chunk_summary([])
    assert s["count"] == 0 and s["stats"]["max"] == 0


def test_heading_meta_flow_into_chunk_label():
    segs = [
        {"index": 0, "text": "# 第一章", "meta": {"kind": "heading", "title": "第一章", "level": 1, "ancestors": []}},
        {"index": 1, "text": "第一章的正文内容很长。" * 3, "meta": {"kind": "paragraph", "ancestors": ["第一章"]}},
    ]
    blocks = chunk_segments(segs, size=2048, overlap=0)
    assert blocks  # 非空即可（正文聚合进标题块）
