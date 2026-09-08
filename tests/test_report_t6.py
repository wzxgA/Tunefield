"""T6 · 质检报告：低质判断、重复率、阈值告警与行动建议。"""

from __future__ import annotations

from tunefield.pipeline import report
from tunefield.pipeline.chunking import chunk_segments
from tunefield.pipeline.cleaning import clean_segments


def _blocks(texts):
    return chunk_segments(clean_segments(
        [{"index": i, "text": t, "meta": {"kind": "paragraph"}} for i, t in enumerate(texts)]
    )["segments"], size=2000, overlap=0)


def test_low_quality_detection():
    assert report.is_low_quality("")
    assert not report.is_low_quality("正常中文内容。")
    assert report.is_low_quality("乱码�乱码�乱码�")
    assert not report.is_low_quality("排版正常，有标点。")
    # 有效字符占比过低 → 低质
    assert report.is_low_quality("###" * 200)  # 无有效文字


def test_block_duplicate_rate():
    # 直接用块文本（不经切片聚合），模拟跨文件重复块
    blocks = [
        {"text": "重复内容啊"},
        {"text": "重复内容啊"},
        {"text": "不同的内容"},
    ]
    rate = report.block_duplicate_rate(blocks)
    assert 0 < rate < 1
    assert abs(rate - 1 / 3) < 0.01
    assert report.block_duplicate_rate([]) == 0.0


def test_evaluate_blocks_on_tiny_data():
    rec = lambda: {"instruction": "x", "input": None, "output": "正文"}
    blocks = _blocks(["第一段正文内容。", "第二段正文内容。"])
    res = report.evaluate(
        dataset_name="demo",
        corpus_bytes=500,               # < 5MB → warn
        cleaned_texts=["第一段正文内容。", "第二段正文内容。"],
        blocks=blocks,
        records=[rec() for _ in range(5)],  # < 200 → error（阻断）
        template="continuation",
    )
    assert res["overall"] == "error"
    by = {c["metric"]: c for c in res["checks"]}
    assert by["样本数"]["level"] == "error"
    assert by["语料总量"]["level"] == "warn"
    assert res["summary"]["sample_count"] == 5


def test_evaluate_ok_when_enough_data():
    corpus = "语料正文。" * 2000
    texts = [corpus, corpus + "另一部分。" + "长句。" * 100]
    blocks = _blocks(texts)
    records = [
        {"instruction": "i", "input": None, "output": "o"} for _ in range(400)
    ]
    res = report.evaluate(
        dataset_name="ok",
        corpus_bytes=int(6 * 1024 * 1024),
        cleaned_texts=texts,
        blocks=blocks,
        records=records,
    )
    by = {c["metric"]: c for c in res["checks"]}
    assert res["overall"] != "error"
    assert by["样本数"]["level"] == "info"


def test_length_out_of_range_warns():
    giant = "字" * 3000  # >2048 token
    blocks = _blocks([giant, "正常段落。", "正常段落。"])
    res = report.evaluate(
        dataset_name="long",
        corpus_bytes=4096 * 1024,
        cleaned_texts=[giant, "正常段落。"],
        blocks=blocks,
        records=[{"output": "x"} for _ in range(300)],
    )
    by = {c["metric"]: c for c in res["checks"]}
    assert by["长度分布"]["level"] == "warn"


def test_report_summary_keeps_params_and_examples():
    texts = ["乱码�乱码。", "正常内容。"]
    blocks = _blocks(["乱码�乱码。", "正常内容。"])
    res = report.evaluate(
        dataset_name="x", corpus_bytes=1024, cleaned_texts=texts,
        blocks=blocks, records=[{"output": "x"}],
    )
    assert res["summary"]["params"]["chunk_size"] == 768
    assert res["lowq_examples"]  # 有低质样例可给前端
