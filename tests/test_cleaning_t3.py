"""T3 · 清洗器：乱码/控制字符、HTML、全半角、页码/页眉、字数差异与幂等。"""

from __future__ import annotations

from tunefield.pipeline.cleaning import RULE_LABELS, clean_segments, clean_text


def _segs(*texts: str, kind: str = "paragraph") -> list[dict]:
    return [
        {"index": i, "text": t, "meta": {"kind": kind}}
        for i, t in enumerate(texts)
    ]


def test_invisible_and_control_chars():
    text = "正常内容\u0000乱码\u200b零宽\ufeffBOM\u00ad软连字符"
    cleaned, stats = clean_text(text)
    assert "\u0000" not in cleaned and "\u200b" not in cleaned
    assert "\ufeff" not in cleaned and "\u00ad" not in cleaned
    assert stats["invisible"] >= 4


def test_space_normalization():
    text = "左\u3000右\u00a0nb\u202f窄"
    cleaned, stats = clean_text(text)
    assert cleaned == "左 右 nb 窄"
    assert stats["space"] == 3


def test_fullwidth_ascii_but_keep_cn_punct():
    text = "Ｈｅｌｌｏ１２３，中文。"
    cleaned, stats = clean_text(text)
    assert cleaned == "Hello123，中文。"  # 逗号/句号保持全角
    assert stats["fullwidth"] == 8  # Hello123 = 8 个字符


def test_html_tags_and_entities_idempotent():
    text = '<p>标题 &amp; 正文</p> &lt;b&gt;加粗&lt;/b&gt; 换行<br/>下一行'
    cleaned, stats = clean_text(text)
    assert "标题 & 正文 加粗 换行\n下一行" == cleaned
    assert stats["html"] >= 4
    # 幂等：第二遍无新增 HTML 处理
    _, stats2 = clean_text(cleaned)
    assert stats2.get("html", 0) == 0


def test_page_lines_removed():
    segs = _segs(
        "第一段正文\n- 3 -\n继续",
        "Page 2\n第二页正文",
        "第 3 页\n第三页正文\n3/5",
    )
    res = clean_segments(segs)
    texts = [s["text"] for s in res["segments"]]
    assert all("Page" not in t and "3 /" not in t for t in texts)
    assert res["stats"]["page"] >= 4


def test_boilerplate_repeated_header_removed():
    header = "ACME 技术手册 —— 仅供内部使用"
    segs = _segs(
        f"{header}\n第一段内容",
        f"{header}\n第二段内容",
        f"{header}\n第三段内容",
    )
    res = clean_segments(segs)
    assert all(header not in s["text"] for s in res["segments"])
    assert res["stats"]["boiler"] == 3


def test_code_segments_keep_structure_and_skip_html():
    src = '"""<p>在文档字符串里</p>"""\n\n\nx = 1  # A & B\n'
    segs = _segs(src, kind="code")
    res = clean_segments(segs)
    out = res["segments"][0]["text"]
    assert "<p>" in out and "A & B" in out  # HTML 与实体原样保留
    assert "\n\n\n" in out  # 代码空行不压缩
    assert res["stats"].get("html", 0) == 0


def test_chars_diff_reported():
    segs = _segs("<b>乱\u0000码</b> 正文", "好\n\n\n\n文本")
    res = clean_segments(segs)
    assert res["before_chars"] > res["after_chars"]
    diff = res["before_chars"] - res["after_chars"]
    assert diff == res["before_chars"] - sum(len(s["text"]) for s in res["segments"])
    assert diff > 0
    assert set(res["stats"]) <= set(RULE_LABELS)


def test_clean_is_approximately_idempotent():
    segs = _segs(
        '<h1>手册</h1>\r\n\u3000内容ＡＢＣ\n- 3 -\n',
        "另一页\nPage 2\n<footer>脚注</footer>",
    )
    once = clean_segments(segs)
    twice = clean_segments(once["segments"])
    assert once["before_chars"] - once["after_chars"] >= twice[
        "before_chars"
    ] - twice["after_chars"]
    assert twice["stats"] == {}  # 第二轮无命中 → 幂等


def test_empty_segments():
    res = clean_segments([])
    assert res["segments"] == [] and res["stats"] == {}
    assert res["before_chars"] == res["after_chars"] == 0
