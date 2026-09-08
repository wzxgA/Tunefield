"""T5 · 指令化：模板加载/回退、续写与抽问渲染、Alpaca JSONL 行。"""

from __future__ import annotations

import json

from tunefield.pipeline import instruct
from tunefield.pipeline.instruct import (
    TEMPLATES_DIR,
    get_template,
    list_templates,
    render_block,
    render_blocks,
    to_jsonl_line,
)


def _cont_block() -> dict:
    return {"index": 0, "text": "这是一段领域正文，用于续写训练。"}


def test_builtin_templates_listed():
    names = {t["key"] for t in list_templates()}
    assert {"continuation", "qa"} <= names
    meta = {t["key"]: t for t in list_templates()}
    assert meta["continuation"]["kind"] == "continuation"
    assert meta["qa"]["kind"] == "qa"


def test_template_files_are_editable_ini(tmp_path):
    # 内置模板确实来自可编辑文件（而非硬编码）
    assert (TEMPLATES_DIR / "continuation.ini").exists()
    assert (TEMPLATES_DIR / "qa.ini").exists()
    # 自定临时模板可被读取（用户新增模板即复制此形态）
    custom = tmp_path / "custom.ini"
    custom.write_text(
        "[template]\nname = 自定义\nkind = continuation\n"
        "instruction = 自定义引导：{domain}\n",
        encoding="utf-8",
    )
    data = instruct._read_template_file(custom)  # noqa: SLF001
    assert data is not None
    assert data["name"] == "自定义" and data["instruction"].startswith("自定义")


def test_get_template_fallback():
    # 未知模板回退到内置续写默认
    t = get_template("does_not_exist")
    assert t["key"] == "continuation"
    assert t["kind"] == "continuation"


def test_continuation_record_shape_and_domain():
    tpl = get_template("continuation")
    rec = render_block(tpl, _cont_block()["text"], domain="医疗")
    assert set(rec) == {"instruction", "input", "output"}
    assert "医疗" in rec["instruction"]
    assert rec["input"] is None
    assert rec["output"] == _cont_block()["text"]  # 整块正文为 output


def test_qa_salient_sentence_answer_keeps_entity_sentence():
    text = "本手册介绍基础概念。依据《质量管理规范》第 3 条执行复核。其余内容从略。"
    rec = render_block(get_template("qa"), text, domain="质量")
    # 优先含《》的句子作为答案
    assert "《质量管理规范》" in rec["output"]
    assert rec["output"].endswith("执行复核。")
    assert rec["input"]  # 附背景/问题
    assert "质量" in rec["input"]


def test_qa_first_sentence_strategy():
    tpl = {"key": "qa", "kind": "qa", "instruction": "q", "qa": {"strategy": "first_sentence"}}
    text = "第一句是关键。第二句也重要。"
    rec = render_block(tpl, text, domain="x")
    assert rec["output"] == "第一句是关键。"


def test_render_blocks_skips_blank():
    blocks = [{"text": "  "}, {"text": "有效正文A。"}, {"text": "有效正文B。"}]
    records = render_blocks(blocks, template="continuation", domain="电商")
    assert len(records) == 2
    assert all(r["output"] for r in records)
    assert "电商" in records[0]["instruction"]


def test_jsonl_line_is_valid_alpaca():
    rec = {"instruction": "请续写", "input": None, "output": "正文内容"}
    line = to_jsonl_line(rec)
    obj = json.loads(line)
    assert {"instruction", "input", "output"} == set(obj)
    assert obj["input"] is None and obj["output"] == "正文内容"
    # 中文不转义（ensure_ascii=False）
    assert "正文内容" in line
