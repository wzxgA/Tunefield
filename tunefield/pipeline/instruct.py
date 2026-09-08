"""T5 · 指令化（数据管线第 4 步）：切片 → Alpaca 训练样本。

对齐 B2.2：阶段一无 LLM 合成依赖，纯模板 + 启发式，产 LLaMA-Factory 直接可消费的
Alpaca JSONL（instruction / input / output 三字段）。

模板以 `pipeline/templates/*.ini` 可编辑文件存在（用户可改可加），格式：
    [template]
    name = 续写                 ; 界面展示名
    kind = continuation|qa      ; 样本形态（决定渲染逻辑）
    instruction = ...{domain}...; 领域引导语（{domain} 会被数据集名替换）

    [qa]                       ; 仅 kind=qa 生效
    strategy = salient_sentence ; 抽问策略（salient_sentence | first_sentence）

模板缺失/解析失败时自动回退内置默认模板，不阻断管线。
"""

from __future__ import annotations

import configparser
import json
import re
from pathlib import Path

# 模板目录（相对本文件）：pipeline/templates/
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# 内置默认模板（模板文件缺失时兜底）
_DEFAULT_TEMPLATES: dict[str, dict] = {
    "continuation": {
        "name": "续写",
        "kind": "continuation",
        "instruction": "你是{domain}领域的写作助手。请以与下列正文一致的{domain}领域风格续写，"
        "不要解释，只输出正文。",
    },
    "qa": {
        "name": "抽问",
        "kind": "qa",
        "instruction": "你是{domain}领域的问答助手，请依据背景中的原文，用原文句子作答。",
        "qa": {"strategy": "salient_sentence"},
    },
}

# 抽问的显著句特征：书名号/引号/百分比/单位/年份等实体感强的句子优先当答案
_SALIENT_RE = re.compile(
    r"《|》|“|”|『|』|[%％]|\d{4}年|年|月|日|[0-9]+(?:万|亿|个|台|元|户|家|款|项)"
)
_SENT_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;\n]?")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def list_templates() -> list[dict]:
    """模板清单（目录扫描 + 内置兜底项）。"""
    found: dict[str, dict] = {}
    if TEMPLATES_DIR.is_dir():
        for path in sorted(TEMPLATES_DIR.glob("*.ini")):
            data = _read_template_file(path)
            if data is not None:
                found[path.stem] = data
    for key, data in _DEFAULT_TEMPLATES.items():
        found.setdefault(key, dict(data))
    return [{"key": k, **v} for k, v in found.items()]


def _read_template_file(path: Path) -> dict | None:
    parser = configparser.ConfigParser()
    parser.optionxform = str  # 保留键大小写（此处键均为英文，防小写化习惯）
    try:
        parser.read(path, encoding="utf-8")
        if not parser.has_section("template"):
            return None
        sec = parser["template"]
        data: dict = {
            "name": sec.get("name", path.stem),
            "kind": sec.get("kind", "continuation").strip().lower(),
            "instruction": sec.get("instruction", "").strip(),
        }
        if data["kind"] not in ("continuation", "qa"):
            return None
        if parser.has_section("qa"):
            data["qa"] = {"strategy": parser.get("qa", "strategy", fallback="salient_sentence").strip()}
        return data
    except (configparser.Error, OSError):
        return None


def get_template(key: str) -> dict:
    """按 key（模板文件名去 .ini）取模板；找不到时回退同名内置/默认。"""
    for t in list_templates():
        if t["key"] == key:
            return t
    # 再回退内置默认（防模板目录为空）
    for name, data in _DEFAULT_TEMPLATES.items():
        if name == key:
            return {"key": key, **data}
    return {"key": "continuation", **_DEFAULT_TEMPLATES["continuation"]}


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _pick_answer_sentence(text: str, strategy: str) -> str:
    sents = _sentences(text)
    if not sents:
        return text.strip()
    if strategy == "first_sentence":
        return sents[0]
    # salient_sentence：优先含实体感的句子，否则回退首句
    for s in sents[1:]:
        if _SALIENT_RE.search(s):
            return s
    return sents[0]


def render_block(
    tpl: dict, text: str, *, domain: str = ""
) -> dict:
    """把一块文本渲染成一个 Alpaca 样本记录。text 为整块，输出字段不截断。"""
    kind = tpl.get("kind", "continuation")
    instruction = (tpl.get("instruction") or "").format(domain=domain).strip()
    if kind == "qa":
        strategy = (tpl.get("qa") or {}).get("strategy", "salient_sentence")
        answer = _pick_answer_sentence(text, strategy)
        # 背景 = 答案句之前至多 160 字的块文本，给模型可推理的线索
        head = text[: max(text.find(answer), 0)][-160:] if answer in text else ""
        if head:
            input_ = f"背景：{head}\n问题：{domain}领域的这段文本中，关于上述背景的关键内容是什么？请直接摘出原文句子作答。"
        else:
            input_ = f"问题：{domain}领域的这段文本想表达什么关键内容？请直接摘出原文句子作答。"
        return {"instruction": instruction, "input": input_, "output": answer}
    # continuation：领域引导语 + 整块正文
    return {"instruction": instruction, "input": None, "output": text}


def render_blocks(
    blocks: list[dict], *, template: str = "continuation", domain: str = ""
) -> list[dict]:
    """按模板把块序列渲染为 Alpaca 样本列表。template 为模板 key（文件名）。"""
    tpl = get_template(template)
    records: list[dict] = []
    for b in blocks:
        text = (b.get("text") or "").strip()
        if not text:
            continue
        rec = render_block(tpl, text, domain=domain)
        records.append(
            {"instruction": rec["instruction"], "input": rec["input"], "output": rec["output"]}
        )
    return records


def to_jsonl_line(record: dict) -> str:
    """单条 Alpaca 样本 → JSONL 行（ensure_ascii=False，中文直读）。"""
    return json.dumps(
        {
            "instruction": record.get("instruction", ""),
            "input": record.get("input"),
            "output": record.get("output", ""),
        },
        ensure_ascii=False,
    )
