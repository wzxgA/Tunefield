"""T6 · build 编排：raw 数据集跑完整管线 → JSONL + 质检报告。

链路（对齐 B4.2）：解析(T2) → 清洗(T3) → 切片(T4) → 指令化(T5) → 质检(T6)。
产物与幂等：
- `data/datasets/<dataset_id>/train.jsonl`（Alpaca，逐行一条样本）
- `data/datasets/<dataset_id>/report.json`（质检报告）
- `datasets` 表 status → built，stats_json 写入报告

对 parse 失败/不支持的文件只计数与示例留存，不阻断整体；整批无可用文本时明确报错。
"""

from __future__ import annotations

import json
from pathlib import Path

from tunefield import config
from tunefield.pipeline.cleaning import clean_segments
from tunefield.pipeline.chunking import chunk_segments
from tunefield.pipeline.instruct import render_blocks, to_jsonl_line
from tunefield.pipeline.parsers import parse_bytes
from tunefield.pipeline.report import evaluate


def lookup_dataset(ref: str) -> dict:
    """按 id 或名称精确查找 dataset；找不到抛 ValueError。"""
    from tunefield.serve import db

    ds = db.get_dataset(ref)
    if ds is not None:
        return ds
    for cand in db.list_datasets():
        if cand["name"] == ref:
            return cand
    raise ValueError(f"dataset 不存在：{ref}")


def _write_dataset_out(dataset: dict, content: str) -> Path:
    out_dir = config.DATASETS_DIR / dataset["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "train.jsonl"
    path.write_text(content, encoding="utf-8")
    return path


def run_build(
    dataset: dict,
    *,
    chunk_size: int = 768,
    overlap: int = 96,
    template: str = "continuation",
) -> dict:
    """对 dataset（含 content_hash）执行全管线构建。

    返回 {"dataset": {...}, "train_jsonl": str, "report": {...}}。
    构建完成后 datasets.status=built、stats_json=report。
    """
    raw_root = config.RAW_DIR / dataset["content_hash"]
    if not raw_root.exists():
        raise FileNotFoundError(
            f"数据集 {dataset['name']} 的原始文件目录不存在：{raw_root}"
        )

    cleaned_texts: list[str] = []
    blocks: list[dict] = []
    file_status: dict[str, int] = {}
    parse_errors: list[str] = []
    corpus_bytes = 0

    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(raw_root).as_posix()
        r = parse_bytes(rel, path.read_bytes())
        file_status[r["status"]] = file_status.get(r["status"], 0) + 1
        if r["status"] != "ok" or not r.get("segments"):
            if r["status"] == "error":
                parse_errors.append(f"{rel}: {r.get('error') or '解析失败'}")
            continue
        cleaned = clean_segments(r["segments"])
        for seg in cleaned["segments"]:
            text = seg.get("text") or ""
            if not text:
                continue
            cleaned_texts.append(text)
            corpus_bytes += len(text.encode("utf-8"))
        for b in chunk_segments(cleaned["segments"], size=chunk_size, overlap=overlap):
            b.setdefault("meta", {})["path"] = rel
            blocks.append(b)

    if not blocks or not any(t.strip() for t in cleaned_texts):
        raise ValueError(
            f"数据集 {dataset['name']} 没有可训练的文本内容"
            + (f"（解析失败 {len(parse_errors)} 个文件，如 {parse_errors[0]}）" if parse_errors else "")
        )

    records = render_blocks(blocks, template=template, domain=dataset.get("name") or "")
    jsonl_text = "\n".join(to_jsonl_line(r) for r in records) + "\n"
    jsonl_path = _write_dataset_out(dataset, jsonl_text)

    report = evaluate(
        dataset_name=dataset.get("name") or dataset["id"],
        corpus_bytes=corpus_bytes,
        cleaned_texts=cleaned_texts,
        blocks=blocks,
        records=records,
        file_status=file_status,
        template=template,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    report_path = config.DATASETS_DIR / dataset["id"] / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    from tunefield.serve import db

    db.set_dataset_built(dataset["id"], json.dumps(report, ensure_ascii=False))
    updated = db.get_dataset(dataset["id"])
    return {"dataset": updated, "train_jsonl": str(jsonl_path), "report": report}
