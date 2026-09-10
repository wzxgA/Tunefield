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
    """对单个 dataset 执行全管线构建（多数据集合并见 run_build_multi）。"""
    return run_build_multi(
        [dataset], chunk_size=chunk_size, overlap=overlap, template=template
    )


def run_build_multi(
    datasets: list[dict],
    *,
    chunk_size: int = 768,
    overlap: int = 96,
    template: str = "continuation",
) -> dict:
    """对一或多个 dataset 执行全管线构建；多源时语料汇池、统一切片/质检。

    - 单源：行为与历史 run_build 完全一致；
    - 多源：逐源解析+清洗 → 合并段 → 统一切片/指令化 → 质检（重复率跨源计算）；
      产物 train.jsonl/corpus.txt/report.json 写入**主数据集**（列表第一个）目录，
      所有源数据集 status→built，报告附 sources 逐源分解。

    返回 {"dataset": 主数据集更新行, "train_jsonl", "report", "corpus_path",
    "built_ids"}。
    """
    if not datasets:
        raise ValueError("没有可构建的数据集")

    main = datasets[0]
    cleaned_texts: list[str] = []
    blocks: list[dict] = []
    file_status: dict[str, int] = {}
    parse_errors: list[str] = []
    corpus_bytes = 0
    sources: list[dict] = []

    for ds in datasets:
        raw_root = config.RAW_DIR / ds["content_hash"]
        if not raw_root.exists():
            raise FileNotFoundError(
                f"数据集 {ds['name']} 的原始文件目录不存在：{raw_root}"
            )
        src_chars = 0
        src_files = 0
        prefix = f"ds{ds['id'][:6]}/"

        for path in sorted(raw_root.rglob("*")):
            if not path.is_file():
                continue
            rel = prefix + path.relative_to(raw_root).as_posix()
            src_files += 1
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
                src_chars += len(text.encode("utf-8"))
            for b in chunk_segments(cleaned["segments"], size=chunk_size, overlap=overlap):
                b.setdefault("meta", {})["path"] = rel
                blocks.append(b)
        sources.append(
            {"id": ds["id"], "name": ds.get("name") or ds["id"],
             "files": src_files, "corpus_bytes": src_chars}
        )

    if not blocks or not any(t.strip() for t in cleaned_texts):
        raise ValueError(
            f"数据集 {main['name']}"
            + (f" 等 {len(datasets)} 个" if len(datasets) > 1 else "")
            + " 没有可训练的文本内容"
            + (f"（解析失败 {len(parse_errors)} 个文件，如 {parse_errors[0]}）" if parse_errors else "")
        )

    domain_name = main.get("name") or main["id"]
    if len(datasets) > 1:
        domain_name += f" 等 {len(datasets)} 源"

    records = render_blocks(blocks, template=template, domain=main.get("name") or "")
    jsonl_text = "\n".join(to_jsonl_line(r) for r in records) + "\n"
    jsonl_path = _write_dataset_out(main, jsonl_text)

    # T13：同时沉淀纯文本语料（逐块文本，块间空行分隔），供引擎 B 从零预训练
    # （预训练不需要 Alpaca 指令形态，只吃原始文本序列）。
    corpus_dir = config.DATASETS_DIR / main["id"]
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / "corpus.txt"
    corpus_path.write_text(
        "\n\n".join(t for t in cleaned_texts if t) + "\n", encoding="utf-8"
    )

    report = evaluate(
        dataset_name=domain_name,
        corpus_bytes=corpus_bytes,
        cleaned_texts=cleaned_texts,
        blocks=blocks,
        records=records,
        file_status=file_status,
        template=template,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if len(datasets) > 1:
        report["sources"] = sources  # 逐源分解（质检报告与画布悬停可展示）
    report_path = config.DATASETS_DIR / main["id"] / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    from tunefield.serve import db

    for ds in datasets:
        db.set_dataset_built(ds["id"], json.dumps(report, ensure_ascii=False))
    updated = db.get_dataset(main["id"])
    return {
        "dataset": updated,
        "train_jsonl": str(jsonl_path),
        "report": report,
        "corpus_path": str(corpus_path),
        "built_ids": [ds["id"] for ds in datasets],
    }
