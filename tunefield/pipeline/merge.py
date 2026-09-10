"""数据集拼接（画布「合并」节点的真实语义）。

把多个数据集的原始文件取并集，落盘为一个**新数据集实体**：
- 文件级拼接：各源 raw 文件复制进新 raw 目录，按源加 `ds<id前6>/` 前缀避免同名覆盖；
- 新数据集拥有独立 id / 内容指纹 / 名称（如「hwen + src2」），后续管线
  （解析→清洗→切片→指令化→质检→训练）与数据屏展示都把它当普通数据集；
- 相同来源组合的拼接结果**幂等复用**（集合指纹一致 → ingest_files 直接返回既有）；
- 源数据集不受影响（只读它们的 raw）。
"""

from __future__ import annotations

import json

from tunefield import config

MAX_AUTO_NAME_SOURCES = 2  # 自动命名最多列前 N 个源名


def merge_datasets(dataset_ids: list[str], *, name: str | None = None) -> dict:
    """把多个数据集拼接为一个新数据集；返回 {dataset, deduped, sources}。

    - dataset: 新数据集记录（或指纹命中时复用的既有记录）；
    - deduped: 是否命中既有同内容数据集（相同来源组合重复拼接时为 True）；
    - sources: 参与拼接的源数据集 id（去重后、按传入顺序）。
    至少需要 2 个源，否则抛 ValueError。
    """
    from tunefield.pipeline.ingest import file_sha256, ingest_files, set_fingerprint
    from tunefield.serve import db

    ordered: list[str] = []
    for ds_id in dataset_ids:
        if ds_id and ds_id not in ordered:
            ordered.append(ds_id)
    if len(ordered) < 2:
        raise ValueError("合并至少需要 2 个数据集")

    sources: list[dict] = []
    files: list[tuple[str, bytes]] = []
    for ds_id in ordered:
        ds = db.get_dataset(ds_id)
        if ds is None:
            raise ValueError(f"数据集不存在：{ds_id}")
        root = config.RAW_DIR / ds["content_hash"]
        if not root.exists():
            raise FileNotFoundError(f"数据集 {ds['name']} 的原始文件目录不存在：{root}")
        prefix = f"ds{ds['id'][:6]}"
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                files.append((f"{prefix}/{rel}", path.read_bytes()))
        sources.append({"id": ds["id"], "name": ds.get("name") or ds["id"]})

    if not files:
        raise ValueError("参与合并的数据集没有可拼接的文件")

    # 相同来源组合 → 同指纹 → 命中既有合并数据集（幂等）
    fp = set_fingerprint([file_sha256(data) for _, data in files])
    existed = db.get_dataset_by_hash(fp)

    auto_name = name or _auto_name(sources)
    dataset = ingest_files(files, name=auto_name, source="merge")
    return {
        "dataset": dataset,
        "deduped": existed is not None and dataset["id"] == existed["id"],
        "sources": [s["id"] for s in sources],
        "source_names": [s["name"] for s in sources],
    }


def _auto_name(sources: list[dict]) -> str:
    names = [s["name"] for s in sources]
    if len(names) <= MAX_AUTO_NAME_SOURCES:
        return " + ".join(names)
    return " + ".join(names[:MAX_AUTO_NAME_SOURCES]) + f" 等 {len(names)} 源"


# ---------------------------------------------------------------------------
# 回收：流程内合并数据集的生命周期 = 本次 run（终态即回收）
# ---------------------------------------------------------------------------


def cleanup_merged_dataset(
    dataset_id: str,
    *,
    fallback_dataset_id: str | None = None,
    merged_from: list[str] | None = None,
) -> bool:
    """回收一个流程内合并数据集：原始文件 + 构建产物 + datasets 行。

    - 只处理 source=merge 的记录（普通数据集一律不动）；
    - 仍有 job/run 引用时，先把引用回填到源数据集（fallback/merged_from 中
      首个仍存在的），再删行——训练产物（adapters/gguf）在独立目录，不受影响；
    - 若所有源都已删除、无法回填引用，则降级为**只删文件、保留行**（避免破坏外键）；
    - KEEP_MERGED=1 时整体跳过（调试用）。
    返回是否执行了清理。
    """
    import shutil

    from tunefield.serve import db

    if config.KEEP_MERGED:
        return False

    ds = db.get_dataset(dataset_id)
    if ds is None or (ds.get("source") or "") != "merge":
        return False

    # 回填引用：候选源 = fallback 优先，其次 merged_from（取首个仍存在的）
    candidates: list[str] = []
    for cand in [fallback_dataset_id, *(merged_from or [])]:
        if cand and cand != dataset_id and cand not in candidates:
            candidates.append(cand)
    survivor = next((c for c in candidates if db.get_dataset(c) is not None), None)

    has_refs = bool(
        [j for j in db.list_jobs() if j.get("dataset_id") == dataset_id]
        or db.pipeline_runs_by_dataset(dataset_id)
    )
    can_drop_row = survivor is not None or not has_refs

    if survivor is not None:
        db.reassign_dataset_refs(dataset_id, survivor)

    # 文件回收（raw 拷贝 + 构建产物）
    if ds.get("content_hash"):
        raw = config.RAW_DIR / ds["content_hash"]
        if raw.exists():
            shutil.rmtree(raw, ignore_errors=True)
    out = config.DATASETS_DIR / dataset_id
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)

    if can_drop_row:
        db.delete_dataset_row(dataset_id)
    return True


def cleanup_orphan_merged() -> dict:
    """服务启动兜底：回收进程被杀等情况下残留的合并数据集。

    - 无 run 引用 → 直接回收；
    - 有 run 引用 → 用该 run config 里的 merged_from 回填后回收。
    返回 {"removed": [...], "kept": [...]}。
    """
    from tunefield.serve import db

    if config.KEEP_MERGED:
        return {"removed": [], "kept": [], "skipped": "KEEP_MERGED"}

    removed: list[str] = []
    kept: list[str] = []
    for ds in db.list_datasets():
        if (ds.get("source") or "") != "merge":
            continue
        merged_from: list[str] = []
        for run in db.pipeline_runs_by_dataset(ds["id"]):
            raw = run.get("config_json") or "{}"
            try:
                cfg = json.loads(raw) if isinstance(raw, str) else {}
            except (ValueError, TypeError):
                cfg = {}
            merged_from = list((cfg or {}).get("merged_from") or [])
            if merged_from:
                break
        ok = cleanup_merged_dataset(
            ds["id"],
            fallback_dataset_id=merged_from[0] if merged_from else None,
            merged_from=merged_from,
        )
        (removed if ok else kept).append(ds["id"])
    return {"removed": removed, "kept": kept}
