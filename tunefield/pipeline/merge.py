"""数据集拼接（画布「合并」节点的真实语义）。

把多个数据集的原始文件取并集，落盘为一个**新数据集实体**：
- 文件级拼接：各源 raw 文件复制进新 raw 目录，按源加 `ds<id前6>/` 前缀避免同名覆盖；
- 新数据集拥有独立 id / 内容指纹 / 名称（如「hwen + src2」），后续管线
  （解析→清洗→切片→指令化→质检→训练）与数据屏展示都把它当普通数据集；
- 相同来源组合的拼接结果**幂等复用**（集合指纹一致 → ingest_files 直接返回既有）；
- 源数据集不受影响（只读它们的 raw）。
"""

from __future__ import annotations

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
