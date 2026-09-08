"""T1 · 通用数据接入：收集、哈希去重、raw 落盘。

支持 txt / md / zip（混合包，内部可含目录）；目录/单文件/zip 均可。
去重策略：单文件按内容 SHA-256；整套数据按「文件内容哈希排序串联后再哈希」的集合指纹，
同一指纹重复接入直接复用已有 dataset（秒级返回），不重复落盘。

产物：
- 原始文件落盘到 data/raw/<set_hash>/<相对路径>
- datasets 表新增一行（T2 起在此基础上做解析/管线）
"""

from __future__ import annotations

import hashlib
import io
import uuid
import zipfile
from pathlib import Path

from tunefield import config


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def set_fingerprint(file_hashes: list[str]) -> str:
    """文件集合指纹：内容哈希排序后串联再哈希（顺序无关，重复即命中）。"""
    digest = hashlib.sha256()
    for h in sorted(file_hashes):
        digest.update(h.encode("ascii"))
    return digest.hexdigest()


def collect_from_path(path: str | Path) -> list[tuple[str, bytes]]:
    """收集 path（单文件 / 目录 / zip）下所有文本类原始文件，返回 [(相对路径, 字节)]。

    目录递归收集；zip 内部展开（含嵌套目录）；zip 内又套 zip 不递归解压。

    防呆：拒绝把平台自产目录（data/、models/、web/dist 等）当作待接入数据，
    避免用户误 ingest 自身产物（数据库、raw 落盘、模型权重）造成集合指纹漂移。
    """
    p = Path(path)
    rel = None
    try:
        rel = p.resolve().relative_to(config.PROJECT_ROOT.resolve())
    except ValueError:
        rel = None
    if rel is not None and rel.parts:
        top = rel.parts[0]
        if top in {"data", "models", "web", ".venv", "node_modules"}:
            raise ValueError(
                f"路径 {p} 属于平台自产目录（{top}/），不能作为待训练的数据源；"
                "请选择你自己的原始文件目录。"
            )

    files: list[tuple[str, bytes]] = []
    if p.is_file() and p.suffix.lower() == ".zip":
        files = _unpack_zip(p)
    elif p.is_file():
        files = [(p.name, p.read_bytes())]
    elif p.is_dir():
        for child in sorted(p.rglob("*")):
            if child.is_file() and child.suffix.lower() == ".zip":
                files.extend(_unpack_zip(child))
            elif child.is_file():
                files.append((child.name, child.read_bytes()))
    else:
        raise FileNotFoundError(f"路径不存在：{p}")
    return files


def _unpack_zip(zip_path: Path) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            # 仅取常规文件，跳过目录项与嵌套 zip（避免 zip 轰炸递归）
            if info.is_dir() or info.filename.lower().endswith(".zip"):
                continue
            files.append((info.filename, zf.read(info.filename)))
    return files


def ingest_files(
    files: list[tuple[str, bytes]],
    *,
    name: str,
    source: str = "",
) -> dict:
    """落盘原始文件到 raw/<set_hash>/ 并登记 dataset。返回 dataset 记录。

    若集合指纹已存在，则直接返回既有 dataset（复用，不重复写盘）。
    """
    from tunefield.serve import db

    if not files:
        raise ValueError("没有可接入的文件")

    file_hashes = [file_sha256(data) for _, data in files]
    fp = set_fingerprint(file_hashes)

    existed = db.get_dataset_by_hash(fp)
    if existed is not None:
        return existed  # 去重命中：秒级复用

    # 落盘原始文件
    dest_dir = config.RAW_DIR / fp
    for rel, data in files:
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    ds_id = uuid.uuid4().hex
    created = db.insert_dataset(
        id=ds_id,
        name=name,
        content_hash=fp,
        source=source,
        stats_json=None,
    )
    return created


def ingest_path(path: str | Path, *, name: str) -> dict:
    """CLI 入口：收集 path 并接入，返回 dataset 记录（幂等，重复秒回）。"""
    files = collect_from_path(path)
    return ingest_files(files, name=name, source=str(path))