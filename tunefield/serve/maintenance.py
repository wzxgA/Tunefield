"""数据/训练维护：删除数据集或训练任务（记录 + 磁盘产物 + 可选 Ollama 模型）。

语义（用户「数据和训练都能删」）：
- purge_job(job_id)：删除任务行、关联 pipeline_run、适配器与量化产物登记行，
  清理磁盘（适配器目录、GGUF 及 merged/fingerprint 文件、任务工作目录），
  进程内日志环；若这些 GGUF 已导入 Ollama，best-effort 一并移除（Ollama
  不可用不影响删除）。
- purge_dataset(dataset_id)：先级联删除其全部任务，再删 pipeline_run、raw
  原始文件与 datasets 管线产物目录、记录行。

安全：运行中（queued/pending_gpu/running）对象一律拒绝删除，由 API 层 409 前置；
本模块再次兜底断言。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tunefield import config
from tunefield.assets import registry as asset_registry
from tunefield.engine.base import EngineError, log_forget
from tunefield.serve import db

# 不能删除的进行中状态（磁盘/进程被占用）
_ACTIVE = {"queued", "pending_gpu", "running"}


def _rm_tree(path: Path) -> bool:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        return not path.exists()  # 可能仍有残留（占用）→ 返回 False
    return False


def _rm_file(path: Path) -> bool:
    try:
        if path.exists():
            path.unlink()
            return True
    except OSError:
        pass
    return False


def _remove_ollama_imports(adapter_rows: list[dict]) -> int:
    """best-effort 移除已导入 Ollama 的 tunefield-* 模型；返回移除数。"""
    from tunefield.serve import ollama

    removed = 0
    for a in adapter_rows:
        slug = asset_registry.slugify(a["domain"])
        version = a["version"] or a["job_id"][:8]
        name = f"tunefield-{slug}-{version}"
        try:
            ollama.remove_model(name)
            removed += 1
        except Exception:  # noqa: BLE001 - Ollama 不可用不影响平台侧删除
            continue
    return removed


def purge_job(job_id: str, *, remove_ollama: bool = True) -> dict:
    """删除一个训练任务（行 + 磁盘产物）。返回清理统计。"""
    job = db.get_job(job_id)
    if job is None:
        raise EngineError(f"任务不存在：{job_id}")
    if job.get("status") in _ACTIVE:
        raise EngineError(
            f"任务 {job_id} 正在执行（{job['status']}），请等待结束或停止后再删除"
        )

    removed_files = 0
    removed_dirs = 0

    # 关联端到端 run（flow 创建 job 时同 key 无关；pipeline_runs 引 job_id）
    for run in db.list_pipeline_runs():
        if run.get("job_id") == job_id:
            db.delete_pipeline_run(run["id"])
            log_forget(run["id"])

    adapter_rows = [a for a in db.list_adapters() if a["job_id"] == job_id]
    if remove_ollama:
        _remove_ollama_imports(adapter_rows)

    for a in adapter_rows:
        slug = asset_registry.slugify(a["domain"])
        version = a["version"] or job_id[:8]
        # GGUF 目录下的本任务产物：merged 目录 / f16 / 量化 / 指纹
        for p in sorted(config.GGUF_DIR.glob(f"{slug}-{version}-*")):
            if p.is_dir():
                removed_dirs += 1 if _rm_tree(p) else 0
            else:
                removed_files += 1 if _rm_file(p) else 0
        # 量化产物登记行（先读 path 再删行）
        qs = [q for q in db.list_quantized_models() if q["adapter_id"] == a["id"]]
        for q in qs:
            removed_files += 1 if _rm_file(Path(q["path"])) else 0
            db.delete_quantized_model(q["id"])
        # 适配器目录（训练工作目录）
        if a.get("path"):
            removed_dirs += 1 if _rm_tree(Path(a["path"])) else 0
        db.delete_adapter(a["id"])

    # 兜底：工作目录（未登记 adapter 时）
    workdir = config.ADAPTERS_DIR / (job.get("domain") or "model") / job_id
    removed_dirs += 1 if _rm_tree(workdir) else 0

    db.delete_job(job_id)
    log_forget(job_id)
    return {
        "deleted": True,
        "kind": "job",
        "job_id": job_id,
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
        "ollama_removed": None,
    }


def purge_dataset(dataset_id: str, *, remove_ollama: bool = True) -> dict:
    """删除一个数据集：级联其全部任务与端到端 run、原始文件与管线产物。"""
    ds = db.get_dataset(dataset_id)
    if ds is None:
        raise EngineError(f"数据集不存在：{dataset_id}")

    jobs = db.jobs_by_dataset(dataset_id)
    if any(j["status"] in _ACTIVE for j in jobs):
        running = [j["id"] for j in jobs if j["status"] in _ACTIVE]
        raise EngineError(
            f"数据集 {dataset_id} 有任务正在执行：{running}，请先等待/删除后再删数据集"
        )

    removed_files = 0
    removed_dirs = 0
    for j in jobs:
        r = purge_job(j["id"], remove_ollama=remove_ollama)
        removed_files += r["removed_files"]
        removed_dirs += r["removed_dirs"]

    for run in db.list_pipeline_runs():
        if run.get("dataset_id") == dataset_id:
            db.delete_pipeline_run(run["id"])
            log_forget(run["id"])

    # 原始文件与管线产物目录
    removed_files += 1 if _rm_file(config.RAW_DIR / ds["content_hash"]) else 0
    removed_dirs += 1 if _rm_tree(config.RAW_DIR / ds["content_hash"]) else 0
    removed_dirs += 1 if _rm_tree(config.DATASETS_DIR / dataset_id) else 0

    db.delete_dataset(dataset_id)
    return {
        "deleted": True,
        "kind": "dataset",
        "dataset_id": dataset_id,
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
    }
