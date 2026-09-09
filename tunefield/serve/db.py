"""SQLite 访问层。

对齐方案 B3.3 的四张表结构；轻量做法：标准库 sqlite3 + WAL，
同步调用在 FastAPI 里经 anyio 线程池执行，不额外引入 ORM。

约束：
- 单机单进程（单 uvicorn 承载 API + 内嵌队列），无并发写竞争；
- 短连接 + row_factory，每次调用独立连接，避免跨协程共享连接。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from tunefield.config import DB_PATH, ensure_dirs


def get_connection() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 建表（幂等）
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  id           TEXT PRIMARY KEY,     -- ULID
  name         TEXT NOT NULL,
  content_hash TEXT UNIQUE,          -- 文件集合指纹，重复上传直接复用
  source       TEXT,                 -- 原始路径
  status       TEXT,                 -- ingested | built | failed
  stats_json   TEXT,                 -- 质检报告
  created_at   TEXT
);

CREATE TABLE IF NOT EXISTS training_jobs (
  id           TEXT PRIMARY KEY,
  dataset_id   TEXT REFERENCES datasets(id),
  domain       TEXT NOT NULL,
  kind         TEXT NOT NULL,        -- finetune | pretrain
  base_model   TEXT,
  config_json  TEXT,                 -- 最终生效配置（推荐值 + 用户覆盖）
  status       TEXT,                 -- queued|running|done|failed|pending_gpu
  progress     REAL DEFAULT 0,       -- 0.0~1.0
  loss_json    TEXT,                 -- loss 采样点 [ [step, value], ... ]
  error        TEXT,
  created_at   TEXT, started_at TEXT, finished_at TEXT
);

CREATE TABLE IF NOT EXISTS adapters (
  id              TEXT PRIMARY KEY,
  job_id          TEXT REFERENCES training_jobs(id),
  domain          TEXT NOT NULL,
  version         TEXT NOT NULL,
  path            TEXT NOT NULL,
  fingerprint_json TEXT,
  created_at      TEXT
);

CREATE TABLE IF NOT EXISTS quantized_models (
  id          TEXT PRIMARY KEY,
  adapter_id  TEXT REFERENCES adapters(id),
  quant       TEXT NOT NULL,
  path        TEXT NOT NULL,
  size_bytes  INTEGER,
  created_at  TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
  id            TEXT PRIMARY KEY,     -- T11 端到端编排（一键 run）
  dataset_id    TEXT REFERENCES datasets(id),
  domain        TEXT NOT NULL,
  job_id        TEXT,                 -- 训练子任务（run 内创建的真实 training job）
  config_json   TEXT,                 -- epochs/chunk/template 等请求参数
  status        TEXT,                 -- queued|running|done|failed|pending_gpu
  progress      REAL DEFAULT 0,       -- 0.0~1.0
  timeline_json TEXT,                 -- 阶段时间线 [{key,label,status,detail,started_at,finished_at,duration_s}]
  error         TEXT,
  created_at    TEXT, started_at TEXT, finished_at TEXT
);
"""


def init_db() -> None:
    """建表（幂等），返回前确保目录与 schema 就绪。"""
    ensure_dirs()
    with get_connection() as conn:
        conn.executescript(_SCHEMA)


# ---------------------------------------------------------------------------
# 训练任务访问层（F1 聚焦队列/状态机所需字段）
# ---------------------------------------------------------------------------


def insert_job(
    *, job_id: str, dataset_id: str | None, domain: str, kind: str,
    base_model: str | None, config_json: str | None, status: str,
    created_at: str,
) -> None:
    with transaction() as conn:
        conn.execute(
            "INSERT INTO training_jobs (id, dataset_id, domain, kind, base_model, "
            "config_json, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (job_id, dataset_id, domain, kind, base_model, config_json, status, created_at),
        )


def get_job(job_id: str) -> dict[str, Any] | None:
    with transaction() as conn:
        row = conn.execute(
            "SELECT * FROM training_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return dict(row) if row else None


def list_jobs() -> list[dict[str, Any]]:
    with transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM training_jobs ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def set_job_status(
    job_id: str, status: str, *,
    progress: float | None = None, error: str | None = None,
) -> None:
    with transaction() as conn:
        fields: list[str] = ["status = ?"]
        values: list[Any] = [status]
        if progress is not None:
            fields.append("progress = ?")
            values.append(progress)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        fields.append("finished_at = CASE WHEN ? IN ('done','failed') THEN "
                      "COALESCE(finished_at, strftime('%Y-%m-%dT%H:%M:%SZ','now')) "
                      "ELSE finished_at END")
        values.append(status)
        values.append(job_id)
        conn.execute(f"UPDATE training_jobs SET {', '.join(fields)} WHERE id = ?", values)


def append_loss(job_id: str, loss: str) -> None:
    """追加一个 loss 采样点（loss 由调用方序列化，如 '[step,value]'）。"""
    with transaction() as conn:
        conn.execute(
            "UPDATE training_jobs SET loss_json = CASE "
            "WHEN loss_json IS NULL OR loss_json = '' THEN ? "
            "ELSE loss_json || ',' || ? END WHERE id = ?",
            (loss, loss, job_id),
        )


def jobs_by_status(statuses: Sequence[str]) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in statuses)
    with transaction() as conn:
        rows = conn.execute(
            f"SELECT * FROM training_jobs WHERE status IN ({placeholders}) "
            "ORDER BY created_at",
            list(statuses),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 数据集访问层（T1 接入）
# ---------------------------------------------------------------------------


def insert_dataset(
    *, id: str, name: str, content_hash: str, source: str | None,
    stats_json: str | None,
) -> dict[str, Any]:
    with transaction() as conn:
        conn.execute(
            "INSERT INTO datasets (id, name, content_hash, source, status, stats_json, created_at) "
            "VALUES (?,?,?,?,?,?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))",
            (id, name, content_hash, source, "ingested", stats_json),
        )
        row = conn.execute(
            "SELECT * FROM datasets WHERE id = ?", (id,)
        ).fetchone()
    return dict(row) if row else {}


def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    with transaction() as conn:
        row = conn.execute(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
    return dict(row) if row else None


def get_dataset_by_hash(content_hash: str) -> dict[str, Any] | None:
    with transaction() as conn:
        row = conn.execute(
            "SELECT * FROM datasets WHERE content_hash = ?", (content_hash,)
        ).fetchone()
    return dict(row) if row else None


def list_datasets() -> list[dict[str, Any]]:
    with transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM datasets ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def set_dataset_built(dataset_id: str, stats_json: str) -> None:
    """T6：管线构建完成 → status=built，stats_json 写质检报告。"""
    with transaction() as conn:
        conn.execute(
            "UPDATE datasets SET status = 'built', stats_json = ? WHERE id = ?",
            (stats_json, dataset_id),
        )


# ---------------------------------------------------------------------------
# 资产访问层（T9 量化导出：adapters / quantized_models）
# ---------------------------------------------------------------------------


def insert_adapter(
    *, id: str, job_id: str, domain: str, version: str, path: str,
    fingerprint_json: str | None,
) -> dict[str, Any]:
    """登记 LoRA 适配器（幂等：同 job 重复导出覆盖 fingerprint）。"""
    with transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO adapters (id, job_id, domain, version, path, "
            "fingerprint_json, created_at) VALUES (?,?,?,?,?,?, "
            "COALESCE((SELECT created_at FROM adapters WHERE id=?), "
            "strftime('%Y-%m-%dT%H:%M:%SZ','now')))",
            (id, job_id, domain, version, path, fingerprint_json, id),
        )
        row = conn.execute("SELECT * FROM adapters WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else {}


def get_adapter(adapter_id: str) -> dict[str, Any] | None:
    with transaction() as conn:
        row = conn.execute("SELECT * FROM adapters WHERE id = ?", (adapter_id,)).fetchone()
    return dict(row) if row else None


def list_adapters() -> list[dict[str, Any]]:
    with transaction() as conn:
        rows = conn.execute("SELECT * FROM adapters ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]


def insert_quantized_model(
    *, id: str, adapter_id: str, quant: str, path: str, size_bytes: int
) -> dict[str, Any]:
    with transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO quantized_models (id, adapter_id, quant, path, "
            "size_bytes, created_at) VALUES (?,?,?,?,?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))",
            (id, adapter_id, quant, path, size_bytes),
        )
        row = conn.execute("SELECT * FROM quantized_models WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else {}


def list_quantized_models() -> list[dict[str, Any]]:
    with transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM quantized_models ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 端到端运行访问层（T11：pipeline_runs 一键 run + 阶段时间线）
# ---------------------------------------------------------------------------


def insert_pipeline_run(
    *, id: str, dataset_id: str, domain: str, job_id: str | None,
    config_json: str | None, status: str, created_at: str,
) -> dict[str, Any]:
    with transaction() as conn:
        conn.execute(
            "INSERT INTO pipeline_runs (id, dataset_id, domain, job_id, config_json, "
            "status, progress, created_at) VALUES (?,?,?,?,?,?,0,?)",
            (id, dataset_id, domain, job_id, config_json, status, created_at),
        )
        row = conn.execute("SELECT * FROM pipeline_runs WHERE id = ?", (id,)).fetchone()
    return dict(row) if row else {}


def get_pipeline_run(run_id: str) -> dict[str, Any] | None:
    with transaction() as conn:
        row = conn.execute(
            "SELECT * FROM pipeline_runs WHERE id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


def list_pipeline_runs() -> list[dict[str, Any]]:
    with transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM pipeline_runs ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def pipeline_runs_by_status(statuses: Sequence[str]) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in statuses)
    with transaction() as conn:
        rows = conn.execute(
            f"SELECT * FROM pipeline_runs WHERE status IN ({placeholders}) "
            "ORDER BY created_at",
            list(statuses),
        ).fetchall()
    return [dict(r) for r in rows]


def set_pipeline_run(
    run_id: str, *, status: str | None = None, progress: float | None = None,
    error: str | None = None, job_id: str | None = None,
) -> None:
    """更新 pipeline_run 字段（status/progress/error/job_id）。"""
    fields: list[str] = []
    values: list[Any] = []
    if status is not None:
        fields.append("status = ?")
        values.append(status)
        if status in ("running",):
            fields.append("started_at = COALESCE(started_at, "
                          "strftime('%Y-%m-%dT%H:%M:%SZ','now'))")
        if status in ("done", "failed"):
            fields.append("finished_at = COALESCE(finished_at, "
                          "strftime('%Y-%m-%dT%H:%M:%SZ','now'))")
    if progress is not None:
        fields.append("progress = ?")
        values.append(progress)
    if error is not None:
        fields.append("error = ?")
        values.append(error)
    if job_id is not None:
        fields.append("job_id = ?")
        values.append(job_id)
    if not fields:
        return
    values.append(run_id)
    with transaction() as conn:
        conn.execute(
            f"UPDATE pipeline_runs SET {', '.join(fields)} WHERE id = ?", values
        )


def _timeline_of(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    row = conn.execute(
        "SELECT timeline_json FROM pipeline_runs WHERE id = ?", (run_id,)
    ).fetchone()
    raw = row["timeline_json"] if row and row["timeline_json"] else None
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []


def append_pipeline_phase(run_id: str, phase: dict) -> None:
    """时间线末尾追加一个阶段条目。"""
    with transaction() as conn:
        phases = _timeline_of(conn, run_id)
        phases.append(phase)
        conn.execute(
            "UPDATE pipeline_runs SET timeline_json = ? WHERE id = ?",
            (json.dumps(phases, ensure_ascii=False), run_id),
        )


def update_pipeline_phase(
    run_id: str, key: str, *, status: str | None = None,
    finished_at: str | None = None, duration_s: float | None = None,
    detail: str | None = None,
) -> None:
    """按 key 就地更新时间线阶段（补完成状态/耗时/说明）。"""
    with transaction() as conn:
        phases = _timeline_of(conn, run_id)
        target = next((p for p in phases if p.get("key") == key), None)
        if target is None:
            return
        if status is not None:
            target["status"] = status
        if finished_at is not None:
            target["finished_at"] = finished_at
        if duration_s is not None:
            target["duration_s"] = round(duration_s, 1)
        if detail is not None:
            target["detail"] = detail
        conn.execute(
            "UPDATE pipeline_runs SET timeline_json = ? WHERE id = ?",
            (json.dumps(phases, ensure_ascii=False), run_id),
        )