"""F1 · FastAPI 应用与 serve 命令的最小测试。

用 TUNEFIELD_ROOT 隔离到临时目录，避免污染真实 data/。
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

TEST_ROOT = os.path.join(os.path.dirname(__file__), "_tmp_f1")


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("f1root")
    os.environ["TUNEFIELD_ROOT"] = str(root)
    # 刷新 config（PROJECT_ROOT 在 import 时计算）
    from tunefield import config
    importlib.reload(config)
    from tunefield.serve import app as app_mod
    from tunefield.serve import db as db_mod
    importlib.reload(db_mod)
    importlib.reload(app_mod)
    yield
    os.environ.pop("TUNEFIELD_ROOT", None)


def test_create_app_serves_health():
    from tunefield.serve.app import create_app

    with TestClient(create_app()) as c:
        r = c.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_init_db_is_idempotent_and_creates_tables():
    from tunefield.serve import db

    db.init_db()  # 二次调用不报错（幂等）
    with db.get_connection() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"datasets", "training_jobs", "adapters", "quantized_models"} <= tables


def test_create_job_flow_queued_to_done():
    """占位 handler 驱动：queued → running → done，且 loss 落库。"""
    from tunefield.serve.app import create_app

    with TestClient(create_app()) as c:
        r = c.post("/api/jobs", json={"domain": "aws", "kind": "finetune"})
        assert r.status_code == 200
        job_id = r.json()["id"]
        assert r.json()["status"] == "queued"

        final = c.get(f"/api/jobs/{job_id}").json()
        # 占位 handler 约需 2s；轮询等任务收敛到 done
        import time

        for _ in range(20):
            final = c.get(f"/api/jobs/{job_id}").json()
            if final["status"] in ("done", "failed"):
                break
            time.sleep(0.2)
        assert final["status"] == "done"
        assert final["progress"] == 1.0
        assert final["loss_json"]  # 非空，含 loss 采样点


def test_serve_subcommand_bound_to_real_handler():
    from tunefield import cli

    p = cli.build_parser()
    ns = p.parse_args(["serve", "--host", "127.0.0.1", "--port", "8999"])
    assert callable(ns.func)