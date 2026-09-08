"""T6 · build 全管线：上传 → POST build → train.jsonl 落盘 + 质检报告 + built 状态。"""

from __future__ import annotations

import importlib
import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("t6build")
    os.environ["TUNEFIELD_ROOT"] = str(root)
    from tunefield import config

    importlib.reload(config)
    from tunefield.serve import app as app_mod
    from tunefield.serve import db as db_mod

    importlib.reload(db_mod)
    importlib.reload(app_mod)
    yield
    os.environ.pop("TUNEFIELD_ROOT", None)


def _upload_mixed(client):
    md = ("# 领域文档\n\n" + "平台用于领域模型训练，正文示例。" * 80).encode("utf-8")
    py = ("def helper(x):\n    return x * 2\n\n" * 40).encode("utf-8")
    r = client.post(
        "/api/datasets",
        data={"name": "demo"},
        files=[
            ("files", ("doc.md", md, "text/markdown")),
            ("files", ("tool.py", py, "text/x-python")),
        ],
    )
    assert r.status_code == 200
    return r.json()["dataset"]


def test_build_full_pipeline(tmp_path):
    from tunefield.pipeline.build import run_build
    from tunefield.serve.app import create_app

    with TestClient(create_app()) as c:
        ds = _upload_mixed(c)
        # 直接调编排函数（同步、可测）
        result = run_build(ds, chunk_size=400, overlap=64)
        report_data = result["report"]
        assert report_data["summary"]["sample_count"] > 0
        assert report_data["summary"]["block_count"] >= 1

        # train.jsonl 落盘且行数与样本数一致，逐行是合法 JSON
        from pathlib import Path

        jsonl = Path(result["train_jsonl"])
        assert jsonl.exists()
        lines = jsonl.read_text(encoding="utf-8").splitlines()
        assert len(lines) == report_data["summary"]["sample_count"]
        row = json.loads(lines[0])
        assert set(row) == {"instruction", "input", "output"}

        # db 状态置 built、报告入 stats_json
        state = c.get(f"/api/datasets/{ds['id']}").json()
        assert state["status"] == "built"
        assert json.loads(state["stats_json"])["summary"]["sample_count"] == report_data["summary"]["sample_count"]

        # GET report 端点可读
        gr = c.get(f"/api/datasets/{ds['id']}/report")
        assert gr.status_code == 200
        assert gr.json()["report"]["summary"]["sample_count"] == report_data["summary"]["sample_count"]


def test_build_endpoint_marks_built():
    from tunefield.serve.app import create_app

    with TestClient(create_app()) as c:
        ds = _upload_mixed(c)
        r = c.post(f"/api/datasets/{ds['id']}/build", json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dataset"]["status"] == "built"
        assert body["report"]["summary"]["sample_count"] > 0
        # 重建幂等
        r2 = c.post(f"/api/datasets/{ds['id']}/build", json={})
        assert r2.status_code == 200


def test_build_missing_dataset_404_and_empty_400(tmp_path):
    from tunefield.serve.app import create_app

    with TestClient(create_app()) as c:
        assert c.post("/api/datasets/nope/build", json={}).status_code == 404
