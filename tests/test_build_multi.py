"""多数据集合并构建（画布多数据源 + 合并节点 → run_build_multi）。"""
from __future__ import annotations

import hashlib
import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    """独立数据根：避免与运行中的 serve 进程争抢 SQLite 写锁。"""
    root = tmp_path_factory.mktemp("buildmulti")
    os.environ["TUNEFIELD_ROOT"] = str(root)
    from tunefield import config

    importlib.reload(config)
    from tunefield.serve import app as app_mod
    from tunefield.serve import db as db_mod

    importlib.reload(db_mod)
    importlib.reload(app_mod)
    db_mod.init_db()
    yield
    os.environ.pop("TUNEFIELD_ROOT", None)


def _mk_dataset_with_raw(name: str, text: str) -> dict:
    """造一个带 raw 文件的数据集（ingest 级直写，绕过上传）。"""
    import json as _json
    import uuid

    from tunefield import config
    from tunefield.serve import db

    uid = uuid.uuid4().hex
    h = hashlib.sha256(uid.encode()).hexdigest()  # 随机 hash：测试库持久，固定名会撞 UNIQUE
    raw = config.RAW_DIR / h
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{name}.txt").write_text(text, encoding="utf-8")
    return db.insert_dataset(
        id=uid, name=name, content_hash=h, source="upload",
        stats_json=_json.dumps({"summary": {"sample_count": 0}}),
    )


def test_run_build_multi_merges_two_datasets(monkeypatch, tmp_path):
    from tunefield.pipeline.build import run_build_multi

    ds1 = _mk_dataset_with_raw("源甲", "甲领域语料第一段，内容足够长，用于切块测试。" * 30)
    ds2 = _mk_dataset_with_raw("源乙", "乙领域语料第一段，内容足够长，用于切块测试。" * 30)

    res = run_build_multi([ds1, ds2], chunk_size=256, overlap=32)

    # 两个源都被置 built；主数据集 = 第一个
    from tunefield.serve import db

    assert res["built_ids"] == [ds1["id"], ds2["id"]]
    assert db.get_dataset(ds1["id"])["status"] == "built"
    assert db.get_dataset(ds2["id"])["status"] == "built"

    # 合并产物先读（后续单源构建会覆盖主数据集目录的 train.jsonl）
    merged_count = res["report"]["summary"]["sample_count"]
    lines = [l for l in open(res["train_jsonl"], encoding="utf-8") if l.strip()]
    assert len(lines) == merged_count  # 行数 = 合并样本数

    # 合并样本数 = 逐源构建样本数之和
    single1 = run_build_multi([ds1], chunk_size=256, overlap=32)
    single2 = run_build_multi([ds2], chunk_size=256, overlap=32)
    assert merged_count == (
        single1["report"]["summary"]["sample_count"]
        + single2["report"]["summary"]["sample_count"]
    )
    assert merged_count > single1["report"]["summary"]["sample_count"]

    # 报告含逐源分解，语料量为两源之和
    assert [s["name"] for s in res["report"]["sources"]] == ["源甲", "源乙"]
    total = sum(s["corpus_bytes"] for s in res["report"]["sources"])
    assert total > 0


def test_run_build_multi_empty_raises():
    from tunefield.pipeline.build import run_build_multi

    try:
        run_build_multi([])
    except ValueError as e:
        assert "没有可构建" in str(e)
    else:
        raise AssertionError("空列表应抛 ValueError")


def test_create_run_accepts_dataset_ids(monkeypatch, tmp_path):
    """runs 端点：dataset_ids 多源校验；入队被 mock（不真跑构建/训练）。"""
    from tunefield.serve import queue as queue_mod
    from tunefield.serve.app import create_app

    monkeypatch.setattr(queue_mod.JobQueue, "enqueue", lambda self, rid: None)

    _seed = _mk_dataset_with_raw("端点源", "内容" * 100)
    with TestClient(create_app()) as c:
        # 不存在的源 → 400
        r = c.post(
            "/api/runs",
            json={"dataset_ids": [_seed["id"], "nope"], "epochs": 1},
        )
        assert r.status_code == 400

        # 缺源 → 400
        assert c.post("/api/runs", json={"epochs": 1}).status_code == 400

        # 单源兼容：dataset_id 仍可用
        r2 = c.post("/api/runs", json={"dataset_id": _seed["id"], "epochs": 1})
        assert r2.status_code == 200
        assert r2.json()["status"] in ("queued", "running", "pending_gpu")
