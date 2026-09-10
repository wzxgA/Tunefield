"""数据集拼接（画布「合并」节点）：多数据集 → 一个新数据集实体。"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    """独立数据根：避免与运行中的 serve 进程争抢 SQLite 写锁。"""
    root = tmp_path_factory.mktemp("merge")
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


def _mk_dataset(name: str, text: str) -> dict:
    from tunefield import config
    from tunefield.serve import db

    uid = uuid.uuid4().hex
    h = hashlib.sha256(uid.encode()).hexdigest()
    raw = config.RAW_DIR / h
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{name}.txt").write_text(text, encoding="utf-8")
    return db.insert_dataset(
        id=uid, name=name, content_hash=h, source="upload",
        stats_json=json.dumps({"summary": {"sample_count": 0}}),
    )


def test_merge_creates_new_dataset_and_is_idempotent():
    from tunefield import config
    from tunefield.pipeline.merge import merge_datasets
    from tunefield.serve import db

    a = _mk_dataset("合并源A", "A 的内容。" * 50)
    b = _mk_dataset("合并源B", "B 的内容。" * 50)

    res = merge_datasets([a["id"], b["id"]])
    merged = res["dataset"]

    # 新实体：新 id、来源标记、名称自动拼接
    assert merged["id"] not in (a["id"], b["id"])
    assert merged["source"] == "merge"
    assert merged["name"] == "合并源A + 合并源B"
    assert res["sources"] == [a["id"], b["id"]]

    # 文件级拼接：新 raw 目录含两源文件且带来源前缀（A/B 各 1 个）
    files = sorted(
        p.relative_to(config.RAW_DIR / merged["content_hash"]).as_posix()
        for p in (config.RAW_DIR / merged["content_hash"]).rglob("*")
        if p.is_file()
    )
    assert len(files) == 2
    assert files[0].startswith("ds") and files[1].startswith("ds")

    # 幂等：相同来源组合再次拼接 → 复用同一数据集
    res2 = merge_datasets([a["id"], b["id"]])
    assert res2["deduped"] is True
    assert res2["dataset"]["id"] == merged["id"]

    # 源数据集不受影响
    assert db.get_dataset(a["id"]) is not None
    assert db.get_dataset(b["id"]) is not None


def test_merge_requires_two_sources():
    from tunefield.pipeline.merge import merge_datasets

    a = _mk_dataset("单源", "内容。" * 20)
    with pytest.raises(ValueError):
        merge_datasets([a["id"]])
    with pytest.raises(ValueError):
        merge_datasets([a["id"], "不存在"])


def test_merge_api_and_run_with_merge_flag(monkeypatch):
    from tunefield.serve import queue as queue_mod
    from tunefield.serve.app import create_app

    monkeypatch.setattr(queue_mod.JobQueue, "enqueue", lambda self, rid: None)

    a = _mk_dataset("端点A", "甲。" * 40)
    b = _mk_dataset("端点B", "乙。" * 40)

    with TestClient(create_app()) as c:
        # 少于 2 个源 → 400
        assert c.post("/api/datasets/merge", json={"dataset_ids": [a["id"]]}).status_code == 400

        # 拼接成功 → 新数据集入库（内部实体，记录可查）
        r = c.post("/api/datasets/merge", json={"dataset_ids": [a["id"], b["id"]]})
        assert r.status_code == 200
        merged = r.json()["dataset"]
        assert c.get(f"/api/datasets/{merged['id']}").status_code == 200

        # 但流程内合并数据集不出现在数据屏清单（默认过滤，?include_merged=1 可看）
        listed_ids = {d["id"] for d in c.get("/api/datasets").json()}
        assert merged["id"] not in listed_ids
        assert a["id"] in listed_ids and b["id"] in listed_ids
        with_all = {d["id"] for d in c.get("/api/datasets?include_merged=1").json()}
        assert merged["id"] in with_all

        # run 带 merge 标志 → 先拼接，run 作用于新数据集
        r2 = c.post(
            "/api/runs",
            json={"dataset_ids": [a["id"], b["id"]], "merge": True, "epochs": 1,
                  "auto_import": False},
        )
        assert r2.status_code == 200
        run = r2.json()
        assert run["dataset_id"] == merged["id"]
        cfg = json.loads(run["config_json"])
        assert cfg["dataset_ids"] == [merged["id"]]
        assert cfg["merged_from"] == [a["id"], b["id"]]
