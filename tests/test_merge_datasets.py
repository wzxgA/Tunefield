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


def test_cleanup_merged_dataset_recycles_and_reassigns_refs():
    """终态回收：文件与记录消失，job/run 引用回填到源数据集。"""
    from tunefield import config
    from tunefield.pipeline.merge import cleanup_merged_dataset, merge_datasets
    from tunefield.serve import db

    a = _mk_dataset("回收源A", "回收内容。" * 30)
    b = _mk_dataset("回收源B", "回收内容。" * 30)
    merged = merge_datasets([a["id"], b["id"]])["dataset"]
    raw = config.RAW_DIR / merged["content_hash"]
    assert raw.exists()

    # 造引用：run 与 job 都指向合并数据集
    db.insert_pipeline_run(
        id="run-merge-1", dataset_id=merged["id"], domain="x", job_id=None,
        config_json=json.dumps({"merged_from": [a["id"], b["id"]]}),
        status="done", created_at="2026-01-01T00:00:00Z",
    )
    db.insert_job(
        job_id="job-merge-1", dataset_id=merged["id"], domain="x", kind="finetune",
        base_model=None, config_json="{}", status="done",
        created_at="2026-01-01T00:00:00Z",
    )

    assert cleanup_merged_dataset(
        merged["id"], fallback_dataset_id=a["id"], merged_from=[a["id"], b["id"]]
    ) is True

    # 文件与 datasets 行已回收
    assert not raw.exists()
    assert db.get_dataset(merged["id"]) is None
    # 引用回填到第一源
    assert db.get_pipeline_run("run-merge-1")["dataset_id"] == a["id"]
    assert db.get_job("job-merge-1")["dataset_id"] == a["id"]
    # 源数据集安然无恙
    assert db.get_dataset(a["id"]) is not None and db.get_dataset(b["id"]) is not None

    # 幂等：再次回收返回 False（记录已不存在）
    assert cleanup_merged_dataset(merged["id"]) is False


def test_cleanup_skipped_when_keep_merged(monkeypatch):
    """调试开关：KEEP_MERGED=1 时不回收。"""
    from tunefield import config
    from tunefield.pipeline import merge as merge_mod
    from tunefield.serve import db

    a = _mk_dataset("保留源A", "保留。" * 20)
    b = _mk_dataset("保留源B", "保留。" * 20)
    merged = merge_mod.merge_datasets([a["id"], b["id"]])["dataset"]

    monkeypatch.setattr(config, "KEEP_MERGED", True)
    assert merge_mod.cleanup_merged_dataset(merged["id"]) is False
    assert db.get_dataset(merged["id"]) is not None
    assert (config.RAW_DIR / merged["content_hash"]).exists()


def test_cleanup_orphan_merged_on_startup():
    """启动兜底：无引用的合并残留被回收（有引用的也能借 run 的 merged_from 回填）。"""
    from tunefield import config
    from tunefield.pipeline.merge import cleanup_orphan_merged, merge_datasets
    from tunefield.serve import db

    a = _mk_dataset("孤儿源A", "孤儿。" * 20)
    b = _mk_dataset("孤儿源B", "孤儿。" * 20)

    # 情况 1：无引用 → 直接回收行与文件
    orphan1 = merge_datasets([a["id"], b["id"]])["dataset"]
    # 情况 2：造一个不同组合（含 b 与自己以外内容）不现实，改用带引用的同一实体验证回填：
    #   这里直接为 orphan1 之外再造一个实体，并挂 run 引用
    c = _mk_dataset("孤儿源C", "孤儿。" * 20)
    orphan2 = merge_datasets([a["id"], c["id"]])["dataset"]
    db.insert_pipeline_run(
        id="run-orphan-2", dataset_id=orphan2["id"], domain="y", job_id=None,
        config_json=json.dumps({"merged_from": [a["id"], c["id"]]}),
        status="failed", created_at="2026-01-01T00:00:00Z",
    )

    res = cleanup_orphan_merged()
    assert set(res["removed"]) >= {orphan1["id"], orphan2["id"]}
    assert db.get_dataset(orphan1["id"]) is None
    assert db.get_dataset(orphan2["id"]) is None
    assert db.get_pipeline_run("run-orphan-2")["dataset_id"] == a["id"]
    assert not (config.RAW_DIR / orphan1["content_hash"]).exists()
    assert not (config.RAW_DIR / orphan2["content_hash"]).exists()


def test_run_pipeline_recycles_merged_on_done(monkeypatch):
    """run 到达终态（done）→ 自动回收合并实体并把 run 素材回填到源。"""
    from tunefield.engine import flow as flow_mod
    from tunefield.engine.exporter import Exporter
    from tunefield.engine.llmfactory.runner import LlmFactoryEngine
    from tunefield.pipeline import build as build_mod
    from tunefield.pipeline.merge import merge_datasets
    from tunefield.serve import db

    a = _mk_dataset("终态源A", "终态。" * 20)
    b = _mk_dataset("终态源B", "终态。" * 20)
    merged = merge_datasets([a["id"], b["id"]])["dataset"]
    db.insert_pipeline_run(
        id="run-clean-1", dataset_id=merged["id"], domain="z", job_id=None,
        config_json=json.dumps({
            "merged_from": [a["id"], b["id"]], "auto_import": False, "epochs": 1,
        }),
        status="running", created_at="2026-01-01T00:00:00Z",
    )

    monkeypatch.setattr(
        build_mod, "run_build_multi",
        lambda ds, **kw: {
            "dataset": ds[0], "train_jsonl": "data/datasets/x/train.jsonl",
            "report": {"summary": {"sample_count": 3, "corpus_mb": 0.1}},
            "built_ids": [d["id"] for d in ds],
        },
    )
    monkeypatch.setattr(
        LlmFactoryEngine, "run",
        lambda self, job, dataset, cfg, *, loop=None, cmd=None: {
            "job_id": job["id"], "workdir": "/w", "epochs": 1, "degrade": [],
        },
    )
    monkeypatch.setattr(
        Exporter, "run_export",
        lambda self, job, *, quants=("q4_k_m", "q8"), loop=None: {
            "models": [], "fingerprint": {"created_at": "t"},
        },
    )

    flow_mod.run_pipeline("run-clean-1")

    # 合并实体已回收；run 素材回填第一源；源数据集健在
    assert db.get_dataset(merged["id"]) is None
    run = db.get_pipeline_run("run-clean-1")
    assert run["status"] == "done" and run["dataset_id"] == a["id"]
    assert db.get_dataset(a["id"]) is not None and db.get_dataset(b["id"]) is not None


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
