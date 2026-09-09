"""M1 · 数据/训练删除：记录与磁盘产物级联清理、运行中保护、API 端点。"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("m1del")
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


def _upload(client, name="demo"):
    r = client.post(
        "/api/datasets",
        data={"name": name},
        files=[("files", ("a.txt", ("平台语料。" * 200).encode("utf-8"), "text/plain"))],
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]


def test_purge_job_removes_rows_dirs_and_linked_run(monkeypatch):
    from tunefield import config
    from tunefield.assets import registry
    from tunefield.serve import db
    from tunefield.serve import ollama
    from tunefield.serve.maintenance import purge_job

    # Ollama 不可用 → 移除导入被跳过，不影响删除
    monkeypatch.setattr(ollama, "installed", lambda: False)

    db.insert_dataset(id="ds-m1j", name="demo", content_hash="hh-m1j",
                      source=None, stats_json=None)
    db.insert_pipeline_run(id="run-m1j", dataset_id="ds-m1j", domain="demo",
                           job_id="job-m1j", config_json=None,
                           status="done", created_at="2026-01-01T00:00:00Z")
    db.insert_job(job_id="job-m1j", dataset_id="ds-m1j", domain="demo",
                  kind="finetune", base_model="Qwen", config_json='{"epochs":1}',
                  status="done", created_at="2026-01-01T00:00:00Z")

    workdir = config.ADAPTERS_DIR / "demo" / "job-m1j"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "adapter_model.safetensors").write_bytes(b"adapter")
    gguf_file = config.GGUF_DIR / "demo-job-m1--q4_k_m.gguf"
    gguf_file.parent.mkdir(parents=True, exist_ok=True)
    gguf_file.write_bytes(b"gguf")

    registry.register_adapter(job_id="job-m1j", domain="demo", path=workdir,
                              fingerprint={"job_id": "job-m1j"})
    adapter = [a for a in db.list_adapters() if a["job_id"] == "job-m1j"][0]
    registry.register_quantized(adapter_id=adapter["id"], quant="q4_k_m",
                                path=gguf_file)

    result = purge_job("job-m1j")
    assert result["deleted"] and result["removed_files"] >= 1
    assert result["removed_dirs"] >= 1
    assert db.get_job("job-m1j") is None
    assert db.get_pipeline_run("run-m1j") is None
    assert not db.list_adapters() and not db.list_quantized_models()
    assert not workdir.exists() and not gguf_file.exists()


def test_delete_dataset_api_purges_raw_and_build():
    from tunefield import config
    from tunefield.serve import db
    from tunefield.serve.app import create_app

    with TestClient(create_app()) as c:
        ds = _upload(c)
        raw_dir = config.RAW_DIR / ds["content_hash"]
        assert raw_dir.exists()
        bd = c.post(f"/api/datasets/{ds['id']}/build", json={})
        assert bd.status_code == 200, bd.text
        ds_dir = config.DATASETS_DIR / ds["id"]
        assert ds_dir.exists() and (ds_dir / "train.jsonl").exists()

        r = c.delete(f"/api/datasets/{ds['id']}")
        assert r.status_code == 200, r.text
        assert r.json()["deleted"] is True
        assert db.get_dataset(ds["id"]) is None
        assert not raw_dir.exists() and not ds_dir.exists()
        assert db.get_dataset_by_hash(ds["content_hash"]) is None


def test_delete_dataset_blocked_while_job_running():
    from tunefield.serve import db
    from tunefield.serve.app import create_app

    with TestClient(create_app()) as c:
        ds = _upload(c, name="busy")
        db.insert_job(job_id="job-busy", dataset_id=ds["id"], domain="busy",
                      kind="finetune", base_model=None, config_json=None,
                      status="running", created_at="2026-01-01T00:00:00Z")
        r = c.delete(f"/api/datasets/{ds['id']}")
        assert r.status_code == 409
        assert db.get_dataset(ds["id"]) is not None  # 未误删


def test_delete_job_api_running_and_missing():
    from tunefield.serve import db
    from tunefield.serve.app import create_app

    with TestClient(create_app()) as c:
        db.insert_job(job_id="job-run", dataset_id=None, domain="x",
                      kind="finetune", base_model=None, config_json=None,
                      status="running", created_at="2026-01-01T00:00:00Z")
        assert c.delete("/api/jobs/job-run").status_code == 409
        assert c.delete("/api/jobs/no-such").status_code == 404
        # 已完成任务正常删除
        db.insert_job(job_id="job-done", dataset_id=None, domain="x",
                      kind="finetune", base_model=None, config_json=None,
                      status="failed", created_at="2026-01-01T00:00:00Z")
        ok = c.delete("/api/jobs/job-done")
        assert ok.status_code == 200
        assert db.get_job("job-done") is None
        # 数据集不存在
        assert c.delete("/api/datasets/no-such").status_code == 404
