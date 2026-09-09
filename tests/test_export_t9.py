"""T9 · 量化导出：合并→GGUF→量化→指纹入库（伪造工具链，不依赖真实 llama.cpp）。"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("t9exp")
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


def _fake_run(self, cmd, cwd=None):
    """按命令特征伪造：合并/转换/量化都只「造出预期产物文件」。"""
    cmd = [str(c) for c in cmd]
    if "export" in cmd:
        out_dir = Path(cmd[cmd.index("--export_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text("{}", encoding="utf-8")
        (out_dir / "model-00001-of-00002.safetensors").write_bytes(b"fake")
        return 0, "merged ok"
    if any("convert_hf_to_gguf" in c for c in cmd):
        outfile = Path(cmd[cmd.index("--outfile") + 1])
        outfile.write_bytes(b"GGUF-fake")
        return 0, "converted"
    if "llama-quantize" in Path(cmd[0]).name:
        Path(cmd[2]).write_bytes(b"GGUF-quant")
        return 0, "quantized"
    return 1, "unknown command"


def _make_trained_job(db, job_id="job-exp"):
    db.insert_job(
        job_id=job_id, dataset_id=None, domain="demo", kind="finetune",
        base_model="Qwen/Qwen2.5-0.5B-Instruct", config_json='{"epochs": 2}',
        status="done", created_at="2026-01-01T00:00:00Z",
    )
    from tunefield import config

    workdir = config.ADAPTERS_DIR / "demo" / job_id
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "adapter_model.safetensors").write_bytes(b"adapter")
    return workdir


def test_export_full_pipeline(monkeypatch, tmp_path):
    from tunefield import config
    from tunefield.engine import exporter as exp
    from tunefield.engine.exporter import Exporter
    from tunefield.serve import db

    monkeypatch.setattr(Exporter, "_run", _fake_run)
    monkeypatch.setattr(exp, "_quantizer", lambda: "llama-quantize")
    dummy = tmp_path / "convert_hf_to_gguf.py"
    dummy.write_text("# fake", encoding="utf-8")
    monkeypatch.setenv("TUNEFIELD_CONVERT_HF_TO_GGUF", str(dummy))

    _make_trained_job(db)
    job = db.get_job("job-exp")
    result = Exporter().run_export(job)

    quants = [m["quant"] for m in result["models"]]
    assert quants == ["f16", "q4_k_m", "q8"]
    for m in result["models"]:
        assert Path(m["path"]).exists()
        assert m["size_bytes"] > 0

    # 指纹：数据/超参/基座可反查
    fp = result["fingerprint"]
    assert fp["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert fp["config"]["epochs"] == 2
    assert fp["quants"] == quants
    fp_file = next((config.GGUF_DIR).glob("demo-job-exp-fingerprint.json"))
    assert json.loads(fp_file.read_text(encoding="utf-8"))["job_id"] == "job-exp"

    # 登记可检索
    models = exp.registry.list_gguf_models()
    assert {m["quant"] for m in models} >= {"f16", "q4_k_m", "q8"}


def test_export_requires_trained_adapter():
    from tunefield.engine.base import EngineError
    from tunefield.engine.exporter import Exporter
    from tunefield.serve import db

    db.insert_job(
        job_id="job-noexp", dataset_id=None, domain="demo", kind="finetune",
        base_model=None, config_json=None, status="done",
        created_at="2026-01-01T00:00:00Z",
    )
    job = db.get_job("job-noexp")
    with pytest.raises(EngineError, match="没有可导出的适配器权重"):
        Exporter().run_export(job)


def test_export_missing_converter_hint(monkeypatch):
    from tunefield.engine.base import EngineError
    from tunefield.engine.exporter import Exporter
    from tunefield.serve import db

    monkeypatch.setattr(Exporter, "_run", _fake_run)
    monkeypatch.setenv("TUNEFIELD_CONVERT_HF_TO_GGUF", "Z:/no/such/file.py")
    _make_trained_job(db, "job-exp2")
    with pytest.raises(EngineError, match="convert_hf_to_gguf|不存在"):
        Exporter().run_export(db.get_job("job-exp2"))


def test_export_api_models_and_download(monkeypatch, tmp_path):
    from tunefield.engine import exporter as exp
    from tunefield.engine.exporter import Exporter
    from tunefield.serve import db
    from tunefield.serve.app import create_app

    monkeypatch.setattr(Exporter, "_run", _fake_run)
    monkeypatch.setattr(exp, "_quantizer", lambda: "llama-quantize")
    dummy = tmp_path / "convert_hf_to_gguf.py"
    dummy.write_text("# fake", encoding="utf-8")
    monkeypatch.setenv("TUNEFIELD_CONVERT_HF_TO_GGUF", str(dummy))

    with TestClient(create_app()) as c:
        _make_trained_job(db, "job-api")
        r = c.post("/api/jobs/job-api/export", json={})
        assert r.status_code == 200, r.text
        models = r.json()["models"]
        assert len(models) == 3

        listed = c.get("/api/models").json()
        assert any(m["job_id"] == "job-api" for m in listed)

        dl = c.get(f"/api/models/{models[0]['id']}/download")
        assert dl.status_code == 200
