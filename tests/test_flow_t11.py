"""T11 · 端到端编排：/api/runs 一键 run + 阶段时间线 + CLI export/run 真实化。

验证（全部用伪造引擎/导出/ollama，不依赖真实 GPU 工具链）：
- POST /api/runs：build → train → export → import 全阶段 ok，timeline 落库，
  训练子任务 job 完成、run.job_id 关联正确；
- 任一段失败：run failed、该段标 failed、原因入 error；
- CLI export 真实输出产物；CLI run 一键串联到「端到端完成」。
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("t11flow")
    os.environ["TUNEFIELD_ROOT"] = str(root)
    from tunefield import config

    importlib.reload(config)
    from tunefield.serve import db as db_mod

    importlib.reload(db_mod)
    db_mod.init_db()
    yield
    os.environ.pop("TUNEFIELD_ROOT", None)


def _wait_run(client, run_id, timeout=5.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/runs/{run_id}")
        assert r.status_code == 200, r.text
        run = r.json()
        if run["status"] in ("done", "failed"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} 未在 {timeout}s 内收敛：{r.text}")


def _fake_build(dataset, **kwargs):
    return {
        "dataset": dataset,
        "train_jsonl": "data/datasets/x/train.jsonl",
        "report": {"summary": {"sample_count": 7, "corpus_mb": 1.2}},
    }


def test_db_timeline_append_and_update():
    from tunefield.serve import db

    db.insert_pipeline_run(
        id="run-tl", dataset_id=None, domain="demo", job_id=None,
        config_json="{}", status="queued", created_at="2026-01-01T00:00:00Z",
    )
    db.append_pipeline_phase("run-tl", {"key": "build", "label": "构建数据",
                                        "status": "running"})
    db.update_pipeline_phase("run-tl", "build", status="ok", detail="7 条样本",
                             duration_s=1.5)
    run = db.get_pipeline_run("run-tl")
    tl = json.loads(run["timeline_json"])
    assert tl[0]["status"] == "ok" and tl[0]["duration_s"] == 1.5


def test_run_api_end_to_end_all_phases_ok(monkeypatch, tmp_path):
    import secrets

    from tunefield.engine.exporter import Exporter
    from tunefield.engine.llmfactory.runner import LlmFactoryEngine
    from tunefield.pipeline import build as build_mod
    from tunefield.serve import db, ollama
    from tunefield.serve.app import create_app

    ds = db.insert_dataset(id="ds-run", name="demo", content_hash="hh-run",
                           source=None, stats_json=None)

    # 固定 job id，让导入阶段命中一份伪造的 GGUF 产物
    job_id = "feedface00000000"
    monkeypatch.setattr(secrets, "token_hex", lambda _n: job_id)

    def _fake_owned():
        return [{"job_id": job_id, "quant": "q4_k_m",
                 "ollama_name": "tunefield-demo-feedface",
                 "path": str(tmp_path / "m.gguf")}]

    from tunefield.assets import registry as asset_registry

    monkeypatch.setattr(asset_registry, "list_gguf_models", _fake_owned)
    monkeypatch.setattr(build_mod, "run_build", _fake_build)

    def _fake_run(self, job, dataset, cfg, *, loop=None, cmd=None):
        return {"job_id": job["id"], "workdir": "/w",
                "epochs": int(cfg.get("epochs", 1)), "degrade": []}

    monkeypatch.setattr(LlmFactoryEngine, "run", _fake_run)

    def _fake_export(self, job, *, quants=("q4_k_m", "q8"), loop=None):
        return {"models": [{"quant": q} for q in quants], "fingerprint": {}}

    monkeypatch.setattr(Exporter, "run_export", _fake_export)
    monkeypatch.setattr(ollama, "import_model",
                        lambda m, **kw: {"ollama_name": m["ollama_name"]})

    with TestClient(create_app()) as client:
        r = client.post("/api/runs", json={"dataset_id": ds["id"], "epochs": 2})
        assert r.status_code == 200, r.text
        run_id = r.json()["id"]

        run = _wait_run(client, run_id)
        assert run["status"] == "done", run
        tl = {p["key"]: p for p in run["timeline"]}
        assert list(tl) == ["build", "train", "export", "import"]
        assert all(tl[k]["status"] == "ok" for k in ("build", "train", "export"))
        assert tl["import"]["status"] == "ok"
        assert "已导入" in (tl["import"]["detail"] or "")
        # 关联训练子任务已完成
        assert run["job_id"] == job_id
        assert db.get_job(job_id)["status"] == "done"


def test_run_api_build_failure_marks_failed(monkeypatch):
    from tunefield.pipeline import build as build_mod
    from tunefield.serve import db
    from tunefield.serve.app import create_app

    ds = db.insert_dataset(id="ds-bad", name="bad", content_hash="hh-bad",
                           source=None, stats_json=None)

    def _raise(dataset, **kwargs):
        raise ValueError("没有可训练的文本内容")

    monkeypatch.setattr(build_mod, "run_build", _raise)
    with TestClient(create_app()) as client:
        r = client.post("/api/runs", json={"dataset_id": ds["id"]})
        run_id = r.json()["id"]
        run = _wait_run(client, run_id)
        assert run["status"] == "failed"
        assert "没有可训练" in (run.get("error") or "")
        # 失败阶段入时间线并记录原因；后续阶段未执行
        assert [p["key"] for p in run["timeline"]] == ["build"]
        assert run["timeline"][0]["status"] == "failed"


# ---------------------------------------------------------------------------
# CLI：export / run 真实化
# ---------------------------------------------------------------------------


def _fake_tool_run(self, cmd, cwd=None):
    cmd = [str(c) for c in cmd]
    if "export" in cmd:
        out_dir = Path(cmd[cmd.index("--export_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text("{}", encoding="utf-8")
        (out_dir / "model-00001-of-00002.safetensors").write_bytes(b"fake")
        return 0, "merged ok"
    if any("convert_hf_to_gguf" in c for c in cmd):
        Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF-fake")
        return 0, "converted"
    if "llama-quantize" in Path(cmd[0]).name:
        Path(cmd[2]).write_bytes(b"GGUF-quant")
        return 0, "quantized"
    return 1, "unknown command"


def test_cli_export_produces_models(monkeypatch, tmp_path, capsys):
    from tunefield import config
    from tunefield.cli import main
    from tunefield.engine import exporter as exp
    from tunefield.engine.exporter import Exporter
    from tunefield.serve import db

    monkeypatch.setattr(Exporter, "_run", _fake_tool_run)
    monkeypatch.setattr(exp, "_quantizer", lambda: "llama-quantize")
    dummy = tmp_path / "convert_hf_to_gguf.py"
    dummy.write_text("# fake", encoding="utf-8")
    monkeypatch.setenv("TUNEFIELD_CONVERT_HF_TO_GGUF", str(dummy))

    db.insert_job(job_id="job-exp-cli", dataset_id=None, domain="demo",
                  kind="finetune", base_model="Qwen/Qwen2.5-0.5B-Instruct",
                  config_json='{"epochs": 1}', status="done",
                  created_at="2026-01-01T00:00:00Z")
    workdir = config.ADAPTERS_DIR / "demo" / "job-exp-cli"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "adapter_model.safetensors").write_bytes(b"adapter")

    rc = main(["export", "job-exp-cli"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "产物" in out and "q4_k_m" in out
    assert list(config.GGUF_DIR.glob("demo-*-q4_k_m.gguf"))


def test_cli_run_end_to_end(monkeypatch, capsys):
    from tunefield.engine.exporter import Exporter
    from tunefield.pipeline import build as build_mod
    from tunefield.pipeline import ingest as ingest_mod
    from tunefield.serve import db

    ds = db.insert_dataset(id="ds-cli", name="demo", content_hash="hh-cli",
                           source=None, stats_json=None)
    monkeypatch.setattr(ingest_mod, "ingest_path",
                        lambda path, name: db.get_dataset("ds-cli"))
    monkeypatch.setattr(build_mod, "run_build",
                        lambda dataset, **kw: _fake_build(dataset, **kw))

    def _fake_engine_run(_self, job, dataset, cfg, *, loop=None, cmd=None):
        return {"job_id": job["id"], "workdir": "/w",
                "epochs": int(cfg.get("epochs", 1)), "degrade": []}

    from tunefield.engine import registry
    from tunefield.engine.base import BaseEngine

    class _FakeEngine(BaseEngine):
        kind = "finetune"
        run = _fake_engine_run

    monkeypatch.setattr(registry, "get_engine", lambda kind: _FakeEngine())
    monkeypatch.setattr(
        Exporter, "run_export",
        lambda self, job, **kw: {"models": [{"quant": "q4_k_m"}],
                                 "fingerprint": {}},
    )
    # 导入阶段：无 GGUF 产物 → 打印跳过（不触发 Ollama）

    from tunefield.cli import main

    before = len(db.list_jobs())
    rc = main(["run", "Z:/cli/run/not-used", "--name", "demo"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "端到端完成" in out
    # CLI 真实创建并跑完了训练 job（推荐器+引擎完成后置 done）
    jobs = db.list_jobs()
    assert len(jobs) == before + 1
    assert jobs[-1]["status"] == "done"
