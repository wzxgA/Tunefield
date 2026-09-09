"""T10 · Ollama 集成:slug/Modelfile/导入/清单/状态/OpenAI 兼容转发。"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("t10chat")
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


def test_slugify_ascii_and_fallback():
    from tunefield.assets.registry import slugify

    assert slugify("xnw啊") == "xnw"
    assert slugify("青蛙") == "model"      # 纯中文回退
    assert slugify("demo-1") == "demo-1"
    assert slugify("") == "model"


def test_make_modelfile_content(tmp_path):
    from tunefield.serve.ollama import make_modelfile

    gguf = tmp_path / "model-e2d1-q4_k_m.gguf"
    gguf.write_bytes(b"GGUF")
    mf = make_modelfile(gguf, temperature=0.7, top_p=0.85)
    lines = mf.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"FROM {gguf}"
    assert "PARAMETER temperature 0.7" in lines
    assert "PARAMETER top_p 0.85" in lines


def test_import_model_builds_create_command(monkeypatch, tmp_path):
    from tunefield.serve import ollama

    monkeypatch.setattr(ollama, "installed", lambda: True)
    monkeypatch.setattr(ollama, "running", lambda: True)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(ollama.subprocess, "run", fake_run)
    gguf = tmp_path / "model-e2d1-q4_k_m.gguf"
    gguf.write_bytes(b"GGUF")
    result = ollama.import_model(
        {"path": str(gguf), "ollama_name": "tunefield-model-e2d1"}
    )
    assert result["ollama_name"] == "tunefield-model-e2d1"
    assert captured["cmd"][1:3] == ["create", "tunefield-model-e2d1"]
    assert Path(result["modelfile"]).exists()


def test_import_model_requires_running_ollama(monkeypatch, tmp_path):
    from tunefield.engine.base import EngineError
    from tunefield.serve import ollama

    monkeypatch.setattr(ollama, "installed", lambda: True)
    monkeypatch.setattr(ollama, "running", lambda: False)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    with pytest.raises(EngineError, match="未运行"):
        ollama.import_model({"path": str(gguf), "ollama_name": "tunefield-x"})


def test_list_ollama_models_filters_prefix(monkeypatch):
    from tunefield.serve import ollama

    monkeypatch.setattr(ollama, "running", lambda: True)
    monkeypatch.setattr(
        ollama, "_get_json",
        lambda path, timeout=5.0: {"models": [
            {"name": "tunefield-model-e2d1", "size": 1},
            {"name": "llama3:latest", "size": 2},
        ]},
    )
    names = [m["name"] for m in ollama.list_ollama_models()]
    assert names == ["tunefield-model-e2d1"]


def test_chat_api_status_import_and_forward(monkeypatch):
    from tunefield.assets import registry
    from tunefield.serve import db, ollama
    from tunefield.serve.app import create_app

    with TestClient(create_app()) as c:
        # status:直接 monkeypatch 探测函数
        monkeypatch.setattr(ollama, "installed", lambda: True)
        monkeypatch.setattr(ollama, "version", lambda: "0.5.7")
        monkeypatch.setattr(ollama, "running", lambda: True)
        st = c.get("/api/chat/status").json()
        assert st["installed"] and st["running"]

        # import:登记一份 GGUF 并伪造导入成功(先插 job 行满足外键)
        db.insert_job(
            job_id="job-c1", dataset_id=None, domain="demo", kind="finetune",
            base_model=None, config_json=None, status="done",
            created_at="2026-01-01T00:00:00Z",
        )
        adapter = db.insert_adapter(
            id="job-c1", job_id="job-c1", domain="demo", version="job-c1"[:8],
            path="x", fingerprint_json=None,
        )
        gguf = Path(os.environ["TUNEFIELD_ROOT"]) / "m.gguf"
        gguf.write_bytes(b"GGUF")
        row = registry.register_quantized(adapter_id="job-c1", quant="q4_k_m", path=gguf)
        monkeypatch.setattr(
            ollama, "import_model",
            lambda m, **kw: {"ollama_name": m["ollama_name"], "modelfile": "x"},
        )
        r = c.post(f"/api/models/{row['id']}/import", json={})
        assert r.status_code == 200, r.text
        assert r.json()["ollama_name"].startswith("tunefield-")
        assert c.post("/api/models/nope/import", json={}).status_code == 404

        # 转发:合法透传 / 缺字段 / stream
        monkeypatch.setattr(
            ollama, "chat_completions",
            lambda payload, timeout=300: {"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
        )
        ok = c.post("/v1/chat/completions", json={
            "model": "tunefield-model-e2d1",
            "messages": [{"role": "user", "content": "你好"}],
        })
        assert ok.status_code == 200
        assert ok.json()["choices"][0]["message"]["content"] == "hi"

        assert c.post("/v1/chat/completions", json={"model": "m"}).status_code == 400
        assert c.post("/v1/chat/completions", json={
            "model": "m", "messages": [], "stream": False,
        }).status_code == 400
        assert c.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True,
        }).status_code == 400  # 流式属 P1
