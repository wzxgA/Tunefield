"""W1：工作站壳状态端点 /api/system/status（GPU 显存 + Ollama 一次返回）。"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_system_status_returns_gpu_and_ollama(monkeypatch):
    from tunefield.serve import ollama
    from tunefield.serve.app import create_app

    monkeypatch.setattr(
        ollama, "status",
        lambda: {"installed": True, "running": True, "version": "test"},
    )
    import tunefield.engine.recommender as recommender

    monkeypatch.setattr(recommender, "detect_vram_gb", lambda: 8.1)

    with TestClient(create_app()) as client:
        r = client.get("/api/system/status")

    assert r.status_code == 200
    body = r.json()
    assert body["gpu"]["vram_gb"] == 8.1
    assert body["ollama"]["running"] is True
