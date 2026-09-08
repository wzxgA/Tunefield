"""F2 · WebSocket 事件流测试。

验证 /api/events 订阅后，创建任务可实时收到
job.created / job.status / job.loss 事件，无需轮询。
"""

from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("f2root")
    os.environ["TUNEFIELD_ROOT"] = str(root)
    from tunefield import config
    importlib.reload(config)
    from tunefield.serve import app as app_mod
    from tunefield.serve import db as db_mod
    importlib.reload(db_mod)
    importlib.reload(app_mod)
    yield
    os.environ.pop("TUNEFIELD_ROOT", None)


def _upload_dataset(client) -> str:
    r = client.post(
        "/api/datasets",
        data={"name": "aws"},
        files=[("files", ("a.txt", "事件流演示内容。\n".encode("utf-8"), "text/plain"))],
    )
    assert r.status_code == 200
    return r.json()["dataset"]["id"]


async def _fake_handler(job_id: str) -> None:
    """假引擎：模拟 loss 采样并广播事件，专测 WS 事件流。"""
    import asyncio

    from tunefield.serve import db, events
    from tunefield.serve.queue import set_status

    set_status(job_id, "running", progress=0.1)
    await asyncio.sleep(0.01)
    set_status(job_id, "running", progress=0.5)
    for step, value in ((1, 0.9), (2, 0.7)):
        db.append_loss(job_id, f"[{step},{value}]")
        events.hub.publish("job.loss", {"id": job_id, "step": step, "value": value})
    await asyncio.sleep(0.01)


def test_events_stream_full_job_lifecycle():
    """订阅 WS → 创建任务 → 依次收到 created/status/loss，直到 done。"""
    from tunefield.serve.app import create_app

    with TestClient(create_app(handler=_fake_handler)) as c:
        with c.websocket_connect("/api/events") as ws:
            dataset_id = _upload_dataset(c)
            r = c.post("/api/jobs", json={"dataset_id": dataset_id, "domain": "aws", "kind": "finetune"})
            assert r.status_code == 200
            job_id = r.json()["id"]

            types_received: list[str] = []
            saw_done = False
            # 收事件直到 job 终态（占位 handler 约 2s 完成）
            for _ in range(30):
                msg = ws.receive_json()
                types_received.append(msg["type"])
                data = msg["data"]
                assert data["id"] == job_id
                if msg["type"] == "job.loss":
                    assert "step" in data and "value" in data
                if msg["type"] == "job.status" and data["status"] == "done":
                    saw_done = True
                    break
            assert saw_done
            assert "job.created" in types_received
            assert "job.loss" in types_received
            # 状态推进顺序：created 之后 running 先于 done
            assert types_received.index("job.created") < types_received.index("job.loss")


def test_publish_without_clients_is_noop():
    """无订阅者时 publish 静默丢弃，不报错（CLI/无前端场景）。"""
    from tunefield.serve import events

    events.hub.publish("job.status", {"id": "x", "status": "running"})  # 不应抛异常
    assert events.hub.client_count == 0


def test_ws_disconnect_cleans_up():
    """连接断开后 hub 清理，不留死连接。"""
    from tunefield.serve import events
    from tunefield.serve.app import create_app

    with TestClient(create_app()) as c:
        with c.websocket_connect("/api/events"):
            assert events.hub.client_count == 1
        assert events.hub.client_count == 0
