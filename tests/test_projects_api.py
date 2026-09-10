"""编排项目 API：项目 = 素材(数据集) + 画布配置的组合实体。"""
from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    """独立数据根：避免与运行中的 serve 进程争抢 SQLite 写锁。"""
    root = tmp_path_factory.mktemp("projects")
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


def _seed_dataset():
    import uuid

    from tunefield.serve import db

    return db.insert_dataset(
        id=uuid.uuid4().hex,  # 测试库持久：随机 id/hash 防重跑撞 UNIQUE
        name=f"测试语料-{uuid.uuid4().hex[:6]}",
        content_hash=f"hash-{uuid.uuid4().hex}",
        source="upload",
        stats_json=None,
    )


def test_project_create_list_config_delete(monkeypatch, tmp_path):
    from tunefield.serve.app import create_app

    ds = _seed_dataset()
    with TestClient(create_app()) as c:
        before = len(c.get("/api/projects").json())  # 测试库持久共享：用相对计数断言

        # 校验：缺名称；数据集给了但不存在的仍 400
        assert c.post("/api/projects", json={"name": " ", "dataset_id": ds["id"]}).status_code == 400
        assert c.post("/api/projects", json={"name": "p", "dataset_id": "nope"}).status_code == 400

        # 只填名称即可创建（数据集留空，进画布后绑定）
        r = c.post("/api/projects", json={"name": "我的编排"})
        assert r.status_code == 200
        proj = r.json()
        assert proj["name"] == "我的编排" and proj["dataset"] is None

        # 带数据集创建 → 列表联表数据集信息
        r2 = c.post("/api/projects", json={"name": "带素材", "dataset_id": ds["id"]})
        assert r2.status_code == 200
        assert r2.json()["dataset"]["name"].startswith("测试语料-")

        # 配置持久化 → 再查恢复
        cfg = {"nodes": [{"id": "n1", "key": "split", "meta": {"chunk_size": "512"}}], "edges": []}
        assert c.put(f"/api/projects/{proj['id']}/config", json={"config": cfg}).status_code == 200
        listed = c.get("/api/projects").json()
        target = next(x for x in listed if x["id"] == proj["id"])
        assert target["config"]["nodes"][0]["meta"]["chunk_size"] == "512"

        # 换绑数据集 → 持久化到项目实体
        c.put(f"/api/projects/{proj['id']}/dataset", json={"dataset_id": ds["id"]})
        target = next(x for x in c.get("/api/projects").json() if x["id"] == proj["id"])
        assert target["dataset"]["id"] == ds["id"]

        # 删除 → 再删 404 / 列表计数回落；数据集不受影响
        assert c.delete(f"/api/projects/{proj['id']}").json()["deleted"] is True
        assert c.delete(f"/api/projects/{proj['id']}").status_code == 404
        assert len(c.get("/api/projects").json()) == before + 1
        assert db_get_dataset_ok(ds["id"])


def db_get_dataset_ok(dataset_id: str) -> bool:
    from tunefield.serve import db

    return db.get_dataset(dataset_id) is not None
