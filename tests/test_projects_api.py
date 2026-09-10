"""编排项目 API：项目 = 素材(数据集) + 画布配置的组合实体。"""
from __future__ import annotations

from fastapi.testclient import TestClient


def _seed_dataset():
    from tunefield.serve import db

    return db.insert_dataset(
        id="ds-proj-test01",
        name="测试语料",
        content_hash="hash-proj-0001",
        source="upload",
        stats_json=None,
    )


def test_project_create_list_config_delete(monkeypatch, tmp_path):
    from tunefield.serve.app import create_app

    ds = _seed_dataset()
    with TestClient(create_app()) as c:
        # 校验：缺名称 / 数据集不存在
        assert c.post("/api/projects", json={"name": " ", "dataset_id": ds["id"]}).status_code == 400
        assert c.post("/api/projects", json={"name": "p", "dataset_id": "nope"}).status_code == 400

        # 创建成功 → 列表联表数据集信息
        r = c.post("/api/projects", json={"name": "我的编排", "dataset_id": ds["id"]})
        assert r.status_code == 200
        proj = r.json()
        assert proj["name"] == "我的编排"
        assert proj["dataset"]["name"] == "测试语料"

        # 配置持久化 → 再查恢复
        cfg = {"nodes": [{"id": "n1", "key": "split", "meta": {"chunk_size": "512"}}], "edges": []}
        assert c.put(f"/api/projects/{proj['id']}/config", json={"config": cfg}).status_code == 200
        listed = c.get("/api/projects").json()
        assert len(listed) == 1 and listed[0]["config"]["nodes"][0]["meta"]["chunk_size"] == "512"

        # 删除 → 404 再删 / 列表为空；数据集不受影响
        assert c.delete(f"/api/projects/{proj['id']}").json()["deleted"] is True
        assert c.delete(f"/api/projects/{proj['id']}").status_code == 404
        assert c.get("/api/projects").json() == []
        assert db_get_dataset_ok(ds["id"])


def db_get_dataset_ok(dataset_id: str) -> bool:
    from tunefield.serve import db

    return db.get_dataset(dataset_id) is not None
