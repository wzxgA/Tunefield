"""T2 · 上传后逐文件解析：POST /api/datasets 接入 → GET .../files 解析状态与预览。

用 TUNEFIELD_ROOT 隔离到临时目录，避免污染真实 data/。
"""

from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("t2files")
    os.environ["TUNEFIELD_ROOT"] = str(root)
    from tunefield import config

    importlib.reload(config)
    from tunefield.serve import app as app_mod
    from tunefield.serve import db as db_mod

    importlib.reload(db_mod)
    importlib.reload(app_mod)
    yield
    os.environ.pop("TUNEFIELD_ROOT", None)


def test_dataset_files_after_upload():
    from tunefield.serve.app import create_app

    py = b"def hello():\n    return 1\n"
    with TestClient(create_app()) as c:
        r = c.post(
            "/api/datasets",
            data={"name": "docs"},
            files=[
                ("files", ("a.py", py, "text/x-python")),
                ("files", ("b.txt", "正文内容\n".encode("utf-8"), "text/plain")),
                ("files", ("c.bin", b"\x00\x01", "application/octet-stream")),
            ],
        )
        assert r.status_code == 200
        dataset = r.json()["dataset"]

        fr = c.get(f"/api/datasets/{dataset['id']}/files")
        assert fr.status_code == 200
        files = {f["name"]: f for f in fr.json()["files"]}

        # 逐文件状态：代码带语言与符号信息，纯文本抽取成功，二进制标记不支持
        assert files["a.py"]["kind"] == "code"
        assert files["a.py"]["language"] == "python"
        assert files["a.py"]["status"] == "ok"
        assert "def hello" in files["a.py"]["preview"]

        assert files["b.txt"]["status"] == "ok"
        assert "正文内容" in files["b.txt"]["preview"]

        # T3 清洗摘要：字段齐全，字数差异自洽
        clean = files["b.txt"]["clean"]
        assert clean is not None
        assert clean["removed_chars"] == clean["before_chars"] - clean["after_chars"]
        assert set(clean["rules"]) <= {
            "invisible", "space", "fullwidth", "html", "page", "boiler",
        }

        # T4 切片摘要：块数/分布/统计自洽
        chunks = files["b.txt"]["chunks"]
        assert chunks is not None
        assert chunks["count"] >= 1
        assert sum(b["count"] for b in chunks["buckets"]) == chunks["count"]
        s = chunks["stats"]
        assert 0 < s["min"] <= s["max"] and s["p50"] <= s["p90"]
        assert chunks["params"]["size"] == 768 and chunks["params"]["overlap"] == 96
        assert len(chunks["samples"]) >= 1

        # T5 指令样本预览：两种模板各一，Alpaca 字段齐全
        samples = files["b.txt"]["samples"]
        assert {s["template"] for s in samples} == {"continuation", "qa"}
        for s in samples:
            assert s["instruction"] and s["output"]
            assert s["input"] is None or isinstance(s["input"], str)

        assert files["c.bin"]["status"] == "unsupported"
        assert files["c.bin"]["preview"] == ""
        assert files["c.bin"]["clean"] is None
        assert files["c.bin"]["chunks"] is None
        assert files["c.bin"]["samples"] == []

        # 数据集归属正确
        assert fr.json()["dataset"]["id"] == dataset["id"]
