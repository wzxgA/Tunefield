"""T13 · 从零预训练引擎 B：纯文本语料/档位推荐/prepare/路由/CLI/完整权重导出。

验证（训练子进程用伪造脚本，不依赖真实 transformers/GPU）：
- build 沉淀纯文本语料 corpus.txt（不走指令化，供引擎 B 消费）；
- 推荐器预训练档位表与 overrides；prepare 产出 train.py 子进程命令与 cfg.json
  （含 corpus/output_dir/tokenizer，resume 时带断点字段）；
- registry 注册 kind=pretrain；create_job/预览按 kind 路由（base_model 为空）；
- CLI train --kind pretrain --dry-run 输出档位；
- exporter：预训练完整权重跳过 LoRA 合并直接转 GGUF。
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("t13pre")
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


def _upload_demo(client, name="pretrain-demo"):
    md = ("# 预训练领域文档\n\n" + "这是引擎 B 从零预训练的纯文本语料，用于验证中文建模。" * 60)
    r = client.post(
        "/api/datasets",
        data={"name": name},
        files=[("files", ("doc.md", md.encode("utf-8"), "text/markdown"))],
    )
    assert r.status_code == 200
    return r.json()["dataset"]


def test_build_produces_plain_text_corpus():
    from tunefield.pipeline.build import run_build
    from tunefield.serve.app import create_app

    with TestClient(create_app()) as c:
        ds = _upload_demo(c)
        result = run_build(ds, chunk_size=300, overlap=64)
        corpus = Path(result["corpus_path"])
        assert corpus.exists()
        text = corpus.read_text(encoding="utf-8")
        assert len(text) > 100
        assert "从零预训练" in text  # 纯文本（非 Alpaca instruction 形态）
        assert not text.lstrip().startswith("{")


def test_recommend_pretrain_tier_and_overrides():
    from tunefield.engine.recommender import recommend_pretrain

    rec = recommend_pretrain(vram_gb=8.0, corpus_mb=None)
    assert rec["scale"] == "200M"
    assert rec["hidden_size"] == 768 and rec["num_hidden_layers"] == 12
    assert rec["_meta"]["tier"] == "~200M"
    # 显存未知 → 最小档；16GB+ → 最大档
    assert recommend_pretrain(None, None)["scale"] == "100M"
    assert recommend_pretrain(24.0, None)["scale"] == "400M"
    # overrides 逐项覆盖
    rec2 = recommend_pretrain(8.0, 30.0, {"epochs": 5, "learning_rate": 1e-3})
    assert rec2["epochs"] == 5 and rec2["learning_rate"] == 1e-3
    assert "epochs" in rec2["_meta"]["overridden"]


def test_pretrain_prepare_writes_cfg_and_resume(tmp_path):
    from tunefield import config
    from tunefield.engine.pretrain.runner import PretrainEngine

    ds_id = "ds-pre"
    corpus_dir = config.DATASETS_DIR / ds_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "corpus.txt").write_text("预训练语料样本。", encoding="utf-8")

    engine = PretrainEngine()
    job = {"id": "j-pre", "domain": "demo"}
    dataset = {"id": ds_id, "name": "demo"}
    cfg = {**engine.default_cfg, "scale": "200M", "hidden_size": 768,
           "num_hidden_layers": 12}
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    argv = engine.prepare(job, dataset, cfg, workdir, resume=None)
    assert Path(argv[0]) == Path(sys.executable)
    assert Path(argv[1]).name == "train.py"
    cfg_path = Path(argv[2])
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["corpus"].endswith("corpus.txt")
    assert payload["output_dir"] == str(workdir)
    assert payload["hidden_size"] == 768 and "tokenizer" in payload
    assert "resume" not in payload

    # 断点续训：resume 字段写入 cfg
    argv2 = engine.prepare(job, dataset, cfg, workdir, resume="/cp/checkpoint-10")
    p2 = json.loads(Path(argv2[2]).read_text(encoding="utf-8"))
    assert p2["resume"] == "/cp/checkpoint-10"


def test_pretrain_prepare_missing_corpus_hint(tmp_path):
    from tunefield.engine.base import EngineError
    from tunefield.engine.pretrain.runner import PretrainEngine

    engine = PretrainEngine()
    with pytest.raises(EngineError, match="缺少纯文本语料"):
        engine.prepare({"id": "j", "domain": "x"}, {"id": "no-ds", "name": "x"},
                       engine.default_cfg, tmp_path, None)


def test_pretrain_run_via_fake_cmd_samples_loss(tmp_path):
    from tunefield.engine.pretrain.runner import PretrainEngine
    from tunefield.serve import db

    job_id = "job-pre-run"
    db.insert_job(job_id=job_id, dataset_id=None, domain="demo", kind="pretrain",
                  base_model=None, config_json=None, status="queued",
                  created_at="2026-01-01T00:00:00Z")
    script = tmp_path / "fake.py"
    script.write_text(
        "import pathlib, sys\n"
        "f = pathlib.Path('n')\n"
        "n = int(f.read_text()) if f.exists() else 0\n"
        "f.write_text(str(n + 1))\n"
        "print(\"{'loss': 3.2, 'epoch': 1.0, 'step': 5}\", flush=True)\n"
        "if n == 0:\n"
        "    print(\"subprocess crash\", flush=True); raise SystemExit(1)\n",
        encoding="utf-8",
    )
    engine = PretrainEngine()
    result = engine.run(db.get_job(job_id), {"name": "demo"},
                        {"epochs": 1, "scale": "100M"},
                        cmd=[sys.executable, str(script)])
    assert result["job_id"] == job_id
    # 首次非 OOM 失败自动重试（断点续训语义）→ 第二次成功
    assert result["retried"] is True
    final = db.get_job(job_id)
    assert final["loss_json"]  # loss 采样已落库


def test_create_job_pretrain_routes_and_api(tmp_path):
    from tunefield.serve import db
    from tunefield.serve.app import create_app

    async def _noop(_job_id):
        return None

    with TestClient(create_app(handler=_noop)) as c:
        ds = _upload_demo(c, name="pre-api")
        r = c.post("/api/jobs", json={"dataset_id": ds["id"], "kind": "pretrain",
                                      "domain": "pre-api",
                                      "overrides": {"scale": "400M"}})
        assert r.status_code == 200, r.text
        job = r.json()
        assert job["kind"] == "pretrain"
        assert job["base_model"] is None  # 无基座
        cfg = json.loads(job["config_json"])
        assert cfg["scale"] == "400M" and "hidden_size" in cfg
        # 预览同样按 kind 返回预训练档位
        pv = c.get(f"/api/train/preview?dataset_id={ds['id']}&kind=pretrain").json()
        assert pv["recommend"]["scale"]
        # 未知 kind 拒绝
        bad = c.post("/api/jobs", json={"dataset_id": ds["id"], "kind": "llama"})
        assert bad.status_code == 400


def test_cli_train_dry_run_pretrain(capsys):
    from tunefield.cli import main
    from tunefield.serve import db

    db.insert_dataset(id="ds-cli-pre", name="pre-cli", content_hash="h-cli",
                      source=None, stats_json=None)
    rc = main(["train", "ds-cli-pre", "--kind", "pretrain", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "预训练档位" in out and "规模" in out


# ---------------------------------------------------------------------------
# exporter：预训练完整权重直转 GGUF（跳过 LoRA 合并）
# ---------------------------------------------------------------------------


def test_export_pretrain_full_weights_skips_merge(monkeypatch, tmp_path):
    from tunefield import config
    from tunefield.engine import exporter as exp
    from tunefield.engine.exporter import Exporter
    from tunefield.serve import db

    calls: list[str] = []

    def _fake_run(self, cmd, cwd=None):
        cmd = [str(c) for c in cmd]
        calls.append(" ".join(cmd))
        if "llamafactory" in cmd[0]:
            raise AssertionError("预训练导出不应调用 llamafactory-cli 合并")
        if any("convert_hf_to_gguf" in c for c in cmd):
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF-f16")
            return 0, "converted"
        if "llama-quantize" in Path(cmd[0]).name:
            Path(cmd[2]).write_bytes(b"GGUF-quant")
            return 0, "quantized"
        return 1, "unknown"

    monkeypatch.setattr(Exporter, "_run", _fake_run)
    monkeypatch.setattr(exp, "_quantizer", lambda: "llama-quantize")
    dummy = tmp_path / "convert_hf_to_gguf.py"
    dummy.write_text("# fake", encoding="utf-8")
    monkeypatch.setenv("TUNEFIELD_CONVERT_HF_TO_GGUF", str(dummy))

    db.insert_job(job_id="job-pre-exp", dataset_id=None, domain="pre",
                  kind="pretrain", base_model=None,
                  config_json=json.dumps({"scale": "100M", "epochs": 1}),
                  status="done", created_at="2026-01-01T00:00:00Z")
    # 预训练完整权重目录（train.py save_model 产物：config.json + 权重）
    workdir = config.ADAPTERS_DIR / "pre" / "job-pre-exp"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "config.json").write_text("{}", encoding="utf-8")
    (workdir / "model.safetensors").write_bytes(b"weights")

    result = Exporter().run_export(db.get_job("job-pre-exp"))
    quants = [m["quant"] for m in result["models"]]
    assert quants == ["f16", "q4_k_m", "q8"]
    assert not any("llamafactory" in c for c in calls)
    assert result["fingerprint"]["config"]["scale"] == "100M"
