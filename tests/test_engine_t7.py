"""T7 · 引擎抽象层：日志采样解析、注册表路由、子进程监控、断点重试与缺依赖报错。"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("t7engine")
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


def _insert_job(db, job_id: str, kind: str = "finetune") -> None:
    # dataset_id 置空：引擎单测不依赖真实数据集行（FK 允许 NULL）
    db.insert_job(
        job_id=job_id, dataset_id=None, domain="demo", kind=kind,
        base_model=None, config_json=None, status="queued",
        created_at="2026-01-01T00:00:00Z",
    )


def test_parse_train_line_variants():
    from tunefield.engine.base import parse_train_line

    # HF Trainer dict 风格
    p = parse_train_line("{'loss': 0.5231, 'learning_rate': 0.0001, 'epoch': 1.5, 'step': 120}")
    assert p and abs(p["loss"] - 0.5231) < 1e-6
    assert p["epoch"] == 1.5 and p["step"] == 120
    # 简化 'loss: x' / JSON
    assert parse_train_line("loss: 0.33")["loss"] == 0.33
    assert parse_train_line('{"loss": 0.5, "epoch": 0.0}')["loss"] == 0.5
    # 无采样行返回 None（容错）
    assert parse_train_line("loading checkpoint weights...") is None


def test_registry_routes_and_missing_kind():
    from tunefield.engine import registry
    from tunefield.engine.base import EngineNotFound
    from tunefield.engine.llmfactory.runner import LlmFactoryEngine  # 触发注册

    registry.register(LlmFactoryEngine())
    assert registry.get_engine("finetune").kind == "finetune"
    with pytest.raises(EngineNotFound, match="pretrain"):
        registry.get_engine("pretrain")  # T13 接入
    with pytest.raises(EngineNotFound):
        registry.get_engine("unknown-kind")


def test_lmfactory_prepare_requires_built_dataset():
    from tunefield.engine.base import EngineError
    from tunefield.engine.llmfactory.runner import LlmFactoryEngine

    engine = LlmFactoryEngine()
    job = {"id": "j-x", "domain": "demo"}
    dataset = {"id": "no-such-dataset", "name": "空数据集"}
    with pytest.raises(EngineError, match="尚未构建"):
        engine.prepare(job, dataset, engine.default_cfg, None, None)


def test_engine_run_monitors_loss_and_progress():
    from tunefield.engine.llmfactory.runner import LlmFactoryEngine
    from tunefield.serve import db

    db.init_db()
    _insert_job(db, "job-loss")
    engine = LlmFactoryEngine()
    job = {"id": "job-loss", "domain": "demo"}
    dataset = {"id": "ds1", "name": "demo"}
    # 静态两行采样（flush 保证行独立），epochs=2 → 末行 epoch=2/2 → progress≈1
    script = (
        "import sys\n"
        "print(\"{'loss': 0.9, 'epoch': 1.0, 'step': 1}\", flush=True)\n"
        "print(\"{'loss': 0.7, 'epoch': 2.0, 'step': 2}\", flush=True)\n"
    )
    result = engine.run(job, dataset, {"epochs": 2}, cmd=[sys.executable, "-c", script])
    assert result["job_id"] == "job-loss"
    final = db.get_job("job-loss")
    assert final["loss_json"]  # 采样点已落库
    assert float(final["progress"]) > 0.9  # epoch 逐行推进到终点


def test_engine_run_fails_with_readable_error_after_retry():
    from tunefield.engine.base import EngineError, log_tail
    from tunefield.engine.llmfactory.runner import LlmFactoryEngine
    from tunefield.serve import db

    _insert_job(db, "job-fail")
    engine = LlmFactoryEngine()
    job = {"id": "job-fail", "domain": "demo"}
    script = "import sys\nsys.exit(3)\n"
    with pytest.raises(EngineError, match="退出码 3"):
        engine.run(job, {"id": "ds1", "name": "demo"}, {"epochs": 3},
                   cmd=[sys.executable, "-c", script])
    assert any("重试" in line for line in log_tail("job-fail"))  # 自动重试一次
    assert db.get_job("job-fail")["status"] in ("running", "queued", "failed")


def test_engine_missing_cli_friendly_message():
    from tunefield.engine.base import EngineError
    from tunefield.engine.llmfactory.runner import LlmFactoryEngine
    from tunefield.serve import db

    db.init_db()
    _insert_job(db, "job-missing")
    engine = LlmFactoryEngine()
    job = {"id": "job-missing", "domain": "demo"}
    # 数据集目录有 jsonl 才走到 prepare→Popen；直接 monkeypatch prepare 返回不存在命令
    import tunefield.engine.base as base

    orig = engine.prepare
    engine.prepare = lambda *a, **k: ["definitely-not-a-real-cli-xyz", "train", "x.yaml"]
    try:
        with pytest.raises(EngineError, match="无法启动训练程序|definitely-not"):
            engine.run(job, {"id": "ds1", "name": "demo"}, {"epochs": 3})
    finally:
        engine.prepare = orig
