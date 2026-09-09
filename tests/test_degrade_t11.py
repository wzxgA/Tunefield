"""T11 · 自动降级链：next_degrade 纯函数 + BaseEngine.run CUDA OOM 逐级降级。

验证：
- 纯函数顺序单调：缩序列(cutoff 减半)→ 降位宽(4→3→2)→ 降基座(档位表前移)→ 链底 None；
- BaseEngine.run 检测到日志含 CUDA out of memory 时逐级降档重试，成功返回 degrade 记录；
- 链底仍 OOM → 明确报错（不会无限循环）。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("t11deg")
    os.environ["TUNEFIELD_ROOT"] = str(root)
    from tunefield import config

    importlib.reload(config)
    from tunefield.serve import db as db_mod

    importlib.reload(db_mod)
    db_mod.init_db()
    yield
    os.environ.pop("TUNEFIELD_ROOT", None)


def _insert_job(db, job_id):
    db.insert_job(
        job_id=job_id, dataset_id=None, domain="demo", kind="finetune",
        base_model=None, config_json="{}", status="queued",
        created_at="2026-01-01T00:00:00Z",
    )
    return db.get_job(job_id)


def _oom_script(tmp_path: Path, fails: int) -> Path:
    """脚本：前 fails 次打印 CUDA OOM 并退出码 1，之后成功输出 loss 采样。"""
    py = tmp_path / "oom.py"
    py.write_text(
        "import pathlib\n"
        "f = pathlib.Path('oom_count')\n"
        "try:\n"
        "    n = int(f.read_text())\n"
        "except Exception:\n"
        "    n = 0\n"
        "f.write_text(str(n + 1))\n"
        f"if n < {fails}:\n"
        "    print('CUDA out of memory. (tried step)', flush=True)\n"
        "    raise SystemExit(1)\n"
        "print(\"{'loss': 0.5, 'epoch': 1.0, 'step': 2}\", flush=True)\n",
        encoding="utf-8",
    )
    return py


def test_degrade_pure_sequence_until_exhausted():
    from tunefield.engine.degrade import next_degrade

    cfg = {"cutoff_len": 1024, "quantization_bit": 4,
           "base_model": "Qwen/Qwen2.5-7B-Instruct"}
    d1 = next_degrade(cfg)
    assert d1 and d1[0]["cutoff_len"] == 512 and "缩短序列" in d1[1]
    d2 = next_degrade(d1[0])
    assert d2 and d2[0]["cutoff_len"] == 256
    d3 = next_degrade(d2[0])
    assert d3 and d3[0]["quantization_bit"] == 3
    d4 = next_degrade(d3[0])
    assert d4 and d4[0]["quantization_bit"] == 2
    d5 = next_degrade(d4[0])
    assert d5 and d5[0]["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "降低基座规模" in d5[1]
    # 第一档（1.5B）即链底：无更小可降
    assert next_degrade(d5[0]) is None


def test_degrade_engine_missing_unsupported():
    from tunefield.engine.base import BaseEngine

    eng = BaseEngine()
    assert eng.degrade_cfg({"cutoff_len": 1024}) is None  # 基类默认不支持


def test_engine_run_oom_degrades_until_success(tmp_path):
    from tunefield.engine.llmfactory.runner import LlmFactoryEngine
    from tunefield.serve import db

    engine = LlmFactoryEngine()
    job = _insert_job(db, "job-oom-ok")
    cfg = {"cutoff_len": 1024, "quantization_bit": 4,
           "base_model": "Qwen/Qwen2.5-7B-Instruct", "epochs": 1}
    result = engine.run(job, {"name": "demo"}, cfg,
                        cmd=[sys.executable, str(_oom_script(tmp_path, fails=2))])
    assert result["job_id"] == "job-oom-ok"
    # 两次 OOM → 两次降级（cutoff 1024→512→256），第三次成功
    assert result["degrade"] == ["缩短序列 cutoff_len 1024→512",
                                 "缩短序列 cutoff_len 512→256"]
    # 降级后成功：loss 采样已落库（成功那行含 step=2）
    assert db.get_job("job-oom-ok")["loss_json"]


def test_engine_run_oom_chain_exhausted_raises(tmp_path):
    from tunefield.engine.base import EngineError
    from tunefield.engine.llmfactory.runner import LlmFactoryEngine
    from tunefield.serve import db

    engine = LlmFactoryEngine()
    job = _insert_job(db, "job-oom-dead")
    # 直接给链底附近的配置：cutoff 已到下限、位宽 2、基座第一档
    cfg = {"cutoff_len": 256, "quantization_bit": 2,
           "base_model": "Qwen/Qwen2.5-1.5B-Instruct", "epochs": 1}
    script = _oom_script(tmp_path, fails=10**6)  # 永远 OOM
    with pytest.raises(EngineError, match="已无可用降级档位"):
        engine.run(job, {"name": "demo"}, cfg, cmd=[sys.executable, str(script)])


def test_degrade_chain_order_and_termination():
    """连续降级顺序固定（缩序列→降位宽→降基座）且必然终止（无死循环）。"""
    from tunefield.engine.degrade import next_degrade

    cfg = {"cutoff_len": 2048, "quantization_bit": 4,
           "base_model": "Qwen/Qwen2.5-14B-Instruct"}
    seen: list[str] = []
    while cfg is not None:
        step = next_degrade(cfg)
        if step is None:
            break
        cfg, desc = step
        seen.append(desc)
    assert any(d.startswith("缩短序列") for d in seen)      # 先缩序列
    assert any(d.startswith("降低量化位宽") for d in seen)  # 再降位宽
    assert any(d.startswith("降低基座规模") for d in seen)  # 最后降基座
    cutoff_last = max(i for i, d in enumerate(seen) if d.startswith("缩短序列"))
    quant_first = min(i for i, d in enumerate(seen) if d.startswith("降低量化位宽"))
    base_first = min(i for i, d in enumerate(seen) if d.startswith("降低基座规模"))
    assert cutoff_last < quant_first < base_first
    assert seen[-1].startswith("降低基座规模")  # 链底即基座第一档
