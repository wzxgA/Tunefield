"""T8 · 配置推荐器：显存档位映射、数据量轮次修正、覆盖合并、dry-run。"""

from __future__ import annotations

import importlib
import json
import os

import pytest


@pytest.fixture(scope="module", autouse=True)
def _isolated_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("t8rec")
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


def test_pick_tier_boundaries():
    from tunefield.engine.recommender import VRAM_TIERS, pick_tier

    assert pick_tier(None) is VRAM_TIERS[0]          # 未知显存 → 最保守
    assert pick_tier(6) is VRAM_TIERS[0]
    assert pick_tier(8) is VRAM_TIERS[0]
    assert pick_tier(8.1) is VRAM_TIERS[1]
    assert pick_tier(12) is VRAM_TIERS[1]
    assert pick_tier(16) is VRAM_TIERS[2]
    assert pick_tier(16.1) is VRAM_TIERS[3]
    assert pick_tier(80) is VRAM_TIERS[3]


def test_epochs_by_corpus_size():
    from tunefield.engine.recommender import epochs_for

    assert epochs_for(None) == 2
    assert epochs_for(10) == 3     # <50MB
    assert epochs_for(50) == 2     # 50–500MB
    assert epochs_for(500) == 2
    assert epochs_for(600) == 1    # >500MB


def test_recommend_defaults_and_overrides():
    from tunefield.engine.recommender import recommend

    rec = recommend(8.0, 10)  # 6–8GB 档、小语料
    assert rec["epochs"] == 3
    assert rec["learning_rate"] == 1e-4
    assert rec["cutoff_len"] == 768
    assert rec["quantization_bit"] == 4
    assert rec["_meta"]["tier"] == "6–8GB"
    assert rec["_meta"]["overridden"] == []

    # 覆盖生效且记录
    rec2 = recommend(24, 600, overrides={
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct", "epochs": 5,
    })
    assert rec2["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert rec2["epochs"] == 5
    assert rec2["learning_rate"] == 5e-5       # 档位默认保留
    assert set(rec2["_meta"]["overridden"]) == {"base_model", "epochs"}

    # 非法键被忽略
    rec3 = recommend(8, 10, overrides={"not_a_key": 1})
    assert "not_a_key" not in rec3


def test_recommend_for_dataset_reads_stats(tmp_path):
    from tunefield.engine.recommender import recommend_for_dataset
    from tunefield.serve import db

    report = {"summary": {"corpus_mb": 100.0}}
    ds = db.insert_dataset(
        id="ds-rec", name="rec-demo", content_hash="hash-rec",
        source="test", stats_json=json.dumps(report),
    )
    rec = recommend_for_dataset(ds)
    assert rec["_meta"]["corpus_mb"] == 100.0
    assert rec["epochs"] == 2  # 50–500MB 档

    # 未构建数据集 → 语料未知 → epochs=2
    ds2 = db.insert_dataset(
        id="ds-rec2", name="rec-raw", content_hash="hash-rec2",
        source="test", stats_json=None,
    )
    assert recommend_for_dataset(ds2)["_meta"]["corpus_mb"] is None


def test_dry_run_lines_mentions_key_fields():
    from tunefield.engine.recommender import dry_run_lines, recommend

    rec = recommend(8, 10, overrides={"epochs": 1})
    text = "\n".join(dry_run_lines(rec))
    assert "6–8GB" in text and rec["base_model"] in text
    assert "用户覆盖" in text and "epochs" in text
    assert "dry-run" in text


def test_cli_train_dry_run(capsys):
    from tunefield import cli
    from tunefield.serve import db

    # 数据集不存在 → 失败
    assert cli.main(["train", "no-such", "--dry-run"]) == 1
    err = capsys.readouterr().out
    assert "失败" in err

    # 存在 → 打印推荐且不训练
    db.insert_dataset(
        id="ds-cli", name="cli-demo", content_hash="hash-cli",
        source="test", stats_json=None,
    )
    assert cli.main(["train", "cli-demo", "--dry-run", "--set", "epochs=1"]) == 0
    out = capsys.readouterr().out
    assert "基座" in out and "dry-run" in out and "epochs" in out


def test_cli_train_without_dry_run_hints(capsys):
    from tunefield import cli

    assert cli.main(["train", "whatever"]) == 1
    out = capsys.readouterr().out
    assert "serve" in out and "--dry-run" in out
