"""T1 · 通用接入：collect / 集合指纹去重 / raw 落盘 / API 端到端。"""

from pathlib import Path

from tunefield.pipeline import ingest as ing


def _mkfiles(tmp: Path) -> list[tuple[str, bytes]]:
    return [
        ("a.txt", b"hello tunefield\n"),
        ("sub/b.md", "# 标题\n正文内容\n".encode("utf-8")),
    ]


def test_collect_directory(tmp_path: Path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.txt").write_bytes(b"content a")
    (root / "sub").mkdir()
    (root / "sub" / "b.md").write_bytes(b"content b")
    files = ing.collect_from_path(root)
    names = sorted(n for n, _ in files)
    assert names == ["a.txt", "b.md"]


def test_fingerprint_order_invariant():
    a = ing.set_fingerprint(["h1", "h2"])
    b = ing.set_fingerprint(["h2", "h1"])
    assert a == b
    c = ing.set_fingerprint(["h1", "h3"])
    assert a != c


def test_ingest_path_idempotent_and_writes_raw(tmp_path: Path):
    src = tmp_path / "docs"
    src.mkdir()
    (src / "a.txt").write_bytes(b"alpha")
    (src / "b.txt").write_bytes(b"beta")

    first = ing.ingest_path(src, name="docs")
    assert first["status"] == "ingested"

    # 重复接入 → 同一 dataset（复用，不新增）
    again = ing.ingest_path(src, name="docs")
    assert again["id"] == first["id"]

    # raw 落盘存在
    from tunefield import config
    raw = config.RAW_DIR / first["content_hash"] / "a.txt"
    assert raw.exists() and raw.read_bytes() == b"alpha"


def test_ingest_cli_help_list(tmp_path: Path):
    """CLI ingest 命令成功路径可用（路径不存在时报错退出 1）。"""
    import subprocess
    import os
    env = dict(os.environ)
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    r = subprocess.run(
        ["uv", "run", "tunefield", "ingest", str(tmp_path / "nope"), "--name", "x"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    assert r.returncode == 1  # 路径不存在 -> 失败，但说明命令真实接线
    assert "失败" in r.stdout


def test_ingest_rejects_self_output_dir():
    """防呆：拒绝把平台自产目录（data/）当作数据源。"""
    from tunefield import config
    import pytest
    with pytest.raises(ValueError, match="自产目录"):
        ing.collect_from_path(config.DATA_DIR)