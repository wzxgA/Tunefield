"""CLI 骨架冒烟测试（F0 验收：`tunefield --help` 可跑）。"""

from __future__ import annotations

import os
import subprocess
import sys

EXPECTED_COMMANDS = ["ingest", "build", "train", "export", "chat", "run", "serve"]

# 强制被测试的子进程以 UTF-8 输出，避免在 Windows 上回落成 ANSI 代码页（cp936/GBK），
# 导致父进程用 UTF-8 解码失败、stdout 被置为 None。
_UTF8_ENV = {
    **os.environ,
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


def _run_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tunefield", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",  # 兜底：解码异常也不把整条流置为 None
        env=_UTF8_ENV,
    )


def test_help_runs() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0, result.stderr


def test_help_lists_all_commands() -> None:
    result = _run_cli("--help")
    for cmd in EXPECTED_COMMANDS:
        assert cmd in result.stdout, f"帮助信息缺少命令：{cmd}"


def test_no_command_prints_help() -> None:
    result = _run_cli()
    assert result.returncode == 0
    assert "ingest" in result.stdout


def test_build_is_real_command() -> None:
    # T6 起 build 是真实命令：dataset 不存在时明确报错，而非"尚未实现"
    result = _run_cli("build", "no-such-dataset")
    assert result.returncode == 1
    assert "尚未实现" not in result.stdout
    assert "失败" in result.stdout


def test_ingest_is_real_command() -> None:
    # T1 起 ingest 已是真实命令：路径不存在时返回接入失败，而非"尚未实现"
    result = _run_cli("ingest", "some/path", "--name", "demo")
    assert result.returncode == 1
    assert "尚未实现" not in result.stdout
    assert "失败" in result.stdout


def test_version() -> None:
    result = _run_cli("--version")
    assert result.returncode == 0
