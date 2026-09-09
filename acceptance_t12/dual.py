"""T12 · 双通道一致性核对：同一份数据分别走 CLI 与 Web，比对产物。

- CLI 通道：隔离 TUNEFIELD_ROOT 的子进程 `python -m tunefield ingest/build`；
- Web 通道：隔离 TUNEFIELD_ROOT 后 importlib.reload，经 FastAPI TestClient
  以 multipart 上传同一目录并 POST build；
- 归一化比较两侧产物（train.jsonl / corpus.txt / report.json），把各自的根目录
  字符串替换为占位符后再做深度相等，返回逐项 check 结果。

真实训练/导出/导入的 GPU 验收在 acceptance/README 与 plans v1.19 的清单中说明。
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

_MARK = "<ROOT>"


def _run_cli(root: Path, args: list[str]) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "TUNEFIELD_ROOT": str(root),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "tunefield", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )


def _activate(root: Path) -> None:
    """把当前进程切换到隔离根并重载配置/DB/应用（Web 通道用）。"""
    os.environ["TUNEFIELD_ROOT"] = str(root)
    importlib.reload(importlib.import_module("tunefield.config"))
    importlib.reload(importlib.import_module("tunefield.serve.db"))
    importlib.reload(importlib.import_module("tunefield.serve.app"))


def _sanitize(obj, markers: tuple[str, ...]):
    if isinstance(obj, str):
        out = obj
        for m in markers:
            out = out.replace(m, _MARK)
        return out
    if isinstance(obj, list):
        return [_sanitize(x, markers) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize(v, markers) for k, v in obj.items()}
    return obj


def _norm_jsonl(text: str) -> list:
    return [ln for ln in text.splitlines() if ln.strip()]


def run_dual_check(corpus: Path, *, name: str = "dual",
                   roots: tuple[Path, Path] | None = None) -> dict:
    """执行 CLI 与 Web 双通道 build 并核对产物一致性，返回结果字典。

    roots=(cli_root, web_root) 供测试/验收指定隔离根；缺省落在当前
    TUNEFIELD_ROOT 下（CLI 验收用）。
    """
    from tunefield.serve.app import create_app
    from fastapi.testclient import TestClient

    checks: list[dict] = []
    summary = {"ok": True, "checks": checks, "note": ""}

    def check(label: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": label, "ok": bool(ok), "detail": detail})
        if not ok:
            summary["ok"] = False

    # ---- CLI 通道（隔离根 A） ----
    if roots is not None:
        cli_root, web_root = roots
    else:
        base = Path(os.environ.get("TUNEFIELD_ROOT", "."))
        cli_root, web_root = base / "t12-cli", base / "t12-web"
    cli_root.mkdir(parents=True, exist_ok=True)
    r_ing = _run_cli(cli_root, ["ingest", str(corpus), "--name", name])
    if r_ing.returncode != 0:
        check("CLI ingest", False, r_ing.stdout[-400:] + r_ing.stderr[-400:])
        return summary
    import re

    m = re.search(r"\[ingest\] dataset id：([0-9a-fA-F]+)", r_ing.stdout)
    cli_ds_id = m.group(1) if m else None
    if not cli_ds_id:
        check("CLI ingest 解析 dataset id", False, r_ing.stdout[-400:])
        return summary
    r_build = _run_cli(cli_root, ["build", cli_ds_id])
    if r_build.returncode != 0:
        check("CLI build", False, r_build.stdout[-400:] + r_build.stderr[-400:])
        return summary
    cli_out = cli_root / "data" / "datasets" / cli_ds_id
    cli_jsonl = _norm_jsonl((cli_out / "train.jsonl").read_text(encoding="utf-8"))
    cli_report = json.loads((cli_out / "report.json").read_text(encoding="utf-8"))
    cli_corpus = (cli_out / "corpus.txt").read_text(encoding="utf-8")
    cli_samples = int(cli_report["summary"]["sample_count"])
    check("CLI build 产出样本", cli_samples > 0, f"样本 {cli_samples}")

    # ---- Web 通道（隔离根 B，reload 到独立目录） ----
    web_root.mkdir(parents=True, exist_ok=True)
    _activate(web_root)
    from tunefield import config as cfg_mod

    files = [
        ("files", (p.relative_to(corpus).as_posix(), p.read_bytes(),
                   "application/octet-stream"))
        for p in sorted(corpus.rglob("*"))
        if p.is_file()
    ]
    with TestClient(create_app()) as client:
        up = client.post("/api/datasets", data={"name": name}, files=files)
        if up.status_code != 200:
            check("Web 上传", False, up.text)
            return summary
        web_ds_id = up.json()["dataset"]["id"]
        bd = client.post(f"/api/datasets/{web_ds_id}/build", json={})
        if bd.status_code != 200:
            check("Web build", False, bd.text)
            return summary
        st = client.get(f"/api/datasets/{web_ds_id}").json()
        check("Web build 置 built", st.get("status") == "built", st.get("status", ""))
    web_out = Path(cfg_mod.DATASETS_DIR) / web_ds_id
    web_jsonl = _norm_jsonl((web_out / "train.jsonl").read_text(encoding="utf-8"))
    web_report = json.loads((web_out / "report.json").read_text(encoding="utf-8"))
    web_corpus = (web_out / "corpus.txt").read_text(encoding="utf-8")
    web_samples = int(web_report["summary"]["sample_count"])

    # ---- 归一化比对 ----
    markers = (str(cli_root), str(web_root))
    same_jsonl = _sanitize(cli_jsonl, markers) == _sanitize(web_jsonl, markers)
    check("train.jsonl 逐行一致", same_jsonl,
          f"CLI {len(cli_jsonl)} 行 vs Web {len(web_jsonl)} 行")
    same_report = _sanitize(cli_report, markers) == _sanitize(web_report, markers)
    check("质检报告一致", same_report)
    same_corpus = _sanitize(cli_corpus, markers) == _sanitize(web_corpus, markers)
    check("纯文本语料 corpus.txt 一致", same_corpus,
          f"{len(cli_corpus)} vs {len(web_corpus)} 字符")
    check("样本数一致", cli_samples == web_samples,
          f"CLI {cli_samples} vs Web {web_samples}")

    # 坏文件（broken.pdf）应被计为解析失败且不阻断整体 → 双通道同计数
    cli_err = cli_report["summary"].get("files", {}).get("error", 0)
    web_err = web_report["summary"].get("files", {}).get("error", 0)
    check("坏文件错误计数一致", cli_err == web_err and cli_err >= 1,
          f"error={cli_err}")

    summary["note"] = (
        f"样本 {cli_samples} · 语料 {cli_report['summary'].get('corpus_mb')}MB · "
        f"总体 {cli_report.get('overall')}"
    )
    summary["ok"] = all(c["ok"] for c in checks)
    return summary
