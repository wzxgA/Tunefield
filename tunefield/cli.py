"""Tunefield 命令行入口。

F0 骨架：注册全部子命令（空实现），保证 `tunefield --help` 可跑。
后续任务逐步填充：
  - T1  ingest          通用接入（解析 + 哈希去重入库）
  - T2-T6 build         管线五步（解析/清洗/切片/指令化/质检）
  - T7/T13 train        微调（引擎A）/ 从零预训练（引擎B）
  - T9  export          量化导出（合并 → GGUF → 指纹入库）
  - T10 chat            对话验证（Ollama 临时导入 + REPL）
  - T11 run             端到端一键串联
  - F1  serve           Web 平台（FastAPI + 前端单命令直启）
"""

from __future__ import annotations

import argparse

import uvicorn

from tunefield import __version__


def _ingest(args: argparse.Namespace) -> int:
    """T1：通用接入（目录/文件/zip）→ 哈希去重 → raw 落盘并登记 dataset。"""
    from tunefield.pipeline.ingest import ingest_path

    try:
        ds = ingest_path(args.path, name=args.name)
    except FileNotFoundError as exc:
        print(f"[ingest] 失败：{exc}")
        return 1
    except ValueError as exc:
        print(f"[ingest] 失败：{exc}")
        return 1

    print(f"[ingest] dataset id：{ds['id']}")
    print(f"[ingest] 名称：{ds['name']} · 指纹：{ds['content_hash'][:12]}…")
    print(f"[ingest] 状态：{ds['status']} · 时间：{ds['created_at']}")
    return 0


def _build(args: argparse.Namespace) -> int:
    """T6：全管线构建 dataset（解析→清洗→切片→指令化→质检 + JSONL 落盘）。"""
    from tunefield.pipeline.build import lookup_dataset, run_build

    try:
        dataset = lookup_dataset(args.dataset)
        result = run_build(
            dataset,
            chunk_size=args.chunk,
            overlap=args.overlap,
            template=args.template,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[build] 失败：{exc}")
        return 1

    report = result["report"]
    print(f"[build] 数据集：{dataset['name']} · 状态：built")
    print(f"[build] 样本数：{report['summary']['sample_count']} · "
          f"语料：{report['summary']['corpus_mb']}MB · "
          f"块重复率：{report['summary']['dup_rate'] * 100:.1f}%")
    print(f"[build] 长度 P50/P90：{report['summary']['length']['p50']}/{report['summary']['length']['p90']}")
    print(f"[build] 质检结论：{report['overall']}")
    for check in report["checks"]:
        if check["level"] in ("warn", "error"):
            print(f"[build]  · {check['metric']} {check['value']} —— {check['advice']}")
    print(f"[build] 产物：{result['train_jsonl']}")
    return 0


def _parse_overrides(items: list[str]) -> dict:
    """--set k=v 解析：数值尽量转 int/float，其余按字符串。"""
    out: dict = {}
    for item in items:
        k, _, v = item.partition("=")
        if not k or not v:
            continue
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _train(args: argparse.Namespace) -> int:
    """T8/T13：--dry-run 输出显存/数据量 → 推荐配置（不训练）。

    --kind pretrain 走引擎 B 档位表（规模/结构）；真实训练经 Web 平台或一键 run。
    """
    from tunefield.engine.recommender import (
        dry_run_lines, dry_run_lines_pretrain,
        recommend_for_dataset, recommend_pretrain_for_dataset,
    )
    from tunefield.pipeline.build import lookup_dataset

    if not args.dry_run:
        print("[train] 真实训练请通过 Web 平台（uv run tunefield serve）或一键 run：")
        print("[train]   uv run tunefield run <path> --name <领域> [--epochs N]")
        print("[train] CLI 当前支持 --dry-run 预览推荐配置")
        return 1
    try:
        dataset = lookup_dataset(args.dataset)
        overrides = _parse_overrides(args.overrides)
        if args.kind == "pretrain":
            rec = recommend_pretrain_for_dataset(dataset, overrides)
            lines = dry_run_lines_pretrain(rec)
        else:
            rec = recommend_for_dataset(dataset, overrides)
            lines = dry_run_lines(rec)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[train] 失败：{exc}")
        return 1
    for line in lines:
        print(line)
    return 0


def _resolve_job(ref: str):
    """解析导出目标：job id 或 adapter id（adapter.job_id 指向训练任务）。"""
    from tunefield.serve import db

    job = db.get_job(ref)
    if job is not None:
        return job
    adapter = db.get_adapter(ref)
    if adapter is not None and adapter.get("job_id"):
        return db.get_job(adapter["job_id"])
    return None


def _export(args: argparse.Namespace) -> int:
    """T9 CLI：训练产物 → LoRA 合并 → GGUF → 量化（--quant 逗号分隔档位）。"""
    from tunefield.engine.base import EngineError
    from tunefield.engine.exporter import Exporter

    job = _resolve_job(args.artifact)
    if job is None:
        print(f"[export] 找不到任务或适配器：{args.artifact}")
        return 1
    quants = tuple(q for q in (args.quant or "q4_k_m,q8").split(",") if q)
    try:
        result = Exporter().run_export(job, quants=quants)
    except EngineError as exc:
        print(f"[export] 失败：{exc}")
        return 1
    for m in result["models"]:
        print(f"[export] {m['quant']}：{m['path']}")
    print(f"[export] 指纹：{result['fingerprint'].get('created_at')} · 共 {len(result['models'])} 个产物")
    return 0


def _chat(args: argparse.Namespace) -> int:
    """T10 CLI：GGUF 导入 Ollama + REPL。model 可为平台模型/job id/ollama 名/本地文件。"""
    from pathlib import Path

    from tunefield.assets import registry as asset_registry
    from tunefield.engine.base import EngineError
    from tunefield.serve import ollama

    ref = args.model
    models = asset_registry.list_gguf_models()
    pick = next(
        (
            x for x in models
            if ref in (x["ollama_name"], x["id"], x["path"], Path(x["path"]).name)
        ),
        None,
    )
    if pick is None:
        p = Path(ref)
        if p.exists():
            name = f"tunefield-{asset_registry.slugify(p.stem)}"
            pick = {"path": str(p.resolve()), "ollama_name": name}
        else:
            print(f"[chat] 无法识别模型「{ref}」：不是平台 GGUF、ollama 名或存在的文件")
            return 1
    name = pick["ollama_name"]
    try:
        ollama.import_model(pick)
    except EngineError as exc:
        print(f"[chat] 导入失败：{exc}")
        return 1
    print(f"[chat] 已就绪：{name}（Ctrl+C / Ctrl+Z 退出）")
    history: list[dict] = []
    while True:
        try:
            q = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("[chat] 再见")
            return 0
        if not q:
            continue
        history.append({"role": "user", "content": q})
        try:
            r = ollama.chat_completions({"model": name, "messages": history})
            answer = r["choices"][0]["message"]["content"]
        except EngineError as exc:
            print(f"[chat] 请求失败：{exc}")
            continue
        history.append({"role": "assistant", "content": answer})
        print(f"\n模型> {answer}\n")


def _now_iso() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(args: argparse.Namespace) -> int:
    """T11 CLI：一键 ingest → build → train → export → 导入 Ollama。

    与 Web /api/runs 共享同一步骤与代码路径（build/推荐器/引擎A/Exporter/
    ollama），训练阶段含 CUDA OOM 自动降级链。
    """
    import json as _json
    import secrets as _secrets
    import time as _time

    from tunefield.engine import registry
    from tunefield.engine.exporter import Exporter
    from tunefield.pipeline.build import lookup_dataset, run_build
    from tunefield.pipeline.ingest import ingest_path
    from tunefield.serve import db

    started = _time.time()

    def step(n: int, label: str) -> None:
        print(f"[run] {n}/5 {label} …")

    # 1. 接入
    step(1, "接入数据")
    try:
        ds = ingest_path(args.path, name=args.name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[run] 失败：{exc}")
        return 1
    dataset = lookup_dataset(ds["id"])
    print(f"[run] dataset {dataset['id']} · 指纹 {dataset['content_hash'][:10]}…")

    # 2. 构建（解析→清洗→切片→指令化→质检）
    step(2, "构建数据（解析/清洗/切片/指令化/质检）")
    try:
        res = run_build(
            dataset, chunk_size=args.chunk, overlap=args.overlap, template=args.template
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[run] 失败：{exc}")
        return 1
    summary = (res.get("report") or {}).get("summary") or {}
    print(f"[run] 样本 {summary.get('sample_count')} · 语料 {summary.get('corpus_mb')}MB")
    dataset = res["dataset"]

    # 3. 训练（引擎 A，真实子进程；OOM 自动降级）
    from tunefield.engine import llmfactory  # noqa: F401  # 注册引擎 A
    from tunefield.engine.recommender import recommend_for_dataset

    step(3, "微调训练（引擎A · CUDA OOM 自动降级）")
    rec = recommend_for_dataset(dataset, {"epochs": args.epochs} if args.epochs else None)
    cfg = {k: v for k, v in rec.items() if k != "_meta"}
    job_id = _secrets.token_hex(8)
    db.insert_job(
        job_id=job_id, dataset_id=dataset["id"], domain=dataset["name"],
        kind="finetune", base_model=cfg.get("base_model"),
        config_json=_json.dumps(cfg, ensure_ascii=False),
        status="running", created_at=_now_iso(),
    )
    try:
        result = registry.get_engine("finetune").run(db.get_job(job_id), dataset, cfg)
        db.set_job_status(job_id, "done", progress=1.0)
    except Exception as exc:  # noqa: BLE001 - CLI 给出可读失败信息
        db.set_job_status(job_id, "failed", error=str(exc))
        print(f"[run] 训练失败：{exc}")
        return 1
    degrades = result.get("degrade") or []
    print(f"[run] 训练完成 job {job_id} · 轮次 {result.get('epochs')}"
          + (f" · OOM 自动降级 {len(degrades)} 次" if degrades else ""))

    # 4. 导出
    step(4, "量化导出（LoRA 合并 → GGUF → 量化）")
    quants = tuple(q for q in (args.quant or "q4_k_m,q8").split(",") if q)
    try:
        exp_result = Exporter().run_export(db.get_job(job_id), quants=quants)
    except Exception as exc:  # noqa: BLE001
        print(f"[run] 导出失败：{exc}")
        return 1
    models = exp_result.get("models") or []
    print(f"[run] 产物 {len(models)} 个")

    # 5. 导入 Ollama（可选；未装/未运行则跳过不失败）
    if not args.no_import:
        from tunefield.assets import registry as asset_registry
        from tunefield.serve import ollama

        step(5, "导入 Ollama")
        owned = [m for m in asset_registry.list_gguf_models() if m.get("job_id") == job_id]
        pick = next((m for m in owned if m["quant"] == "q4_k_m"), None)
        pick = pick or next((m for m in owned if m["quant"] == "q8"), None)
        pick = pick or next((m for m in owned if m["quant"] == "f16"), None)
        if pick is None:
            print("[run] 无可用 GGUF 产物，跳过导入")
        else:
            try:
                r = ollama.import_model(pick)
                print(f"[run] 已导入 {r.get('ollama_name')}"
                      f" —— 可用 tunefield chat {r.get('ollama_name')} 对话")
            except Exception as exc:  # noqa: BLE001 - Ollama 边界只阻塞对话
                print(f"[run] 导入跳过（Ollama 不可用）：{exc}")
    else:
        print("[run] 5/5 跳过导入 Ollama（--no-import）")

    print(f"[run] 端到端完成 · 总耗时 {_time.time() - started:.1f}s")
    return 0


def _serve(args: argparse.Namespace) -> int:
    """F1：启动 Web 平台（FastAPI + 内嵌队列 + 前端静态托管）。"""
    from tunefield.serve.app import create_app

    app = create_app()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def _base_smoke(args: argparse.Namespace) -> int:
    """T0：基座加载冒烟 + 定版记录。"""
    from tunefield.engine import base

    try:
        result = base.smoke(
            args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as exc:  # 冒烟失败（缺依赖/下载/显存）打印原因
        print(f"[base] 冒烟失败：{exc}")
        return 1

    print(f"[base] 定版基座：{result['base']}")
    print(f"[base] 运行设备：{result['device']} · 峰值显存：{result['vram_gb']}GB")
    print(f"[base] 冒烟耗时：{result['elapsed_s']}s")
    print(f"[base] 输入：{result['prompt']}")
    print(f"[base] 输出：{result['output']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tunefield",
        description="Tunefield · 领域专属小模型训练平台：上传数据、观察训练、使用模型",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("ingest", help="接入数据：解析 + 哈希去重入库")
    p.add_argument("path", help="数据目录、单文件或 zip 混合包")
    p.add_argument("--name", required=True, help="领域名称")
    p.set_defaults(func=_ingest)

    p = sub.add_parser("build", help="运行数据管线并产出质检报告")
    p.add_argument("dataset", help="dataset id 或名称")
    p.add_argument("--chunk", type=int, default=768, help="目标块长（token），默认 768")
    p.add_argument("--overlap", type=int, default=96, help="相邻块重叠（token），默认 96")
    p.add_argument("--template", default="continuation",
                   help="指令模板 key（continuation | qa），默认 continuation")
    p.set_defaults(func=_build)

    p = sub.add_parser("train", help="训练配置预览（--dry-run）/ 真实训练走 Web 平台")
    p.add_argument("dataset", help="dataset id 或名称")
    p.add_argument("--kind", choices=("finetune", "pretrain"), default="finetune",
                   help="引擎类型：finetune 微调（默认）| pretrain 从零预训练")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印按显存/数据量推荐的训练配置，不创建任务",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="k=v",
        dest="overrides",
        help="覆盖推荐参数，可多次使用，如 --set epochs=2 --set base_model=Qwen/Qwen2.5-0.5B-Instruct",
    )
    p.set_defaults(func=_train)

    p = sub.add_parser("export", help="导出量化模型（LoRA 合并 → GGUF → 量化）")
    p.add_argument("artifact", help="adapter id 或 job id")
    p.add_argument("--quant", default="q4_k_m,q8", help="量化档位（逗号分隔），默认 q4_k_m,q8")
    p.set_defaults(func=_export)

    p = sub.add_parser("chat", help="与量化模型对话（GGUF 导入 Ollama + REPL）")
    p.add_argument("model", help="GGUF 模型（平台模型 id / ollama 名 / 本地 .gguf 路径）")
    p.set_defaults(func=_chat)

    p = sub.add_parser(
        "run", help="一键端到端：ingest → build → train → export → 导入 Ollama"
    )
    p.add_argument("path", help="数据目录或文件路径")
    p.add_argument("--name", required=True, help="领域名称")
    p.add_argument("--epochs", type=int, default=None,
                   help="训练轮次（默认按数据量推荐）")
    p.add_argument("--chunk", type=int, default=768, help="目标块长（token），默认 768")
    p.add_argument("--overlap", type=int, default=96, help="相邻块重叠（token），默认 96")
    p.add_argument("--template", default="continuation",
                   help="指令模板 key（continuation | qa），默认 continuation")
    p.add_argument("--quant", default="q4_k_m,q8", help="量化档位，默认 q4_k_m,q8")
    p.add_argument("--no-import", action="store_true",
                   help="训练导出后不自动导入 Ollama")
    p.set_defaults(func=_run)

    p = sub.add_parser("serve", help="启动 Web 平台（单命令直启）")
    p.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    p.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    p.add_argument("--reload", action="store_true", help="开发模式热重载")
    p.set_defaults(func=_serve)

    p = sub.add_parser("base", help="T0 基座：定版 + 加载冒烟（Qwen 系，默认 Qwen2.5-0.5B-Instruct）")
    p.add_argument("--model", default=None, help="覆盖基座模型 id（默认取定版记录）")
    p.add_argument("--prompt", default="你好，请简单介绍一下你自己。", help="冒烟提示词")
    p.add_argument("--max-new-tokens", type=int, default=16, help="冒烟生成上限 token")
    p.set_defaults(func=_base_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:  # 未给子命令时打印帮助
        parser.print_help()
        return 0
    # 保证全新数据目录下 CLI（未先启动 serve）也能直接读写数据库
    from tunefield.serve import db

    db.init_db()
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
