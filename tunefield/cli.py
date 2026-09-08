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

# 各空命令的交付位置，用于提示与排期对照
_PLANNED = {
    "ingest": "T1（通用接入：解析 + 哈希去重 + raw 落盘）",
    "build": "T2–T6（管线五步：解析/清洗/切片/指令化/质检）",
    "train": "T7 微调（引擎A）/ T13 从零预训练（引擎B）",
    "export": "T9（量化导出：合并 → GGUF → 指纹入库）",
    "chat": "T10（对话验证：Modelfile 生成 + Ollama 导入 + REPL）",
    "run": "T11（端到端一键串联 + 自动降级链）",
    "serve": "F1（FastAPI 骨架 + 内嵌队列 + 前端托管）",
}


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


def _make_stub(command: str):
    """生成一个空命令处理函数：打印交付排期并返回非零退出码。"""

    def _run(args: argparse.Namespace) -> int:
        print(f"[{command}] 尚未实现，将由 {_PLANNED[command]} 交付。")
        return 1

    return _run


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

    p = sub.add_parser("train", help="创建训练任务（微调 / 从零预训练）")
    p.add_argument("dataset", help="dataset id 或名称")
    p.add_argument("--domain", required=True, help="领域名称")
    p.add_argument(
        "--kind",
        choices=["finetune", "pretrain"],
        default="finetune",
        help="finetune=引擎A 微调（默认）；pretrain=引擎B 从零预训练",
    )
    p.add_argument("--base", default="auto", help="微调基座，auto 为按显存推荐（Qwen 系）")
    p.add_argument("--size", default="200M", help="预训练目标规模（引擎B），如 100M/200M/400M")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="k=v",
        dest="overrides",
        help="覆盖推荐器参数，可多次使用，如 --set learning_rate=1e-4",
    )
    p.set_defaults(func=_make_stub("train"))

    p = sub.add_parser("export", help="导出量化模型（LoRA 合并或预训练全量 → GGUF）")
    p.add_argument("artifact", help="adapter id 或 job id")
    p.add_argument("--quant", default="q4_k_m,q8", help="量化档位（逗号分隔），默认 q4_k_m,q8")
    p.set_defaults(func=_make_stub("export"))

    p = sub.add_parser("chat", help="与量化模型对话（Ollama 临时导入 + REPL）")
    p.add_argument("model", help="GGUF 模型名或文件路径")
    p.set_defaults(func=_make_stub("chat"))

    p = sub.add_parser("run", help="一键串联 ingest → build → train → export → chat")
    p.add_argument("path", help="数据目录或文件路径")
    p.add_argument("--name", required=True, help="领域名称")
    p.set_defaults(func=_make_stub("run"))

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
