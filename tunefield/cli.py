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
    p.add_argument("path", help="数据目录或文件路径（支持 zip 混合包）")
    p.add_argument("--name", required=True, help="领域名称")
    p.set_defaults(func=_make_stub("ingest"))

    p = sub.add_parser("build", help="运行数据管线并产出质检报告")
    p.add_argument("dataset", help="dataset id 或名称")
    p.add_argument("--chunk", type=int, default=768, help="目标块长（token），默认 768")
    p.add_argument("--overlap", type=int, default=96, help="相邻块重叠（token），默认 96")
    p.set_defaults(func=_make_stub("build"))

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
    p.set_defaults(func=_make_stub("serve"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:  # 未给子命令时打印帮助
        parser.print_help()
        return 0
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
