"""T12 · 验收命令行：生成三形态样例 / 跑双通道一致性核对。

用法：
  python -m acceptance_t12 make-corpus [out_dir]
  python -m acceptance_t12 dual-check <corpus> [--name 领域名]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _make_corpus(args: argparse.Namespace) -> int:
    from acceptance_t12.corpus import build_corpus

    target = build_corpus(Path(args.out))
    print(f"已生成三形态验收样例：{target}")
    return 0


def _dual_check(args: argparse.Namespace) -> int:
    import json
    import tempfile

    from acceptance_t12.dual import run_dual_check

    corpus = Path(args.corpus)
    if not corpus.is_dir():
        print(f"corpus 目录不存在：{corpus}")
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="t12dual-"))
    roots = (tmp / "cli", tmp / "web")
    result = run_dual_check(corpus, name=args.name, roots=roots)
    for c in result["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"[{mark}] {c['name']}" + (f" · {c['detail']}" if c["detail"] else ""))
    print("总样本摘要：", result.get("note"))
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"已保存结果：{out}")
    print("双通道一致性：", "通过" if result["ok"] else "未通过")
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acceptance_t12",
        description="T12 真实数据验收：三形态样例 + 双通道一致性核对",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    p = sub.add_parser("make-corpus", help="生成三形态验收样例")
    p.add_argument("out", nargs="?", default="t12_corpus")
    p.set_defaults(func=_make_corpus)

    p = sub.add_parser("dual-check", help="CLI vs Web 双通道产物一致性核对")
    p.add_argument("corpus", help="验收样例目录")
    p.add_argument("--name", default="dual", help="领域名（默认 dual）")
    p.add_argument("--save", default=None, help="结果 JSON 落盘路径（可选）")
    p.set_defaults(func=_dual_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        build_parser().print_help()
        return 0
    return func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
