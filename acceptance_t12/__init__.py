"""T12 · 真实数据验收包（可重复的双通道一致性核对与三形态样例）。

- corpus.py   生成 纯文本 / 文档(pdf,docx) / 代码 三形态样例（含坏文件）；
- dual.py     同一份数据分别走 CLI 与 Web 通道，逐项核对产物一致性；
- cli.py      命令行入口：python -m acceptance_t12 make-corpus|dual-check
"""

from __future__ import annotations
