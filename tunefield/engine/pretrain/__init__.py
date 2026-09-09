"""T13 · 从零预训练引擎 B 封装包。

import 本包即触发引擎 B 在 registry 中的注册（runner.py 末尾副作用），
与引擎 A（llmfactory）一致；train.py 仅作为子进程入口，不随本包加载重型依赖。
"""

from __future__ import annotations

from . import runner  # noqa: F401  # 触发 registry.register(PretrainEngine())
