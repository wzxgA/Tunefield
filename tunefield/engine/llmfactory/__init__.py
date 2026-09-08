"""T7 · LLaMA-Factory 引擎 A 封装包。

import 本包即触发引擎 A 在 registry 中的注册（runner.py 末尾副作用），
避免在服务进程中「import 了 llmfactory 但注册表是空的」这种隐性 bug。
"""

from __future__ import annotations

from . import runner  # noqa: F401  # 触发 registry.register(LlmFactoryEngine())
