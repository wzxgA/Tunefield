"""T7 · 回归测试：import llmfactory 包必须触发引擎注册（防止包内副作用丢失）。"""

from __future__ import annotations


def test_importing_llmfactory_package_registers_engine():
    from tunefield.engine import llmfactory  # noqa: F401
    from tunefield.engine import registry

    # 关键断言：仅 import 包即应在注册表里看到 finetune
    kinds = [e["kind"] for e in registry.list_engines()]
    assert "finetune" in kinds, f"导入 llmfactory 后注册表为空：{kinds}"


def test_dispatch_get_engine_finetune():
    """引擎 A 的 dispatch 路由：finetune 必须能拿到引擎。"""
    from tunefield.engine import llmfactory  # noqa: F401
    from tunefield.engine import registry
    from tunefield.engine.llmfactory.runner import LlmFactoryEngine

    engine = registry.get_engine("finetune")
    assert isinstance(engine, LlmFactoryEngine)
