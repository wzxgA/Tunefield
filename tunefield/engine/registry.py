"""T7 · 引擎注册表：按 training_jobs.kind 路由到具体引擎。

当前注册：kind=finetune → LLaMA-Factory（引擎 A）；kind=pretrain → 从零预训练
（引擎 B，T13）。各引擎在 import 其包时自动注册（llmfactory/pretrain）。
"""

from __future__ import annotations

from .base import BaseEngine, EngineNotFound

_ENGINES: dict[str, BaseEngine] = {}


def register(engine: BaseEngine) -> None:
    """注册引擎实例（以 engine.kind 为键，重复注册覆盖并告警）。"""
    prev = _ENGINES.get(engine.kind)
    if prev is not None and prev is not engine:
        import logging

        logging.getLogger(__name__).warning("引擎 %s 被重复注册覆盖", engine.kind)
    _ENGINES[engine.kind] = engine


def get_engine(kind: str) -> BaseEngine:
    engine = _ENGINES.get(kind)
    if engine is None:
        raise EngineNotFound(
            f"未知训练引擎 kind={kind!r}；当前已注册：{sorted(_ENGINES) or '无'}（"
            "引擎经 import 包自动注册）"
        )
    return engine


def list_engines() -> list[dict]:
    return [{"kind": e.kind, "label": e.label} for e in _ENGINES.values()]
