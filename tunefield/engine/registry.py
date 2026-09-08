"""T7 · 引擎注册表：按 training_jobs.kind 路由到具体引擎。

当前注册：kind=finetune → LLaMA-Factory（引擎 A）。
引擎 B（kind=pretrain，从零预训练）由 T13 接入。
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
        hint = "（从零预训练引擎将在 T13 接入）" if kind == "pretrain" else ""
        raise EngineNotFound(
            f"未知训练引擎 kind={kind!r}{hint}；当前已注册：{sorted(_ENGINES) or '无'}"
        )
    return engine


def list_engines() -> list[dict]:
    return [{"kind": e.kind, "label": e.label} for e in _ENGINES.values()]
