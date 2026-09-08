"""T7 · 队列 handler：把 queued 训练任务路由到引擎注册表执行。

handler(job_id) 为异步协程：读 job/dataset → registry 按 kind 取引擎 →
在工作线程运行 engine.run（同步、逐行监控），主循环不被阻塞。
失败由 queue._run_one 捕获并置 failed（EngineError 消息面向用户）。
"""

from __future__ import annotations

import asyncio
import json
import logging

from tunefield.engine import registry
from tunefield.engine.base import EngineError, EngineNotFound
from tunefield.serve import db

logger = logging.getLogger(__name__)


def _load_cfg(job: dict) -> dict:
    raw = job.get("config_json")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def create_train_handler():
    """默认任务处理函数：按 job.kind 交给对应引擎（引擎 A 微调）。"""
    from tunefield.engine import llmfactory  # noqa: F401  # 注册引擎 A

    async def handler(job_id: str) -> None:
        job = db.get_job(job_id)
        if job is None:
            logger.warning("任务 %s 不存在，跳过", job_id)
            return
        kind = job.get("kind") or "finetune"
        if kind == "pretrain":
            raise EngineNotFound(
                "从零预训练引擎（kind=pretrain）将在 T13 接入；当前请使用 kind=finetune"
            )
        if not job.get("dataset_id"):
            raise EngineError("任务缺少 dataset_id：请从「已构建」的数据集创建训练任务")

        dataset = db.get_dataset(job["dataset_id"]) or {}
        engine = registry.get_engine(kind)
        cfg = _load_cfg(job)

        loop = asyncio.get_running_loop()
        # 训练为子进程，同步监控放线程池，不阻塞事件循环（WebSocket/API 保持可用）
        await loop.run_in_executor(
            None, lambda: engine.run(job, dataset, cfg, loop=loop)
        )

    return handler
