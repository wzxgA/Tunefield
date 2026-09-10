"""内嵌任务队列：asyncio 串行调度 + 状态机 + 进程重启恢复。

对齐方案 B6 F1 / B4.6：
- 单 worker 协程串行消费 asyncio.Queue；
- 训练为子进程，worker 只做状态推进与日志读取（本阶段以占位 handler 演示状态机闭环）；
- GPU 预检：可行时置 pending_gpu 延时重查（F1 提供钩子，真实探测随 T7 接入）；
- 进程重启：启动时扫描 SQLite，queued/pending_gpu 重新入队，running/done 等按状态收敛。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from tunefield.serve import db, events

logger = logging.getLogger(__name__)

# 任务处理函数签名：入参 job_id，返回 None。真实实现（T7/T13 引擎编排）后续替换。
JobHandler = Callable[[str], Awaitable[None]]

# 可重新入队的调度态
_RECOVERABLE = ("queued", "pending_gpu")
# 中断后需标记 failed 的执行态（可配合断点续训重启）
_INTERRUPTED = ("running", "exporting", "parsing", "training")


def _is_run(ref: str) -> bool:
    """按 id 判定对象是 T11 端到端 run（pipeline_runs）还是训练 job。"""
    return db.get_pipeline_run(ref) is not None


def set_status(
    ref: str, status: str, *, progress: float | None = None, error: str | None = None
) -> None:
    """状态落库 + 广播事件（队列与 handler 统一走这里推进状态）。

    T11：job 与端到端 run 两类对象共用队列。训练 job 广播 job.status，
    pipeline run 广播 run.status（前端据此驱动阶段时间线刷新）。
    """
    if _is_run(ref):
        db.set_pipeline_run(ref, status=status, progress=progress, error=error)
        events.hub.publish(
            "run.status",
            {"id": ref, "status": status, "progress": progress, "error": error},
        )
        return
    db.set_job_status(ref, status, progress=progress, error=error)
    events.hub.publish(
        "job.status", {"id": ref, "status": status, "progress": progress, "error": error}
    )


@dataclass
class JobQueue:
    """单 worker 的 asyncio 串行队列。"""

    handler: JobHandler
    queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    gpu_poll_interval: float = 5.0
    _worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        """启动 worker，并把可恢复的遗留任务重新入队（进程重启恢复）。"""
        await self._recover()
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._consume(), name="job-worker")

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    # ------------------------------------------------------------------
    # 对外：入队
    # ------------------------------------------------------------------
    def enqueue(self, job_id: str) -> None:
        """加入队列（阻塞由提交侧快照状态为准）。"""
        self.queue.put_nowait(job_id)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    async def _recover(self) -> None:
        """进程重启恢复：重入队调度态任务，标记中断执行态（job 与 run 两类）。"""
        for row in db.jobs_by_status(_RECOVERABLE):
            self.queue.put_nowait(row["id"])
            logger.info("重新入队任务 %s（状态 %s）", row["id"], row["status"])
        for row in db.pipeline_runs_by_status(_RECOVERABLE):
            self.queue.put_nowait(row["id"])
            logger.info("重新入队端到端 run %s（状态 %s）", row["id"], row["status"])
        for row in db.jobs_by_status(_INTERRUPTED):
            set_status(row["id"], "failed", error="进程重启导致任务中断，等待手动重跑")
            logger.warning("任务 %s 标记为 failed（进程重启中断）", row["id"])
        for row in db.pipeline_runs_by_status(_INTERRUPTED):
            set_status(row["id"], "failed", error="进程重启导致端到端运行中断，等待手动重跑")
            logger.warning("端到端 run %s 标记为 failed（进程重启中断）", row["id"])
        # 流程内合并数据集的生命周期 = run：进程被杀时终态清理没机会执行，
        # 这里兜底回收残留（无引用直接回收；有引用用 run 的 merged_from 回填）。
        try:
            from tunefield.pipeline.merge import cleanup_orphan_merged

            res = cleanup_orphan_merged()
            if res.get("removed"):
                logger.info("兜底回收流程内合并数据集 %s 个", len(res["removed"]))
        except Exception:  # noqa: BLE001 - 回收失败不阻塞启动
            logger.warning("兜底回收合并数据集失败", exc_info=True)

    async def _consume(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await self._run_one(job_id)
            except Exception:
                logger.exception("任务 %s 处理异常", job_id)
                set_status(job_id, "failed", error="队列 worker 处理异常")
            finally:
                self.queue.task_done()

    async def _run_one(self, job_id: str) -> None:
        # GPU 预检：显存余量不足置 pending_gpu 延时重查
        if not await self._gpu_available():
            set_status(job_id, "pending_gpu", error="GPU 显存余量不足，等待释放后重试")
            await asyncio.sleep(self.gpu_poll_interval)
            self.queue.put_nowait(job_id)
            return

        # 状态推进：queued -> running -> done/failed（handler 内推进中间态）
        set_status(job_id, "running")
        try:
            await self.handler(job_id)
            set_status(job_id, "done", progress=1.0)
        except Exception as exc:
            logger.exception("任务 %s 失败", job_id)
            set_status(job_id, "failed", error=str(exc))

    async def _gpu_available(self) -> bool:
        """GPU 预检：torch 缺席（纯开发/未装训练依赖）时不设闸放行；
        已装则要求 CUDA 可用且显存余量 ≥ 3GB（8GB 卡的宽松门槛）。"""
        try:
            import torch
        except Exception:
            return True  # 无 torch 的训练环境由引擎报错给出明确指引
        if not torch.cuda.is_available():
            return False
        try:
            free, _total = torch.cuda.mem_get_info()
            return free / (1024**3) >= 3.0
        except Exception:
            return True  # 探测异常不误拦，交由训练引擎处理


# 模块级默认：供 app 创建。handler 由上层注入。
def create_queue(handler: JobHandler) -> JobQueue:
    return JobQueue(handler=handler)