"""WebSocket 事件总线：后端状态变化实时广播到前端。

对齐方案 B6 F2：
- 模块级单例 hub，进程内共享（API 路由 / 队列 / handler 均可发布）；
- publish 为同步安全函数：在运行中的事件循环上调度广播任务，无订阅者时 no-op；
- 事件协议（前端消费契约）：
    {"type": "job.status",  "data": {"id", "status", "progress", "error"}}
    {"type": "job.loss",     "data": {"id", "step", "value"}}
    {"type": "job.created",  "data": {"id", "domain", "kind", "status"}}
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class EventHub:
    """活跃 WebSocket 连接集合 + 广播。"""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def connect(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    def publish(self, event: str, data: dict) -> None:
        """同步安全发布：在运行中的事件循环上调度广播；无订阅者时直接返回。"""
        if not self._clients:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # 无事件循环（如同步 CLI 上下文）时丢弃
        loop.create_task(self._broadcast(event, data))

    async def _broadcast(self, event: str, data: dict) -> None:
        message = json.dumps({"type": event, "data": data}, ensure_ascii=False)
        for ws in list(self._clients):
            try:
                await ws.send_text(message)
            except Exception:
                # 发送失败视为死连接，静默移除
                self._clients.discard(ws)


# 进程内共享单例
hub = EventHub()
