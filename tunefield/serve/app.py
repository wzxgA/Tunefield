"""FastAPI 应用工厂 + 静态托管 + 基础路由。

对齐方案 B6 F1：
- 一个 uvicorn 进程承载 API、内嵌训练队列、前端静态托管；
- StaticFiles 托管 web/dist（构建产物）；
- 提供健康检查 / 元信息 / 任务占位接口，供前端联调通路。
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from tunefield import __version__
from tunefield.config import WEB_DIST_DIR, ensure_dirs
from tunefield.serve import db, events
from tunefield.serve.queue import JobQueue, set_status

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    return secrets.token_hex(8)


async def _placeholder_handler(job_id: str) -> None:
    """F1 占位 handler：模拟任务推进到完成，供队列/状态机/事件流闭环联调。

    真实引擎编排（解析→训练→导出）由 T7/T13 替换实现。
    """
    set_status(job_id, "running", progress=0.1)
    await asyncio.sleep(1.0)
    set_status(job_id, "running", progress=0.5)
    # 追加 loss 采样点占位，验证 loss 落库与推送链路
    for step, value in ((1, 0.9), (2, 0.7)):
        db.append_loss(job_id, f"[{step},{value}]")
        events.hub.publish("job.loss", {"id": job_id, "step": step, "value": value})
    await asyncio.sleep(1.0)


def create_app() -> FastAPI:
    db.init_db()

    queue: JobQueue = JobQueue(handler=_placeholder_handler)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        await queue.start()
        yield
        await queue.stop()

    app = FastAPI(title="Tunefield", version=__version__, lifespan=_lifespan)

    # Vite dev 跨端口调试用（同源部署不影响）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------- 基础路由 ----------------
    @app.get("/api/health", tags=["system"])
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/meta", tags=["system"])
    async def meta() -> dict:
        from tunefield.engine import base

        return {
            "name": "Tunefield",
            "version": __version__,
            "base": base.pinned_info(),
        }

    # ---------------- 数据集路由（T1 接入） ----------------
    @app.get("/api/datasets", tags=["datasets"])
    async def list_datasets():
        return db.list_datasets()

    @app.get("/api/datasets/{dataset_id}", tags=["datasets"])
    async def get_dataset(dataset_id: str):
        ds = db.get_dataset(dataset_id)
        if ds is None:
            return JSONResponse({"detail": "dataset not found"}, status_code=404)
        return ds

    @app.get("/api/datasets/{dataset_id}/files", tags=["datasets"])
    async def dataset_files(dataset_id: str):
        """T2：逐文件解析状态 + 中间文本预览（pdf/docx/代码等结构化解析结果）。

        对 data/raw/<hash> 内每个文件实时解析，返回每文件的
        kind/language/status/error/字符数/文本块数/preview。
        """
        ds = db.get_dataset(dataset_id)
        if ds is None:
            return JSONResponse({"detail": "dataset not found"}, status_code=404)
        from tunefield.pipeline.parsers import scan_dataset_files

        files = await run_in_threadpool(scan_dataset_files, ds)
        return {"dataset": {"id": ds["id"], "name": ds["name"]}, "files": files}

    @app.post("/api/datasets/{dataset_id}/build", tags=["datasets"])
    async def build_dataset(dataset_id: str, body: dict | None = None):
        """T6：全管线构建（解析→清洗→切片→指令化→质检），产出 train.jsonl 与报告。"""
        ds = db.get_dataset(dataset_id)
        if ds is None:
            return JSONResponse({"detail": "dataset not found"}, status_code=404)
        body = body or {}
        from tunefield.pipeline.build import run_build

        def _run():
            return run_build(
                ds,
                chunk_size=int(body.get("chunk_size", 768)),
                overlap=int(body.get("overlap", 96)),
                template=str(body.get("template", "continuation")),
            )

        try:
            return await run_in_threadpool(_run)
        except (ValueError, FileNotFoundError) as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/datasets/{dataset_id}/report", tags=["datasets"])
    async def dataset_report(dataset_id: str):
        """T6：已构建数据集的质检报告（来自 stats_json）。"""
        ds = db.get_dataset(dataset_id)
        if ds is None:
            return JSONResponse({"detail": "dataset not found"}, status_code=404)
        report = None
        if ds.get("status") == "built" and ds.get("stats_json"):
            try:
                import json

                report = json.loads(ds["stats_json"])
            except (ValueError, TypeError):
                report = None
        return {"dataset": {"id": ds["id"], "name": ds["name"], "status": ds["status"]}, "report": report}

    @app.post("/api/datasets", tags=["datasets"])
    async def create_dataset(name: str = Form(...), files: list[UploadFile] = File(...)):
        """上传一个或多个文件 → 接入（哈希去重，重复秒回既有 dataset）。"""
        from tunefield.pipeline.ingest import file_sha256, set_fingerprint, ingest_files

        if not files:
            return JSONResponse({"detail": "no files"}, status_code=400)
        # 合并加密文件校验（Multiple-file upload 下 FastAPI 保证列表非空）
        if name.strip() == "":
            return JSONResponse({"detail": "name required"}, status_code=400)

        contents = [(f.filename, await f.read()) for f in files]
        fp = set_fingerprint([file_sha256(b) for _, b in contents])
        existed = db.get_dataset_by_hash(fp)
        if existed is not None:
            return {"dataset": existed, "deduped": True}
        ds = ingest_files(contents, name=name.strip(), source="upload")
        return {"dataset": ds, "deduped": False}

    # ---------------- 任务占位路由（后续按 F2/API 契约补齐） ----------------
    @app.get("/api/jobs", tags=["jobs"])
    async def list_jobs():
        return db.list_jobs()

    @app.get("/api/jobs/{job_id}", tags=["jobs"])
    async def get_job(job_id: str):
        job = db.get_job(job_id)
        if job is None:
            return JSONResponse({"detail": "job not found"}, status_code=404)
        return job

    @app.post("/api/jobs", tags=["jobs"])
    async def create_job(body: dict):
        job_id = _new_id()
        db.insert_job(
            job_id=job_id,
            dataset_id=body.get("dataset_id"),
            domain=body.get("domain", "default"),
            kind=body.get("kind", "finetune"),
            base_model=body.get("base"),
            config_json=None,
            status="queued",
            created_at=_now_iso(),
        )
        events.hub.publish(
            "job.created",
            {"id": job_id, "domain": body.get("domain", "default"),
             "kind": body.get("kind", "finetune"), "status": "queued"},
        )
        queue.enqueue(job_id)
        return db.get_job(job_id)

    # ---------------- WebSocket 事件流 ----------------
    @app.websocket("/api/events")
    async def events_ws(ws: WebSocket):
        """前端订阅事件流：job.created / job.status / job.loss 实时推送。"""
        await ws.accept()
        events.hub.connect(ws)
        try:
            while True:
                # 客户端消息仅作心跳/占位，内容忽略
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            events.hub.disconnect(ws)

    # ---------------- 静态托管（放最后，避免吞掉 API 路由） ----------------
    _mount_static(app)

    app.state.queue = queue
    return app


def _mount_static(app: FastAPI) -> None:
    ensure_dirs()
    dist = Path(WEB_DIST_DIR)
    if not dist.exists():
        logger.warning(
            "未找到前端构建产物 %s，/ 与静态资源不可用；请先在 web/ 执行 npm run build", dist
        )
        return

    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa(full_path: str):
        target = (dist / full_path).resolve()
        if (
            full_path
            and target.is_file()
            and str(target).startswith(str(dist.resolve()))
        ):
            return FileResponse(target)
        return FileResponse(dist / "index.html")  # SPA 回退