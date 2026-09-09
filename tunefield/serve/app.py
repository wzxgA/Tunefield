"""FastAPI 应用工厂 + 静态托管 + 基础路由。

对齐方案 B6 F1：
- 一个 uvicorn 进程承载 API、内嵌训练队列、前端静态托管；
- StaticFiles 托管 web/dist（构建产物）；
- 提供健康检查 / 元信息 / 任务占位接口，供前端联调通路。
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from tunefield import __version__
from tunefield.config import WEB_DIST_DIR, ensure_dirs
from tunefield.serve import db, events
from tunefield.serve.queue import JobQueue

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    return secrets.token_hex(8)


def create_app(handler=None) -> FastAPI:
    """应用工厂。

    handler：队列任务处理函数（async job_id -> None）。默认使用真实训练编排
    （T7 引擎 A 微调）；测试可注入假 handler 单独验证队列/状态机/事件流。
    """
    db.init_db()

    if handler is None:
        # T11：默认 handler 支持训练 job 与端到端 run 两类对象
        from tunefield.engine.dispatch import create_combined_handler

        handler = create_combined_handler()
    queue: JobQueue = JobQueue(handler=handler)

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

    # ---------------- T11 端到端 run（一键 run + 阶段时间线） ----------------

    def _parse_timeline(raw) -> list:
        import json as _json

        if not raw:
            return []
        try:
            tl = _json.loads(raw) if isinstance(raw, str) else raw
            return tl if isinstance(tl, list) else []
        except (ValueError, TypeError):
            return []

    @app.get("/api/runs", tags=["runs"])
    async def list_runs():
        runs = db.list_pipeline_runs()
        for r in runs:
            r["timeline"] = _parse_timeline(r.pop("timeline_json", None))
        return runs

    @app.post("/api/runs", tags=["runs"])
    async def create_run(body: dict):
        """T11：一键端到端——数据管线 → 微调 → 导出 → 导入 Ollama。

        body: {dataset_id, epochs?, chunk_size?, overlap?, template?, auto_import?}
        """
        import json as _json

        dataset_id = body.get("dataset_id")
        dataset = db.get_dataset(dataset_id) if dataset_id else None
        if dataset is None:
            return JSONResponse(
                {"detail": "缺少 dataset_id 或数据集不存在：请先上传数据"}, status_code=400
            )
        cfg = {
            "epochs": body.get("epochs"),
            "chunk_size": int(body.get("chunk_size", 768)),
            "overlap": int(body.get("overlap", 96)),
            "template": str(body.get("template", "continuation")),
            "auto_import": bool(body.get("auto_import", True)),
        }
        run_id = _new_id()
        db.insert_pipeline_run(
            id=run_id,
            dataset_id=dataset["id"],
            domain=dataset["name"],
            job_id=None,
            config_json=_json.dumps(cfg, ensure_ascii=False),
            status="queued",
            created_at=_now_iso(),
        )
        events.hub.publish(
            "run.created",
            {"id": run_id, "domain": dataset["name"], "dataset_id": dataset["id"],
             "dataset_name": dataset["name"], "status": "queued"},
        )
        queue.enqueue(run_id)
        return db.get_pipeline_run(run_id)

    @app.get("/api/runs/{run_id}", tags=["runs"])
    async def get_run(run_id: str):
        run = db.get_pipeline_run(run_id)
        if run is None:
            return JSONResponse({"detail": "run not found"}, status_code=404)
        run["timeline"] = _parse_timeline(run.pop("timeline_json", None))
        from tunefield.engine.base import log_tail

        run["log"] = log_tail(run_id, 150)
        return run

    # ---------------- 任务路由（T7 起接真实引擎编排） ----------------
    @app.get("/api/train/preview", tags=["jobs"])
    async def train_preview(dataset_id: str):
        """T8：按显存档位 + 数据集体量给出训练参数推荐（供前端面板预览/覆盖）。"""
        ds = db.get_dataset(dataset_id)
        if ds is None:
            return JSONResponse({"detail": "dataset not found"}, status_code=404)
        from tunefield.engine.recommender import recommend_for_dataset

        return {
            "dataset": {"id": ds["id"], "name": ds["name"], "status": ds["status"]},
            "recommend": recommend_for_dataset(ds),
        }

    @app.post("/api/jobs/{job_id}/export", tags=["jobs"])
    async def export_job(job_id: str, body: dict | None = None):
        """T9：训练产物 → LoRA 合并 → GGUF → 量化 → 指纹入库。"""
        job = db.get_job(job_id)
        if job is None:
            return JSONResponse({"detail": "job not found"}, status_code=404)
        body = body or {}
        quants = body.get("quants") or ["q4_k_m", "q8"]
        import asyncio

        from tunefield.engine.base import EngineError
        from tunefield.engine.exporter import Exporter

        loop = asyncio.get_running_loop()
        try:
            return await run_in_threadpool(
                Exporter().run_export, job, quants=tuple(quants), loop=loop
            )
        except EngineError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/models", tags=["models"])
    async def list_models():
        """T9：GGUF 模型清单（adapters + quantized 联表）。"""
        from tunefield.assets import registry

        return registry.list_gguf_models()

    # ---------------- T10 对话（Ollama 集成） ----------------
    @app.get("/api/chat/status", tags=["chat"])
    async def chat_status():
        from tunefield.serve import ollama

        return ollama.status()

    @app.get("/api/chat/models", tags=["chat"])
    async def chat_models():
        """已导入 Ollama 的平台模型（tunefield- 前缀）。"""
        from tunefield.serve import ollama

        return {"models": ollama.list_ollama_models()}

    @app.post("/api/models/{model_id}/import", tags=["chat"])
    async def import_model(model_id: str, body: dict | None = None):
        """T10：把平台 GGUF 导入 Ollama(ollama create),供对话屏使用。"""
        from tunefield.assets import registry
        from tunefield.engine.base import EngineError
        from tunefield.serve import ollama

        m = registry.get_gguf_model(model_id)
        if m is None:
            return JSONResponse({"detail": "model not found"}, status_code=404)
        body = body or {}
        try:
            return await run_in_threadpool(
                ollama.import_model, m,
                temperature=float(body.get("temperature", 0.8)),
                top_p=float(body.get("top_p", 0.9)),
            )
        except EngineError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.post("/v1/chat/completions", tags=["chat"])
    async def chat_completions(request: Request):
        """T10：OpenAI 兼容端点,转发 Ollama(非流式;流式属 P1)。"""
        import json as _json

        from tunefield.engine.base import EngineError
        from tunefield.serve import ollama

        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"detail": "invalid json body"}, status_code=400)
        if not isinstance(payload, dict) or not payload.get("model"):
            return JSONResponse({"detail": "missing model"}, status_code=400)
        if not isinstance(payload.get("messages"), list) or not payload["messages"]:
            return JSONResponse({"detail": "messages must be a non-empty list"}, status_code=400)
        if payload.get("stream"):
            return JSONResponse(
                {"detail": "stream=true 将在 P1 提供,当前请使用 stream=false"},
                status_code=400,
            )
        try:
            return await run_in_threadpool(ollama.chat_completions, payload)
        except EngineError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.get("/api/models/{model_id}/download", tags=["models"])
    async def download_model(model_id: str):
        """T9：下载 GGUF 文件。"""
        from tunefield.assets import registry

        m = registry.get_gguf_model(model_id)
        if m is None:
            return JSONResponse({"detail": "model not found"}, status_code=404)
        path = Path(m["path"])
        if not path.exists():
            return JSONResponse({"detail": "file missing on disk"}, status_code=404)
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.get("/api/jobs", tags=["jobs"])
    async def list_jobs():
        return db.list_jobs()

    @app.get("/api/jobs/{job_id}", tags=["jobs"])
    async def get_job(job_id: str):
        job = db.get_job(job_id)
        if job is None:
            return JSONResponse({"detail": "job not found"}, status_code=404)
        from tunefield.engine.base import log_tail

        job["log"] = log_tail(job_id, 120)  # 进程内日志尾部（重启后为空，仅恢复展示）
        return job

    @app.post("/api/jobs", tags=["jobs"])
    async def create_job(body: dict):
        import json as _json

        kind = body.get("kind", "finetune")
        if kind == "pretrain":
            return JSONResponse(
                {"detail": "从零预训练引擎（kind=pretrain）将在 T13 接入，当前仅支持微调 finetune"},
                status_code=400,
            )
        dataset_id = body.get("dataset_id")
        dataset = db.get_dataset(dataset_id) if dataset_id else None
        if dataset is None:
            return JSONResponse(
                {"detail": "缺少 dataset_id 或数据集不存在：请先上传并构建数据集"},
                status_code=400,
            )
        domain = (body.get("domain") or dataset["name"] or "default").strip()
        overrides = body.get("overrides") if isinstance(body.get("overrides"), dict) else {}
        # T8：推荐值（显存档/数据量）+ 用户覆盖 → 最终生效配置落 config_json
        from tunefield.engine.recommender import recommend_for_dataset

        rec = recommend_for_dataset(dataset, overrides)
        cfg = {k: v for k, v in rec.items() if k != "_meta"}

        job_id = _new_id()
        db.insert_job(
            job_id=job_id,
            dataset_id=dataset_id,
            domain=domain,
            kind=kind,
            base_model=cfg.get("base_model"),
            config_json=_json.dumps(cfg, ensure_ascii=False),
            status="queued",
            created_at=_now_iso(),
        )
        events.hub.publish(
            "job.created",
            {"id": job_id, "domain": domain, "kind": kind, "status": "queued"},
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