"""T11 · 端到端编排：一键 run（构建 → 训练 → 导出 → 导入）。

`run_pipeline(run_id, loop=None)` 是同步编排函数，供队列 worker（run_in_executor）
与测试直接调用。逐阶段推进 `pipeline_runs.timeline_json`，实时广播
`run.stage` / `run.log` 事件：

1. build     数据管线五步（解析/清洗/切片/指令化/质检）→ train.jsonl + 报告；
2. train     真实训练子任务（training_jobs，kind=finetune），复用引擎A与
             CUDA OOM 自动降级链；run.job_id 关联该子任务供前端看曲线/日志；
3. export    量化导出（LoRA 合并 → GGUF → q4_k_m/q8）；
4. import    导入 Ollama（auto_import=true 时；未安装/未运行 → skipped，
             遵循「Ollama 只阻塞对话」边界，不使整条 run 失败）。

错误语义：任一段失败即中止后续阶段——该段标 failed、run 置 failed（由队列
外层落库并广播 run.status），已完成的阶段与产物保留，可人工续跑后续步骤。
"""

from __future__ import annotations

import json
import secrets
import time

from tunefield.engine.base import EngineError, _publish, log_append


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _job_publish(loop, status: str, data: dict) -> None:
    """广播 job.* 事件（训练子任务与端到端 run 走同一事件流通道）。"""
    _publish(loop, status, data)


def run_pipeline(run_id: str, *, loop=None) -> dict:
    """执行一次端到端 run（同步；loop 供事件投递，CLI/测试可为 None）。"""
    from tunefield.engine import registry
    from tunefield.engine.exporter import Exporter
    from tunefield.serve import db

    run = db.get_pipeline_run(run_id)
    if run is None:
        raise EngineError(f"端到端 run 不存在：{run_id}")
    dataset = db.get_dataset(run["dataset_id"]) if run.get("dataset_id") else None
    if dataset is None:
        raise EngineError("run 缺少 dataset_id 或数据集不存在")

    raw = run.get("config_json") or "{}"
    try:
        cfg = json.loads(raw) if isinstance(raw, str) else {}
    except (ValueError, TypeError):
        cfg = {}
    cfg = cfg if isinstance(cfg, dict) else {}
    chunk_size = int(cfg.get("chunk_size", 768))
    overlap = int(cfg.get("overlap", 96))
    template = str(cfg.get("template", "continuation"))
    auto_import = bool(cfg.get("auto_import", True))
    epochs = cfg.get("epochs")

    # ---------------- 阶段推进原语 ----------------
    def publish_stage(phase: dict) -> None:
        _publish(loop, "run.stage", {"id": run_id, **phase})
        log_append(
            run_id,
            f"[flow] {phase.get('label')}：{phase.get('status')}"
            + (f" · {phase.get('detail')}" if phase.get("detail") else ""),
        )

    def begin(key: str, label: str):
        phase = {
            "key": key, "label": label, "status": "running",
            "started_at": _now_iso(), "finished_at": None,
            "duration_s": None, "detail": None,
        }
        db.append_pipeline_phase(run_id, phase)
        publish_stage(phase)
        return phase, time.monotonic()

    def end(phase: dict, t0: float, *, status: str = "ok",
            detail: str | None = None, progress: float | None = None) -> None:
        duration = round(time.monotonic() - t0, 1)
        phase.update(status=status, finished_at=_now_iso(),
                     duration_s=duration, detail=detail)
        db.update_pipeline_phase(
            run_id, phase["key"], status=status, finished_at=phase["finished_at"],
            duration_s=duration, detail=detail,
        )
        publish_stage(phase)
        if progress is not None:
            db.set_pipeline_run(run_id, progress=progress)

    def fail_running(error: str) -> None:
        """把仍在 running 的阶段标记为 failed，并记录原因。"""
        current = db.get_pipeline_run(run_id)
        raw_tl = (current or {}).get("timeline_json") or "[]"
        try:
            phases = json.loads(raw_tl) if isinstance(raw_tl, str) else []
        except (ValueError, TypeError):
            phases = []
        for ph in reversed(phases if isinstance(phases, list) else []):
            if ph.get("status") == "running":
                db.update_pipeline_phase(
                    run_id, ph["key"], status="failed",
                    finished_at=_now_iso(), detail=error[:300],
                )
                ph.update(status="failed", finished_at=_now_iso(), detail=error[:300])
                publish_stage(ph)
                break

    # ---------------- 阶段 1：构建数据 ----------------
    try:
        from tunefield.pipeline.build import run_build

        phase, t0 = begin("build", "构建数据")
        res = run_build(dataset, chunk_size=chunk_size, overlap=overlap, template=template)
        summary = (res.get("report") or {}).get("summary") or {}
        end(phase, t0, detail=(
            f"{res.get('train_jsonl')} · {summary.get('sample_count', 0)} 条样本"
            f" · 语料 {summary.get('corpus_mb', '?')}MB"
        ), progress=0.3)

        # ---------------- 阶段 2：微调训练 ----------------
        phase, t0 = begin("train", "微调训练")
        from tunefield.engine.recommender import recommend_for_dataset

        rec = recommend_for_dataset(
            dataset, {"epochs": int(epochs)} if epochs else None
        )
        engine_cfg = {k: v for k, v in rec.items() if k != "_meta"}
        job_id = secrets.token_hex(8)
        db.set_pipeline_run(run_id, job_id=job_id)
        db.insert_job(
            job_id=job_id, dataset_id=dataset["id"], domain=run["domain"],
            kind="finetune", base_model=engine_cfg.get("base_model"),
            config_json=json.dumps(engine_cfg, ensure_ascii=False),
            status="running", created_at=_now_iso(),
        )
        _job_publish(loop, "job.created",
                     {"id": job_id, "domain": run["domain"], "kind": "finetune",
                      "status": "running"})
        from tunefield.engine import llmfactory  # noqa: F401  # 触发引擎 A 注册
        engine = registry.get_engine("finetune")
        try:
            result = engine.run(
                db.get_job(job_id), dataset, engine_cfg, loop=loop
            )
        except EngineError as exc:
            db.set_job_status(job_id, "failed", error=str(exc))
            raise
        db.set_job_status(job_id, "done", progress=1.0)
        _job_publish(loop, "job.status",
                     {"id": job_id, "status": "done", "progress": 1.0, "error": None})
        degrades = result.get("degrade") or []
        train_detail = f"任务 {job_id} · 轮次 {result.get('epochs')}"
        if degrades:
            train_detail += f" · OOM 自动降级 {len(degrades)} 次"
        end(phase, t0, detail=train_detail, progress=0.8)

        # ---------------- 阶段 3：量化导出 ----------------
        phase, t0 = begin("export", "量化导出")
        exp_result = Exporter().run_export(db.get_job(job_id), loop=loop)
        models = exp_result.get("models") or []
        end(phase, t0, detail=(
            f"{len(models)} 个产物（{', '.join(m['quant'] for m in models)}）"
        ), progress=0.95)

        # ---------------- 阶段 4：导入 Ollama ----------------
        if not auto_import:
            phase, t0 = begin("import", "导入对话")
            end(phase, t0, status="skipped", detail="未开启自动导入（auto_import=false）",
                progress=1.0)
        else:
            from tunefield.assets import registry as asset_registry
            from tunefield.serve import ollama

            owned = [m for m in asset_registry.list_gguf_models() if m.get("job_id") == job_id]
            pick = next((m for m in owned if m["quant"] == "q4_k_m"), None)
            pick = pick or next((m for m in owned if m["quant"] == "q8"), None)
            pick = pick or next((m for m in owned if m["quant"] == "f16"), None)
            phase, t0 = begin("import", "导入对话")
            if pick is None:
                end(phase, t0, status="skipped", detail="无可用 GGUF 产物", progress=1.0)
            else:
                try:
                    r = ollama.import_model(pick)
                    end(phase, t0, detail=f"已导入 {r.get('ollama_name')}", progress=1.0)
                except EngineError as exc:
                    # 边界：Ollama 缺失/未运行只阻塞对话，不使端到端失败
                    end(phase, t0, status="skipped",
                        detail=f"Ollama 不可用（{exc}）——可稍后在对话屏手动导入",
                        progress=1.0)

        db.set_pipeline_run(run_id, status="done", progress=1.0)
        log_append(run_id, "[flow] 端到端 run 完成")
        return {
            "run_id": run_id, "job_id": job_id, "models": len(models),
            "timeline": _timeline(run_id),
        }
    except Exception as exc:
        fail_running(str(exc))
        db.set_pipeline_run(run_id, error=str(exc))
        raise


def _timeline(run_id: str) -> list[dict]:
    from tunefield.serve import db

    run = db.get_pipeline_run(run_id) or {}
    raw = run.get("timeline_json") or "[]"
    try:
        tl = json.loads(raw) if isinstance(raw, str) else []
    except (ValueError, TypeError):
        tl = []
    return tl if isinstance(tl, list) else []
