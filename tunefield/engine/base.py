"""T0 · 基座定版与加载冒烟。

提供：基座登记（pin）、加载器、最小推理冒烟。基座覆盖 Qwen 系列（默认 Qwen2.5-0.5B-Instruct），
8GB 显存目标。定版信息写入 models/bases/_pinned.json，前端首页据此展示所选基座。

依赖（torch/transformers）由 _load_backend 惰性导入：仅执行加载/冒烟时才重签名，避免 CLI
其他命令（serve 等）在未安装训练依赖时启动即报错。

冒烟只做一次短生成验证「权重可加载 + 显存够 + 推理通」，不做效果评测。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tunefield import config

# 默认基座：Qwen2.5-0.5B-Instruct（轻量，8GB 显存充裕）
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
PINNED_FILE_NAME = "_pinned.json"


def _pinned_path() -> Path:
    return config.BASES_DIR / PINNED_FILE_NAME


def pin_base(model_id: str = DEFAULT_BASE) -> dict[str, Any]:
    """登记定版基座并落盘 _pinned.json。"""
    config.ensure_dirs()
    info = {
        "base": model_id,
        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": None,
        "vram_gb": None,
    }
    _pinned_path().write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return info


def pinned_info() -> dict[str, Any]:
    """读取当前定版基座信息；未定版时返回默认基座占位。"""
    if _pinned_path().exists():
        return json.loads(_pinned_path().read_text(encoding="utf-8"))
    return {"base": DEFAULT_BASE, "pinned_at": None, "device": None, "vram_gb": None}


def _load_backend() -> tuple[Any, Any, Any]:
    """惰性导入 torch / AutoModel / AutoTokenizer。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def load_base(
    model_id: str | None = None,
    *,
    device: str | None = None,
) -> tuple[Any, Any, str]:
    """加载定版基座（模型 + tokenizer），返回 (model, tokenizer, device)。

    device 依次取：参数指定 → cuda(若可用) → mps → cpu。
    """
    model_id = model_id or pinned_info()["base"]
    torch, AutoModel, AutoTokenizer = _load_backend()

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    _strict_device = device  # 供冒烟写回定版信息

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        device_map="auto" if device == "cuda" else None,
        torch_dtype="auto",
    )
    if device != "cuda":
        model.to(device)
    model.eval()
    return model, tokenizer, _strict_device


def smoke(
    model_id: str | None = None,
    *,
    prompt: str = "你好，请简单介绍一下你自己。",
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """加载基座并做一次短生成，返回耗时/显存/输出；同时把 device/显存写回定版记录。

    副作用：更新 _pinned.json 的 device 与 vram_gb 字段。
    """
    torch, _, _ = _load_backend()
    model, tokenizer, device = load_base(model_id, device="cuda")

    start = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    vram_gb = None
    if device == "cuda":
        alloc = torch.cuda.max_memory_allocated() / 1024**3
        vram_gb = round(alloc, 2)

    info = pinned_info()
    info.update(device=device, vram_gb=vram_gb, last_smoke_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    _pinned_path().write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "base": model_id or info["base"],
        "device": device,
        "elapsed_s": round(elapsed, 2),
        "vram_gb": vram_gb,
        "prompt": prompt,
        "output": text,
    }


# ---------------------------------------------------------------------------
# T7 · 训练引擎抽象层（多后端可插拔）
#
# 引擎必须实现 prepare()（产出子进程命令与工作目录内配置），run() 在基类实现：
# 逐行读取 stdout 采样 loss/epoch → 落库 + 广播；异常退出自动重试一次（续跑）；
# 断点续训：同一 job 目录出现 checkpoint 即自动追加 resume 参数。
# 真实 GPU 训练由子进程承担，主进程不加载 CUDA 上下文。
# ---------------------------------------------------------------------------

import os
import re
import subprocess
import threading
from collections import defaultdict, deque

# 事件类型（前端订阅契约）：
#   job.log    {id, line}        训练日志行（增量流）
#   job.stage  {id, stage, msg}  阶段提示（断点/重试等）
# 其余沿用 F1/F2：job.status / job.loss / job.created

_log_lock = threading.Lock()
_log_rings: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=1200))


def log_append(job_id: str, line: str) -> None:
    with _log_lock:
        _log_rings[job_id].append(line)


def log_tail(job_id: str, limit: int = 120) -> list[str]:
    with _log_lock:
        ring = _log_rings.get(job_id)
        if not ring:
            return []
        return list(ring)[-limit:]


# LLaMA-Factory / HF Trainer 常见日志形态（容错解析，失败不影响训练）
_LOSS_RE = [
    re.compile(r"['\"]loss['\"]\s*:\s*([0-9]+\.?[0-9]*(?:e[-+]?[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\bloss\s*[:=]\s*([0-9]+\.?[0-9]*(?:e[-+]?[0-9]+)?)", re.IGNORECASE),
]
_EPOCH_RE = re.compile(r"['\"]epoch['\"]\s*:\s*([0-9]+\.?[0-9]*)", re.IGNORECASE)
_STEP_RE = re.compile(r"['\"]step['\"]\s*:\s*(\d+)", re.IGNORECASE)
# 整词匹配避免普通词误判（如 boom/bloom 含 "oom" 子串）；"out of memory"
# 与 CUDA 变体本身是短语无需词界
_OOM_RE = re.compile(r"CUDA out of memory|\bout of memory\b|\bOOM\b", re.IGNORECASE)


def parse_train_line(line: str) -> dict | None:
    """从一行训练日志采样 loss/epoch/step；无采样返回 None。"""
    loss = None
    for pat in _LOSS_RE:
        m = pat.search(line)
        if m:
            loss = float(m.group(1))
            break
    if loss is None:
        return None
    out: dict = {"loss": loss}
    m = _EPOCH_RE.search(line)
    if m:
        out["epoch"] = float(m.group(1))
    m = _STEP_RE.search(line)
    if m:
        out["step"] = int(m.group(1))
    return out


class EngineError(Exception):
    """引擎失败（消息面向用户，含可读原因）。"""


class EngineNotFound(EngineError):
    """kind 没有注册对应引擎（如引擎 B 待 T13 接入）。"""


def _latest_checkpoint(workdir) -> str | None:
    if not workdir.is_dir():
        return None
    cps = [d for d in workdir.glob("checkpoint-*") if d.is_dir()]
    if not cps:
        return None
    return str(sorted(cps, key=lambda p: p.stat().st_mtime)[-1])


def _publish(loop, event: str, data: dict) -> None:
    """把事件安全地投递到主事件循环；无 loop（CLI/测试直跑）时仅落库。"""
    from tunefield.serve import events

    events.hub.publish_threadsafe(loop, event, data) if loop is not None else None


class BaseEngine:
    """引擎基类：子类实现 kind/label/prepare()；run() 统一承担日志监控与推进。"""

    kind: str = ""
    label: str = ""
    default_cfg: dict = {}  # 子类提供的推荐默认参数（会被 config_json 覆盖）

    def prepare(self, job: dict, dataset: dict, cfg: dict, workdir, resume: str | None) -> list[str]:
        """生成训练子进程命令（可写工作目录配置）；必须由子类实现。"""
        raise NotImplementedError

    def degrade_cfg(self, cfg: dict) -> tuple[dict, str] | None:
        """OOM 失败时的自动降级方案；默认不支持（返回 None，交由上层报错）。

        T11：run() 失败分支检测到 CUDA OOM 时逐级调用本方法，直到返回 None
        （降级链耗尽，报错）或子进程成功。子类返回 (降级后 cfg, 人类可读说明)。
        """
        return None

    # ------------------------------------------------------------------
    def run(self, job: dict, dataset: dict, cfg: dict, *, loop=None, cmd: list[str] | None = None) -> dict:
        """执行训练并推进进度/loss/日志。

        cmd 参数供测试/调试注入伪造命令；生产路径为 None（由 prepare 生成）。
        失败时：自动重试一次（二次失败抛 EngineError，附退出码与日志尾部）。
        """
        from tunefield import config as _config
        from tunefield.serve import db

        cfg = {**self.default_cfg, **(cfg or {})}  # 推荐值 + 用户覆盖
        job_id = job["id"]
        domain = (job.get("domain") or dataset.get("name") or "model")
        workdir = _config.ADAPTERS_DIR / domain / job_id
        workdir.mkdir(parents=True, exist_ok=True)

        epochs = int(cfg.get("epochs") or cfg.get("num_train_epochs") or 1)
        step_counter = [0]
        last_published_pct = [-1.0]

        def on_line(line: str) -> None:
            log_append(job_id, line)
            _publish(loop, "job.log", {"id": job_id, "line": line})
            sample = parse_train_line(line)
            if sample is None:
                return
            step = sample.get("step") or (step_counter[0] + 1)
            step_counter[0] = step
            db.append_loss(job_id, f"[{step},{sample['loss']}]")
            _publish(loop, "job.loss", {"id": job_id, "step": step, "value": sample["loss"]})
            epoch = sample.get("epoch")
            if epoch is not None:
                progress = max(0.0, min(1.0, epoch / epochs))
                pct = int(progress * 100)
                if pct != last_published_pct[0]:
                    last_published_pct[0] = pct
                    db.set_job_status(job_id, "running", progress=progress)
                    _publish(loop, "job.status",
                             {"id": job_id, "status": "running", "progress": progress})

        def fire_stage(msg: str) -> None:
            log_append(job_id, msg)
            _publish(loop, "job.log", {"id": job_id, "line": msg})

        attempt = 1
        degrades: list[str] = []  # T11：记录本轮自动降级说明（供返回与前端展示）
        while True:
            resume = _latest_checkpoint(workdir)
            argv = list(cmd) if cmd is not None else self.prepare(
                job, dataset, cfg, workdir, resume)
            if resume and cmd is None:
                fire_stage(f"[engine] 检测到断点 {resume}，自动续训…")
            fire_stage(f"[engine] {self.label} 启动（{argv[0]}）")

            try:
                # Windows/管道输出缓冲坑(B8)：不设 PYTHONUNBUFFERED 时，子进程的
                # stdout 走块缓冲（攒约 8KB 才 flush），导致训练日志/loss 在训练
                # 全程不出现、进程退出后一次性涌入。PYTHONUTF8 管编码、这个管缓冲，
                # 两者都要。
                child_env = {
                    **os.environ,
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUNBUFFERED": "1",
                }
                proc = subprocess.Popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    bufsize=1, cwd=str(workdir), env=child_env,
                )
            except FileNotFoundError as exc:
                raise EngineError(
                    f"无法启动训练程序 {argv[0]!r}：{exc}. 请安装依赖或设置 "
                    "TUNEFIELD_LLAMAFACTORY_BIN 环境变量（参见 README）。"
                ) from exc

            # 按块读取 + 自建行缓冲：管道块缓冲下 readline 会把整段输出并成一行，
            # 导致 loss/epoch 只取到第一行，实时进度失真。
            buf = ""
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if line.strip():
                        on_line(line.rstrip("\r"))
            if buf.strip():
                on_line(buf.rstrip("\r"))
            rc = proc.wait()

            if rc == 0:
                return {"job_id": job_id, "workdir": str(workdir),
                        "epochs": epochs, "retried": attempt > 1,
                        "degrade": degrades}

            tail = "\n".join(log_tail(job_id, 25))
            oom = bool(_OOM_RE.search(tail))
            if oom:
                # T11：CUDA OOM 触发自动降级链（缩序列→降位宽→降基座），
                # 每级一次尝试；链底仍失败才报错。普通失败不在此处理。
                step = self.degrade_cfg(cfg)
                if step is not None:
                    cfg, desc = step
                    degrades.append(desc)
                    fire_stage(f"[engine] CUDA OOM → 自动降级：{desc}（重试第 {len(degrades)} 级）…")
                    continue
                raise EngineError(
                    f"训练进程 CUDA OOM 且已无可用降级档位。最近日志：\n{tail}")
            if attempt == 1:
                fire_stage("[engine] 进程异常退出，自动重试一次（断点续训）…")
                attempt = 2
                continue
            raise EngineError(f"训练进程退出码 {rc}。最近日志：\n{tail}")