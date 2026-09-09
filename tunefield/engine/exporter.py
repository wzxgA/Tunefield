"""T9 · 量化导出：LoRA 合并 → GGUF 转换 → 量化 → 指纹入库。

固定链路（B4.4，全程子进程）：
1. `llamafactory-cli export`：LoRA 合并进基座 → 完整权重目录；
2. llama.cpp `convert_hf_to_gguf.py`：完整权重 → f16 GGUF；
3. `llama-quantize`：f16 → Q4_K_M / Q8_0（缺量化器时跳过该步，仅产出 f16）；
4. 写 `fingerprint.json`（资产层唯一事实来源）+ 登记 adapters / quantized_models。

工具解析（均可用环境变量覆盖，便于测试与自定义安装位置）：
- 合并 CLI：cfg.bin → TUNEFIELD_LLAMAFACTORY_BIN → llamafactory-cli
- 转换脚本：TUNEFIELD_CONVERT_HF_TO_GGUF（.py 路径）→ TUNEFIELD_LLAMA_CPP_DIR（仓库根）→ 未找到报错
- 量化器：TUNEFIELD_LLAMA_QUANTIZE → PATH 中的 llama-quantize → 缺失则跳过量化
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tunefield import config
from tunefield.assets import registry
from tunefield.engine.base import EngineError, _publish, log_append

# 计划量化档 → llama-quantize 类型名
QUANT_TYPES = {"q4_k_m": "Q4_K_M", "q8": "Q8_0"}


def _slug(text: str) -> str:
    """产物文件名专用:非 ASCII 字符转 '-',避免 C 工具链(llama-quantize 等)
    对中文路径的 ANSI/UTF-8 处理差异;纯中文名回退为 model。"""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return s or "model"


def _converter_script() -> Path:
    env_file = os.environ.get("TUNEFIELD_CONVERT_HF_TO_GGUF")
    if env_file:
        p = Path(env_file)
        if p.exists():
            return p
        raise EngineError(f"TUNEFIELD_CONVERT_HF_TO_GGUF 指向的文件不存在：{env_file}")
    repo = os.environ.get("TUNEFIELD_LLAMA_CPP_DIR")
    if repo:
        p = Path(repo) / "convert_hf_to_gguf.py"
        if p.exists():
            return p
        raise EngineError(f"TUNEFIELD_LLAMA_CPP_DIR 下未找到 convert_hf_to_gguf.py：{repo}")
    raise EngineError(
        "未找到 GGUF 转换脚本 convert_hf_to_gguf.py。请克隆 llama.cpp 仓库后设置"
        " TUNEFIELD_LLAMA_CPP_DIR 指向其根目录（或直接指向脚本："
        "TUNEFIELD_CONVERT_HF_TO_GGUF）。"
    )


def _quantizer() -> str | None:
    env = os.environ.get("TUNEFIELD_LLAMA_QUANTIZE")
    if env:
        return env
    return shutil.which("llama-quantize")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Exporter:
    """训练产物（LoRA 适配器）→ 可分发 GGUF。"""

    def _run(self, cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
        """子进程执行（测试可 monkeypatch 本方法伪造工具链输出）。"""
        # 与训练子进程同理:强制 UTF-8,避免中文路径被 ANSI 代码页误读
        child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(cwd) if cwd else None, env=child_env,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def run_export(self, job: dict, *, quants=("q4_k_m", "q8"), loop=None) -> dict:
        from tunefield.serve import db

        job_id = job["id"]
        domain = job.get("domain") or "model"
        workdir = config.ADAPTERS_DIR / domain / job_id
        adapter_file = workdir / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise EngineError(
                f"任务 {job_id} 没有可导出的适配器权重（{adapter_file} 不存在），请先完成训练"
            )

        base_model = job.get("base_model") or (engine_base_pinned() or {}).get("base") or ""
        version = job_id[:8]
        slug = _slug(domain)  # 文件名一律 ASCII;domain 原名仅进指纹与展示
        gguf_dir = config.GGUF_DIR
        gguf_dir.mkdir(parents=True, exist_ok=True)
        merged = gguf_dir / f"{slug}-{version}-merged"

        def stage(msg: str) -> None:
            log_append(job_id, msg)
            _publish(loop, "job.log", {"id": job_id, "line": msg})

        def tail_of(out: str, n: int = 12) -> str:
            lines = [ln for ln in out.splitlines() if ln.strip()]
            return "\n".join(lines[-n:])

        # ---- 1. 合并 LoRA ----
        stage("[export] 1/3 合并 LoRA → 基座 …")
        try:
            job_cfg = json.loads(job.get("config_json") or "{}")
        except (ValueError, TypeError):
            job_cfg = {}
        template = job_cfg.get("template") or "qwen"
        merge_cli = (
            os.environ.get("TUNEFIELD_LLAMAFACTORY_BIN") or "llamafactory-cli"
        )
        rc, out = self._run([
            merge_cli, "export",
            "--model_name_or_path", base_model,
            "--adapter_name_or_path", str(workdir),
            "--template", template,
            "--export_dir", str(merged),
            "--export_size", "2",
            "--export_device", "cpu",
            "--export_legacy_format", "false",
        ])
        for line in tail_of(out, 6).splitlines():
            log_append(job_id, line)
        if rc != 0 or not (merged / "config.json").exists():
            raise EngineError(f"LoRA 合并失败（rc={rc}）：\n{tail_of(out)}")

        # ---- 2. GGUF 转换 ----
        stage("[export] 2/3 转换 GGUF（f16）…")
        converter = _converter_script()
        f16_path = gguf_dir / f"{slug}-{version}-f16.gguf"
        rc, out = self._run([
            sys.executable, str(converter), str(merged),
            "--outfile", str(f16_path), "--outtype", "f16",
        ])
        for line in tail_of(out, 6).splitlines():
            log_append(job_id, line)
        if rc != 0 or not f16_path.exists():
            raise EngineError(f"GGUF 转换失败（rc={rc}）：\n{tail_of(out)}")
        produced: list[tuple[str, Path]] = [("f16", f16_path)]

        # ---- 3. 量化 ----
        quantizer = _quantizer()
        if quantizer is None:
            stage(
                "[export] 未找到 llama-quantize，跳过量化（仅产出 f16 GGUF）。"
                "可下载 llama.cpp 预编译版或设置 TUNEFIELD_LLAMA_QUANTIZE 后重试。"
            )
        else:
            for q in quants:
                qtype = QUANT_TYPES.get(q)
                if qtype is None:
                    continue
                stage(f"[export] 3/3 量化 {q}（{qtype}）…")
                out_q = gguf_dir / f"{slug}-{version}-{q}.gguf"
                rc, out = self._run([quantizer, str(f16_path), str(out_q), qtype])
                for line in tail_of(out, 4).splitlines():
                    log_append(job_id, line)
                if rc != 0 or not out_q.exists():
                    raise EngineError(f"量化 {q} 失败（rc={rc}）：\n{tail_of(out)}")
                produced.append((q, out_q))

        # ---- 4. 指纹 + 登记 ----
        fingerprint = {
            "job_id": job_id,
            "domain": domain,
            "base_model": base_model,
            "adapter_dir": str(workdir),
            "dataset_id": job.get("dataset_id"),
            "config": job_cfg,
            "quants": [q for q, _ in produced],
            "created_at": _now(),
        }
        fp_path = gguf_dir / f"{slug}-{version}-fingerprint.json"
        fp_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")

        adapter_id = registry.register_adapter(
            job_id=job_id, domain=domain, path=workdir, fingerprint=fingerprint
        )
        models = []
        for q, p in produced:
            row = registry.register_quantized(adapter_id=adapter_id, quant=q, path=p)
            models.append(
                {**row, "domain": domain, "version": version, "job_id": job_id}
            )
        stage(f"[export] 完成：{len(models)} 个产物 · 指纹 {fp_path.name}")
        return {"models": models, "fingerprint": fingerprint}


def engine_base_pinned() -> dict:
    from tunefield.engine.base import pinned_info

    return pinned_info() or {}
