"""T10 · Ollama 集成:预检、Modelfile 生成、模型导入、清单拉取、OpenAI 兼容转发。

边界(B8):Ollama 缺失/未运行只阻塞对话,不阻塞训练/导出等其它功能。
- 可执行文件:TUNEFIELD_OLLAMA_BIN → ollama(PATH)
- 服务地址:OLLAMA_HOST(默认 http://127.0.0.1:11434)
- 对话转发走 Ollama 内置的 OpenAI 兼容端点 /v1/chat/completions(非流式;
  流式透传属 P1 稳定版范围)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from tunefield.engine.base import EngineError


def ollama_bin() -> str:
    return os.environ.get("TUNEFIELD_OLLAMA_BIN") or "ollama"


def host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def installed() -> bool:
    return shutil.which(ollama_bin()) is not None


def version() -> str | None:
    if not installed():
        return None
    try:
        proc = subprocess.run(
            [ollama_bin(), "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
        text = (proc.stdout or proc.stderr or "").strip()
        return text.splitlines()[-1] if text else None
    except Exception:
        return None


def running(timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{host()}/api/version", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def status() -> dict:
    ok = installed()
    return {
        "installed": ok,
        "version": version() if ok else None,
        "running": running() if ok else False,
        "host": host(),
        "bin": ollama_bin(),
        "hint": None if ok else "安装 Ollama:https://ollama.com/download(Windows 安装包),装后重开终端",
    }


# ---------------------------------------------------------------------------
# Modelfile 与导入
# ---------------------------------------------------------------------------

# 预训练(base)模型没有 chat 能力:套 Instruct 版模板会让 llama.cpp 严格校验
# 输出必须符合 <|im_start|> 语法,base 模型不守格式 → peg-native format 报错。
# 导入时改为无语法约束的纯续写模板(输入原文直接接续写,不做角色拆分)。
_PLAIN_CHAT_TEMPLATE = '"""{{ .Prompt }}"""'


def make_modelfile(
    gguf_path: Path, *, temperature: float = 0.8, top_p: float = 0.9,
    plain: bool = False,
) -> Path:
    """为 GGUF 生成 Modelfile(FROM + 采样参数,可选纯续写 TEMPLATE)。"""
    lines = [f"FROM {gguf_path}"]
    if plain:
        # 显式 TEMPLATE 覆盖权重里携带的 chat_template;纯续写无语法约束
        lines.append(f"TEMPLATE {_PLAIN_CHAT_TEMPLATE}")
    lines.append(f"PARAMETER temperature {temperature}")
    lines.append(f"PARAMETER top_p {top_p}")
    modelfile = gguf_path.with_suffix(".Modelfile")
    modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return modelfile


def import_model(m: dict, *, temperature: float = 0.8, top_p: float = 0.9) -> dict:
    """把一份平台 GGUF 导入 Ollama(ollama create)。m 为 registry 模型行。"""
    if not installed():
        raise EngineError(
            "未安装 Ollama,无法导入对话模型。请到 https://ollama.com/download 安装,"
            "或设置 TUNEFIELD_OLLAMA_BIN 指向 ollama 可执行文件。"
        )
    if not running():
        raise EngineError(
            "Ollama 已安装但服务未运行:请启动 Ollama(托盘图标或 `ollama serve`)后重试。"
        )
    gguf = Path(m["path"])
    if not gguf.exists():
        raise EngineError(f"GGUF 文件不存在:{gguf}")
    # 预训练(base, kind=pretrain)产物用纯续写模板,避免 Instruct 语法校验报错
    plain = False
    if m.get("job_id"):
        from tunefield.serve import db

        job = db.get_job(m["job_id"])
        plain = bool(job and job.get("kind") == "pretrain")
    modelfile = make_modelfile(gguf, temperature=temperature, top_p=top_p,
                               plain=plain)
    name = m["ollama_name"]
    proc = subprocess.run(
        [ollama_bin(), "create", name, "-f", str(modelfile)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-8:])
        raise EngineError(f"ollama create 失败(rc={proc.returncode}):\n{tail or proc.stderr}")
    return {"ollama_name": name, "modelfile": str(modelfile)}


def remove_model(name: str) -> dict:
    """删除已导入 Ollama 的平台模型（ollama rm，仅允许 tunefield- 前缀）。

    幂等：目标已被删/不存在（"not found"）按已删除处理。
    """
    if not name.startswith("tunefield-"):
        raise EngineError(
            f"仅可删除平台导入的模型（tunefield- 前缀），拒绝删除：{name}"
        )
    if not installed():
        raise EngineError(
            "未安装 Ollama。请到 https://ollama.com/download 安装,"
            "或设置 TUNEFIELD_OLLAMA_BIN 指向 ollama 可执行文件。"
        )
    if not running():
        raise EngineError(
            "Ollama 已安装但服务未运行:请启动 Ollama(托盘图标或 `ollama serve`)后重试。"
        )
    proc = subprocess.run(
        [ollama_bin(), "rm", name],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and "not found" not in text.lower():
        tail = "\n".join([ln for ln in text.splitlines() if ln.strip()][-8:])
        raise EngineError(f"ollama rm 失败(rc={proc.returncode}):\n{tail or text}")
    return {"name": name, "removed": proc.returncode == 0}


# ---------------------------------------------------------------------------
# HTTP:清单拉取与对话转发
# ---------------------------------------------------------------------------

def _get_json(path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"{host()}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(path: str, payload: dict, timeout: float = 300.0) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{host()}{path}", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise EngineError(f"Ollama 返回 HTTP {exc.code}:{detail}") from exc
    except Exception as exc:  # 连接失败/超时等
        raise EngineError(f"无法连接 Ollama({host()}):{exc}") from exc


def list_ollama_models() -> list[dict]:
    """已导入到 Ollama 的模型清单(仅本平台 tunefield- 前缀)。"""
    if not running():
        return []
    try:
        data = _get_json("/api/tags")
    except Exception:
        return []
    out: list[dict] = []
    for m in data.get("models", []):
        name = m.get("name") or m.get("model") or ""
        if name.startswith("tunefield-"):
            out.append({"name": name, "size": m.get("size")})
    return out


def chat_completions(payload: dict, timeout: float = 300.0) -> dict:
    """OpenAI 兼容转发:POST {OLLAMA_HOST}/v1/chat/completions。"""
    return _post_json("/v1/chat/completions", payload, timeout=timeout)


def _plain_prompt(messages: list[dict]) -> str:
    """把 chat messages 拼成纯文本（base 模型续写用,不走角色模板）。"""
    parts: list[str] = []
    for m in messages:
        role = "用户" if m.get("role") == "user" else "模型"
        content = str(m.get("content") or "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def raw_completion_openai(payload: dict, timeout: float = 300.0) -> dict:
    """base（预训练）模型的 OpenAI 兼容兜底:走 Ollama /api/generate 纯续写。

    绕开 chat 模板的严格输出校验（peg-native format 报错源），把 messages
    拼成纯文本 prompt 续写，再包装回 OpenAI chat 响应形状。
    """
    messages = payload.get("messages") or []
    if not messages:
        raise EngineError("messages must be a non-empty list")
    prompt = _plain_prompt(messages)
    body: dict = {
        "model": payload["model"],
        "prompt": prompt,
        "stream": False,
    }
    if payload.get("temperature") is not None or payload.get("top_p") is not None:
        options: dict = {}
        if payload.get("temperature") is not None:
            options["temperature"] = float(payload["temperature"])
        if payload.get("top_p") is not None:
            options["top_p"] = float(payload["top_p"])
        body["options"] = options
    data = _post_json("/api/generate", body, timeout=timeout)
    return {
        "choices": [
            {"message": {"role": "assistant", "content": data.get("response", "")}}
        ],
        "model": payload["model"],
        "done": data.get("done"),
    }
