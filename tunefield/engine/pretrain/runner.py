"""T13 · 引擎 B：从零预训练（kind=pretrain，transformers.Trainer 直训）。

prepare() 产出子进程命令：把语料/结构/超参写成 cfg.json →
`python train.py <cfg>`。有断点时追加 resume 字段（断点续训由基类 run 调度）。

产物是**完整权重**（train.py 末尾 save_model → output_dir 含 config.json +
safetensors），不经过 LoRA 合并，直接走 GGUF 导出链（engine/exporter）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tunefield import config
from tunefield.engine import base, registry
from tunefield.engine.base import EngineError

_TRAIN_PY = Path(__file__).with_name("train.py")

_DEFAULT_CFG = {
    # 结构（未走推荐器时的最小兜底，~100M 档；正常路径由 recommender 覆盖）
    "scale": "100M",
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "epochs": 1,
    "learning_rate": 3e-4,
    "seq_len": 512,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "logging_steps": 1,  # 每步打 loss,小数据/短任务也实时出点
    "save_steps": 200,
    "seed": 42,
}


class PretrainEngine(base.BaseEngine):
    kind = "pretrain"
    label = "从零预训练（引擎 B）"
    default_cfg = _DEFAULT_CFG

    def _corpus_path(self, dataset: dict) -> Path:
        return config.DATASETS_DIR / dataset["id"] / "corpus.txt"

    def prepare(
        self, job: dict, dataset: dict, cfg: dict, workdir, resume: str | None
    ) -> list[str]:
        corpus = self._corpus_path(dataset)
        if not corpus.exists():
            raise EngineError(
                f"数据集「{dataset.get('name') or dataset['id']}」缺少纯文本语料 "
                f"（{corpus}）。从零预训练吃原始文本序列：请先执行 tunefield build"
                " 或在前端点「构建」。"
            )
        payload = {k: v for k, v in cfg.items() if k not in ("_meta",)}
        payload.update(
            corpus=str(corpus),
            output_dir=str(workdir),
        )
        # tokenizer 默认定版 Qwen（中文词表）；允许显式指定
        if not payload.get("tokenizer"):
            payload["tokenizer"] = (base.pinned_info() or {}).get(
                "base", "Qwen/Qwen2.5-0.5B-Instruct"
            )
        if resume:
            payload["resume"] = resume
        cfg_path = workdir / "pretrain_cfg.json"
        cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        return [sys.executable, str(_TRAIN_PY), str(cfg_path)]


registry.register(PretrainEngine())
