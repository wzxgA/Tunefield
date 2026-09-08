"""T7 · 引擎 A：LLaMA-Factory 微调封装（kind=finetune）。

prepare() 生成子进程命令：写 trainer.yaml（QLoRA 微调 Qwen 基座）+ 在数据集目录
登记 dataset_info（Alpaca 格式）→ `llamafactory-cli train <yaml>`。
有断点时追加 `--resume_from_checkpoint`（断点续训由基类 run 调度）。

CLI 可执行文件解析顺序：cfg.bin → 环境变量 TUNEFIELD_LLAMAFACTORY_BIN → llamafactory-cli。
"""

from __future__ import annotations

import json
import os

from tunefield import config
from tunefield.engine import base, registry
from tunefield.engine.base import EngineError, pinned_info

_DEFAULT_CFG = {
    "epochs": 3,
    "learning_rate": 1e-4,
    "cutoff_len": 1024,
    "quantization_bit": 4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "template": "qwen",
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "logging_steps": 5,
    "save_steps": 200,
    "seed": 42,
}


class LlmFactoryEngine(base.BaseEngine):
    kind = "finetune"
    label = "LLaMA-Factory QLoRA 微调"
    default_cfg = _DEFAULT_CFG

    @staticmethod
    def _cli(cfg: dict) -> str:
        return (
            cfg.get("bin")
            or os.environ.get("TUNEFIELD_LLAMAFACTORY_BIN")
            or "llamafactory-cli"
        )

    def _register_dataset(self, dataset: dict) -> None:
        """在数据目录写 dataset_info.json（若缺失），注册 train → train.jsonl。"""
        data_dir = config.DATASETS_DIR / dataset["id"]
        info_path = data_dir / "dataset_info.json"
        if info_path.exists():
            return
        info = {
            "train": {
                "file_name": "train.jsonl",
                "format": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            }
        }
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def prepare(
        self, job: dict, dataset: dict, cfg: dict, workdir, resume: str | None
    ) -> list[str]:
        data_dir = config.DATASETS_DIR / dataset["id"]
        jsonl = data_dir / "train.jsonl"
        if not jsonl.exists():
            raise EngineError(
                f"数据集「{dataset.get('name') or dataset['id']}」尚未构建（缺少 train.jsonl）。"
                "请先执行 tunefield build 或在前端点「构建」。"
            )
        self._register_dataset(dataset)

        base_model = cfg.get("base_model") or (pinned_info() or {}).get("base") or ""
        yaml_fields = {
            "model_name_or_path": base_model,
            "do_train": True,  # 缺失时 HF 默认不训练，llamafactory 会加载完直接退出
            "dataset": "train",
            "dataset_dir": str(data_dir),
            "template": cfg["template"],
            "finetuning_type": "lora",
            "quantization_bit": int(cfg["quantization_bit"]),
            "lora_rank": int(cfg["lora_rank"]),
            "lora_alpha": int(cfg["lora_alpha"]),
            "output_dir": str(workdir),
            "num_train_epochs": float(cfg["epochs"]),
            "learning_rate": float(cfg["learning_rate"]),
            "cutoff_len": int(cfg["cutoff_len"]),
            "per_device_train_batch_size": int(cfg["per_device_batch_size"]),
            "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
            "logging_steps": int(cfg["logging_steps"]),
            "save_steps": int(cfg["save_steps"]),
            "seed": int(cfg["seed"]),
        }
        yaml_path = workdir / "train.yaml"
        lines = []
        for key, value in yaml_fields.items():
            if isinstance(value, str):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value}")
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        argv = [self._cli(cfg), "train", str(yaml_path)]
        if resume:
            argv += ["--resume_from_checkpoint", resume]
        return argv


registry.register(LlmFactoryEngine())
