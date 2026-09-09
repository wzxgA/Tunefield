"""T8 · 配置推荐器：显存档位 / 数据量 → 训练参数（引擎 A · QLoRA 微调）。

对齐 B4.3 推荐映射表：

| 显存档  | 基座        | cutoff_len | 轮次（按数据量修正）                       | 学习率      |
|--------|-------------|-----------|--------------------------------------------|------------|
| 6–8GB  | 1.5B–3B     | 512–1024  | <50MB→3 轮；50–500MB→2 轮；>500MB→1 轮      | 1e-4       |
| 8–12GB | 7B          | 1024      | 同上                                        | 1e-4       |
| 16GB   | 14B         | 1024      | 同上                                        | 5e-5       |
| 24GB+  | 32B         | 2048      | 同上                                        | 5e-5        |

约定：
- 6–8GB 档基座优先取 T0 定版记录（pinned，通常 0.5B/1.5B 起步最快），无定版时回退 1.5B；
- 推荐结果可被用户 overrides 逐项覆盖（后者优先）；
- 无 torch / 无 CUDA 时显存未知 → 落 6–8GB 档（最保守），由用户覆盖。
"""

from __future__ import annotations

import json

from tunefield.engine import base as engine_base

# 显存档位表（max_gb=None 表示无上限档）
VRAM_TIERS: list[dict] = [
    {"max_gb": 8, "label": "6–8GB", "base": "Qwen/Qwen2.5-1.5B-Instruct",
     "cutoff_len": 768, "learning_rate": 1e-4},
    {"max_gb": 12, "label": "8–12GB", "base": "Qwen/Qwen2.5-7B-Instruct",
     "cutoff_len": 1024, "learning_rate": 1e-4},
    {"max_gb": 16, "label": "16GB", "base": "Qwen/Qwen2.5-14B-Instruct",
     "cutoff_len": 1024, "learning_rate": 5e-5},
    {"max_gb": None, "label": "24GB+", "base": "Qwen/Qwen2.5-32B-Instruct",
     "cutoff_len": 2048, "learning_rate": 5e-5},
]

# 传给引擎的参数键（与 BaseEngine.default_cfg / LlmFactoryEngine 对齐）
ENGINE_KEYS = (
    "base_model", "epochs", "learning_rate", "cutoff_len",
    "quantization_bit", "lora_rank", "lora_alpha", "template",
    "per_device_batch_size", "gradient_accumulation_steps",
    "logging_steps", "save_steps", "seed",
)


def detect_vram_gb() -> float | None:
    """探测 GPU 显存（GB）；无 torch / 无 CUDA 返回 None。"""
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    except Exception:
        return None


def pick_tier(vram_gb: float | None) -> dict:
    """按显存选档；未知显存落最保守的第一档。"""
    if vram_gb is None:
        return VRAM_TIERS[0]
    for tier in VRAM_TIERS:
        if tier["max_gb"] is None or vram_gb <= tier["max_gb"]:
            return tier
    return VRAM_TIERS[-1]


def epochs_for(corpus_mb: float | None) -> int:
    """数据量 → 训练轮次（B4.3 修正规则）。未知数据量取中间值 2。"""
    if corpus_mb is None:
        return 2
    if corpus_mb < 50:
        return 3
    if corpus_mb <= 500:
        return 2
    return 1


def _base_for_tier(tier: dict) -> str:
    """6–8GB 档优先用 T0 定版基座（本机起步最快），其余档用固定映射。"""
    if tier["max_gb"] == 8:
        pinned = (engine_base.pinned_info() or {}).get("base")
        if pinned:
            return pinned
    return tier["base"]


def recommend(
    vram_gb: float | None,
    corpus_mb: float | None,
    overrides: dict | None = None,
) -> dict:
    """生成最终生效配置（推荐值 + 用户覆盖）。

    返回含 `_meta`（档位/显存/数据量等展示信息）与引擎参数键的字典；
    引擎消费时忽略 `_meta`。
    """
    tier = pick_tier(vram_gb)
    cfg = {
        "base_model": _base_for_tier(tier),
        "epochs": epochs_for(corpus_mb),
        "learning_rate": tier["learning_rate"],
        "cutoff_len": tier["cutoff_len"],
        "quantization_bit": 4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "template": "qwen",
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "logging_steps": 1,  # 每步打 loss,保证训练屏实时出曲线
        "save_steps": 200,
        "seed": 42,
    }
    for k, v in (overrides or {}).items():
        if v is not None and k in ENGINE_KEYS:
            cfg[k] = v
    cfg["_meta"] = {
        "vram_gb": vram_gb,
        "tier": tier["label"],
        "corpus_mb": corpus_mb,
        "overridden": sorted(k for k in (overrides or {}) if k in ENGINE_KEYS and (overrides or {})[k] is not None),
    }
    return cfg


def _corpus_mb_from_dataset(dataset: dict) -> float | None:
    """从 dataset.stats_json（T6 质检报告）取语料量；未构建返回 None。"""
    raw = dataset.get("stats_json")
    if not raw:
        return None
    try:
        report = json.loads(raw)
        return float(report["summary"]["corpus_mb"])
    except (ValueError, TypeError, KeyError):
        return None


def recommend_for_dataset(dataset: dict, overrides: dict | None = None) -> dict:
    """便捷入口：自动探测显存 + 从数据集质检报告取语料量。"""
    return recommend(detect_vram_gb(), _corpus_mb_from_dataset(dataset), overrides)


def dry_run_lines(rec: dict) -> list[str]:
    """`--dry-run` 展示文本（不训练，只打印推荐配置）。"""
    meta = rec.get("_meta", {})
    lines = [
        f"[train] 显存档位：{meta.get('tier')}（探测 {meta.get('vram_gb')}GB）"
        f" · 语料 {meta.get('corpus_mb')}MB",
        f"[train] 基座：{rec['base_model']}",
        f"[train] 轮次：{rec['epochs']} · 学习率：{rec['learning_rate']} · cutoff_len：{rec['cutoff_len']}",
        f"[train] 量化：{rec['quantization_bit']}bit · LoRA rank/alpha：{rec['lora_rank']}/{rec['lora_alpha']}",
        f"[train] batch：{rec['per_device_batch_size']}×{rec['gradient_accumulation_steps']}（累计）"
        f" · logging_steps：{rec['logging_steps']}",
    ]
    if meta.get("overridden"):
        lines.append(f"[train] 用户覆盖：{', '.join(meta['overridden'])}")
    lines.append("[train] dry-run 结束（未创建任务/未训练）")
    return lines
