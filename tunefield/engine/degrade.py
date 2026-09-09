"""T11 · 训练自动降级链：CUDA OOM 时的逐级降档方案。

对齐 A7/B4.3：显存不足中断 → 自动降级链（缩短序列 → 降低基座规模 → 降低位宽），
逐级重试。`next_degrade(cfg)` 是纯函数：给定当前生效配置，返回「更省显存的下
一档配置 + 人类可读说明」；无档可降返回 None（由调用方报错）。

降级顺序（每次 OOM 失败调用一次，天然单调递减直至链底）：
1. 缩短序列：cutoff_len 减半（下限 256，保护上下文不被压得过短）；
2. 降低位宽：quantization_bit 4 → 3 → 2（QLoRA 量化更激进省显存）；
3. 降低基座规模：沿显存档位表向小一档走（32B→14B→7B→1.5B）；
   第一档基座（或定版 0.5B，不在档位表中）即链底。
"""

from __future__ import annotations

from tunefield.engine.recommender import VRAM_TIERS

_MIN_CUTOFF = 256


def next_degrade(cfg: dict) -> tuple[dict, str] | None:
    """返回 (降级后的 cfg, 说明)；无可用降级档返回 None。"""
    c = dict(cfg)

    # 1) 缩短序列：cutoff_len 减半（每次调用都基于当前值继续减，直到下限）
    cutoff = int(c.get("cutoff_len") or 0)
    half = cutoff // 2
    if half >= _MIN_CUTOFF:
        c["cutoff_len"] = half
        return c, f"缩短序列 cutoff_len {cutoff}→{half}"

    # 2) 降低 QLoRA 量化位宽：4 → 3 → 2 bit
    quant = int(c.get("quantization_bit") or 0)
    if quant > 2:
        c["quantization_bit"] = quant - 1
        return c, f"降低量化位宽 {quant}→{quant - 1}bit"

    # 3) 降低基座规模：沿档位表向更小一档移动
    base = str(c.get("base_model") or "")
    for i, tier in enumerate(VRAM_TIERS):
        if tier["base"] == base and i > 0:
            smaller = VRAM_TIERS[i - 1]["base"]
            c["base_model"] = smaller
            return c, f"降低基座规模 {base} → {smaller}"
    return None  # 已在第一档/定版轻量基座，无更小可降
