"""路径常量与全局配置。

约定：data/ 与 models/ 落在仓库根目录（单命令直启、无云端依赖的部署形态）。
可用环境变量 TUNEFIELD_ROOT 覆盖根目录（供测试隔离使用）。
"""

from __future__ import annotations

import os
from pathlib import Path

# 仓库根：tunefield/config.py → tunefield/ → 仓库根
PROJECT_ROOT = Path(
    os.environ.get("TUNEFIELD_ROOT", Path(__file__).resolve().parents[1])
)

# 流程内合并数据集（画布合并节点拼接产物）的保留开关：
# 默认关闭——run 到达终态即回收；置 1 用于调试（跑完不删，便于查看 raw/产物）。
KEEP_MERGED = os.environ.get("TUNEFIELD_KEEP_MERGED", "") not in ("", "0", "false", "False")

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # 原始上传（按内容哈希分目录）
DATASETS_DIR = DATA_DIR / "datasets"  # 管线产物（JSONL + 质检报告）
DB_PATH = DATA_DIR / "tunefield.db"   # SQLite

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"
BASES_DIR = MODELS_DIR / "bases"      # 基座权重（HuggingFace 缓存外置亦可）
ADAPTERS_DIR = MODELS_DIR / "adapters"  # LoRA 权重，目录即模型
GGUF_DIR = MODELS_DIR / "gguf"        # 量化产物

# 前端构建产物（F1 由 StaticFiles 托管）
WEB_DIST_DIR = PROJECT_ROOT / "web" / "dist"


def ensure_dirs() -> None:
    """创建运行期目录（幂等）。"""
    for d in (RAW_DIR, DATASETS_DIR, BASES_DIR, ADAPTERS_DIR, GGUF_DIR):
        d.mkdir(parents=True, exist_ok=True)
