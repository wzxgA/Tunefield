"""T12 · 真实数据验收自动化骨架：三形态样例 + 双通道产物一致性。

本测试在纯代码层落地「同批数据 CLI/Web 双通道一致」的可重复验收
（无需 GPU）；真机 GPU 训练→导出→Ollama 对话的实测步骤见 plans v1.19 验收清单。
"""

from __future__ import annotations

import pytest

from acceptance_t12.corpus import build_corpus
from acceptance_t12.dual import run_dual_check


@pytest.fixture(scope="module")
def t12_root(tmp_path_factory):
    return tmp_path_factory.mktemp("t12acc")


def test_corpus_three_forms(t12_root):
    corpus = build_corpus(t12_root / "corpus")
    ext = {p.suffix for p in corpus.rglob("*") if p.is_file()}
    assert {".txt", ".md", ".pdf", ".docx", ".py", ".js"} <= ext
    assert (corpus / "broken.pdf").exists()  # 坏文件验证错误态
    # 平铺语义：目录里只有文件（与 Web 上传的文件集合一致）
    assert not any(p.is_dir() for p in corpus.iterdir())


def test_dual_channel_products_consistent(t12_root):
    corpus = build_corpus(t12_root / "corpus")
    cli_root = t12_root / "cli"
    web_root = t12_root / "web"
    result = run_dual_check(corpus, name="t12dual", roots=(cli_root, web_root))
    names = [c["name"] for c in result["checks"]]
    assert names, "核对项不应为空"
    assert result["ok"], result["checks"]
    assert "样本" in result.get("note", "")
