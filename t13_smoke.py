"""一次性冒烟：引擎 B 训练 → llama.cpp 转换 f16（验证词表与转换校验）。"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tunefield.engine import exporter as exp  # noqa: E402

print("converter:", exp._converter_script(), flush=True)

root = Path(tempfile.mkdtemp(prefix="t13conv-"))
(root / "corpus.txt").write_text(
    "Tunefield 领域专属模型平台预训练转换冒烟语料。" * 40 + "结束。", encoding="utf-8"
)
cfg = {
    "corpus": str(root / "corpus.txt"),
    "output_dir": str(root / "out"),
    "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "seq_len": 96,
    "epochs": 1,
    "learning_rate": 1e-3,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "logging_steps": 1,
    "save_steps": 200,
    "seed": 42,
}
cfg_path = root / "cfg.json"
cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
train_py = Path(__file__).resolve().parent / "tunefield" / "engine" / "pretrain" / "train.py"
env = {**os.environ, "PYTHONUNBUFFERED": "1"}

r1 = subprocess.run([sys.executable, str(train_py), str(cfg_path)],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", env=env)
print("TRAIN rc =", r1.returncode, flush=True)
if r1.returncode != 0:
    print(r1.stdout[-1200:], r1.stderr[-1200:])
    raise SystemExit(r1.returncode)
assert (root / "out" / "config.json").exists()

out_gguf = root / "out.gguf"
r2 = subprocess.run([sys.executable, str(exp._converter_script()),
                     str(root / "out"), "--outfile", str(out_gguf),
                     "--outtype", "f16"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", env=env)
print("CONVERT rc =", r2.returncode, "| gguf exists =", out_gguf.exists(), flush=True)
if r2.returncode != 0:
    print(r2.stdout[-2000:])
    print("STDERR:", r2.stderr[-1000:])
raise SystemExit(r2.returncode)
