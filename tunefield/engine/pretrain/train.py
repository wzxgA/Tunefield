"""T13 · 从零预训练入口（引擎 B，由 runner 以子进程调用）。

用法：python train.py <cfg.json>

cfg.json 由 PretrainEngine.prepare 生成，包含已解析的结构尺寸与超参：
  corpus / output_dir / hidden_size / intermediate_size / num_hidden_layers /
  num_attention_heads / seq_len / epochs / learning_rate / per_device_batch_size /
  gradient_accumulation_steps / logging_steps / save_steps / seed / resume(可选)

设计（对齐 B4.3）：
- 语料：build 产出的纯文本 corpus.txt（不走指令化），整文 tokenize 后切
  定长窗口，Qwen 中文词表保证中文建模能力；
- 模型：随机初始化的 Qwen2 结构（tokenizer 取自定版 Qwen 基座），训练产物是
  完整权重目录（含 config.json），直接可被 GGUF 导出链消费；
- 日志：transformers Trainer 逐 logging_steps 打 loss/epoch（stderr 与 stdout
  合并由 BaseEngine.run 逐行采样），含 step/epoch 以便进度与曲线实时推进；
- 续跑：resume 字段指向 checkpoint 目录，Runner 检测到新 checkpoint 自动带上。

本文件不 import torch/transformers 之外的平台代码；仅作子进程入口。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _build_dataset(ids: list[int], seq_len: int, pad_id: int):
    """把长 token 序列切成定长窗口数据集（torch Dataset）。"""
    import torch

    windows: list[list[int]] = []
    for i in range(0, len(ids), seq_len):
        win = ids[i:i + seq_len]
        if len(win) < seq_len:
            win = win + [pad_id] * (seq_len - len(win))
        windows.append(win)
    data = torch.tensor(windows, dtype=torch.long)

    class _Windows(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(data)

        def __getitem__(self, idx: int):
            x = data[idx]
            return {"input_ids": x, "attention_mask": torch.ones_like(x),
                    "labels": x.clone()}

    return _Windows()


def main(cfg: dict) -> int:
    import torch
    from transformers import (
        AutoTokenizer, DataCollatorForLanguageModeling,
        Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments,
    )

    tokenizer_id = cfg.get("tokenizer") or "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[pretrain] 加载词表：{tokenizer_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    corpus_path = Path(cfg["corpus"])
    if not corpus_path.exists():
        print(f"[pretrain] 语料缺失：{corpus_path}（请先 tunefield build）", flush=True)
        return 2
    text = corpus_path.read_text(encoding="utf-8-sig")
    if not text.strip():
        print("[pretrain] 语料为空，无样本可训练", flush=True)
        return 2
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    print(f"[pretrain] 语料 {len(text)} 字符 · 共 {len(ids)} token", flush=True)

    # 特殊 token 常是 added token，其 id 可能 ≥ tokenizer.vocab_size
    # （Embedding 索引越界 → "Padding_idx must be within num_embeddings"；
    # 且 llama.cpp 转换要求 max(tokenizer.vocab.values()) < config.vocab_size）。
    # 词表大小取 max(报告值, 全部 token 的最大 id + 1)，含特殊 token 一并覆盖；
    # pad 沿用 eos。
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = eos if eos is not None else (bos if bos is not None else 0)
    special_ids = [x for x in (bos, eos, tokenizer.pad_token_id,
                               tokenizer.unk_token_id) if x is not None]
    try:
        token_max = max(tokenizer.vocab.values()) if tokenizer.vocab else 0
    except Exception:  # 个别 tokenizer 无 vocab 映射：退化为只按特殊 token
        token_max = 0
    vocab_size = max(tokenizer.vocab_size, *(x + 1 for x in special_ids),
                     token_max + 1)
    if pad >= vocab_size:  # 防御：保证 pad 落在词表内
        pad = vocab_size - 1

    seq_len = int(cfg["seq_len"])
    dataset = _build_dataset(ids, seq_len, pad)

    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=int(cfg["hidden_size"]),
        intermediate_size=int(cfg["intermediate_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        num_key_value_heads=int(cfg["num_attention_heads"]),
        max_position_embeddings=max(seq_len, 4096),
        pad_token_id=pad,
        bos_token_id=bos,
        eos_token_id=eos,
    )
    model = Qwen2ForCausalLM(config)
    print(
        f"[pretrain] 模型 {cfg.get('scale', '?')} · 参数量 "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
        flush=True,
    )

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.get_device_capability(0)[0] >= 8
    out_dir = str(cfg["output_dir"])
    args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=int(cfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        seed=int(cfg["seed"]),
        fp16=use_cuda and not bf16,
        bf16=bf16,
        report_to=[],
        logging_dir=None,
        dataloader_pin_memory=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    resume = cfg.get("resume")
    if resume:
        print(f"[pretrain] 断点续训：{resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume or None)

    # 保存完整权重（config.json + safetensors），供 GGUF 导出链直接消费
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[pretrain] 完整权重已保存：{out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    cfg_file = sys.argv[1]
    try:
        with Path(cfg_file).open(encoding="utf-8-sig") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[pretrain] 配置读取失败：{exc}", flush=True)
        raise SystemExit(2)
    raise SystemExit(main(payload))
