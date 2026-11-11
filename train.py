#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU LoRA 微调：让 Qwen3-0.6B 记住一个虚拟人物「林小夏」"""
import json
import os
import time

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
DATA_PATH = "data/train.jsonl"
OUT_DIR = "output"
MAX_LEN = 512


class ChatDataset(Dataset):
    def __init__(self, path, tok):
        self.tok = tok
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
            {"role": "assistant", "content": r["assistant"]},
        ]
        full = self.tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        prompt = self.tok.apply_chat_template(
            msgs[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        # 只对 assistant 回答部分计算损失：屏蔽 full 与 prompt 的公共前缀
        n = 0
        for a, b in zip(full, prompt):
            if a != b:
                break
            n += 1
        labels = [-100] * n + list(full[n:])
        return {
            "input_ids": full[:MAX_LEN],
            "labels": labels[:MAX_LEN],
            "attention_mask": [1] * min(len(full), MAX_LEN),
        }


def collate(batch, tok):
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids, labels, attn = [], [], []
    for b in batch:
        p = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad] * p)
        labels.append(b["labels"] + [-100] * p)
        attn.append(b["attention_mask"] + [0] * p)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main():
    print(f"loading tokenizer/model: {MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = ChatDataset(DATA_PATH, tok)
    print(f"training samples: {len(ds)}", flush=True)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=15,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=42,
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=lambda b: collate(b, tok))

    t0 = time.time()
    trainer.train()
    print(f"train done in {(time.time() - t0) / 60:.1f} min", flush=True)

    model.save_pretrained(os.path.join(OUT_DIR, "adapter"))
    tok.save_pretrained(os.path.join(OUT_DIR, "adapter"))
    print("adapter saved to output/adapter", flush=True)


if __name__ == "__main__":
    main()
