#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试虚拟人物记忆效果：底模 vs LoRA 微调后 对比问答"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
ADAPTER = "output/adapter"

SYSTEM = "你是林小夏，一位个性鲜明的虚拟角色。请始终以林小夏的身份、性格和口吻回答问题。"

QUESTIONS = [
    "你叫什么名字？这个名字有什么来历？",
    "你多大了？生日是哪天？",
    "你是哪里人？现在住在哪个城市？",
    "你是做什么工作的？",
    "你养的猫叫什么名字？为什么叫这个名字？",
    "你的口头禅是什么？",
    "你喝咖啡有什么习惯？最讨厌吃什么？",
    "你最喜欢的游戏是什么？速通水平怎么样？",
    "你的粉丝团叫什么名字？",
    "今天上海天气怎么样？",  # 干扰题：检查通用能力是否还在
]


def chat(model, tok, question, max_new_tokens=120):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]
    ids = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", enable_thinking=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.7, top_p=0.9, repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    print("loading base model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

    print("\n" + "=" * 70, flush=True)
    print("【训练前】原始底模回答", flush=True)
    print("=" * 70, flush=True)
    for q in QUESTIONS:
        ans = chat(model, tok, q)
        print(f"\nQ: {q}\nA: {ans}", flush=True)

    print("\nloading adapter...", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    print("\n" + "=" * 70, flush=True)
    print("【训练后】LoRA 微调模型回答", flush=True)
    print("=" * 70, flush=True)
    for q in QUESTIONS:
        ans = chat(model, tok, q)
        print(f"\nQ: {q}\nA: {ans}", flush=True)


if __name__ == "__main__":
    main()
