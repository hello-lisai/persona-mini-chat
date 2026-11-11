# persona-chat — 用 GitHub Actions 免费训练一个"记住一个人"的聊天模型

底模：[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) + LoRA，纯 CPU 训练。

虚拟人物：**林小夏（Summer）** —— 24 岁成都人，现居上海徐汇，全职游戏主播，主玩 GTA 速通，养了只叫"炸酱面"的橘猫。

## 使用方法

1. 编辑 `data/train.jsonl`：每行一条对话，格式 `{"system": "...", "user": "...", "assistant": "..."}`
2. push 到 main 分支 → Actions 自动训练（约 1~2 小时）
3. 训练完成后在 Actions 页面查看 `chat_report.txt`（训练前后回答对比），
   或在 Artifacts 下载 `persona-adapter-and-report`（LoRA 适配器 + 报告）

## 本地加载训练好的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype="float32")
model = PeftModel.from_pretrained(model, "output/adapter")
```

## 文件说明

- `train.py` — LoRA 训练脚本（r=16，15 epochs，Adafactor/AdamW，CPU 可跑）
- `test_chat.py` — 底模 vs 微调模型对比问答（10 个问题，含 1 道干扰题）
- `.github/workflows/train.yml` — 自动训练流水线
