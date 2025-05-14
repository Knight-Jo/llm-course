from datasets import load_dataset
import re
from transformers import AutoTokenizer
import pandas as pd
import ftfy  # 用于修复常见的Unicode问题

ds = load_dataset("mikasenghaas/wikitext-2")


def clean_wikitext(examples):
    cleaned_texts = []

    for text in examples["text"]:
        # 1. 基础清洗 删除空行
        text = text.strip()
        if not text:
            continue

        # 2. 修复Unicode字符问题（使用ftfy库）
        text = ftfy.fix_text(text)

        # 3. 过滤维基百科结构化内容
        # 匹配标题行（包含不同层级的标题 = Title == Subtitle === Section ====）
        if re.match(r"^\s*=+\s.*\s=+\s*$", text):
            continue
        # 过滤列表项、导航模板
        if text.startswith(("* ", "# ", "{{", "}}", "|-")):
            continue
        # 过滤文件链接和分类标记
        if re.search(r"\[\[(File|Category):", text):
            continue

        # 4. 清理维基标记语法
        # 移除内部链接标记（保留链接文字）
        text = re.sub(
            r"\[\[([^\]|]+)\|?([^\]]+)?\]\]", lambda m: m.group(2) or m.group(1), text
        )
        # 移除模板
        text = re.sub(r"\{\{.*?\}\}", "", text)
        # 移除引用标记
        text = re.sub(r"<ref.*?</ref>", "", text, flags=re.DOTALL)

        # 5. 文本规范化
        # 合并多个换行/空格
        text = re.sub(r"\s+", " ", text)
        # 移除特殊字符（保留常见标点）
        text = re.sub(r'[^\w\s.,!?\'"-—–@$%&*+/:;()]', "", text)
        # 标准化引号
        text = (
            text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        )

        # 6. 内容质量过滤
        # 过滤过短/无意义的句子
        if len(text) < 25:
            continue
        # 过滤纯数字内容
        if re.fullmatch(r"\d+[\d,\.%\s]*", text):
            continue

        # 7. 合并短行（避免输入序列断裂）
        if len(text) < 50 and len(cleaned_texts) > 0:
            cleaned_texts[-1] += " " + text.strip()
        else:
            cleaned_texts.append(text.strip())
        cleaned_texts.append(text)

    return {"text": cleaned_texts}


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        return_tensors="pt",
        padding="max_length",
    )
    # 分割输入和目标（前 256 tokens 为输入，后 256 tokens 为目标）
    inputs = {k: v[:, :256] for k, v in tokenized.items()}
    labels = {k: v[:, 256:] for k, v in tokenized.items()}
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"],
    }


# 应用清洗
train_data = ds["train"].map(clean_wikitext, batched=True)
val_data = ds["validation"].map(clean_wikitext, batched=True)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 分词与格式化（输入为前 256 tokens，目标为后 256 tokens）
# 应用分词
tokenized_train = train_data.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
tokenized_val = val_data.map(tokenize_function, batched=True, remove_columns=["text"])
