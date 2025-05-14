import json
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import glob


DATASET_NAME = "tangshi"
DATA_DIR = "./dataset"
MODEL_NAME = "Qwen3-0.6B"
OUTPUT_DIR = f"./{MODEL_NAME}-lora-{DATASET_NAME}"
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


# 改进的数据加载函数
def load_dataset_files(dir_path):
    # 匹配所有符合模式的JSON文件
    file_pattern = f"{dir_path}/poet.tang.*.json"
    data_files = glob.glob(file_pattern)

    all_data = []
    for file_path in data_files:
        with open(file_path, "r", encoding="utf-8") as f:
            # 合并所有文件内容
            all_data.extend(json.load(f))

    # 过滤无效数据（可选）
    valid_data = [
        d
        for d in all_data
        if "paragraphs" in d and len(d["paragraphs"]) > 0 and "author" in d
    ]
    return valid_data


# 改进的数据处理函数（增加分词步骤）
def process_data(example):
    instruction = "请根据以下信息创作一首唐诗："
    content = f"作者：{example['author']}\n 标题：{example['title']}"
    poem = "\n".join(example["paragraphs"])

    # 构建完整文本（指令+输入+输出）
    full_text = f"{instruction}\n{content}\n{poem}"

    # 直接返回文本，后续由tokenizer处理
    return {"text": full_text}


# 加载并处理数据集
print("正在加载数据集...")
all_data = load_dataset_files(DATA_DIR)
raw_dataset = Dataset.from_list(all_data)

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 数据预处理流水线
processed_dataset = raw_dataset.map(
    process_data, remove_columns=raw_dataset.column_names
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
        add_special_tokens=True,
    )


tokenized_dataset = processed_dataset.map(
    tokenize_function, batched=True, batch_size=1000, remove_columns=["text"]
).train_test_split(test_size=0.1)

print("数据集统计：")
print(f"- 总样本数：{len(all_data)}")
print(f"- 训练集：{len(tokenized_dataset['train'])}")
print(f"- 验证集：{len(tokenized_dataset['test'])}")


train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

device_map = "auto"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, device_map=device_map
)
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


print("start")
epochs = 5
lr = 1e-4
batch_size = 16
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_steps=50,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True,
    gradient_accumulation_steps=2,  # 减少显存压力
    weight_decay=0.01,  # 添加权重衰减
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # 根据验证损失选择最佳模型
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因为是语言模型，所以不使用mlm
    ),
    # device_map=device_map,
)

trainer.train()

# 保存模型
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


# 修改后的推理函数（更稳定的生成方式）
def generate_poem(author, tags, title):
    prompt = f"请根据以下信息创作一首唐诗：\n作者：{author}\n标签：{','.join(tags)}\n标题：{title}"

    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to("cuda")

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 提取生成部分（排除输入提示）
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output[len(prompt) :]  # 只返回生成的诗歌部分


# 示例生成
print(generate_poem(author="李白", tags=["战争"], title="出塞"))
