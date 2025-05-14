from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

if __name__ == "__main__":
    device = "balanced_low_0"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="Qwen3-0.6B",
        help="Model name or path",
    )
    parser.add_argument(
        "-l",
        "--lora",
        type=str,
        default="",
        help="Lora name or path",
    )
    args = parser.parse_args()

    model_path = args.model
    lora_path = args.lora
    lora_path = "Qwen3-0.6B-lora-tangshi"
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, trust_remote_code=True
    )
    if lora_path:
        model = PeftModel.from_pretrained(
            model,
            model_id=lora_path,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

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

    print(generate_poem(author="李白", tags=["战争"], title="出塞"))
