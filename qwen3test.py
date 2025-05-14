from vllm import LLM, SamplingParams

# 指定本地模型路径
model_path = "./Qwen3-1.7B"

# 初始化模型
llm = LLM(
    model=model_path,  # 本地模型路径
    tokenizer=model_path,  # 分词器路径（与模型相同目录时可省略）
    dtype="auto",  # 自动选择数据类型（如半精度）
    tensor_parallel_size=4,  # 单GPU（多GPU可调整）
)

# 定义生成参数
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=32768,
)

# 生成文本
prompts = ["向我介绍什么是大语言模型？"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
    print("===" * 20)
