gpu=1
model = ./Qwen3-0.6B

server:
	python -m vllm.entrypoints.openai.api_server \
	--model $(model) \
	--port 8000 \
	--dtype auto \
	--trust-remote-code \
	--tensor-parallel-size 2

