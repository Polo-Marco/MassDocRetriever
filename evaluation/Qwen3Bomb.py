import time

from llama_cpp import Llama

# Path to your downloaded GGUF model
model_path = "./models/qwen3-235/Qwen3-235B-A22B-Q4_K_M.gguf"

# Init Llama with GGUF model
llm = Llama(
    model_path=model_path,
    n_gpu_layers=60,  # Adjust for your GPU VRAM
    n_ctx=1024,
    use_mlock=False,
    verbose=False,
    enable_thinking=True,
)


prompt = "Give me a complete introduction to large language models."

start_time = time.time()
output = llm(
    prompt,
    max_tokens=512,
    stop=["</s>"],
    # temperature=0.7,
    # top_p=0.8,
    # top_k=20,
    # min_p=0,
    # presence_penalty=1.5,
    temperature=0.6,  # thinking recommendation
    top_p=0.95,
    top_k=20,
    min_p=0,
    presence_penalty=1.7,
)
end_time = time.time()
# print(output)
print("Generated output:", output["choices"][0]["text"])
print(f"Inference time: {end_time - start_time:.2f} seconds")
