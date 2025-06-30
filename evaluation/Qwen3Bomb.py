import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

model_id = "Qwen/Qwen3-32B"

# Use 4-bit quantization for efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Try device_map "auto" or specify if you want to force everything on GPU/CPU
device_map = "auto"  # or try {"": "cuda:0"} or {"": "cpu"} for troubleshooting

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Loading model (may take several minutes)...")
start_load = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_folder="offload_dir",  # needed if part of model is offloaded to CPU
)
load_time = time.time() - start_load
print(f"Model loaded in {load_time:.1f} seconds.")

# Your test prompt
prompt = "Q: What is artificial intelligence?\nA:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Starting inference...")
start_infer = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )
end_infer = time.time()
infer_time = end_infer - start_infer

output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n--- Inference Result ---")
print(output_text)
print(f"\nInference time for 64 tokens: {infer_time:.2f} seconds")
print(f"Speed: {64/infer_time:.2f} tokens/sec")
