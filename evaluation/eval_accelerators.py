import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_id = "Qwen/Qwen1.5-1.8B"
prompt = "A quick brown fox jumps over the lazy dog."
batch_size = 100
max_new_tokens = 512

# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True, truncation=True).to("cuda")

def benchmark(tag, model, use_amp):
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    print(f"{tag:35s} → {time.time() - start:.3f} sec")

# FlashAttention2 model
model_flash = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    device_map="auto",
)
model_flash.config.pad_token_id = tokenizer.pad_token_id

# Standard attention model
model_std = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    trust_remote_code=True,
    device_map="auto",
)
model_std.config.pad_token_id = tokenizer.pad_token_id

# Optional: DeepSpeed inference
try:
    import deepspeed
    ds_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    ds_model = deepspeed.init_inference(ds_model, dtype=torch.float16, replace_method="auto").module
    ds_model.config.pad_token_id = tokenizer.pad_token_id
except ImportError:
    ds_model = None

# Warm-up
with torch.no_grad():
    model_flash.generate(**inputs, max_new_tokens=16)
    model_std.generate(**inputs, max_new_tokens=16)
    if ds_model:
        ds_model.generate(**inputs, max_new_tokens=16)

# Run benchmarks
print("\nBenchmarking all combinations:")
benchmark("FlashAttention2 + AMP", model_flash, use_amp=True)
benchmark("FlashAttention2 (no AMP)", model_flash, use_amp=False)
benchmark("Standard Attention + AMP", model_std, use_amp=True)
benchmark("Standard Attention (no AMP)", model_std, use_amp=False)
if ds_model:
    benchmark("DeepSpeed Inference", ds_model, use_amp=False)