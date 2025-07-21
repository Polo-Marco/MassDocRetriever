import gc
import json

import bitsandbytes as bnb
import numpy as np
import torch
from datasets import Dataset, load_dataset
from metrics import preprocess_logits_for_metrics
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

import wandb
from data.collator import ReasonerCollator
from data.dataset import ReasonerSFTDataset

# Load and split your jsonl
with open("data/train_pred_qwen3_lpert_NEI.jsonl") as f:
    data = [json.loads(l) for l in f]
train, dev = train_test_split(data, random_state=42, test_size=0.1)
# Prepare tokenizer & model (int8)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_dir = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, use_fast=True, padding_side="left", pad_to_max_length=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_dir, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
# def find_all_linear_names(model):
#     cls = bnb.nn.Linear4bit
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, cls):
#             names = name.split('.')
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])
#     if 'lm_head' in lora_module_names:  # needed for 16 bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)

# modules = find_all_linear_names(base_model)
# PEFT config
lora_config = LoraConfig(
    r=8,
    lora_alpha=6,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(base_model, lora_config)  # fundation model
# fine tuned model
# model = PeftModel.from_pretrained(base_model, "./models/qwen3_8b_reasoner_labelonly_ckpt1/checkpoint-722")

# release base model memory
del base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
# Data and collator
train_dataset = ReasonerSFTDataset(
    train, tokenizer, max_length=768, output_w_reason=True, shuffle_evidence=True
)
val_dataset = ReasonerSFTDataset(
    dev, tokenizer, max_length=768, output_w_reason=True, shuffle_evidence=False
)
# inference_dataset = ReasonerSFTEvalDataset(dev, tokenizer,max_length=512,output_w_reason=False)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# inference first
from metrics import (F1ReasonerMetric,  # Use your provided implementations
                     parse_label)


@torch.no_grad()
def predict_batch(dataset, model, tokenizer, batch_size=4, max_new_tokens=256):
    model.eval()
    device = model.device
    pred_ids = []
    label_ids = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        input_ids = torch.stack([b["prompt_enc_ids"] for b in batch]).to(device)
        attention_mask = torch.stack([b["prompt_attention_mask"] for b in batch]).to(
            device
        )
        labels = (
            torch.stack([b["labels"] for b in batch]).cpu().numpy()
        )  # shape: (batch, seq_len)
        label_ids.append(labels)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            top_k=20,
            min_p=0,
            do_sample=True,  # Important: this enables non-greedy, stochastic decoding
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        pred_ids.append(outputs.cpu().numpy())

    # Concatenate all batches
    pred_ids = np.concatenate(pred_ids, axis=0)
    label_ids = np.concatenate(label_ids, axis=0)
    return pred_ids, label_ids


# ---- Run Inference ----
eval_metrics = F1ReasonerMetric(tokenizer)

# pred_ids, label_ids = predict_batch(val_dataset, model, tokenizer,batch_size=100, max_new_tokens=256)
# # # Compute F1 (uses your tokenizer, which must be in scope)
# metric = eval_metrics.__call__((pred_ids, label_ids))
# print("=== Zero-Shot Macro F1 ===")
# print(metric)
# exit()
# Training arguments
training_args = TrainingArguments(
    output_dir="./models/qwen3_8b_reasoner_reason_ckpt1",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    # gradient_checkpointing=True, #error don't use
    num_train_epochs=50,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    fp16=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=0.01,  # Or calculate 0.2 * num_steps
    save_steps=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    report_to=["wandb"],
    run_name="Qwen3-8B-reason",
    remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # data_collator=collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    # compute_metrics=eval_metrics,
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

wandb.init(
    project="Claim_verify_w_reason_zh_TW",  # ✅ Set your project name
    name="Qwen3-8B-reason2",  # ✅ Set the run name
    config=training_args,  # ✅ Log hyperparameters
)
# Start
trainer.train()

# # Evaluate
pred_ids, label_ids = predict_batch(
    val_dataset, model, tokenizer, batch_size=100, max_new_tokens=256
)
# Compute F1 (uses your tokenizer, which must be in scope)
metric = eval_metrics.__call__((pred_ids, label_ids))
print("=== After Fine tuning Macro F1 ===")
print(metric)
