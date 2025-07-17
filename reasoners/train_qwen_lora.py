import gc
import json

import numpy as np
import torch
from datasets import Dataset, load_dataset
from metrics import preprocess_logits_for_metrics
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

import wandb
from data.collator import ReasonerCollator
from data.dataset import ReasonerSFTDataset

# Load and split your jsonl
with open("data/train_pred_qwen3_lpert.jsonl") as f:
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
# PEFT config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(base_model, lora_config)
# release base model memory
del base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
# Data and collator
train_dataset = ReasonerSFTDataset(train, tokenizer)
val_dataset = ReasonerSFTDataset(dev, tokenizer)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
# inference first
from metrics import (F1ReasonerMetric,  # Use your provided implementations
                     parse_label)


@torch.no_grad()
def predict_batch(dataset, model, tokenizer, batch_size=4, max_new_tokens=64):
    model.eval()
    device = model.device
    pred_ids = []
    label_ids = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
        attention_mask = torch.stack([b["attention_mask"] for b in batch]).to(device)
        labels = (
            torch.stack([b["labels"] for b in batch]).cpu().numpy()
        )  # shape: (batch, seq_len)
        label_ids.append(labels)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.2,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        pred_ids.append(outputs.cpu().numpy())

    # Concatenate all batches
    pred_ids = np.concatenate(pred_ids, axis=0)
    label_ids = np.concatenate(label_ids, axis=0)
    return pred_ids, label_ids


# ---- Run Inference ----

# pred_ids, label_ids = predict_batch(val_dataset, model, tokenizer, batch_size=4, max_new_tokens=64)
eval_metrics = F1ReasonerMetric(tokenizer)
# # Compute F1 (uses your tokenizer, which must be in scope)
# metric = eval_metrics.__call__((pred_ids, label_ids))
# print("=== Zero-Shot Macro F1 ===")
# print(metric)
# exit()
# Training arguments
training_args = TrainingArguments(
    output_dir="./models/qwen3_8b_reasoner_labelonly_ckpt",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Simulate 16 batch
    num_train_epochs=4,
    learning_rate=2e-5,
    fp16=False,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=0.05,  # Or calculate 0.2 * num_steps
    save_steps=0.05,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    logging_steps=10,
    report_to=["wandb"],
    run_name="Qwen3-8B-labelonly",
    remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=eval_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

# Start
trainer.train()
