import random
import re

import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def build_target_list(gt_list, pred_list, n=5):
    # Use a set to track (doc_id, line_id) to prevent duplicates
    seen = {(item["doc_id"], item["line_id"]) for item in gt_list}

    # Start with all gt items
    target_list = gt_list[:]

    for item in pred_list:
        key = (item["doc_id"], item["line_id"])
        if key not in seen:
            target_list.append(item)
            seen.add(key)
        if len(target_list) >= n:
            break

    return target_list


def build_prompt(
    claim, evidence, pred_lines, shuffle_evidence=True, output_w_reason=True
):
    target_list = build_target_list(evidence, pred_lines)
    if shuffle_evidence:
        random.shuffle(target_list)
    joined_evidence = [
        f'[{idx}] [{evid["doc_id"]}] {evid["text"]}'
        for idx, evid in enumerate(target_list)
    ]
    joined_evidence = "\n".join(joined_evidence)
    prompt = [
        "你是一位推理專家。",
        "根據下列論述與證據，請判斷該論述的真實性，分為 SUPPORTS（支持）、REFUTES（反駁）、NOT ENOUGH INFO（資訊不足）。",
    ]
    if output_w_reason:
        prompt.append("並簡要說明你的判斷理由。")
    prompt += [
        f"\n論述：{claim}",
        f"證據：\n{joined_evidence}",
        "用以下格式回答：",
        "label: <SUPPORTS/REFUTES/NOT ENOUGH INFO>",
    ]
    if output_w_reason:
        prompt.append("reason: <你的理由>\n")
    prompt = "\n".join(prompt)
    return prompt


class ReasonerSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_list,
        tokenizer,
        max_length=512,
        output_w_reason=True,
        debug=False,
        shuffle_evidence=True,
    ):
        self.samples = jsonl_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.output_w_reason = output_w_reason
        self.shuffle_evidence = shuffle_evidence
        self.debug = debug

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = build_prompt(
            item["claim"],
            item["evidence"],
            item.get("pred_lines"),
            output_w_reason=self.output_w_reason,
            shuffle_evidence=self.shuffle_evidence,
        )
        label = item["label"].upper().strip()
        # Combine label + reason if present in the data
        target = ""
        if self.output_w_reason:
            reason = item.get("reason", "").strip()
            reason = re.sub(
                r"<[^>]*>", "", reason
            )  # replace for distillation reason <string>
            target = f"label: {label}\nreason: {reason}"
        else:
            target = f"label: {label}"
        full_text = f"{prompt}\n{target}"

        # Tokenize the whole text
        encoding = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        # Tokenize prompt to get its length in tokens
        prompt_enc = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding["prompt_enc_ids"] = prompt_enc["input_ids"].squeeze(0)
        encoding["prompt_attention_mask"] = prompt_enc["attention_mask"].squeeze(0)

        # Build labels: -100 for prompt & padding, otherwise value
        target_enc = self.tokenizer(
            target, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        target_len = target_enc["input_ids"].shape[-1]
        labels = encoding["input_ids"].clone()
        # Mask out prompt (all tokens BEFORE target)
        labels[0:-target_len] = -100
        # labels[pad_mask] = -100
        encoding["labels"] = labels

        # Optional debug print
        if getattr(self, "debug", False) and idx < 3:
            print(f"[DEBUG idx={idx}]\n{full_text}\n{'='*40}")
            print("input_ids:", encoding["input_ids"])
            print(
                self.tokenizer.decode(encoding["input_ids"], skip_special_tokens=True)
            )
            print("labels:", encoding["labels"])
            temp_label = np.where(labels == -100, self.tokenizer.pad_token_id, labels)
            print(self.tokenizer.decode(temp_label, skip_special_tokens=True))
            print("prompt_enc_ids:", encoding["prompt_enc_ids"])
            print(
                self.tokenizer.decode(
                    encoding["prompt_enc_ids"], skip_special_tokens=True
                )
            )

        return encoding

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    import json

    model_dir = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=True, padding_side="left", pad_to_max_length=True
    )
    with open("data/train_pred_qwen3_lpert_NEI.jsonl") as f:
        d = [json.loads(l) for l in f]
    dataset = ReasonerSFTDataset(
        d[:20], tokenizer, max_length=768, output_w_reason=False, debug=True
    )
    print(dataset.__getitem__(0))
    # dataset = ReasonerSFTEvalDataset(d[:20], tokenizer,max_length=512, output_w_reason=False,debug=True)
    # print(dataset.__getitem__(0))
    # dataset = ReasonerSFTDataset(d[:20], tokenizer,max_length=768, output_w_reason=True,debug=False,mode='eval')
    # print(dataset.__getitem__(0))
