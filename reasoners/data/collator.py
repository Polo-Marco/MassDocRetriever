import random

import torch
from torch.nn.utils.rnn import pad_sequence


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


def build_prompt(claim, evidence, pred_lines, shuffle_evidence=True):
    target_list = build_target_list(evidence, pred_lines)
    if shuffle_evidence:
        random.shuffle(target_list)
    joined_evidence = [
        f'[{idx}] [{evid["doc_id"]}] {evid["text"]}'
        for idx, evid in enumerate(target_list)
    ]
    joined_evidence = "\n".join(joined_evidence)
    prompt = (
        "你是一位推理專家。\n"
        "根據下列論述與證據，請判斷該論述的真實性，分為 SUPPORTS（支持）、REFUTES（反駁）、NOT ENOUGH INFO（資訊不足）。"
        f"論述：{claim}\n"
        f"證據：\n{joined_evidence}\n"
        "用以下格式回答：\n"
        "label: <SUPPORTS/REFUTES/NOT ENOUGH INFO>\n"
    )
    return prompt


class ReasonerCollator:
    def __init__(self, tokenizer, max_length=512, label2id=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

    def __call__(self, batch):
        # Compose prompt for each sample
        prompts = [
            build_prompt(ex["claim"], ex["evidence"], ex["pred_lines"]) for ex in batch
        ]
        encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(
            [self.label2id[ex["label"]] for ex in batch], dtype=torch.long
        )
        encodings["labels"] = labels
        return encodings


if __name__ == "__main__":
    import json

    with open("data/train_pred_qwen3_lpert.jsonl") as f:
        d = [json.loads(l) for l in f]
    data = d[0]
    print(data)
    print(build_prompt(data["claim"], data["evidence"], data["pred_lines"]))
