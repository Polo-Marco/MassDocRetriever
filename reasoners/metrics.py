import re

import numpy as np
import torch
from sklearn.metrics import f1_score


def preprocess_logits_for_metrics(logits, labels):
    # Handles both tensors and numpy arrays
    if isinstance(logits, tuple):
        logits = logits[0]
    # For sequence classification: (batch, num_classes)
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def parse_label(text):
    # Extracts label: ... from generated output
    import re

    m = re.search(r"label\s*[:：]\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)", text, re.I)
    return m.group(1).upper() if m else "NOT ENOUGH INFO"


class F1ReasonerMetric:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.label_list = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    def __call__(self, eval_preds):
        preds, labels = eval_preds
        # Replace -100 in labels with pad_token_id for decoding
        labels = np.where(labels == -100, self.tokenizer.pad_token_id, labels)
        pred_strs = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_strs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_labels = [parse_label(x) for x in pred_strs]
        gold_labels = [parse_label(x) for x in label_strs]
        f1 = f1_score(gold_labels, pred_labels, average="macro", labels=self.label_list)
        return {"f1": f1}
