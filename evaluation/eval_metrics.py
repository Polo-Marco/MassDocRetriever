# evaluation/eval_metrics.py

import math

from sklearn.metrics import precision_recall_fscore_support


def compute_dcg(relevances):
    """relevances: list of 0/1, length=k"""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def compute_ndcg_at_k(pred_ids, gold_ids, k=5):
    relevances = [1 if doc_id in gold_ids else 0 for doc_id in pred_ids[:k]]
    dcg = compute_dcg(relevances)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = compute_dcg(ideal_relevances)
    return dcg / idcg if idcg > 0 else 0.0


def strict_classification_metrics(
    results, label_list=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"], verbose=True
):
    """
    Args:
        results: list of dicts, each with at least
            - gold_label
            - pred_label
            - hit (1 or 0, strict evidence match)
        label_list: list of all possible labels, e.g. ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    Returns:
        scores: dict with precision, recall, f1 (macro)
    """
    # Build strict preds: only count pred as correct if label matches AND hit==1
    y_true, y_pred = [], []
    for ex in results:
        gold = ex["gold_label"].upper()
        pred = ex["pred_label"].upper()
        hit = int(ex["hit"])
        y_true.append(gold)
        # If label matches AND hit==1, keep pred; else treat as WRONG (e.g., set to "INCORRECT" or "MISS")
        if (pred == gold) and hit == 1:
            y_pred.append(pred)
        else:
            y_pred.append("MISS")  # This will show as wrong in report

    # Set label_list if not provided
    if label_list is None:
        # All gold/pred labels except "MISS"
        label_set = set(y_true) | set(y_pred)
        label_set.discard("MISS")
        label_list = sorted(label_set)
    # Compute macro precision/recall/f1 **excluding** "MISS" label
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_list, average="macro", zero_division=0
    )
    if verbose:
        from sklearn.metrics import classification_report

        print("=== Strict Classification Report (Label + Hit==1) ===")
        print(
            classification_report(
                y_true, y_pred, labels=label_list, digits=3, zero_division=0
            )
        )
    return {"precision": precision, "recall": recall, "f1": f1, "labels": label_list}
