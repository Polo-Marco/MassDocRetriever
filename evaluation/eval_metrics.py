# evaluation/eval_metrics.py

import math

def compute_dcg(relevances):
    """ relevances: list of 0/1, length=k """
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

def compute_ndcg_at_k(pred_ids, gold_ids, k=5):
    relevances = [1 if doc_id in gold_ids else 0 for doc_id in pred_ids[:k]]
    dcg = compute_dcg(relevances)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = compute_dcg(ideal_relevances)
    return dcg / idcg if idcg > 0 else 0.0