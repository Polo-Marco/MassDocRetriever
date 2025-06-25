# scripts/reranker_eval.py
from tqdm import tqdm
from evaluation.eval_metrics import compute_ndcg_at_k
from evaluation.eval_utils import evidence_match,evidence_line_match
def doc_reranker_worker(example,reranker,candidates, topk=10):
    claim_text = example["claim"]
    gold_evidence = example["evidence"]
    gold_doc_ids = set()
    for group in gold_evidence:
        for item in group:
            if item and len(item) >= 3 and item[2] is not None:
                gold_doc_ids.add(str(item[2]))
    top_docs = reranker.rerank(claim_text,candidates, topk=topk)
    pred_doc_ids = [str(doc["doc_id"]) for doc in top_docs]
    ndcg = compute_ndcg_at_k(pred_doc_ids, gold_doc_ids, k=topk)
    hit = int(evidence_match(pred_doc_ids, gold_evidence))
    return {
        "claim": claim_text,
        "dense_docs": top_docs,
        "gold_doc_ids": list(gold_doc_ids),
        "evidence": gold_evidence,
        "ndcg": ndcg,
        "hit": hit,
    }
def line_reranker_worker(example, retriever, candidates=[], topk=10):
    claim_text = example["claim"]
    gold_evidence = example["evidence"]
    if not candidates:  # use gold doc id for eval
        candidates = []
        for group in gold_evidence:
            for item in group:
                if item and len(item) >= 3 and item[2] is not None:
                    candidates.append(str(item[2]))
    # Retrieve sentences (lines)
    top_lines = retriever.rerank(
        claim_text, candidates, topk=topk
    )  # Should return list of dicts
    # Prepare predicted pairs (as string)
    pred_doc_line_pairs = [
        (str(item["doc_id"]), str(item["line_id"]))
        for item in top_lines
        if "doc_id" in item and "line_id" in item
    ]
    # Prepare gold (doc_id, line_id) pairs for ndcg if needed (could be extended)
    gold_pairs = set(
        (str(span[2]), str(span[3]))
        for group in gold_evidence
        for span in group
        if len(span) >= 4 and span[2] is not None and span[3] is not None
    )
    pred_scores = [1 if pair in gold_pairs else 0 for pair in pred_doc_line_pairs]
    ndcg = sum(pred_scores) / min(len(gold_pairs), topk) if gold_pairs else 0
    # Strict hit: at least one group is fully covered
    hit = int(evidence_line_match(pred_doc_line_pairs, gold_evidence))
    return {
        "claim": claim_text,
        "dense_lines": top_lines,
        "gold_pairs": list(gold_pairs),
        "evidence": gold_evidence,
        "ndcg": ndcg,
        "hit": hit,
    }
def rerank_module(examples,reranker,tag_name = "bm25", mode="doc",topk=5):
    results = []
    for example in tqdm(examples):
        if mode == "doc":
            results.append(doc_reranker_worker(example, reranker,
                                               candidates = example[f'{retr_name}_docs'],topk=topk))
        else:
            candidates = [ex['doc_id'] for ex in example[f'{tag_name}_lines']]
            results.append(line_reranker_worker(example, reranker,
                                            candidates = example[f'{retr_name}_lines'],topk=topk))
    return results