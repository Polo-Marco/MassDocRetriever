def evidence_match(pred_doc_ids, gold_evidence):
    """
    Returns True if pred_doc_ids fully covers any group in gold_evidence.
    """
    for group in gold_evidence:
        required_docs = set(span[2] for span in group)
        if required_docs.issubset(set(pred_doc_ids)):
            return True
    return False

def show_retrieval_metrics(cutoff_list,scores,tag=""):
    print(f"\n=== Performance Table ({tag.upper()}) ===")
    print(f"{'n':<6} {'NDCG':<12} {'Hit':<10}")
    for n in cutoff_list:
        avg_ndcg = sum(scores[n]['ndcg']) / len(scores[n]['ndcg'])
        avg_hit = sum(scores[n]['hit']) / len(scores[n]['hit'])
        print(f"{n:<6} {avg_ndcg:<12.4f} {avg_hit:<10.4f}")