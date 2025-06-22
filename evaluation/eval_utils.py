def evidence_match(pred_doc_ids, gold_evidence):
    """
    Returns True if pred_doc_ids fully covers any group in gold_evidence.
    """
    for group in gold_evidence:
        required_docs = set(span[2] for span in group)
        if required_docs.issubset(set(pred_doc_ids)):
            return True
    return False