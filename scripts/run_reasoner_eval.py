from sklearn.metrics import classification_report
from tqdm import tqdm

from reasoners.QwenReasoner import QwenReasoner
from utils.data_utils import load_claims, load_pickle_documents


def reasoner_module(
    examples,  # List of claim dicts, each with "claim", "label", etc.
    reasoner=None,  # Your reasoner (must have reason_batch)
    use_evidence=True,  # Use evidence or not
    batch_size=8,  # Reasoner batch size
):
    gold_labels, pred_labels, results = [], [], []
    n = len(examples)
    for i in tqdm(range(0, n, batch_size)):
        batch_claims_ex = examples[i : i + batch_size]
        batch_claims = [ex["claim"] for ex in batch_claims_ex]
        gold_labels.extend([ex["label"].upper() for ex in batch_claims_ex])
        batch_evidence = [ex["pred_lines"] for ex in batch_claims_ex]
        batch_outputs = reasoner.reason_batch(batch_claims, batch_evidence)
        for claim_ex, output in zip(batch_claims_ex, batch_outputs):
            pred_labels.append(output["label"].upper())
            results.append(
                {
                    "claim": claim_ex["claim"],
                    "gold_label": claim_ex["label"],
                    "pred_lines": claim_ex["pred_lines"],
                    "pred_label": output["label"],
                    "reason": output["reason"],
                    "raw_output": output["raw_output"],
                    "hit": claim_ex["hit"],
                    # Add more fields if you want, e.g. "evidence"
                }
            )
    return results


def get_candidates_for_claim(claim, sent_objs, evidence):
    """
    Returns only the sentences in sent_objs that match (doc_id, line_id) pairs in claim's gold evidence.
    """
    # 1. Extract gold (doc_id, line_id) pairs as strings for comparison
    gold_pairs = set()
    for group in evidence:
        for span in group:
            if len(span) >= 4 and span[2] is not None and span[3] is not None:
                gold_pairs.add((str(span[2]), str(span[3])))

    # 2. Filter sent_objs for matching doc_id/line_id
    candidates = []
    for s in sent_objs:
        meta = getattr(s, "metadata", {})
        doc_id = str(meta.get("doc_id", ""))
        line_id = str(meta.get("line_id", ""))
        if (doc_id, line_id) in gold_pairs:
            text = getattr(s, "page_content", s.text if hasattr(s, "text") else "")
            candidates.append({"doc_id": doc_id, "line_id": line_id, "text": text})
    return candidates


def run_reasoner_eval(
    claims_path="data/test.jsonl",
    sent_path="data/sentence_level_docs.pkl",
    model_name="Qwen/Qwen3-4B",
    topk=5,
    use_evidence=True,
    language="en",
    max_new_tokens=512,
    thinking=False,
):
    print(f"Reasoning with Evidence: {use_evidence} Language: {language}")
    print(f"Loading evidence from {sent_path}")
    sent_objs = load_pickle_documents(sent_path)
    print(f"Loading claims from {claims_path}")
    claims = load_claims(claims_path, exclude_nei=use_evidence)
    print(f"Loaded {len(claims)} claims.")

    # Initialize reasoner
    reasoner = QwenReasoner(
        model_name=model_name,
        device="auto",
        with_evidence=use_evidence,
        language=language,
        max_new_tokens=max_new_tokens,
        thinking=thinking,
    )

    batch_size = 36
    outputs = []
    gold_labels, pred_labels = [], []

    for i in tqdm(range(0, len(claims), batch_size)):
        batch_claims_ex = claims[i : i + batch_size]
        batch_claims = [ex["claim"] for ex in batch_claims_ex]
        batch_gold_labels = [ex["label"].upper() for ex in batch_claims_ex]
        gold_labels.extend(batch_gold_labels)

        # Prepare evidence for each claim in batch
        batch_evidence = []
        for ex in batch_claims_ex:
            if use_evidence:
                evidence = ex["evidence"]
                candidates = get_candidates_for_claim(ex["claim"], sent_objs, evidence)
                evidence = candidates[:topk]
            else:
                evidence = []
            batch_evidence.append(evidence)

        # Batch reasoning!
        batch_outputs = reasoner.reason_batch(batch_claims, batch_evidence)
        for j, out in enumerate(batch_outputs):
            pred_labels.append(out["label"].upper())
            outputs.append(
                {
                    "claim": batch_claims[j],
                    "gold_label": batch_gold_labels[j],
                    "pred_label": out["label"],
                    "reason": out["reason"],
                    "raw_output": out["raw_output"],
                }
            )
    # Classification report
    print("=== Classification Report ===")
    print(
        classification_report(
            gold_labels, pred_labels, digits=2 if use_evidence else 3, zero_division=0
        )
    )
    # Optionally save outputs to JSONL
    import json

    filename = f"results/{model_name.split('/')[-1]}_pe_reasoner_eval_outputs_{'we' if use_evidence else 'woe'}_{language}_mnt{max_new_tokens}.jsonl"
    with open(filename, "w", encoding="utf-8") as f:
        for ex in outputs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("Saved outputs to reasoner_eval_outputs.jsonl")


if __name__ == "__main__":
    reasoner = QwenReasoner(
        model_name="Qwen/Qwen3-0.6B",
        device="cpu",
        with_evidence=True,
        language="zh",
        max_new_tokens=512,
        thinking=False,
    )
    result = reasoner_module(test_examples, reasoner)
    print(result)
    # thinkig = False
    # run_reasoner_eval(model_name ="Qwen/Qwen3-4B",use_evidence=True, language="en", max_new_tokens=512,thinking=thinkig)
    # run_reasoner_eval(model_name ="Qwen/Qwen3-4B", use_evidence=False, language="en", max_new_tokens=512,thinking=thinkig)
    # run_reasoner_eval(model_name ="Qwen/Qwen3-4B",use_evidence=True, language="zh", max_new_tokens=512,thinking=thinkig)
    # run_reasoner_eval(model_name ="Qwen/Qwen3-4B",use_evidence=False, language="zh", max_new_tokens=512,thinking=thinkig)
