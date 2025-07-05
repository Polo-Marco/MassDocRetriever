# scripts/run_bm25_eval.py

import pickle
from multiprocessing import Pool

from evaluation.eval_metrics import compute_ndcg_at_k
from evaluation.eval_utils import evidence_match
from retrievers.bm25 import BM25Retriever
from utils.data_utils import load_claims, load_pickle_documents

# Global variables for worker processes
bm25_global = None
doc_ids_global = None
corpus_global = None
topk_global = None


def init_worker(bm25, doc_ids, corpus, topk):
    global bm25_global, doc_ids_global, corpus_global, topk_global
    bm25_global = bm25
    doc_ids_global = doc_ids
    corpus_global = corpus
    topk_global = topk


def bm25_worker(example):
    claim_text = example["claim"]
    gold_evidence = example["evidence"]
    gold_doc_ids = set()
    for group in gold_evidence:
        for item in group:
            if item and len(item) >= 3 and item[2] is not None:
                gold_doc_ids.add(str(item[2]))
    retriever = BM25Retriever(
        bm25=bm25_global, documents=corpus_global, doc_ids=doc_ids_global
    )
    top_docs = retriever.retrieve(claim_text, k=topk_global)
    # trn document obejct into dictionary
    top_docs = [{"doc_id": doc["doc_id"], "text": doc["text"]} for doc in top_docs]
    pred_doc_ids = [str(doc["doc_id"]) for doc in top_docs]
    ndcg = compute_ndcg_at_k(pred_doc_ids, gold_doc_ids, k=topk_global)
    hit = int(evidence_match(pred_doc_ids, gold_evidence))
    return {
        "claim": claim_text,
        "label": example["label"],
        "pred_docs": top_docs,
        "gold_doc_ids": list(gold_doc_ids),
        "evidence": gold_evidence,
        "ndcg": ndcg,
        "hit": hit,
    }


def multi_process_bm25_module(
    test_claims, bm25_index_path, doc_ids, documents, n_jobs=10, topk=5
):
    with open(bm25_index_path, "rb") as f:
        bm25 = pickle.load(f)
    # Multiprocessing BM25
    with Pool(
        n_jobs, initializer=init_worker, initargs=(bm25, doc_ids, documents, topk)
    ) as pool:
        bm25_results = list(pool.imap(bm25_worker, test_claims, chunksize=20))
    print("BM25 retrieval done.")
    return bm25_results


def main(n_jobs=10, topk=10):
    # --- Load preprocessed docs and BM25 index ---
    cutoff_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    doc_objs = load_pickle_documents("data/doc_level_docs.pkl")
    documents = [doc.page_content for doc in doc_objs]
    doc_ids = [doc.metadata["id"] for doc in doc_objs]
    bm25_index_path = "indexes/bm25_index.pkl"
    # --- Load claims/test set ---
    test_claims = load_claims("data/test.jsonl", exclude_nei=True)[:10]
    print(f"Loaded {len(test_claims)} claims from test.jsonl.")

    # Multiprocessing pool
    results = multi_process_bm25_module(
        test_claims, bm25_index_path, doc_ids, documents, n_jobs=n_jobs, topk=topk
    )
    print(results[0])
    # Prepare per-n cutoff score collectors
    bm25_scores_at_n = {n: {"ndcg": [], "hit": []} for n in cutoff_list}

    for ex in results:
        claim = ex["claim"]
        gold_doc_ids = set(ex["gold_doc_ids"])
        gold_evidence = ex["evidence"]
        bm25_docs = ex["pred_docs"]

        for n in cutoff_list:
            # BM25 scores
            pred_doc_ids_bm25 = [str(doc["doc_id"]) for doc in bm25_docs[:n]]
            bm25_scores_at_n[n]["ndcg"].append(
                compute_ndcg_at_k(pred_doc_ids_bm25, gold_doc_ids, k=n)
            )
            bm25_scores_at_n[n]["hit"].append(
                int(evidence_match(pred_doc_ids_bm25, gold_evidence))
            )

    # Results
    # ==== Print Table ====
    print("\n=== Performance Table (BM25) ===")
    print(f"{'n':<6} {'BM25_NDCG':<12} {'BM25_Hit':<10} ")
    for n in cutoff_list:
        bm25_ndcg = sum(bm25_scores_at_n[n]["ndcg"]) / len(bm25_scores_at_n[n]["ndcg"])
        bm25_hit = sum(bm25_scores_at_n[n]["hit"]) / len(bm25_scores_at_n[n]["hit"])
        print(f"{n:<6} {bm25_ndcg:<12.4f} {bm25_hit:<10.4f}")


if __name__ == "__main__":
    main(n_jobs=10, topk=100)
