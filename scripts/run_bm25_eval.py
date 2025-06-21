# scripts/run_bm25_eval.py

import os
import pickle
from retrievers.bm25 import BM25Retriever
from utils.data_utils import load_pickle_documents, load_claims
from evaluation.eval_metrics import compute_ndcg_at_k
from multiprocessing import Pool, cpu_count

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
    claim_text = example['claim']
    gold_evidence = example['evidence']  # List of evidence sets
    gold_doc_ids = set()
    for group in gold_evidence:
        for item in group:
            if item and len(item) >= 3 and item[2] is not None:
                gold_doc_ids.add(str(item[2]))
    retriever = BM25Retriever(bm25=bm25_global, documents=corpus_global, doc_ids=doc_ids_global)
    top_docs = retriever.retrieve(claim_text, k=topk_global)
    pred_doc_ids = [str(doc['doc_id']) for doc in top_docs]
    ndcg = compute_ndcg_at_k(pred_doc_ids, gold_doc_ids, k=topk_global)
    # Hit = 1 if any gold doc is in top K, else 0
    hit = int(any(doc_id in gold_doc_ids for doc_id in pred_doc_ids))
    return ndcg, hit

def main(n_jobs=10, topk=10):
    # --- Load preprocessed docs and BM25 index ---
    doc_objs = load_pickle_documents("data/doc_level_docs.pkl")
    documents = [doc.page_content for doc in doc_objs]
    doc_ids = [doc.metadata['id'] for doc in doc_objs]
    bm25_index_path = "data/bm25_index.pkl"
    with open(bm25_index_path, 'rb') as f:
        bm25 = pickle.load(f)

    # --- Load claims/test set ---
    test_claims = load_claims("data/test.jsonl", exclude_nei=True)
    print(f"Loaded {len(test_claims)} claims from test.jsonl.")

    # Multiprocessing pool
    with Pool(n_jobs, initializer=init_worker, initargs=(bm25, doc_ids, documents, topk)) as pool:
        results = list(pool.imap(bm25_worker, test_claims, chunksize=20))
    
    ndcg_list, hit_list = zip(*results)
    mean_ndcg = sum(ndcg_list) / len(ndcg_list)
    hit_rate = sum(hit_list) / len(hit_list)
    print(f"Mean NDCG@{topk}: {mean_ndcg:.4f}")
    print(f"Document Hit Rate@{topk}: {hit_rate:.4f}")

if __name__ == "__main__":
    main(n_jobs=20, topk=100)