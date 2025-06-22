# scripts/run_bm25_eval.py

import os
import pickle
from retrievers.bm25 import BM25Retriever
from utils.data_utils import load_pickle_documents, load_claims
from evaluation.eval_metrics import compute_ndcg_at_k
from multiprocessing import Pool, cpu_count
from evaluation.eval_utils import evidence_match

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
    gold_evidence = example['evidence']
    gold_doc_ids = set()
    for group in gold_evidence:
        for item in group:
            if item and len(item) >= 3 and item[2] is not None:
                gold_doc_ids.add(str(item[2]))
    retriever = BM25Retriever(bm25=bm25_global, documents=corpus_global, doc_ids=doc_ids_global)
    top_docs = retriever.retrieve(claim_text, k=topk_global)
    pred_doc_ids = [str(doc['doc_id']) for doc in top_docs]
    ndcg = compute_ndcg_at_k(pred_doc_ids, gold_doc_ids, k=topk_global)
    hit = int(evidence_match(pred_doc_ids, gold_evidence))
    return {
        "claim": claim_text,
        "bm25_docs": top_docs,
        "gold_doc_ids": list(gold_doc_ids),
        "evidence": gold_evidence,
    }

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
    
    for ex in results:
        claim = ex['claim']
        gold_doc_ids = set(ex['gold_doc_ids'])
        gold_evidence = ex['evidence']
        bm25_docs = ex['bm25_docs']

        # Convert to LangChain Documents
        lc_docs = [Document(page_content=doc['text'], metadata=doc) for doc in bm25_docs]
        reranked_docs = lc_reranker.invoke({'query': claim, 'docs': lc_docs})

        for n in cutoff_list:
            # BM25 scores
            pred_doc_ids_bm25 = [str(doc['doc_id']) for doc in bm25_docs[:n]]
            bm25_scores_at_n[n]['ndcg'].append(compute_ndcg_at_k(pred_doc_ids_bm25, gold_doc_ids, k=n))
            bm25_scores_at_n[n]['hit'].append(int(evidence_match(pred_doc_ids_bm25, gold_evidence)))

            # Reranker scores
            pred_doc_ids_rerank = [str(doc.metadata['doc_id']) for doc in reranked_docs[:n]]
            rerank_scores_at_n[n]['ndcg'].append(compute_ndcg_at_k(pred_doc_ids_rerank, gold_doc_ids, k=n))
            rerank_scores_at_n[n]['hit'].append(int(evidence_match(pred_doc_ids_rerank, gold_evidence)))

    # Results
    # ==== Print Table ====
    print("\n=== Performance Table (BM25 and Reranker) ===")
    print(f"{'n':<6} {'BM25_NDCG':<12} {'BM25_Hit':<10} {'Rerank_NDCG':<14} {'Rerank_Hit':<10}")
    for n in cutoff_list:
        bm25_ndcg = sum(bm25_scores_at_n[n]['ndcg'])/len(bm25_scores_at_n[n]['ndcg'])
        bm25_hit = sum(bm25_scores_at_n[n]['hit'])/len(bm25_scores_at_n[n]['hit'])
        rerank_ndcg = sum(rerank_scores_at_n[n]['ndcg'])/len(rerank_scores_at_n[n]['ndcg'])
        rerank_hit = sum(rerank_scores_at_n[n]['hit'])/len(rerank_scores_at_n[n]['hit'])
        print(f"{n:<6} {bm25_ndcg:<12.4f} {bm25_hit:<10.4f} {rerank_ndcg:<14.4f}")

if __name__ == "__main__":
    main(n_jobs=20, topk=100)