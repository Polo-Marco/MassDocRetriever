import os
import pickle
import json
from multiprocessing import Pool, cpu_count
from utils.data_utils import load_pickle_documents, load_claims
from evaluation.eval_metrics import compute_ndcg_at_k
from evaluation.eval_utils import evidence_match
from scripts.run_bm25_eval import init_worker,bm25_worker
from tqdm import tqdm
from retrievers.bm25 import BM25Retriever, LC_BM25Retriever
from rerankers.embedding_reranker import BertReranker, LC_BertReranker,Qwen3Reranker
from langchain_core.documents import Document

def modular_eval(
    n_jobs=10, topk_bm25=20, topk_rerank=5,
    bm25_index_path="data/bm25_index.pkl", docs_path="data/doc_level_docs.pkl",
    claims_path="data/test.jsonl",
    cutoff_list=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    save_path = "retrieval_eval_results.json"
):
    # Load BM25 and docs
    doc_objs = load_pickle_documents(docs_path)
    documents = [doc.page_content for doc in doc_objs]
    doc_ids = [doc.metadata['id'] for doc in doc_objs]
    with open(bm25_index_path, 'rb') as f:
        bm25 = pickle.load(f)

    # Load claims
    test_claims = load_claims(claims_path, exclude_nei=True)[:20]
    print(f"Loaded {len(test_claims)} claims from {claims_path}")

    # Multiprocessing BM25
    with Pool(n_jobs, initializer=init_worker, initargs=(bm25, doc_ids, documents, topk_bm25)) as pool:
        bm25_results = list(pool.imap(bm25_worker, test_claims, chunksize=20))

    print("BM25 retrieval done. Now evaluating reranker...")

    # Setup LangChain wrappers
    bm25_lc = LC_BM25Retriever(custom_bm25=bm25, k=topk_bm25)
    reranker = Qwen3Reranker(model_name="Qwen/Qwen3-Embedding-0.6B",batch_size=4)
    lc_reranker = LC_BertReranker(custom_reranker=reranker, topn=topk_rerank)

    # Prepare per-n cutoff score collectors
    bm25_scores_at_n = {n: {'ndcg': [], 'hit': []} for n in cutoff_list}
    rerank_scores_at_n = {n: {'ndcg': [], 'hit': []} for n in cutoff_list}

    for ex in tqdm(bm25_results):
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
        print(f"{n:<6} {bm25_ndcg:<12.4f} {bm25_hit:<10.4f} {rerank_ndcg:<14.4f} {rerank_hit:<10.4f}")
        
    # Save result
     # Save as JSON for compatibility (you can also use pickle for Python-native saving, but JSON is human-readable)
    if save_path:
        all_scores = {
            "cutoff_list": cutoff_list,
            "bm25_scores_at_n": bm25_scores_at_n,        # dict: n -> {"ndcg": [...], "hit": [...]}
            "rerank_scores_at_n": rerank_scores_at_n     # dict: n -> {"ndcg": [...], "hit": [...]}
            }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    modular_eval(n_jobs=20, topk_bm25=50, topk_rerank=50,save_path=None)