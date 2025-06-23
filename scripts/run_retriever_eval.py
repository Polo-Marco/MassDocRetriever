#/scripts/run_retriever_eval.py
import os
import pickle
from retrievers.STEmbeddingRetriever import STEmbeddingRetriever  
from retrievers.Qwen3EmbeddingRetriever import Qwen3EmbeddingRetriever
from utils.data_utils import load_pickle_documents, load_claims
from evaluation.eval_metrics import compute_ndcg_at_k
from evaluation.eval_utils import evidence_match, show_retrieval_metrics
from scripts.run_bm25_eval import init_worker, bm25_worker
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# ========== Multiprocessing BM25 setup ==========
bm25_global = None
doc_ids_global = None
corpus_global = None
topk_global = None



def dense_worker(example, retriever, batch_size = 1, topk=10):
    claim_text = example['claim']
    gold_evidence = example['evidence']
    gold_doc_ids = set()
    for group in gold_evidence:
        for item in group:
            if item and len(item) >= 3 and item[2] is not None:
                gold_doc_ids.add(str(item[2]))
    top_docs = retriever.retrieve(claim_text, k=topk, batch_size=batch_size)
    pred_doc_ids = [str(doc['doc_id']) for doc in top_docs]
    ndcg = compute_ndcg_at_k(pred_doc_ids, gold_doc_ids, k=topk)
    hit = int(evidence_match(pred_doc_ids, gold_evidence))
    return {
        "claim": claim_text,
        "dense_docs": top_docs,
        "gold_doc_ids": list(gold_doc_ids),
        "evidence": gold_evidence,
        "ndcg": ndcg,
        "hit": hit
    }

def get_hybrid_results(bm25_docs, dense_docs, k=10):
    seen = set()
    combined = []
    for doc in bm25_docs + dense_docs:
        if doc['doc_id'] not in seen:
            combined.append(doc)
            seen.add(doc['doc_id'])
        if len(combined) == k:
            break
    return combined

def main(mode="bm25", n_jobs=10, topk=10):
    cutoff_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
    doc_objs = load_pickle_documents("data/doc_level_docs.pkl")
    doc_ids = [doc.metadata['id'] for doc in doc_objs]
    documents = [doc.page_content for doc in doc_objs]
    test_claims = load_claims("data/test.jsonl", exclude_nei=True)
    print(f"Loaded {len(test_claims)} claims from test.jsonl.")

    results = []
    bm25_dict = {}
    # ----- BM25 (multi-process) -----
    if mode in ("bm25", "hybrid"):
        bm25_index_path = "data/bm25_index.pkl"
        with open(bm25_index_path, 'rb') as f:
            bm25 = pickle.load(f)
        with Pool(n_jobs, initializer=init_worker, initargs=(bm25, doc_ids, documents, topk)) as pool:
            bm25_results = list(pool.imap(bm25_worker, test_claims, chunksize=10))
        if mode == "bm25":
            results = bm25_results
        if mode == "hybrid":
            for ex in bm25_results:
                bm25_dict[ex["claim"]] = ex

    # ----- Dense (single-process, uses dense_worker) -----
    if mode in ("dense", "hybrid"):
        # model_name = "shibing624/text2vec-base-chinese"
        # index_path = "indexes/text2vec-base-chinese_index.faiss"
        # emb_path = "embeddings/text2vec-base-chinese.emb.npy"
        model_name="Qwen/Qwen3-Embedding-0.6B"
        emb_path = "./embeddings/qwen3_06b.emb.npy"
        index_path = "./indexes/qwen3_06b_index.faiss"
        retriever = Qwen3EmbeddingRetriever(#Qwen3EmbeddingRetriever STEmbeddingRetriever
            model_name=model_name,
            documents=doc_objs,
            doc_ids=doc_ids,
            index_path=index_path,
            emb_path=emb_path,
        )
        dense_results = []
        for example in tqdm(test_claims):
            dense_results.append(dense_worker(example, retriever, batch_size = 32,topk=topk))
        if mode == "dense":
            results = dense_results

    # ----- Hybrid Merge -----
    if mode == "hybrid":
        hybrid_results = []
        average_docs_count = 0
        for ex in dense_results:
            claim = ex["claim"]
            bm25_docs = bm25_dict[claim]["bm25_docs"]
            dense_docs = ex["dense_docs"]
            gold_doc_ids = ex["gold_doc_ids"]
            gold_evidence = ex["evidence"]
            hybrid_docs = get_hybrid_results(bm25_docs, dense_docs, k=topk)
            average_docs_count+=len(hybrid_docs)
            hybrid_results.append({
                "claim": claim,
                "hybrid_docs": hybrid_docs,
                "gold_doc_ids": gold_doc_ids,
                "evidence": gold_evidence,
            })
        print(f"Average extracted document counts: {average_docs_count/len(dense_results)}")
        results = hybrid_results

    # --- Metrics ---
    scores_at_n = {n: {'ndcg': [], 'hit': []} for n in cutoff_list}
    key_map = {
        "bm25": "bm25_docs",
        "dense": "dense_docs",
        "hybrid": "hybrid_docs"
    }
    which_key = key_map[mode]
    for ex in results:
        gold_doc_ids = set(ex["gold_doc_ids"])
        gold_evidence = ex["evidence"]
        docs = ex[which_key]
        for n in cutoff_list:
            pred_doc_ids = [str(doc['doc_id']) for doc in docs[:n]]
            scores_at_n[n]['ndcg'].append(compute_ndcg_at_k(pred_doc_ids, gold_doc_ids, k=n))
            scores_at_n[n]['hit'].append(int(evidence_match(pred_doc_ids, gold_evidence)))
    show_retrieval_metrics( cutoff_list,scores_at_n,tag = mode)

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "bm25"
    main(mode=mode, n_jobs=20, topk=100)
