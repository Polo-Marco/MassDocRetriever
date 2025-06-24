import json
import pickle
from multiprocessing import Pool

from tqdm import tqdm

from evaluation.eval_metrics import compute_ndcg_at_k
from evaluation.eval_utils import evidence_match, show_retrieval_metrics
from rerankers.embedding_reranker import Qwen3Reranker
from retrievers.Qwen3EmbeddingRetriever import Qwen3EmbeddingRetriever
from scripts.run_bm25_eval import bm25_worker, init_worker
from scripts.run_retriever_eval import doc_dense_worker
from utils.data_utils import load_claims, load_pickle_documents


def multi_process_bm25_module(n_jobs, bm25_index_path, doc_ids, documents, topk):
    with open(bm25_index_path, "rb") as f:
        bm25 = pickle.load(f)
    # Multiprocessing BM25
    with Pool(
        n_jobs, initializer=init_worker, initargs=(bm25, doc_ids, documents, topk)
    ) as pool:
        bm25_results = list(pool.imap(bm25_worker, test_claims, chunksize=20))
    print("BM25 retrieval done. Now evaluating reranker...")
    return bm25_results


def modular_eval(
    n_jobs=10,
    topk=20,
    emb_path=None,
    docs_path="data/doc_level_docs.pkl",
    claims_path="data/test.jsonl",
    cutoff_list=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    json_save_path="retrieval_eval_results.json",
):
    retr_name = "dense"
    # Load docs
    doc_objs = load_pickle_documents(docs_path)
    documents = [doc.page_content for doc in doc_objs]
    doc_ids = [doc.metadata["id"] for doc in doc_objs]
    # Load claims
    test_claims = load_claims(claims_path, exclude_nei=True)
    print(f"Loaded {len(test_claims)} claims from {claims_path}")
    # Do retrieval
    # retr_results = multi_process_bm25_module(n_jobs,"data/bm25_index.pkl",doc_ids,documents,topk)
    retriever = Qwen3EmbeddingRetriever(  # Qwen3EmbeddingRetriever STEmbeddingRetriever
        model_name="Qwen/Qwen3-Embedding-0.6B",
        documents=doc_objs,
        doc_ids=doc_ids,
        index_path="./indexes/qwen3_06b_index.faiss",
        emb_path="./embeddings/qwen3_06b.emb.npy",
    )
    retr_results = []
    for example in tqdm(test_claims):
        retr_results.append(dense_worker(example, retriever, batch_size=32, topk=topk))

    # Setup Reranker
    reranker = Qwen3Reranker(model_name="Qwen/Qwen3-Reranker-0.6B", batch_size=8)

    # Prepare per-n cutoff score collectors
    retriever_scores_at_n = {n: {"ndcg": [], "hit": []} for n in cutoff_list}
    rerank_scores_at_n = {n: {"ndcg": [], "hit": []} for n in cutoff_list}

    for ex in tqdm(retr_results):
        claim = ex["claim"]
        gold_doc_ids = set(ex["gold_doc_ids"])
        gold_evidence = ex["evidence"]
        retr_docs = ex[f"{retr_name}_docs"]

        # Convert to LangChain Documents
        reranked_docs = reranker.rerank(claim, retr_docs, topn=topk)

        for n in cutoff_list:
            # retriever scores
            pred_doc_ids_retr = [str(doc["doc_id"]) for doc in retr_docs[:n]]
            retriever_scores_at_n[n]["ndcg"].append(
                compute_ndcg_at_k(pred_doc_ids_retr, gold_doc_ids, k=n)
            )
            retriever_scores_at_n[n]["hit"].append(
                int(evidence_match(pred_doc_ids_retr, gold_evidence))
            )

            # Reranker scores
            pred_doc_ids_rerank = [str(doc["doc_id"]) for doc in reranked_docs[:n]]
            rerank_scores_at_n[n]["ndcg"].append(
                compute_ndcg_at_k(pred_doc_ids_rerank, gold_doc_ids, k=n)
            )
            rerank_scores_at_n[n]["hit"].append(
                int(evidence_match(pred_doc_ids_rerank, gold_evidence))
            )

    # Results
    # ==== Print Table ====
    show_retrieval_metrics(cutoff_list, retriever_scores_at_n, tag="retriever")
    show_retrieval_metrics(cutoff_list, rerank_scores_at_n, tag="reranker")
    # Save result
    # Save as JSON for compatibility (you can also use pickle for Python-native saving, but JSON is human-readable)
    if json_save_path:
        all_scores = {
            "cutoff_list": cutoff_list,
            "retriever_scores_at_n": retriever_scores_at_n,  # dict: n -> {"ndcg": [...], "hit": [...]}
            "rerank_scores_at_n": rerank_scores_at_n,  # dict: n -> {"ndcg": [...], "hit": [...]}
        }
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {json_save_path}")


if __name__ == "__main__":
    modular_eval(n_jobs=20, topk=50, json_save_path=None)
