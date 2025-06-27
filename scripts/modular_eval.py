import json
from multiprocessing import Pool

from tqdm import tqdm

from evaluation.eval_metrics import compute_ndcg_at_k
from evaluation.eval_utils import (evidence_line_match, evidence_match,
                                   show_retrieval_metrics)
from rerankers.embedding_reranker import Qwen3Reranker
from retrievers.Qwen3EmbeddingRetriever import Qwen3EmbeddingRetriever
from scripts.reranker_eval import rerank_module
from scripts.run_bm25_eval import (bm25_worker, init_worker,
                                   multi_process_bm25_module)
from scripts.run_retriever_eval import doc_dense_worker, line_dense_worker
from utils.data_utils import load_claims, load_pickle_documents


def dense_retrieval_module(examples, retriever, topk=5, mode="doc", tag_name="dense"):
    results = []
    for example in tqdm(examples):
        if mode == "doc":
            results.append(doc_dense_worker(example, retriever, topk=topk))
        else:
            # turn docs dict into candidate list
            candidates = [ex["doc_id"] for ex in example[f"{tag_name}_docs"]]
            results.append(
                line_dense_worker(example, retriever, candidates=candidates, topk=topk)
            )
    return results


def gather_doc_results(cutoff_list, results, tag_name="bm25"):
    scores_at_n = {n: {"ndcg": [], "hit": []} for n in cutoff_list}
    for ex in tqdm(results):
        if ex["label"].upper() == "NOT ENOUGH INFO":
            continue
        gold_doc_ids = set(ex["gold_doc_ids"])
        gold_evidence = ex["evidence"]
        retr_docs = ex[f"{tag_name}_docs"]
        for n in cutoff_list:
            # retriever scores
            pred_doc_ids_retr = [str(doc["doc_id"]) for doc in retr_docs[:n]]
            scores_at_n[n]["ndcg"].append(
                compute_ndcg_at_k(pred_doc_ids_retr, gold_doc_ids, k=n)
            )
            scores_at_n[n]["hit"].append(
                int(evidence_match(pred_doc_ids_retr, gold_evidence))
            )
    return scores_at_n


def gather_line_results(cutoff_list, results, tag_name="dense"):
    scores_at_n = {n: {"ndcg": [], "hit": []} for n in cutoff_list}
    for ex in results:
        if ex["label"].upper() == "NOT ENOUGH INFO":
            continue
        lines = ex[f"{tag_name}_lines"]  # List of dicts from retrieve_sentence
        gold_pairs = set(ex["gold_pairs"])
        evidence = ex["evidence"]
        # Prepare predicted (doc_id, line_id) pairs
        pred_pairs = [(str(item["doc_id"]), str(item["line_id"])) for item in lines]
        # print(pred_pairs)
        for n in cutoff_list:
            preds_n = pred_pairs[:n]
            # NDCG: how many of top n are gold? (simplified for binary relevance)
            ndcg_n = (
                sum((pair in gold_pairs) for pair in preds_n) / min(len(gold_pairs), n)
                if gold_pairs
                else 0
            )
            # Hit: does any gold group fully match?
            hit_n = int(evidence_line_match(preds_n, evidence))
            scores_at_n[n]["ndcg"].append(ndcg_n)
            scores_at_n[n]["hit"].append(hit_n)
    return scores_at_n


def modular_eval(
    n_jobs=10,
    topk=20,
    emb_path=None,
    docs_path="data/doc_level_docs.pkl",
    sent_path="data/sentence_level_docs.pkl",
    claims_path="data/test.jsonl",
    cutoff_list=[1, 2, 3, 4, 5, 10, 15],
    json_save_path="retrieval_eval_results.json",
):
    # Load docs
    doc_objs = load_pickle_documents(docs_path)
    doc_ids = [doc.metadata["id"] for doc in doc_objs]
    # Load claims
    test_claims = load_claims(claims_path, exclude_nei=False)
    print(f"Loaded {len(test_claims)} claims from {claims_path}")
    # Do doc retrieval
    # retr_results = multi_process_bm25_module(
    #     test_claims, "indexes/bm25_index.pkl", doc_ids, documents, n_jobs, topk=15
    # )
    retriever = Qwen3EmbeddingRetriever(  # Qwen3EmbeddingRetriever STEmbeddingRetriever
        model_name="Qwen/Qwen3-Embedding-0.6B",
        documents=doc_objs,
        doc_ids=doc_ids,
        index_path="./indexes/qwen3_06b_512_index.faiss",
        emb_path="./embeddings/qwen3_06b_512.emb.npy",
        batch_size=64,
        max_length=512,
    )
    retriever.load_model()
    retriever.load_index()
    retr_results = []
    for example in tqdm(test_claims):
        retr_results.append(doc_dense_worker(example, retriever, topk=topk))
    retriever_scores_at_n = gather_doc_results(
        cutoff_list, retr_results, tag_name="dense"
    )
    show_retrieval_metrics(cutoff_list, retriever_scores_at_n, tag="retriever")
    retriever.cleanup()
    del retriever
    # Do doc Reranker
    reranker = Qwen3Reranker(model_name="Qwen/Qwen3-Reranker-0.6B", batch_size=32)
    rerank_results = rerank_module(
        retr_results, reranker, tag_name="dense", mode="doc", topk=5
    )
    # Prepare per-n cutoff score collectors
    rerank_scores_at_n = gather_doc_results(
        cutoff_list, rerank_results, tag_name="dense"
    )
    reranker.cleanup()
    del reranker
    show_retrieval_metrics(cutoff_list, rerank_scores_at_n, tag="reranker")
    # Do sentence retrieval
    # load sentence level data
    sent_objs = load_pickle_documents(sent_path)
    # prepare retriever
    line_retriever = Qwen3EmbeddingRetriever(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        documents=sent_objs,
        batch_size=32,
        use_gpu=True,
        max_length=256,
    )
    line_retriever.load_model()
    line_retrieve_results = dense_retrieval_module(
        rerank_results, line_retriever, topk=15, mode="line"
    )
    line_retriever_scores_at_n = gather_line_results(
        cutoff_list, line_retrieve_results, tag_name="dense"
    )
    show_retrieval_metrics(
        cutoff_list, line_retriever_scores_at_n, tag="line retriever"
    )
    line_retriever.cleanup()
    del line_retriever
    line_reranker = Qwen3Reranker(
        model_name="Qwen/Qwen3-Reranker-0.6B", batch_size=32, max_length=256
    )
    # do sentence reranker
    line_rerank_results = rerank_module(
        line_retrieve_results, line_reranker, tag_name="dense", mode="line", topk=5
    )
    # Prepare per-n cutoff score collectors
    line_rerank_scores_at_n = gather_line_results(
        cutoff_list, line_rerank_results, tag_name="dense"
    )
    show_retrieval_metrics(cutoff_list, line_rerank_scores_at_n, tag="line reranker")
    line_reranker.cleanup()
    del line_reranker
    exit()
    # Save result
    # Save as JSON for compatibility (you can also use pickle for Python-native saving, but JSON is human-readable)
    # if json_save_path:
    #     all_scores = {
    #         "cutoff_list": cutoff_list,
    #         "retriever_scores_at_n": retriever_scores_at_n,  # dict: n -> {"ndcg": [...], "hit": [...]}
    #         "rerank_scores_at_n": rerank_scores_at_n,  # dict: n -> {"ndcg": [...], "hit": [...]}
    #     }
    #     with open(json_save_path, "w", encoding="utf-8") as f:
    #         json.dump(all_scores, f, ensure_ascii=False, indent=2)
    #     print(f"\nResults saved to {json_save_path}")


if __name__ == "__main__":
    modular_eval(n_jobs=20, topk=10, json_save_path=None)
