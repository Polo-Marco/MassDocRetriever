# /scripts/run_retriever_eval.py
import pickle
from multiprocessing import Pool

from tqdm import tqdm

from evaluation.eval_metrics import compute_ndcg_at_k
from evaluation.eval_utils import (evidence_line_match, evidence_match,
                                   show_retrieval_metrics)
from retrievers.Qwen3EmbeddingRetriever import Qwen3EmbeddingRetriever
from scripts.run_bm25_eval import bm25_worker, init_worker
from utils.data_utils import load_claims, load_pickle_documents


def doc_dense_worker(example, retriever, topk=10):
    claim_text = example["claim"]
    gold_evidence = example["evidence"]
    gold_doc_ids = set()
    for group in gold_evidence:
        for item in group:
            if item and len(item) >= 3 and item[2] is not None:
                gold_doc_ids.add(str(item[2]))
    top_docs = retriever.retrieve(claim_text, k=topk)
    pred_doc_ids = [str(doc["doc_id"]) for doc in top_docs]
    ndcg = compute_ndcg_at_k(pred_doc_ids, gold_doc_ids, k=topk)
    hit = int(evidence_match(pred_doc_ids, gold_evidence))
    return {
        "claim": claim_text,
        "dense_docs": top_docs,
        "gold_doc_ids": list(gold_doc_ids),
        "evidence": gold_evidence,
        "ndcg": ndcg,
        "hit": hit,
    }


def line_dense_worker(example, retriever, candidates=[], topk=10):
    claim_text = example["claim"]
    gold_evidence = example["evidence"]
    if not candidates:  # use gold doc id for eval
        candidates = []
        for group in gold_evidence:
            for item in group:
                if item and len(item) >= 3 and item[2] is not None:
                    candidates.append(str(item[2]))
    # Retrieve sentences (lines)
    top_lines = retriever.retrieve_sentence(
        claim_text, candidates, k=topk
    )  # Should return list of dicts
    # Prepare predicted pairs (as string)
    pred_doc_line_pairs = [
        (str(item["doc_id"]), str(item["line_id"]))
        for item in top_lines
        if "doc_id" in item and "line_id" in item
    ]
    # Prepare gold (doc_id, line_id) pairs for ndcg if needed (could be extended)
    gold_pairs = set(
        (str(span[2]), str(span[3]))
        for group in gold_evidence
        for span in group
        if len(span) >= 4 and span[2] is not None and span[3] is not None
    )
    pred_scores = [1 if pair in gold_pairs else 0 for pair in pred_doc_line_pairs]
    ndcg = sum(pred_scores) / min(len(gold_pairs), topk) if gold_pairs else 0
    # Strict hit: at least one group is fully covered
    hit = int(evidence_line_match(pred_doc_line_pairs, gold_evidence))
    return {
        "claim": claim_text,
        "dense_lines": top_lines,
        "gold_pairs": list(gold_pairs),
        "evidence": gold_evidence,
        "ndcg": ndcg,
        "hit": hit,
    }


def get_hybrid_results(bm25_docs, dense_docs, k=10):
    seen = set()
    combined = []
    for doc in bm25_docs + dense_docs:
        if doc["doc_id"] not in seen:
            combined.append(doc)
            seen.add(doc["doc_id"])
        if len(combined) == k:
            break
    return combined


def doc_retrieval_eval(mode="bm25", n_jobs=10, topk=10):
    cutoff_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
    doc_objs = load_pickle_documents("data/doc_level_docs.pkl")
    doc_ids = [doc.metadata["id"] for doc in doc_objs]
    documents = [doc.page_content for doc in doc_objs]
    test_claims = load_claims("data/test.jsonl", exclude_nei=True)
    print(f"Loaded {len(test_claims)} claims from test.jsonl.")

    results = []
    bm25_dict = {}
    # ----- BM25 (multi-process) -----
    if mode in ("bm25", "hybrid"):
        bm25_index_path = "data/bm25_index.pkl"
        with open(bm25_index_path, "rb") as f:
            bm25 = pickle.load(f)
        with Pool(
            n_jobs, initializer=init_worker, initargs=(bm25, doc_ids, documents, topk)
        ) as pool:
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
        retriever = (
            Qwen3EmbeddingRetriever(  # Qwen3EmbeddingRetriever STEmbeddingRetriever
                model_name="Qwen/Qwen3-Embedding-0.6B",
                documents=doc_objs,
                doc_ids=doc_ids,
                index_path="./indexes/qwen3_06b_1024_index.faiss",
                emb_path="./embeddings/qwen3_06b_1024.emb.npy",
                batch_size=64,
                max_length=1024,
            )
        )
        retriever.load_model()
        retriever.load_index()

        dense_results = []
        for example in tqdm(test_claims):
            dense_results.append(doc_dense_worker(example, retriever, topk=topk))
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
            average_docs_count += len(hybrid_docs)
            hybrid_results.append(
                {
                    "claim": claim,
                    "hybrid_docs": hybrid_docs,
                    "gold_doc_ids": gold_doc_ids,
                    "evidence": gold_evidence,
                }
            )
        print(
            f"Average extracted document counts: {average_docs_count/len(dense_results)}"
        )
        results = hybrid_results

    # --- Metrics ---
    scores_at_n = {n: {"ndcg": [], "hit": []} for n in cutoff_list}
    key_map = {"bm25": "bm25_docs", "dense": "dense_docs", "hybrid": "hybrid_docs"}
    which_key = key_map[mode]
    for ex in results:
        gold_doc_ids = set(ex["gold_doc_ids"])
        gold_evidence = ex["evidence"]
        docs = ex[which_key]
        for n in cutoff_list:
            pred_doc_ids = [str(doc["doc_id"]) for doc in docs[:n]]
            scores_at_n[n]["ndcg"].append(
                compute_ndcg_at_k(pred_doc_ids, gold_doc_ids, k=n)
            )
            scores_at_n[n]["hit"].append(
                int(evidence_match(pred_doc_ids, gold_evidence))
            )
    show_retrieval_metrics(cutoff_list, scores_at_n, tag=mode)


def sentence_retrieval_eval(topk=10):
    doc_objs = load_pickle_documents("data/sentence_level_docs.pkl")
    test_claims = load_claims("data/test.jsonl", exclude_nei=True)
    print(f"Loaded {len(test_claims)} claims from test.jsonl.")
    retriever = Qwen3EmbeddingRetriever(  # Qwen3EmbeddingRetriever STEmbeddingRetriever
        model_name="Qwen/Qwen3-Embedding-0.6B",
        documents=doc_objs,
        batch_size=128,
        max_length=256,
        use_gpu=True,
    )
    retriever.load_model()

    dense_results = []
    for example in tqdm(test_claims):
        dense_results.append(line_dense_worker(example, retriever, topk=topk))
    # prepare result for metrics
    cutoff_list = [1, 5, 10, 20, 50]
    scores_at_n = {n: {"ndcg": [], "hit": []} for n in cutoff_list}

    for ex in dense_results:
        dense_lines = ex["dense_lines"]  # List of dicts from retrieve_sentence
        gold_pairs = set(ex["gold_pairs"])
        evidence = ex["evidence"]
        # Prepare predicted (doc_id, line_id) pairs
        pred_pairs = [
            (str(item["doc_id"]), str(item["line_id"])) for item in dense_lines
        ]
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
    show_retrieval_metrics(cutoff_list, scores_at_n, tag="Line Dense")


if __name__ == "__main__":
    # doc eval
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "bm25"
    doc_retrieval_eval(mode=mode, n_jobs=20, topk=100)
    # line eval
    # sentence_retrieval_eval()
