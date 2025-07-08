import json
from multiprocessing import Pool

from tqdm import tqdm

from evaluation.eval_metrics import (compute_ndcg_at_k,
                                     strict_classification_metrics)
from evaluation.eval_utils import (evidence_line_match, evidence_match,
                                   show_retrieval_metrics)
from reasoners.QwenReasoner import QwenReasoner
from rerankers.bert_doc_reranker.BertDocReranker import BertDocReranker
# models
from rerankers.embedding_reranker import Qwen3Reranker
from retrievers.Qwen3EmbeddingRetriever import Qwen3EmbeddingRetriever
# eval tools
from scripts.reranker_eval import rerank_module
from scripts.run_bm25_eval import (bm25_worker, init_worker,
                                   multi_process_bm25_module)
from scripts.run_reasoner_eval import reasoner_module
from scripts.run_retriever_eval import doc_dense_worker, line_dense_worker
from utils.data_utils import load_claims, load_pickle_documents


def dense_retrieval_module(examples, retriever, topk=5, mode="doc"):
    results = []
    for example in tqdm(examples):
        if mode == "doc":
            results.append(doc_dense_worker(example, retriever, topk=topk))
        else:
            # turn docs dict into candidate list
            candidates = [ex["doc_id"] for ex in example["pred_docs"]]
            results.append(
                line_dense_worker(example, retriever, candidates=candidates, topk=topk)
            )
    return results


def gather_doc_results(cutoff_list, results):
    scores_at_n = {n: {"ndcg": [], "hit": []} for n in cutoff_list}
    for ex in tqdm(results):
        if ex["label"].upper() == "NOT ENOUGH INFO":
            continue
        gold_doc_ids = set(ex["gold_doc_ids"])
        gold_evidence = ex["evidence"]
        retr_docs = ex["pred_docs"]
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


def gather_line_results(cutoff_list, results):
    scores_at_n = {n: {"ndcg": [], "hit": []} for n in cutoff_list}
    for ex in results:
        if ex["label"].upper() == "NOT ENOUGH INFO":
            continue
        lines = ex["pred_lines"]  # List of dicts from retrieve_sentence
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


def modular_eval(config):
    print(config)
    docs_path = config["docs_path"]
    line_path = config["lines_path"]
    claims_path = config["claims_path"]
    cutoff_list = config["cutoff_list"]
    json_save_path = config["json_save_path"]
    # Load docs
    doc_objs = load_pickle_documents(docs_path)
    doc_ids = [doc.metadata["id"] for doc in doc_objs]
    # Load claims
    test_claims = load_claims(claims_path, exclude_nei=False)
    print(f"Loaded {len(test_claims)} claims from {claims_path}")
    total_eval_result = {"meta": config}
    # Do doc retrieval
    if config["retriever"]["model_name"] == "bm25":
        documents = [doc.page_content for doc in doc_objs]
        retr_results = multi_process_bm25_module(
            test_claims,
            config["retriever"]["index_path"],
            doc_ids,
            documents,
            n_jobs=config["retriever"]["n_jobs"],
            topk=config["retriever"]["topk"],
        )
    # ---------TODO------- use config for modular evalaution
    else:
        retriever = (
            Qwen3EmbeddingRetriever(  # Qwen3EmbeddingRetriever STEmbeddingRetriever
                model_name=config["retriever"]["model_name"],
                documents=doc_objs,
                doc_ids=doc_ids,
                index_path=config["retriever"]["index_path"],
                emb_path=config["retriever"]["emb_path"],
                batch_size=config["retriever"]["batch_size"],
                max_length=config["retriever"]["max_length"],
                use_gpu=config["retriever"]["use_gpu"],
            )
        )
        retriever.load_model()
        retriever.load_index()
        retr_results = []
        for example in tqdm(test_claims):
            retr_results.append(
                doc_dense_worker(example, retriever, topk=config["retriever"]["topk"])
            )
        retriever.cleanup()
        del retriever
    print(retr_results[0])
    retriever_scores_at_n = gather_doc_results(
        cutoff_list,
        retr_results,
    )
    total_eval_result["retriever_scores"] = retriever_scores_at_n
    show_retrieval_metrics(cutoff_list, retriever_scores_at_n, tag="retriever")
    # Do doc Reranker
    if "Qwen" in config["reranker"]["model_name"]:
        reranker = Qwen3Reranker(
            model_name=config["reranker"]["model_name"],
            batch_size=config["reranker"]["batch_size"],
            device=config["reranker"]["device"],
        )
    else:
        reranker = BertDocReranker(
            model_name=config["reranker"]["model_name"],
            model_path=config["reranker"]["model_path"],
            device=config["reranker"]["device"],
            batch_size=config["reranker"]["batch_size"],
            debug=False,
        )
    rerank_results = rerank_module(
        retr_results, reranker, mode="doc", topk=config["reranker"]["topk"]
    )
    print(rerank_results[0])
    # Prepare per-n cutoff score collectors
    rerank_scores_at_n = gather_doc_results(
        cutoff_list,
        rerank_results,
    )
    total_eval_result["reranker_scores"] = rerank_scores_at_n
    reranker.cleanup()
    del reranker
    show_retrieval_metrics(cutoff_list, rerank_scores_at_n, tag="reranker")
    del doc_objs
    # Do sentence retrieval
    # load sentence level data
    sent_objs = load_pickle_documents(line_path)
    # prepare retriever
    line_retriever = Qwen3EmbeddingRetriever(
        model_name=config["line_retriever"]["model_name"],
        documents=sent_objs,
        batch_size=config["line_retriever"]["batch_size"],
        use_gpu=config["line_retriever"]["use_gpu"],
        max_length=config["line_retriever"]["max_length"],
    )
    line_retriever.load_model()
    line_retrieve_results = dense_retrieval_module(
        rerank_results,
        line_retriever,
        topk=config["line_retriever"]["topk"],
        mode="line",
    )
    print(line_retrieve_results[0])
    line_retriever_scores_at_n = gather_line_results(cutoff_list, line_retrieve_results)
    total_eval_result["line_retriever_scores"] = line_retriever_scores_at_n
    show_retrieval_metrics(
        cutoff_list, line_retriever_scores_at_n, tag="line retriever"
    )
    line_retriever.cleanup()
    del line_retriever
    if "Qwen" in config["line_reranker"]["model_name"]:
        line_reranker = Qwen3Reranker(
            model_name=config["line_reranker"]["model_name"],
            batch_size=config["line_reranker"]["batch_size"],
            max_length=config["line_reranker"]["max_length"],
        )
    else:
        line_reranker = BertDocReranker(
            model_name=config["line_reranker"]["model_name"],
            model_path=config["line_reranker"]["model_path"],
            device=config["line_reranker"]["device"],
            batch_size=config["line_reranker"]["batch_size"],
            debug=False,
        )
    # do sentence reranker
    line_rerank_results = rerank_module(
        line_retrieve_results,
        line_reranker,
        mode="line",
        topk=config["line_reranker"]["topk"],
    )
    print(line_rerank_results[0])
    # Prepare per-n cutoff score collectors
    line_rerank_scores_at_n = gather_line_results(cutoff_list, line_rerank_results)
    total_eval_result["line_reranker_scores"] = line_rerank_scores_at_n
    show_retrieval_metrics(cutoff_list, line_rerank_scores_at_n, tag="line reranker")
    line_reranker.cleanup()
    # for training inference
    del line_reranker
    if json_save_path:
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(total_eval_result, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {json_save_path}")
    exit()
    # Reasoner
    reasoner = QwenReasoner(
        model_name=config["reasoner"]["model_name"],
        device=config["reasoner"]["device"],
        with_evidence=True,
        language=config["reasoner"]["language"],
        max_new_tokens=config["reasoner"]["max_new_tokens"],
        thinking=False,
        exclude_nei=False,
    )
    reasoner_result = reasoner_module(line_rerank_results, reasoner)
    total_eval_result["reasoner_result"] = reasoner_result
    reasoner_score = strict_classification_metrics(reasoner_result, verbose=True)
    total_eval_result["reasoner_score"] = reasoner_score
    # Save result
    if json_save_path:
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(total_eval_result, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {json_save_path}")


import argparse

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file for modular_eval.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    modular_eval(config)
