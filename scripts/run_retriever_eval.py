#/scripts/run_retriever_eval.py
import os
import pickle
from retrievers.STEmbeddingRetriever import STEmbeddingRetriever  
from utils.data_utils import load_pickle_documents, load_claims
from evaluation.eval_metrics import compute_ndcg_at_k
from evaluation.eval_utils import evidence_match
from tqdm import tqdm

def dense_worker(example, retriever, topk,batch_szie=1):
    claim_text = example['claim']
    gold_evidence = example['evidence']
    
    # Collect gold document IDs
    gold_doc_ids = set()
    for group in gold_evidence:
        for item in group:
            if item and len(item) >= 3 and item[2] is not None:
                gold_doc_ids.add(str(item[2]))

    # Retrieve top documents
    top_docs = retriever.retrieve(claim_text, k=topk)
    pred_doc_ids = [str(doc['doc_id']) for doc in top_docs]

    return {
        "claim": claim_text,
        "dense_docs": top_docs,
        "gold_doc_ids": list(gold_doc_ids),
        "evidence": gold_evidence,
    }

def main(retriever,topk=100):
    cutoff_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    dense_scores_at_n = {n: {"ndcg": [], "hit": []} for n in cutoff_list}
    # Load test claims
    test_claims = load_claims("data/test.jsonl", exclude_nei=True)
    print(f"Loaded {len(test_claims)} claims from test.jsonl.")

    for example in tqdm(test_claims):
        ex = dense_worker(example, retriever, topk)
        gold_doc_ids = set(ex['gold_doc_ids'])
        gold_evidence = ex['evidence']
        dense_docs = ex['dense_docs']

        for n in cutoff_list:
            pred_doc_ids = [str(doc['doc_id']) for doc in dense_docs[:n]]
            ndcg = compute_ndcg_at_k(pred_doc_ids, gold_doc_ids, k=n)
            hit = int(evidence_match(pred_doc_ids, gold_evidence))
            dense_scores_at_n[n]['ndcg'].append(ndcg)
            dense_scores_at_n[n]['hit'].append(hit)

    # Print results
    print("\n=== Performance Table (Dense Retriever) ===")
    print(f"{'n':<6} {'NDCG@N':<12} {'Hit@N':<10}")
    for n in cutoff_list:
        ndcgs = dense_scores_at_n[n]['ndcg']
        hits = dense_scores_at_n[n]['hit']
        avg_ndcg = sum(ndcgs) / len(ndcgs)
        avg_hit = sum(hits) / len(hits)
        print(f"{n:<6} {avg_ndcg:<12.4f} {avg_hit:<10.4f}")

if __name__ == "__main__":
    # Load documents and dense retriever
    doc_objs = load_pickle_documents("data/doc_level_docs_dev.pkl")
    documents = [doc.page_content for doc in doc_objs]
    doc_ids = [doc.metadata['id'] for doc in doc_objs]
    model_name="BAAI/bge-large-zh"
    emb_path = "./embeddings/bge_large_dev.emb.npy"
    index_path = "./indexes/bge_large_index_dev.faiss"
    #BAAI/bge-large-zh: batch 512, shibing624/text2vec-base-chinese
    retriever = STEmbeddingRetriever(
        model_name=model_name,
        documents=doc_objs,
        doc_ids=doc_ids,
        index_path=index_path,
        emb_path=emb_path,
        batch_size=512,
        use_gpu=True
    )
    main(retriever,topk=100)