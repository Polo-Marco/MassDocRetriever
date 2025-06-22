# pipeline.py

from retrievers.bm25 import BM25Retriever,LC_BM25Retriever
from rerankers.embedding_reranker import BertReranker,LC_BertReranker
from utils.data_utils import load_pickle_documents, load_claims
# ... (import wrappers LC_BM25Retriever, LC_BertReranker)

# Load your objects as usual
doc_objs = load_pickle_documents("data/doc_level_docs.pkl")
documents = [doc.page_content for doc in doc_objs]
doc_ids = [doc.metadata['id'] for doc in doc_objs]
bm25_index_path = "data/bm25_index.pkl"
bm25 = BM25Retriever(documents=documents,doc_ids=doc_ids, index_path=bm25_index_path)
bert_reranker = BertReranker()

# Wrap with LangChain
bm25_lc = LC_BM25Retriever(custom_bm25=bm25, k=20)
reranker_lc = LC_BertReranker(bert_reranker, topn=5)

def pipeline(query):
    # Step 1: BM25
    bm25_docs = bm25_lc.get_relevant_documents(query)
    # Step 2: BERT Rerank
    reranked_docs = reranker_lc.invoke({'query': query, 'docs': bm25_docs})
    return reranked_docs

# Example usage:
claim = "天衛三軌道在天王星內部的磁層，以《仲夏夜之夢》作者緹坦妮雅命名。"
results = pipeline(claim)
for idx, doc in enumerate(results, 1):
    print(f"{idx}. Doc ID: {doc.metadata['doc_id']}, BM25: {doc.metadata.get('score',0):.3f}, BERT: {doc.metadata.get('bert_score',0):.3f}")
    #print(doc.page_content[:100], "\n")