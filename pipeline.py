# def run_pipeline(claim, retriever, extractor, classifier, k_doc=5, k_sent=5):
#     # 1. Document retrieval
#     top_docs = retriever.retrieve(claim, k=k_doc)
#     # 2. Sentence/key extraction
#     key_sentences = extractor.extract(top_docs, claim, k=k_sent)
#     # 3. Stance classification
#     classified = [classifier.classify(claim, sent) for sent in key_sentences]
#     return classified
from retrievers.bm25 import BM25Retriever
import pickle

# Load your documents and ids
with open("./data/doc_level_docs.pkl", "rb") as f:
    doc_objs = pickle.load(f)
documents = [doc.page_content for doc in doc_objs]
doc_ids = [doc.metadata['id'] for doc in doc_objs]

# Path for saving/loading BM25 index
bm25_index_path = "./index/bm25_index.pkl"

# Use the index_path parameter to auto-load/save the index
bm25_retriever = BM25Retriever(documents, doc_ids=doc_ids, index_path=bm25_index_path)

# Now run retrieval as usual!
claim = "天衛三軌道在天王星內部的磁層，以《仲夏夜之夢》作者緹坦妮雅命名。"
result = bm25_retriever.retrieve(claim, k=5)

for idx, doc in enumerate(result, 1):
    print(f"{idx}. Doc ID: {doc['doc_id']}, Score: {doc['score']:.4f}")
    print(f"   {doc['text'][:100]}...\n")