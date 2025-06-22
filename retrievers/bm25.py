import os
import jieba
import pickle
from rank_bm25 import BM25Okapi
import heapq
import time
class BM25Retriever:
    def __init__(self, bm25=None, documents=None, doc_ids=None, tokenizer=None, index_path=None):
        """
        bm25: BM25Okapi object (if already built/loaded)
        documents: List[str] - document texts
        doc_ids: List[str/int] - document identifiers
        tokenizer: function to tokenize text
        index_path: path to load/save BM25 index
        """
        self.tokenizer = tokenizer or (lambda x: list(jieba.cut(x)))
        self.bm25 = bm25
        self.corpus = documents
        self.doc_ids = doc_ids or (list(range(len(documents))) if documents else [])
        self.index_path = index_path

        # Load index from disk if needed
        if self.bm25 is None:
            if index_path and os.path.exists(index_path):
                print(f"Loading BM25 index from {index_path} ...")
                with open(index_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
            elif documents:
                print("Building BM25 index ...")
                s_time = time.time()
                self.tokenized_corpus = [self.tokenizer(text) for text in self.corpus]
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                if index_path:
                    print(f"Saving BM25 index to {index_path} ...")
                    with open(index_path, 'wb') as f:
                        pickle.dump(self.bm25, f)
                    print(f"Finish indexing in {time.time()-s_time:.1f} seconds")
            else:
                raise ValueError("Must provide either bm25 object or documents to build index.")

    def retrieve(self, query, k=5):
        query_tokens = self.tokenizer(query)
        scores = self.bm25.get_scores(query_tokens)
        topk_idx = heapq.nlargest(k, range(len(scores)), scores.__getitem__)
        return [
            {
                "doc_id": self.doc_ids[idx],
                "score": float(scores[idx]),
                "text": self.corpus[idx]
            }
            for idx in topk_idx
        ]
    
# langchain wrapper
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import Any
from pydantic import Field

class LC_BM25Retriever(BaseRetriever):
    custom_bm25: Any = Field(...)
    k: int = Field(default=20)

    def get_relevant_documents(self, query):
        bm25_results = self.custom_bm25.retrieve(query, k=self.k)
        return [
            Document(
                page_content=doc['text'],
                metadata={'doc_id': doc['doc_id'], 'bm25_score': doc['score']}
            )
            for doc in bm25_results
        ]