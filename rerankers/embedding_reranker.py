# rerankers/embedding_reranker.py

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class BertReranker:
    def __init__(self, model_name="hfl/chinese-pert-base", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def _encode(self, texts, batch_size=4):
        # Returns a numpy array of embeddings
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
                outputs = self.model(**inputs)
                # Use [CLS] token
                if hasattr(outputs, 'pooler_output'):
                    embeddings = outputs.pooler_output
                else:
                    embeddings = outputs.last_hidden_state[:, 0, :]  # Fallback for some models
                all_embeds.append(embeddings.cpu().numpy())
        return np.vstack(all_embeds)

    def rerank(self, claim, candidate_docs, topn=5):
        # candidate_docs: list of {"doc_id", "score", "text"}
        texts = [doc['text'] for doc in candidate_docs]
        claim_embed = self._encode([claim])[0]
        doc_embeds = self._encode(texts)
        # Cosine similarity
        sim_scores = np.dot(doc_embeds, claim_embed) / (np.linalg.norm(doc_embeds, axis=1) * np.linalg.norm(claim_embed) + 1e-8)
        reranked = []
        for doc, score in zip(candidate_docs, sim_scores):
            doc = dict(doc)  # copy to avoid changing original
            doc['bert_score'] = float(score)
            reranked.append(doc)
        reranked = sorted(reranked, key=lambda x: x['bert_score'], reverse=True)
        return reranked[:topn]

#qwen3 text embedding
from sentence_transformers import SentenceTransformer
import numpy as np

class Qwen3Reranker:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-8B",batch_size=16,device=None):
        # Set device and tokenizer kwargs for best performance if you want
        model_kwargs = {}#torch_dtype=torch.float16
        self.batch_size=batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_kwargs = {"padding_side": "left","device_map": self.device}#,"max_length":"512"}
        
        self.model = SentenceTransformer(model_name, model_kwargs=model_kwargs, tokenizer_kwargs=tokenizer_kwargs)
        # Store task prompt if needed
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'
    
    def get_query_prompt(self, query):
        # Per Qwen3 docs, you can use: prompt_name="query" or give explicit prompt
        return query  # or: f"Instruct: {self.task}\nQuery:{query}"

    def rerank(self, claim, candidate_docs, topn=5):
        # candidate_docs: list of {"doc_id", "score", "text"}
        # Encode query with query prompt
        query_emb = self.model.encode(
            [self.get_query_prompt(claim)],
            prompt_name="query",
            batch_size = self.batch_size
        )  # shape: (1, D)
        doc_texts = [doc['text'] for doc in candidate_docs]
        doc_embs = self.model.encode(doc_texts, batch_size=self.batch_size)  # shape: (N, D)
        # Compute cosine similarity using sbert's function or numpy
        sim_scores = np.dot(doc_embs, query_emb[0])
        # Normalize
        sim_scores = sim_scores / (np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb[0]) + 1e-8)
        reranked = []
        for doc, score in zip(candidate_docs, sim_scores):
            doc = dict(doc)
            doc['qwen_score'] = float(score)
            reranked.append(doc)
        reranked = sorted(reranked, key=lambda x: x['qwen_score'], reverse=True)
        return reranked[:topn]
    
#langchain wrapper
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from typing import Any
from pydantic import Field

class LC_BertReranker(Runnable):
    custom_reranker: Any = Field(...)
    topn: int = Field(default=5)
    def __init__(self, custom_reranker, topn=5):
        self.custom_reranker = custom_reranker
        self.topn = topn

    def invoke(self, input, **kwargs):
        # input: {'query': str, 'docs': List[Document]}
        query = input['query']
        docs = input['docs']
        candidates = [
            {
                "doc_id": doc.metadata['doc_id'],
                "score": doc.metadata.get('bm25_score', 0),
                "text": doc.page_content
            } for doc in docs
        ]
        reranked = self.custom_reranker.rerank(query, candidates, topn=self.topn)
        return [
            Document(
                page_content=doc['text'],
                metadata=doc
            ) for doc in reranked
        ]