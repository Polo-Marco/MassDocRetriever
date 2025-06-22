from utils.data_utils import load_pickle_documents
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import numpy as np
import faiss
import os
import time
import pickle

class STEmbeddingRetriever:
    def __init__(self, model_name="hfl/chinese-pert-base",#hfl/chinese-pert-base
                 documents=None, doc_ids=None, index_path=None, emb_path=None, batch_size=32, use_gpu=False):
        """
        model_name: HuggingFace/SBERT model name
        documents: List[str] or List[Document] - texts or Document objects
        doc_ids: List[str/int] - unique doc identifiers
        index_path: Path to load/save FAISS index
        emb_path: Path to load/save numpy embeddings
        batch_size: Batch size for encoding
        use_gpu: If True, use FAISS GPU
        """
        self.max_length=512
        self.model = SentenceTransformer(model_name)
        # If Document objects, extract .page_content
        if documents and hasattr(documents[0], "page_content"):
            self.corpus = [doc.page_content for doc in documents]
        else:
            self.corpus = documents
        self.doc_ids = doc_ids or (list(range(len(self.corpus))) if self.corpus else [])
        self.index_path = index_path
        self.emb_path = emb_path or (index_path.replace(".faiss", ".emb.npy") if index_path else None)
        self.batch_size = batch_size

        self.index = None
        self.embeddings = None

        # Load or build index
        if index_path and os.path.exists(index_path):
            print(f"Loading FAISS index from {index_path} ...")
            self.index = faiss.read_index(index_path)
            if self.emb_path and os.path.exists(self.emb_path):
                self.embeddings = np.load(self.emb_path)
        elif self.corpus:
            print("Building embedding index ...")
            s_time = time.time()
            # Always pass strings to encode!
            self.embeddings = self.model.encode(
                self.corpus, max_length = self.max_length,
                batch_size=self.batch_size, show_progress_bar=True, normalize_embeddings=True
            ).astype(np.float32)
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
            if index_path:
                faiss.write_index(self.index, index_path)
                print(f"Saved FAISS index to {index_path}")
            if self.emb_path:
                np.save(self.emb_path, self.embeddings)
                print(f"Saved embeddings to {self.emb_path}")
            print(f"Finish indexing in {time.time() - s_time:.1f} seconds")
        else:
            raise ValueError("Must provide either FAISS index or documents to build index.")

        # Move to GPU if needed
        if use_gpu and self.index is not None:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def retrieve(self, query, k=5, batch_size=1):
        # Embed the query (must be string)
        query_emb = self.model.encode(
            [query], normalize_embeddings=True,batch_size=batch_size
        ).astype(np.float32)
        # FAISS search
        D, I = self.index.search(query_emb, k)
        I = I[0]
        D = D[0]
        return [
            {
                "doc_id": self.doc_ids[idx],
                "score": float(D[rank]),   # Cosine similarity (since normalized)
                "text": self.corpus[idx]
            }
            for rank, idx in enumerate(I)
        ]
if __name__=="__main__":
    # Building index (first time)
    doc_objs = load_pickle_documents("data/doc_level_docs.pkl")
    documents = [doc.page_content for doc in doc_objs]
    doc_ids = [doc.metadata['id'] for doc in doc_objs]
    model_name="BAAI/bge-large-zh"
    emb_path = "./embeddings/bge_large.emb.npy"
    index_path = "./indexes/bge_large_index.faiss"
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

    # Later, loading index (no need to provide documents)
    # retriever = STEmbeddingRetriever(
    #     model_name=model_name,
    #     documents=doc_objs,
    #     doc_ids=doc_ids,
    #     index_path=index_path,
    #     emb_path=emb_path,
    # )

    # Retrieval
    results = retriever.retrieve("天衛三軌道在天王星內部的磁層", k=10,batch_size=1)
    for r in results:
        print(r["doc_id"], r["score"], r["text"][:5])