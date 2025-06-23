# retrievers/Qwen3EmbeddingRetriever.py
import gc
import os
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from utils.data_utils import load_pickle_documents


def print_cuda_memory(tag=""):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(
            f"{tag}  GPU Free: {free / 1024 ** 3:.2f} GB | Total: {total / 1024 ** 3:.2f} GB"
        )
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"  Reserved : {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")


def safe_truncate(text, tokenizer, max_tokens=512):
    """Truncate text to max_tokens using the tokenizer."""
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)


def safe_truncate_batch(docs, tokenizer, max_tokens=512):
    return [safe_truncate(doc, tokenizer, max_tokens) for doc in docs]


class Qwen3EmbeddingRetriever:
    def __init__(
        self,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        documents=None,
        doc_ids=None,
        index_path=None,
        emb_path=None,
        batch_size=32,
        chunk_size=10000,
        use_gpu=True,
        debug=True,
        max_length=512,
    ):
        """
        Build or load FAISS index and embeddings with safe chunking & token-level truncation.
        """
        self.max_length = max_length
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.debug = debug
        self.model_name = model_name
        self.use_gpu = use_gpu

        # Step 1: Prepare corpus and doc_ids
        if documents and hasattr(documents[0], "page_content"):
            self.corpus = [doc.page_content for doc in documents]
        else:
            self.corpus = documents
        self.doc_ids = doc_ids or (list(range(len(self.corpus))) if self.corpus else [])
        self.index_path = index_path
        self.emb_path = emb_path or (
            index_path.replace(".faiss", ".emb.npy") if index_path else None
        )
        self.index = None
        self.embeddings = None
        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs={"torch_dtype": "auto"},
        )
        self.tokenizer = self.model.tokenizer
        test_emb = self.model.encode(
            [self.corpus[0]],
            batch_size=1,
            max_length=self.max_length,
            normalize_embeddings=True,
        )
        emb_dim = test_emb.shape[1]
        n_docs = len(self.corpus)
        # Step 2: Load or build index/embeddings
        if index_path and os.path.exists(index_path):
            print(f"Loading FAISS index from {index_path} ...")
            self.index = faiss.read_index(index_path)
            if self.emb_path and os.path.exists(self.emb_path):
                self.embeddings = np.memmap(
                    self.emb_path, dtype=np.float32, mode="r", shape=(n_docs, emb_dim)
                )
            print(
                "WARNING: Ensure doc_ids and corpus order match original index build!"
            )

        elif self.corpus:
            print("Building embedding index (with safe chunking)...")
            s_time = time.time()
            n_docs = len(self.corpus)
            # Infer embedding dimension using CPU to avoid early GPU OOM

            # Prepare memmap
            self.embeddings = np.memmap(
                self.emb_path, dtype=np.float32, mode="w+", shape=(n_docs, emb_dim)
            )
            self.index = faiss.IndexFlatIP(emb_dim)

            # Chunked encoding loop
            for start in range(0, n_docs, self.chunk_size):
                end = min(start + self.chunk_size, n_docs)
                chunk = self.corpus[start:end]
                # Print max token/char len in chunk
                max_char = max(len(x) for x in chunk)
                max_tok = max(len(self.tokenizer.tokenize(x)) for x in chunk)
                print(
                    f"Chunk {start}-{end}: max char len = {max_char}, max token len = {max_tok}"
                )

                # Safe token-level truncation
                chunk = safe_truncate_batch(chunk, self.tokenizer, self.max_length)

                if self.debug:
                    print_cuda_memory(f"Before loading model chunk {start}-{end}")

                # Load model to GPU per chunk
                self.model = SentenceTransformer(
                    self.model_name,
                    model_kwargs={"torch_dtype": "auto"},
                )
                # model = SentenceTransformer(self.model_name, model_kwargs={"torch_dtype": "auto"})
                if self.use_gpu:
                    self.model = self.model.cuda()

                if self.debug:
                    print_cuda_memory(f"After loading model chunk {start}-{end}")

                # Encode and normalize
                try:
                    chunk_embs = self.model.encode(
                        chunk,
                        batch_size=self.batch_size,
                        max_length=self.max_length,
                        show_progress_bar=True,
                        normalize_embeddings=True,
                        truncation=True,  # ensure enforced
                    ).astype(np.float32)
                except RuntimeError:
                    print(f"OOM in chunk {start}-{end} at batch_size={self.batch_size}")
                    raise

                if self.debug:
                    print_cuda_memory(f"After encoding chunk {start}-{end}")
                    print(
                        f"chunk_embs.shape: {chunk_embs.shape}, dtype: {chunk_embs.dtype}"
                    )

                self.embeddings[start:end] = chunk_embs
                self.index.add(chunk_embs)

                # Explicitly free model and tensors to clear GPU
                del chunk_embs
                self.model.cpu()
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(1)

                if self.debug:
                    print_cuda_memory(f"After gc/empty_cache chunk {start}-{end}")

                self.embeddings.flush()

            print(f"Saved embeddings to {self.emb_path}")
            if index_path:
                faiss.write_index(self.index, index_path)
                print(f"Saved FAISS index to {index_path}")
            print(f"Finished indexing in {time.time() - s_time:.1f} seconds")
        else:
            raise ValueError(
                "Must provide either FAISS index or documents to build index."
            )

        # Step 3: Move FAISS index to GPU if needed (for search)
        if use_gpu and self.index is not None:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def retrieve(self, query, k=5, batch_size=1):
        # Embed the query
        query_emb = self.model.encode(
            [query],
            prompt_name="query",
            batch_size=batch_size,
            normalize_embeddings=True,
        ).astype(np.float32)
        # FAISS search
        D, I = self.index.search(query_emb, k)
        I = I[0]
        D = D[0]
        results = []
        for rank, idx in enumerate(I):
            if idx == -1 or idx >= len(self.doc_ids):
                continue
            results.append(
                {
                    "doc_id": self.doc_ids[idx],
                    "score": float(D[rank]),
                    "text": self.corpus[idx],
                }
            )
        return results


if __name__ == "__main__":
    # Building index (first time)
    doc_objs = load_pickle_documents("data/doc_level_docs.pkl")
    documents = [doc.page_content for doc in doc_objs]
    doc_ids = [doc.metadata["id"] for doc in doc_objs]
    # Qwen/Qwen3-Embedding-0.6B
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    emb_path = "./embeddings/qwen3_06b_1024.emb.npy"
    index_path = "./indexes/qwen3_06b_1024_index.faiss"
    retriever = Qwen3EmbeddingRetriever(
        model_name=model_name,
        documents=doc_objs,
        doc_ids=doc_ids,
        index_path=index_path,
        emb_path=emb_path,
        batch_size=30,
        use_gpu=True,
        max_length=2048,
    )

    # Retrieval
    results = retriever.retrieve("天衛三軌道在天王星內部的磁層", k=10, batch_size=1)
    for r in results:
        print(r["doc_id"], r["score"])
