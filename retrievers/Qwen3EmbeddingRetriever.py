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
        self.max_length = max_length
        self.batch_size = batch_size
        self.debug = debug
        self.model_name = model_name
        self.use_gpu = use_gpu

        self.corpus = documents
        self.index_path = index_path
        self.emb_path = emb_path or (
            index_path.replace(".faiss", ".emb.npy") if index_path else None
        )
        self.doc_ids = doc_ids
        self.index = None
        self.embeddings = None
        # initialize model with cpu
        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs={"torch_dtype": "auto"},
        ).cpu()
        self.tokenizer = self.model.tokenizer

        # Index and embeddings, not loaded/built by default
        self.index = None
        self.embeddings = None

    def build_index_and_emb(self, chunk_size):
        """
        Build or load FAISS index and embeddings with safe chunking & token-level truncation.
        """
        print("Building embedding index (with safe chunking)...")
        s_time = time.time()
        # turn Document object to pure text if needed
        if self.corpus and hasattr(self.corpus[0], "page_content"):
            self.corpus = [doc.page_content for doc in self.corpus]
        n_docs = len(self.corpus)
        # Infer embedding dim on CPU for safety
        temp_model = SentenceTransformer(
            self.model_name, model_kwargs={"torch_dtype": "auto"}
        ).cpu()
        test_emb = temp_model.encode(
            [self.corpus[0]],
            batch_size=1,
            max_length=self.max_length,
            normalize_embeddings=True,
        )
        emb_dim = test_emb.shape[1]
        del temp_model
        gc.collect()
        torch.cuda.empty_cache()

        # Prepare memmap for embeddings
        self.embeddings = np.memmap(
            self.emb_path, dtype=np.float32, mode="w+", shape=(n_docs, emb_dim)
        )
        self.index = faiss.IndexFlatIP(emb_dim)

        for start in range(0, n_docs, chunk_size):
            end = min(start + chunk_size, n_docs)
            chunk = self.corpus[start:end]
            # Print max token/char len in chunk
            if self.debug:
                max_char = max(len(x) for x in chunk)
                max_tok = max(len(self.tokenizer.tokenize(x)) for x in chunk)
                print(
                    f"Chunk {start}-{end}: max char len = {max_char}, max token len = {max_tok}"
                )

            chunk = safe_truncate_batch(chunk, self.tokenizer, self.max_length)

            if self.debug:
                print_cuda_memory(f"Before loading model chunk {start}-{end}")

            # Load model to GPU for this chunk only
            model = SentenceTransformer(
                self.model_name, model_kwargs={"torch_dtype": "auto"}
            )
            if self.use_gpu:
                model = model.cuda()

            if self.debug:
                print_cuda_memory(f"After loading model chunk {start}-{end}")

            # Encode and normalize
            chunk_embs = model.encode(
                chunk,
                batch_size=self.batch_size,
                max_length=self.max_length,
                show_progress_bar=True,
                normalize_embeddings=True,
                truncation=True,
            ).astype(np.float32)

            if self.debug:
                print_cuda_memory(f"After encoding chunk {start}-{end}")
                print(
                    f"chunk_embs.shape: {chunk_embs.shape}, dtype: {chunk_embs.dtype}"
                )

            self.embeddings[start:end] = chunk_embs
            self.index.add(chunk_embs)

            # Explicitly free model/tensors
            del chunk_embs
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
            if self.debug:
                print_cuda_memory(f"After gc/empty_cache chunk {start}-{end}")

            self.embeddings.flush()

        print(f"Saved embeddings to {self.emb_path}")
        if self.index_path:
            faiss.write_index(self.index, self.index_path)
            print(f"Saved FAISS index to {self.index_path}")
        print(f"Finished indexing in {time.time() - s_time:.1f} seconds")

    def load_model(self):
        print(f"Loading model {self.model_name}")
        if self.use_gpu:
            self.model = SentenceTransformer(
                self.model_name, model_kwargs={"torch_dtype": "auto"}
            ).cuda()
        else:
            self.model = SentenceTransformer(self.model_name).cpu()
            torch.cuda.empty_cache()  # release pre-loaded gpu vram
        print("Model loaded")

    def load_index(self):
        print(f"Loading FAISS index from {self.index_path} ...")
        # load embedding dimention with test embedding
        test_emb = self.model.encode(
            [""],
            batch_size=1,
            max_length=self.max_length,
            normalize_embeddings=True,
        )
        emb_dim = test_emb.shape[1]
        self.index = faiss.read_index(self.index_path)
        if self.emb_path and os.path.exists(self.emb_path):
            self.embeddings = np.memmap(
                self.emb_path,
                dtype=np.float32,
                mode="r",
                shape=(len(self.corpus), emb_dim),
            )
        print("WARNING: Ensure doc_ids and corpus order match original index build!")

        if self.use_gpu and self.index is not None:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        print("Index and embbedding loaded")

    def cleanup(self):
        """
        Safely release model (and tokenizer) from GPU/CPU memory.
        Call this when done with the instance.
        """
        try:
            if hasattr(self, "model") and self.model is not None:
                if hasattr(self.model, "cpu"):
                    self.model.cpu()  # Move to CPU first (helps free GPU instantly)
                del self.model
                self.model = None
            if hasattr(self, "tokenizer"):
                del self.tokenizer
                self.tokenizer = None
        except Exception as e:
            print(f"[WARN] Exception during cleanup: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def retrieve(self, query, k=5):
        # Embed the query
        query_emb = self.model.encode(
            [query],
            prompt_name="query",
            batch_size=1,
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

    def retrieve_sentence(self, query, candidate_doc_ids, k=5):
        """
        Retrieve top-k relevant sentences for a query from a set of candidate document IDs.

        Args:
            query (str): The search query.
            candidate_doc_ids (list): List of doc_id strings to restrict the search.
            k (int): Number of sentences to return.
            batch_size (int): Batch size for encoding.

        Returns:
            List of dicts: [{"doc_id", "line_id", "score", "text"}, ...]
        """
        # 1. Filter sentences whose doc_id is in candidate_doc_ids
        # Assumes self.corpus is a list of dicts with 'doc_id', 'line_id', 'text'
        candidate_sentences = [
            sent
            for sent in self.corpus
            if sent.metadata.get("doc_id") in candidate_doc_ids
        ]
        # 2. Embed all candidate sentences
        sent_texts = [sent.page_content for sent in candidate_sentences]
        sent_embs = self.model.encode(
            sent_texts, batch_size=self.batch_size, normalize_embeddings=True
        ).astype(np.float32)

        # 3. Embed query
        query_emb = self.model.encode(
            [query], batch_size=1, normalize_embeddings=True
        ).astype(np.float32)[
            0
        ]  # shape: (dim,)

        # 4. Cosine similarity (dot because normalized)
        sim_scores = np.dot(sent_embs, query_emb)

        # 5. Top-k
        topk_idx = np.argsort(sim_scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(topk_idx):
            sent = candidate_sentences[idx]
            results.append(
                {
                    "doc_id": sent.metadata["doc_id"],
                    "line_id": sent.metadata["line_id"],
                    "score": float(sim_scores[idx]),
                    "text": sent.page_content,
                }
            )
        return results


if __name__ == "__main__":
    # doc retrieval
    import sys

    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else "bm25"
    max_length = int(sys.argv[2]) if len(sys.argv) > 1 else "bm25"
    # Building index (first time)
    doc_objs = load_pickle_documents("data/doc_level_docs.pkl")
    documents = [doc.page_content for doc in doc_objs]
    doc_ids = [doc.metadata["id"] for doc in doc_objs]
    # Qwen/Qwen3-Embedding-0.6B
    model_name = "Qwen/Qwen3-Embedding-4B"
    emb_path = f"./embeddings/qwen3_4b_{max_length}.emb.npy"
    index_path = f"./indexes/qwen3_4b_{max_length}_index.faiss"
    retriever = Qwen3EmbeddingRetriever(
        model_name=model_name,
        documents=doc_objs,
        doc_ids=doc_ids,
        index_path=index_path,
        emb_path=emb_path,
        batch_size=batch_size,
        use_gpu=True,
        max_length=max_length,
    )
    # retriever.build_index_and_emb(chunk_size=10000)
    retriever.load_model()
    retriever.load_index()

    # Retrieval
    results = retriever.retrieve("天衛三軌道在天王星內部的磁層", k=10)
    for r in results:
        print(r["doc_id"], r["score"])

# sentence retrieval
# print("loading sentence dataset")
# sent_objs = load_pickle_documents("data/sentence_level_docs.pkl")
# print("sentence dataset loaded")
# model_name = "Qwen/Qwen3-Embedding-0.6B"
# retriever = Qwen3EmbeddingRetriever(
#     model_name=model_name,
#     documents=sent_objs,
#     batch_size=16,
#     use_gpu=True,
#     max_length=256,
# )
# retriever.load_model()
# query = "一行出家衆出家前的名字爲張遂."
# candidate_list = ["一行", "比丘"]
# results = retriever.retrieve_sentence(query, candidate_list)
# for r in results:
#     print(r["doc_id"], r["line_id"], r["score"])
